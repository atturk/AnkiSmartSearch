"""Background Qt workers for indexing, searching, and collection operations.

Prevents freezing Anki's main thread while communicating progress through Qt signals.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

try:
    from aqt.qt import QObject, QThread, pyqtSignal  # type: ignore
except ImportError:
    try:
        from PyQt6.QtCore import QObject, QThread, pyqtSignal  # type: ignore
    except ImportError:
        import threading

        class QThread:  # type: ignore
            def __init__(self, parent=None):
                self._thread = None

            def start(self):
                self._thread = threading.Thread(target=self.run, daemon=True)
                self._thread.start()

            def wait(self, timeout=None):
                if self._thread:
                    self._thread.join(timeout=timeout)

        class _BoundSignal:
            def __init__(self):
                self._slots = []

            def connect(self, slot):
                self._slots.append(slot)

            def emit(self, *args, **kwargs):
                for slot in list(self._slots):
                    slot(*args, **kwargs)

        class pyqtSignal:  # type: ignore
            def __init__(self, *types):
                self._types = types
                self._name = None

            def __set_name__(self, owner, name):
                self._name = f"__sig_{name}"

            def __get__(self, instance, owner):
                if instance is None:
                    return self
                if not hasattr(instance, self._name):
                    setattr(instance, self._name, _BoundSignal())
                return getattr(instance, self._name)


from ..ai_client import OpenAICompatibleClient
from ..config_manager import ConfigManager
from ..deck_ops import (
    RollbackResult,
    RunResult,
    _get_op_changes,
    assign_notes_to_best_units,
    create_batch_filtered_decks_op,
    execute_unit_deck_creation,
    rollback_run,
    rollback_run_op,
)
from ..indexer import DeckIndexer, collect_presearch_data
from ..search_pipeline import SmartSearchPipeline, concepts_from_keywords
from ..storage import SearchDatabase
from ..text_utils import parse_markdown_units, parse_keyword_units


def build_browse_result(units_search_data: List[Dict[str, Any]], warning: str = "") -> Dict[str, Any]:
    """Merges per-unit matches into one relevance-ordered list for a search-only run.

    Each note keeps its best score across units. Nothing is written to the collection.
    """
    best: Dict[int, float] = {}
    unit_errors: List[Dict[str, str]] = []
    for data in units_search_data:
        scores = data.get("note_scores", {}) or {}
        for nid in data.get("matched_nids", []) or []:
            sc = float(scores.get(nid, 0.0))
            if nid not in best or sc > best[nid]:
                best[nid] = sc
        if data.get("unit_error"):
            unit = data.get("unit")
            unit_errors.append(
                {"unit_title": getattr(unit, "unit_title", ""), "error": str(data["unit_error"])}
            )
    matched = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return {
        "status": "done",
        "browse_only": True,
        "matched": matched,
        "unit_errors": unit_errors,
        "warning": warning,
        "units_processed": len(units_search_data),
        "results": [],
    }


class IndexWorker(QThread):
    """Background worker for incremental note and embedding indexing."""

    progress_sig = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(dict)
    error_sig = pyqtSignal(str)

    def __init__(
        self,
        col: Any,
        config: ConfigManager,
        storage: SearchDatabase,
        ai_client: OpenAICompatibleClient,
        selected_decks: Optional[List[str]] = None,
        parent: Optional[Any] = None,
        precomputed_notes_data: Optional[Any] = None,
        main_thread_runner: Optional[Callable[[Callable[[], Any]], Any]] = None,
        search_query: Optional[str] = None,
    ):
        super().__init__(parent)
        self.col = col
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.selected_decks = selected_decks
        self.precomputed_notes_data = precomputed_notes_data
        self.main_thread_runner = main_thread_runner
        self.search_query = search_query
        self._is_cancelled = False
        self.indexer = DeckIndexer(self.config, self.storage, self.ai_client)

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:

            def _on_progress(current: int, total: int, msg: str) -> None:
                self.progress_sig.emit(current, total, msg)

            def _check_cancel() -> bool:
                return self._is_cancelled

            notes_data = self.precomputed_notes_data
            effective_q = (
                self.search_query
                if self.search_query is not None
                else (self.config.get_effective_search_query() if self.selected_decks is None else None)
            )
            if notes_data is None and self.main_thread_runner is not None:
                box: Dict[str, Any] = {}

                def _do_scan():
                    if effective_q:
                        nids = self.indexer.get_notes_for_query(self.col, effective_q)
                    else:
                        dids = self.indexer.find_target_deck_ids(
                            self.col, self.selected_decks or self.config.selected_decks
                        )
                        nids = self.indexer.get_notes_in_decks(self.col, dids)
                    box["data"] = self.indexer.scan_notes_metadata(self.col, nids, cancel_check=_check_cancel)

                self.main_thread_runner(_do_scan)
                notes_data = box.get("data")

            res = self.indexer.index_selected_decks(
                col=self.col,
                selected_deck_names=self.selected_decks,
                progress_callback=_on_progress,
                cancel_check=_check_cancel,
                search_query=effective_q,
                precomputed_notes_data=notes_data,
            )
            self.finished_sig.emit(res)
        except Exception as e:
            self.error_sig.emit(str(e))


class SearchBatchWorker(QThread):
    """Background worker executing the search pipeline:

    1. Markdown parsing
    2. Auto-indexing any missing embeddings for selected decks
    3. Multi-stage concept extraction & hybrid search per unit
    4. Safe filtered deck creation delegated to the main thread with CollectionOp / Undo entry
    """

    progress_sig = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(dict)
    error_sig = pyqtSignal(str)

    def __init__(
        self,
        col: Any,
        config: ConfigManager,
        storage: SearchDatabase,
        ai_client: OpenAICompatibleClient,
        raw_text: str,
        base_deck_name: str,
        parent: Optional[Any] = None,
        precomputed_data: Optional[Dict[str, Any]] = None,
        main_thread_runner: Optional[Callable[[Callable[[], Any]], Any]] = None,
        profile: str = "",
        input_mode: str = "text",
        create_deck: bool = True,
    ):
        super().__init__(parent)
        self.col = col
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.raw_text = raw_text
        self.base_deck_name = base_deck_name
        self.precomputed_data = precomputed_data
        self.main_thread_runner = main_thread_runner
        self.input_mode = input_mode
        # False = search only: no unsuspension, no filtered deck, no history entry;
        # the dialog shows the matches in the Browser ordered by relevance.
        self.create_deck = create_deck
        self.last_warning: str = ""
        if not profile:
            try:
                from aqt import mw
                if mw and hasattr(mw, "pm") and mw.pm and hasattr(mw.pm, "name"):
                    profile = str(mw.pm.name or "")
            except Exception:
                profile = ""
        self.profile = profile
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            # 1. Parse markdown units
            if self.input_mode == "keyword":
                units = parse_keyword_units(
                    raw_text=self.raw_text,
                    base_deck_name=self.base_deck_name,
                    naming_style=self.config.batch_deck_naming_style,
                )
                if not units:
                    self.error_sig.emit("Nessuna keyword trovata.")
                    return
            else:
                units = parse_markdown_units(
                    raw_text=self.raw_text,
                    base_deck_name=self.base_deck_name,
                    naming_style=self.config.batch_deck_naming_style,
                )
                if not units:
                    self.error_sig.emit("Nessun testo o unità valida trovata nel testo inserito.")
                    return

            total_units = len(units)
            total_steps = total_units * 100
            self._last_progress_val = 0

            def emit_prog(val: int, msg: str) -> None:
                v = max(self._last_progress_val, min(val, total_steps))
                self._last_progress_val = v
                self.progress_sig.emit(v, total_steps, msg)

            emit_prog(0, f"Rilevate {total_units} unità di studio.")

            # 2. Check and ensure target query notes are indexed
            if self.precomputed_data is not None:
                candidate_nids = self.precomputed_data["candidate_nids"]
                suspended_nids = self.precomputed_data["suspended_nids"]
                allowed_cids = self.precomputed_data["allowed_cids"]
                parent_deck_prefix = self.precomputed_data["parent_deck_prefix"]
                precomputed_notes = self.precomputed_data.get("precomputed_notes_data")
            else:
                if self.main_thread_runner is not None:
                    data_box: Dict[str, Any] = {}

                    def _do_reads():
                        data_box["data"] = collect_presearch_data(
                            self.col, self.config, self.storage, self.ai_client
                        )

                    self.main_thread_runner(_do_reads)
                    r_data = data_box["data"]
                else:
                    r_data = collect_presearch_data(
                        self.col, self.config, self.storage, self.ai_client
                    )
                candidate_nids = r_data["candidate_nids"]
                suspended_nids = r_data["suspended_nids"]
                allowed_cids = r_data["allowed_cids"]
                parent_deck_prefix = r_data["parent_deck_prefix"]
                precomputed_notes = r_data["precomputed_notes_data"]

            indexer = DeckIndexer(self.config, self.storage, self.ai_client)
            active_query = self.config.get_effective_search_query()

            def _check_cancel() -> bool:
                return self._is_cancelled

            def _on_idx_progress(cur: int, tot: int, msg: str) -> None:
                emit_prog(0, f"[Verifica Indice] {msg}")

            idx_res = indexer.index_selected_decks(
                col=self.col,
                selected_deck_names=self.config.selected_decks,
                progress_callback=_on_idx_progress,
                cancel_check=_check_cancel,
                search_query=active_query,
                precomputed_notes_data=precomputed_notes,
            )

            if self._is_cancelled:
                self.finished_sig.emit({"status": "cancelled", "units_processed": 0, "results": []})
                return

            active_model_id = self.config.get_active_model_id()
            pipeline = SmartSearchPipeline(self.config, self.storage, self.ai_client)

            search_id = getattr(self, "search_id", "") or f"search_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            self.search_id = search_id
            precomputed = getattr(self, "precomputed_concepts", None)

            translations: Dict[str, str] = {}
            keyword_warning = ""
            if self.input_mode == "keyword" and not precomputed:
                should_translate = bool(getattr(self.config, "keyword_translate", True))
                chat_configured = bool(self.config.chat_model and self.config.chat_model.strip())
                if should_translate:
                    if not chat_configured:
                        keyword_warning = "Traduzione non riuscita: keyword usate senza traduzione."
                    else:
                        all_kws = [k for u in units for k in getattr(u, "keywords", []) if k.strip()]
                        n_kws = len(all_kws)
                        model_name = self.config.chat_model or "LLM"

                        stop_heartbeat = threading.Event()
                        start_time = time.time()

                        def _heartbeat():
                            while not stop_heartbeat.wait(1.0):
                                elapsed = int(time.time() - start_time)
                                emit_prog(
                                    0,
                                    f"Traduzione di {n_kws} keyword con {model_name}… {elapsed} s",
                                )

                        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
                        hb_thread.start()

                        try:
                            emit_prog(0, f"Traduzione di {n_kws} keyword con {model_name}…")
                            translations = pipeline.translate_keywords(units)
                        except Exception as e:
                            translations = {}
                            keyword_warning = "Traduzione non riuscita: keyword usate senza traduzione."
                        finally:
                            stop_heartbeat.set()
                            hb_thread.join(timeout=1.0)

            self.last_warning = keyword_warning

            if self._is_cancelled:
                self.finished_sig.emit({"status": "cancelled", "units_processed": 0, "results": []})
                return

            units_search_data: List[Dict[str, Any]] = []

            for idx, unit in enumerate(units, start=1):
                if self._is_cancelled:
                    break

                unit_base = (idx - 1) * 100
                prefix = f"[{idx}/{total_units}] " if total_units > 1 else ""

                parsed = urlparse(self.config.chat_base_url)
                host = parsed.netloc or parsed.path or self.config.chat_base_url
                model_name = self.config.chat_model or "LLM"

                # Search pipeline (re-use precomputed concepts if provided to skip Stage 1)
                if precomputed and unit.unit_title in precomputed:
                    concepts = precomputed[unit.unit_title]
                    emit_prog(unit_base + 70, f"{prefix}Riutilizzo di {len(concepts)} concetti: {unit.unit_title}")
                    unit_error = ""
                elif self.input_mode == "keyword":
                    concepts = concepts_from_keywords(unit.keywords, translations=translations)
                    unit_error = ""
                else:
                    # Sotto-passo 1: Invio richiesta (10%)
                    emit_prog(unit_base + 10, f"{prefix}Invio richiesta a {model_name} ({host})…")

                    # Sotto-passo 2: In attesa della risposta dell'IA (60%)
                    stop_heartbeat = threading.Event()
                    start_time = time.time()

                    def _heartbeat():
                        while not stop_heartbeat.wait(1.0):
                            # La barra resta ferma al passo "richiesta inviata": durante l'attesa
                            # avanza solo il cronometro nel messaggio (nessuna percentuale inventata).
                            elapsed = int(time.time() - start_time)
                            emit_prog(
                                unit_base + 10,
                                f"{prefix}In attesa della risposta dell'IA… {elapsed} s",
                            )

                    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
                    hb_thread.start()

                    try:
                        concepts = pipeline.extract_concepts(unit)
                        unit_error = ""
                    except Exception as e:
                        concepts = []
                        unit_error = str(e)
                    finally:
                        stop_heartbeat.set()
                        hb_thread.join(timeout=1.0)

                    emit_prog(unit_base + 70, f"{prefix}Risposta ricevuta: {len(concepts)} concetti")

                if not concepts:
                    scored_notes = []
                    pipeline.last_concept_coverage = []
                else:
                    # Sotto-passo 3: Ricerca vettoriale (20%)
                    num_notes = len(candidate_nids) if candidate_nids is not None else 0
                    if num_notes == 0:
                        try:
                            num_notes = self.storage.count_embeddings(active_model_id)
                        except Exception:
                            num_notes = 0

                    if self.input_mode == "keyword":
                        emit_prog(unit_base + 75, f"{prefix}Ricerca vettoriale di {len(concepts)} keyword tra {num_notes} note…")
                    else:
                        emit_prog(unit_base + 75, f"{prefix}Ricerca vettoriale tra {num_notes} note…")

                    if getattr(self, "override_min_threshold", None) is not None:
                        eff_min_threshold = self.override_min_threshold
                    elif self.input_mode == "keyword":
                        eff_min_threshold = self.config.get_keyword_min_similarity_threshold()
                    else:
                        eff_min_threshold = None

                    scored_notes = pipeline.retrieve_for_concepts(
                        concepts=concepts,
                        candidate_nids=candidate_nids,
                        active_model_id=active_model_id,
                        suspended_nids=suspended_nids,
                        min_threshold=eff_min_threshold,
                        max_notes_per_concept=getattr(self, "override_max_notes_per_concept", None),
                    )

                    if self.input_mode != "keyword" and self.config.enable_llm_verification:
                        scored_notes = pipeline.verify_candidates(unit, scored_notes, concepts=concepts)
                        surviving_nids = {c.nid for c in scored_notes}
                        min_threshold = self.config.get_min_similarity_threshold()
                        for cov in pipeline.last_concept_coverage:
                            cov.matched_nids = [n for n in cov.matched_nids if n in surviving_nids]
                            cov.matched_notes_count = len(cov.matched_nids)
                            if cov.matched_notes_count == 0:
                                cov.status = "uncovered"
                            elif cov.matched_notes_count == 1 or cov.best_score < (min_threshold + 0.06):
                                cov.status = "partial"
                            else:
                                cov.status = "covered"

                    emit_prog(unit_base + 90, f"{prefix}Ricerca completata ({len(scored_notes)} note)")

                matched_nids = [sn.nid for sn in scored_notes]
                note_scores = {sn.nid: sn.score for sn in scored_notes}
                cov_dicts = [c.to_dict() for c in pipeline.last_concept_coverage]

                units_search_data.append(
                    {
                        "unit": unit,
                        "matched_nids": matched_nids,
                        "note_scores": note_scores,
                        "concept_coverage": cov_dicts,
                        "unit_error": unit_error,
                        "search_id": search_id,
                    }
                )

            if self._is_cancelled:
                self.finished_sig.emit({"status": "cancelled", "units_processed": 0, "results": []})
                return

            if not self.create_deck:
                self.finished_sig.emit(build_browse_result(units_search_data, self.last_warning))
                return

            # Disambiguate notes across units: assign each note exclusively to the unit where it scored highest
            assign_notes_to_best_units(units_search_data)

            # Sotto-passo 4: Creazione mazzo filtrato (10%)
            emit_prog(total_units * 95, "Creazione mazzo filtrato…")

            # Delegate collection writes to the main thread with undo entry
            self._delegate_deck_creations(
                units_search_data=units_search_data,
                parent_deck_prefix=parent_deck_prefix,
                allowed_cids=allowed_cids if allowed_cids else None,
                total_units=total_units,
                total_steps=total_steps,
                emit_prog=emit_prog,
            )

        except Exception as e:
            self.error_sig.emit(str(e))

    def _delegate_deck_creations(
        self,
        units_search_data: List[Dict[str, Any]],
        parent_deck_prefix: Optional[str],
        allowed_cids: Optional[Set[int]],
        total_units: int,
        total_steps: Optional[int] = None,
        emit_prog: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        if total_steps is None:
            total_steps = total_units * 100

        def _finish_progress():
            if emit_prog is not None:
                emit_prog(total_steps, "Completato con successo!")
            else:
                self.progress_sig.emit(total_steps, total_steps, "Completato con successo!")

        # Case A: Injected runner (tests or custom host)
        if self.main_thread_runner is not None:

            def _injected_task():
                try:
                    results, changes = create_batch_filtered_decks_op(
                        col=self.col,
                        units_data=units_search_data,
                        config=self.config,
                        storage=self.storage,
                        parent_deck_prefix=parent_deck_prefix,
                        allowed_cids=allowed_cids,
                        profile=self.profile,
                        search_id=getattr(self, "search_id", ""),
                    )
                    _finish_progress()
                    self.finished_sig.emit(
                        {
                            "status": "cancelled" if self._is_cancelled else "done",
                            "units_processed": len(results),
                            "results": results,
                            "warning": getattr(self, "last_warning", ""),
                        }
                    )
                except Exception as e:
                    self.error_sig.emit(str(e))

            self.main_thread_runner(_injected_task)
            return

        # Case B: Inside Anki environment
        try:
            from aqt import mw  # type: ignore
            from aqt.operations import CollectionOp  # type: ignore
        except ImportError:
            mw = None
            CollectionOp = None

        if mw is not None and hasattr(mw, "taskman"):

            def _on_main():
                try:
                    if CollectionOp is not None:
                        op_results: List[RunResult] = []

                        def _col_op(col: Any):
                            res, changes = create_batch_filtered_decks_op(
                                col=col,
                                units_data=units_search_data,
                                config=self.config,
                                storage=self.storage,
                                parent_deck_prefix=parent_deck_prefix,
                                allowed_cids=allowed_cids,
                                profile=self.profile,
                                search_id=getattr(self, "search_id", ""),
                            )
                            op_results.extend(res)
                            return _get_op_changes(changes)

                        def _on_success(changes: Any):
                            _finish_progress()
                            self.finished_sig.emit(
                                {
                                    "status": "cancelled" if self._is_cancelled else "done",
                                    "units_processed": len(op_results),
                                    "results": op_results,
                                    "warning": getattr(self, "last_warning", ""),
                                }
                            )

                        def _on_failure(exc: Exception):
                            self.error_sig.emit(str(exc))

                        CollectionOp(parent=mw, op=_col_op).success(_on_success).failure(_on_failure).run_in_background()
                    else:
                        results, changes = create_batch_filtered_decks_op(
                            col=self.col,
                            units_data=units_search_data,
                            config=self.config,
                            storage=self.storage,
                            parent_deck_prefix=parent_deck_prefix,
                            allowed_cids=allowed_cids,
                            profile=self.profile,
                        )
                        _finish_progress()
                        self.finished_sig.emit(
                            {
                                "status": "cancelled" if self._is_cancelled else "done",
                                "units_processed": len(results),
                                "results": results,
                                "warning": getattr(self, "last_warning", ""),
                            }
                        )
                except Exception as e:
                    self.error_sig.emit(str(e))

            mw.taskman.run_on_main(_on_main)
            return

        # Case C: Fallback outside Anki (direct execution)
        try:
            results, changes = create_batch_filtered_decks_op(
                col=self.col,
                units_data=units_search_data,
                config=self.config,
                storage=self.storage,
                parent_deck_prefix=parent_deck_prefix,
                allowed_cids=allowed_cids,
            )
            _finish_progress()
            self.finished_sig.emit(
                {
                    "status": "cancelled" if self._is_cancelled else "done",
                    "units_processed": len(results),
                    "results": results,
                    "warning": getattr(self, "last_warning", ""),
                }
            )
        except Exception as e:
            self.error_sig.emit(str(e))


class RollbackWorker(QThread):
    """Background worker for safely rolling back a past search run."""

    finished_sig = pyqtSignal(object)
    error_sig = pyqtSignal(str)

    def __init__(
        self,
        col: Any,
        storage: SearchDatabase,
        run_id: str,
        keep_reviewed: bool = True,
        parent: Optional[Any] = None,
        main_thread_runner: Optional[Callable[[Callable[[], Any]], Any]] = None,
        current_profile: str = "",
    ):
        super().__init__(parent)
        self.col = col
        self.storage = storage
        self.run_id = run_id
        self.keep_reviewed = keep_reviewed
        self.main_thread_runner = main_thread_runner
        if not current_profile:
            try:
                from aqt import mw
                if mw and hasattr(mw, "pm") and mw.pm and hasattr(mw.pm, "name"):
                    current_profile = str(mw.pm.name or "")
            except Exception:
                current_profile = ""
        self.current_profile = current_profile

    def run(self) -> None:
        try:
            self._delegate_rollback()
        except Exception as e:
            self.error_sig.emit(str(e))

    def _delegate_rollback(self) -> None:
        # Case A: Injected runner (tests or custom host)
        if self.main_thread_runner is not None:

            def _injected_task():
                try:
                    res, changes = rollback_run_op(
                        col=self.col,
                        storage=self.storage,
                        run_id=self.run_id,
                        keep_reviewed=self.keep_reviewed,
                        current_profile=self.current_profile,
                    )
                    self.finished_sig.emit(res)
                except Exception as e:
                    self.error_sig.emit(str(e))

            self.main_thread_runner(_injected_task)
            return

        # Case B: Inside Anki environment
        try:
            from aqt import mw  # type: ignore
            from aqt.operations import CollectionOp  # type: ignore
        except ImportError:
            mw = None
            CollectionOp = None

        if mw is not None and hasattr(mw, "taskman"):

            def _on_main():
                try:
                    if CollectionOp is not None:
                        rollback_holder: List[RollbackResult] = []

                        def _col_op(col: Any):
                            r, changes = rollback_run_op(
                                col=col,
                                storage=self.storage,
                                run_id=self.run_id,
                                keep_reviewed=self.keep_reviewed,
                                current_profile=self.current_profile,
                            )
                            rollback_holder.append(r)
                            return _get_op_changes(changes)

                        def _on_success(changes: Any):
                            r = rollback_holder[0] if rollback_holder else RollbackResult(
                                run_id=self.run_id, deck_removed=False, resuspended_count=0, kept_reviewed_count=0
                            )
                            self.finished_sig.emit(r)

                        def _on_failure(exc: Exception):
                            self.error_sig.emit(str(exc))

                        CollectionOp(parent=mw, op=_col_op).success(_on_success).failure(_on_failure).run_in_background()
                    else:
                        res, changes = rollback_run_op(
                            col=self.col,
                            storage=self.storage,
                            run_id=self.run_id,
                            keep_reviewed=self.keep_reviewed,
                            current_profile=self.current_profile,
                        )
                        self.finished_sig.emit(res)
                except Exception as e:
                    self.error_sig.emit(str(e))

            mw.taskman.run_on_main(_on_main)
            return

        # Case C: Fallback outside Anki (direct execution)
        try:
            res, changes = rollback_run_op(
                col=self.col,
                storage=self.storage,
                run_id=self.run_id,
                keep_reviewed=self.keep_reviewed,
                current_profile=self.current_profile,
            )
            self.finished_sig.emit(res)
        except Exception as e:
            self.error_sig.emit(str(e))
