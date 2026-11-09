"""Background Qt workers for indexing, searching, and collection operations.

Prevents freezing Anki's main thread while communicating progress through Qt signals.
"""

from __future__ import annotations

import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

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
from ..search_pipeline import SmartSearchPipeline
from ..storage import SearchDatabase
from ..text_utils import parse_markdown_units


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
    ):
        super().__init__(parent)
        self.col = col
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.selected_decks = selected_decks
        self.precomputed_notes_data = precomputed_notes_data
        self.main_thread_runner = main_thread_runner
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
            if notes_data is None and self.main_thread_runner is not None:
                box: Dict[str, Any] = {}

                def _do_scan():
                    q = self.config.get_effective_search_query() if self.selected_decks is None else None
                    if q:
                        nids = self.indexer.get_notes_for_query(self.col, q)
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
                search_query=self.config.get_effective_search_query() if self.selected_decks is None else None,
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
            units = parse_markdown_units(
                raw_text=self.raw_text,
                base_deck_name=self.base_deck_name,
                naming_style=self.config.batch_deck_naming_style,
            )
            if not units:
                self.error_sig.emit("Nessun testo o unità valida trovata nel testo inserito.")
                return

            total_units = len(units)
            self.progress_sig.emit(0, total_units, f"Rilevate {total_units} unità di studio.")

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
                self.progress_sig.emit(0, total_units, f"[Verifica Indice] {msg}")

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

            units_search_data: List[Dict[str, Any]] = []

            for idx, unit in enumerate(units, start=1):
                if self._is_cancelled:
                    break

                self.progress_sig.emit(
                    idx - 1,
                    total_units,
                    f"[{idx}/{total_units}] Elaborazione IA: {unit.unit_title}...",
                )

                # Search pipeline (re-use precomputed concepts if provided to skip Stage 1)
                if precomputed and unit.unit_title in precomputed:
                    concepts = precomputed[unit.unit_title]
                    scored_notes = pipeline.retrieve_for_concepts(
                        concepts=concepts,
                        candidate_nids=candidate_nids,
                        active_model_id=active_model_id,
                        suspended_nids=suspended_nids,
                        min_threshold=getattr(self, "override_min_threshold", None),
                        max_notes_per_concept=getattr(self, "override_max_notes_per_concept", None),
                    )
                    unit_error = ""
                else:
                    scored_notes = pipeline.search_unit(
                        unit=unit,
                        candidate_nids=candidate_nids,
                        active_model_id=active_model_id,
                        suspended_nids=suspended_nids,
                    )
                    unit_error = getattr(pipeline, "last_unit_error", "")

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

            # Disambiguate notes across units: assign each note exclusively to the unit where it scored highest
            assign_notes_to_best_units(units_search_data)

            self.progress_sig.emit(
                total_units,
                total_units,
                "Applicazione modifiche alla collezione (creazione mazzi)...",
            )

            # 4. Delegate collection writes to the main thread with undo entry
            self._delegate_deck_creations(
                units_search_data=units_search_data,
                parent_deck_prefix=parent_deck_prefix,
                allowed_cids=allowed_cids if allowed_cids else None,
                total_units=total_units,
            )

        except Exception as e:
            self.error_sig.emit(str(e))

    def _delegate_deck_creations(
        self,
        units_search_data: List[Dict[str, Any]],
        parent_deck_prefix: Optional[str],
        allowed_cids: Optional[Set[int]],
        total_units: int,
    ) -> None:
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
                    self.progress_sig.emit(total_units, total_units, "Completato con successo!")
                    self.finished_sig.emit(
                        {
                            "status": "cancelled" if self._is_cancelled else "done",
                            "units_processed": len(results),
                            "results": results,
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
                            self.progress_sig.emit(total_units, total_units, "Completato con successo!")
                            self.finished_sig.emit(
                                {
                                    "status": "cancelled" if self._is_cancelled else "done",
                                    "units_processed": len(op_results),
                                    "results": op_results,
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
                        self.progress_sig.emit(total_units, total_units, "Completato con successo!")
                        self.finished_sig.emit(
                            {
                                "status": "cancelled" if self._is_cancelled else "done",
                                "units_processed": len(results),
                                "results": results,
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
            self.progress_sig.emit(total_units, total_units, "Completato con successo!")
            self.finished_sig.emit(
                {
                    "status": "cancelled" if self._is_cancelled else "done",
                    "units_processed": len(results),
                    "results": results,
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
