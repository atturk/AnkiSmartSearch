"""Background Qt workers for indexing, searching, and collection operations.

Prevents freezing Anki's main thread while communicating progress through Qt signals.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional

try:
    from aqt.qt import QObject, QThread, pyqtSignal  # type: ignore
except ImportError:
    try:
        from PyQt6.QtCore import QObject, QThread, pyqtSignal  # type: ignore
    except ImportError:

        class QThread:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        def pyqtSignal(*args, **kwargs):  # type: ignore
            class DummySignal:
                def emit(self, *a, **k):
                    pass

                def connect(self, slot):
                    pass

            return DummySignal()


from ..ai_client import OpenAICompatibleClient
from ..config_manager import ConfigManager
from ..deck_ops import execute_unit_deck_creation, rollback_run
from ..indexer import DeckIndexer
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
    ):
        super().__init__(parent)
        self.col = col
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.selected_decks = selected_decks
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

            res = self.indexer.index_selected_decks(
                col=self.col,
                selected_deck_names=self.selected_decks,
                progress_callback=_on_progress,
                cancel_check=_check_cancel,
            )
            self.finished_sig.emit(res)
        except Exception as e:
            self.error_sig.emit(str(e))


class SearchBatchWorker(QThread):
    """Background worker executing the full pipeline:

    1. Markdown parsing
    2. Auto-indexing any missing embeddings for selected decks
    3. Multi-stage concept extraction & hybrid search per unit
    4. Safe filtered deck creation
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
    ):
        super().__init__(parent)
        self.col = col
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.raw_text = raw_text
        self.base_deck_name = base_deck_name
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

            # 2. Check and ensure target decks are indexed
            indexer = DeckIndexer(self.config, self.storage, self.ai_client)
            target_dids = indexer.find_target_deck_ids(self.col, self.config.selected_decks)
            candidate_nids = indexer.get_notes_in_decks(self.col, target_dids)
            suspended_nids = indexer.get_notes_with_suspended_cards(self.col, target_dids)

            def _check_cancel() -> bool:
                return self._is_cancelled

            def _on_idx_progress(cur: int, tot: int, msg: str) -> None:
                self.progress_sig.emit(0, total_units, f"[Verifica Indice] {msg}")

            idx_res = indexer.index_selected_decks(
                col=self.col,
                selected_deck_names=self.config.selected_decks,
                progress_callback=_on_idx_progress,
                cancel_check=_check_cancel,
            )

            if self._is_cancelled:
                self.finished_sig.emit({"status": "cancelled", "units_processed": 0, "results": []})
                return

            active_model_id = self.config.get_active_model_id()
            pipeline = SmartSearchPipeline(self.config, self.storage, self.ai_client)

            results = []

            for idx, unit in enumerate(units, start=1):
                if self._is_cancelled:
                    break

                self.progress_sig.emit(
                    idx - 1,
                    total_units,
                    f"[{idx}/{total_units}] Elaborazione IA: {unit.unit_title}...",
                )

                # Search pipeline (Stage 1, 2, 3)
                scored_notes = pipeline.search_unit(
                    unit=unit,
                    candidate_nids=candidate_nids,
                    active_model_id=active_model_id,
                    suspended_nids=suspended_nids,
                )

                matched_nids = [sn.nid for sn in scored_notes]

                self.progress_sig.emit(
                    idx - 1,
                    total_units,
                    f"[{idx}/{total_units}] Creazione Mazzo Filtrato ({len(matched_nids)} note)...",
                )

                # Safe filtered deck creation
                run_res = execute_unit_deck_creation(
                    col=self.col,
                    unit=unit,
                    matched_nids=matched_nids,
                    config=self.config,
                    storage=self.storage,
                )
                results.append(run_res)

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
    ):
        super().__init__(parent)
        self.col = col
        self.storage = storage
        self.run_id = run_id
        self.keep_reviewed = keep_reviewed

    def run(self) -> None:
        try:
            res = rollback_run(
                col=self.col,
                storage=self.storage,
                run_id=self.run_id,
                keep_reviewed=self.keep_reviewed,
            )
            self.finished_sig.emit(res)
        except Exception as e:
            self.error_sig.emit(str(e))
