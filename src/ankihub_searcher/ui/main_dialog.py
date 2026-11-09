"""PyQt6 dialogs for Smart Search.

Provides 4 minimal, native-styled standalone dialogs:
1. SearchDialog (compact main search window with hierarchical deck tree and Browser integration)
2. HistoryDialog (run history, safe undo, and delete-deck-only shortcut)
3. IndexDialog (model vector status, incremental indexer, and DB export/import)
4. SettingsDialog (search parameters, LLM/embedding providers, and field indexing)

Alias AnkiHubSearcherDialog = SearchDialog is preserved for backwards compatibility.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

try:
    from aqt import mw  # type: ignore
    from aqt.qt import (  # type: ignore
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QTimer,
        QTreeWidget,
        QTreeWidgetItem,
        Qt,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt6.QtCore import Qt, QTimer  # type: ignore
        from PyQt6.QtWidgets import (  # type: ignore
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSlider,
            QTableWidget,
            QTableWidgetItem,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        mw = None
    except ImportError:
        # Running headless
        class _HeadlessMeta(type):
            def __getattr__(cls, name):
                return _HeadlessWidget()

        class _HeadlessWidget(metaclass=_HeadlessMeta):
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                return _HeadlessWidget()

            def __call__(self, *args, **kwargs):
                return _HeadlessWidget()

        QApplication = _HeadlessWidget  # type: ignore
        QCheckBox = _HeadlessWidget  # type: ignore
        QComboBox = _HeadlessWidget  # type: ignore
        QDialog = _HeadlessWidget  # type: ignore
        QFileDialog = _HeadlessWidget  # type: ignore
        QGroupBox = _HeadlessWidget  # type: ignore
        QHBoxLayout = _HeadlessWidget  # type: ignore
        QHeaderView = _HeadlessWidget  # type: ignore
        QLabel = _HeadlessWidget  # type: ignore
        QLineEdit = _HeadlessWidget  # type: ignore
        QListWidget = _HeadlessWidget  # type: ignore
        QListWidgetItem = _HeadlessWidget  # type: ignore
        QMessageBox = _HeadlessWidget  # type: ignore
        QPlainTextEdit = _HeadlessWidget  # type: ignore
        QProgressBar = _HeadlessWidget  # type: ignore
        QPushButton = _HeadlessWidget  # type: ignore
        QSlider = _HeadlessWidget  # type: ignore
        QTableWidget = _HeadlessWidget  # type: ignore
        QTableWidgetItem = _HeadlessWidget  # type: ignore
        QTreeWidget = _HeadlessWidget  # type: ignore
        QTreeWidgetItem = _HeadlessWidget  # type: ignore
        QVBoxLayout = _HeadlessWidget  # type: ignore
        QWidget = _HeadlessWidget  # type: ignore
        Qt = _HeadlessWidget  # type: ignore
        QTimer = _HeadlessWidget  # type: ignore
        mw = None


from ..ai_client import OpenAICompatibleClient
from ..config_manager import ConfigManager
from ..deck_ops import _get_op_changes, delete_filtered_deck_only, delete_filtered_deck_only_op
from ..indexer import DeckIndexer, collect_presearch_data
from ..storage import SearchDatabase
from ..text_utils import parse_markdown_units
from .worker import IndexWorker, RollbackWorker, SearchBatchWorker

CURRENT_VERSION = "0.4.0"


def compute_coverage_stats(coverage_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes honest coverage metrics:
    - 'covered', 'partial', and 'uncovered' counted separately.
    - Percentage counts ONLY fully covered concepts.
    """
    total = len(coverage_items)
    covered = sum(1 for c in coverage_items if c.get("status") == "covered")
    partial = sum(1 for c in coverage_items if c.get("status") == "partial")
    uncovered = sum(1 for c in coverage_items if c.get("status") == "uncovered")
    pct = int(round((covered / total) * 100)) if total > 0 else 0
    return {
        "total": total,
        "covered": covered,
        "partial": partial,
        "uncovered": uncovered,
        "pct": pct,
        "summary_label": (
            f"Copertura concetti: {covered} / {total} ({pct}%)  —  "
            f"Coperti: {covered}  |  Parziali: {partial}  |  Scoperti: {uncovered}"
        ),
        "history_str": f"{covered}/{total} ({pct}%)" if total > 0 else "-",
    }


def _get_current_profile_name() -> str:
    """Returns the current profile name from Anki if available, or empty string."""
    try:
        from aqt import mw
        if mw and hasattr(mw, "pm") and hasattr(mw.pm, "name"):
            return str(mw.pm.name or "")
    except Exception:
        pass
    return ""


# =====================================================================
# 1. SEARCH DIALOG (Compact Main Window)
# =====================================================================

class CoverageDetailDialog(QDialog):
    """Minimal dialog displaying per-concept coverage details and uncovered claims."""

    def __init__(
        self,
        coverage_items: List[Dict[str, Any]],
        title_context: str = "",
        created_deck_names: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.coverage_items = list(coverage_items or [])
        self.created_deck_names = list(created_deck_names or [])

        win_title = "Smart Search — Dettagli Copertura"
        if title_context:
            win_title += f" ({title_context})"
        self.setWindowTitle(win_title)
        self.resize(800, 480)

        # Sort: uncovered first, then partial, then covered (and ascending by score)
        status_order = {"uncovered": 0, "partial": 1, "covered": 2}
        self.sorted_items = sorted(
            self.coverage_items,
            key=lambda x: (status_order.get(str(x.get("status", "uncovered")), 0), float(x.get("best_score", 0.0))),
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        stats = compute_coverage_stats(self.sorted_items)
        summary_lbl = QLabel(stats["summary_label"])
        layout.addWidget(summary_lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Stato", "Concetto (IT)", "Citazione testo", "Claim USMLE (EN)", "Carte / Score"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setRowCount(len(self.sorted_items))

        status_labels = {
            "uncovered": "Scoperto (0)",
            "partial": "Parziale",
            "covered": "Coperto",
        }

        for r_idx, item in enumerate(self.sorted_items):
            st_raw = str(item.get("status", "uncovered"))
            st_text = status_labels.get(st_raw, st_raw)
            if item.get("shared_with"):
                st_text += f" (con {item['shared_with']})"
            topic_it = str(item.get("topic_it", ""))
            quote_it = str(item.get("quote_it", ""))
            claim_en = str(item.get("claim_en", ""))
            notes_cnt = int(item.get("matched_notes_count", 0))
            best_score = float(item.get("best_score", 0.0))

            self.table.setItem(r_idx, 0, QTableWidgetItem(st_text))
            self.table.setItem(r_idx, 1, QTableWidgetItem(topic_it))
            self.table.setItem(r_idx, 2, QTableWidgetItem(quote_it))
            self.table.setItem(r_idx, 3, QTableWidgetItem(claim_en))
            self.table.setItem(r_idx, 4, QTableWidgetItem(f"{notes_cnt} note ({best_score:.2f})"))

        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 170)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table)

        lbl_detail = QLabel("Dettaglio punto selezionato (selezionabile e copiabile):")
        layout.addWidget(lbl_detail)

        self.txt_detail = QPlainTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setMaximumHeight(120)
        layout.addWidget(self.txt_detail)

        if self.sorted_items:
            self.table.selectRow(0)

        btn_layout = QHBoxLayout()

        btn_copy_uncovered = QPushButton("Copia punti scoperti / parziali")
        btn_copy_uncovered.clicked.connect(self._copy_uncovered_to_clipboard)
        btn_layout.addWidget(btn_copy_uncovered)

        if self.created_deck_names:
            btn_browser = QPushButton("Browser")
            btn_browser.clicked.connect(self._open_browser)
            btn_layout.addWidget(btn_browser)

        btn_layout.addStretch()

        btn_close = QPushButton("Chiudi")
        btn_close.setDefault(True)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_row_selected(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self.sorted_items):
            return
        item = self.sorted_items[row]
        st_map = {"uncovered": "SCOPERTO", "partial": "PARZIALE", "covered": "COPERTO"}
        st = st_map.get(str(item.get("status", "uncovered")), "SCOPERTO")
        kws = ", ".join(item.get("keywords_en", []))
        preview = str(item.get("best_note_preview", "")).strip() or "Nessuna nota corrispondente."
        shared_with = item.get("shared_with", "")
        shared_line = f"\nCondivisa con: {shared_with}" if shared_with else ""
        quote_it = str(item.get("quote_it", "")).strip()
        quote_line = f"Citazione testo: «{quote_it}»\n" if quote_it else ""
        text = (
            f"[{st}] {item.get('topic_it', '')}\n"
            f"{quote_line}"
            f"Claim (EN): {item.get('claim_en', '')}\n"
            f"Keywords: {kws}\n"
            f"Nota più vicina nel mazzo (score {float(item.get('best_score', 0.0)):.2f}): {preview}"
            f"{shared_line}"
        )
        self.txt_detail.setPlainText(text)

    def _copy_uncovered_to_clipboard(self) -> None:
        targets = [c for c in self.sorted_items if c.get("status") in ("uncovered", "partial")]
        if not targets:
            targets = self.sorted_items
        lines = []
        for c in targets:
            st = "SCOPERTO" if c.get("status") == "uncovered" else "PARZIALE" if c.get("status") == "partial" else "COPERTO"
            q = f" (cit: «{c.get('quote_it', '')}»)" if c.get("quote_it") else ""
            lines.append(f"- [{st}] {c.get('topic_it', '')}{q}: {c.get('claim_en', '')}")
        text = "\n".join(lines)
        try:
            cb = QApplication.clipboard()
            if cb:
                cb.setText(text)
            QMessageBox.information(
                self,
                "Copiato",
                f"Copiati {len(targets)} punti chiave negli appunti.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Impossibile copiare negli appunti:\n{e}")

    def _open_browser(self) -> None:
        try:
            import aqt
            if aqt.mw and self.created_deck_names:
                query = " OR ".join(f'"deck:{name}"' for name in self.created_deck_names)
                browser = aqt.dialogs.open("Browser", aqt.mw)
                if browser and hasattr(browser, "search_for"):
                    browser.search_for(query)
        except Exception as e:
            print(f"[SmartSearch] Error opening browser: {e}")


# =====================================================================
# 1. SEARCH DIALOG (Compact Main Window)
# =====================================================================

class SearchDialog(QDialog):
    """Compact, minimal main search dialog with Browser-style query input and history."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.storage = SearchDatabase()
        self.ai_client = OpenAICompatibleClient()
        self.indexer = DeckIndexer(self.config, self.storage, self.ai_client)

        self.setWindowTitle("Smart Search")
        self.resize(540, 410)

        self.search_worker: Optional[SearchBatchWorker] = None
        self._query_timer = QTimer(self)
        self._query_timer.setSingleShot(True)
        self._query_timer.setInterval(250)
        self._query_timer.timeout.connect(self._update_query_stats)

        self._init_ui()
        self._update_query_stats()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 1. Browser-style Search Query Input with History
        lbl_query = QLabel("Mazzi / Query in cui cercare (sintassi Browser Anki):")
        layout.addWidget(lbl_query)

        self.combo_query = QComboBox()
        self.combo_query.setEditable(True)
        self.combo_query.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        if self.config.search_query_history:
            self.combo_query.addItems(self.config.search_query_history)
        self.combo_query.setCurrentText(self.config.get_effective_search_query())
        if self.combo_query.lineEdit():
            self.combo_query.lineEdit().setPlaceholderText(
                'Es. "deck:AnKing Step Deck" oppure "deck:AnKing Step Deck" OR "deck:Mastering Radiology"'
            )
        self.combo_query.currentTextChanged.connect(self._on_query_text_changed)
        layout.addWidget(self.combo_query)

        self.lbl_query_stats = QLabel("Carte selezionate: calcolo in corso...")
        layout.addWidget(self.lbl_query_stats)

        # 2. Filtered Deck Name (Optional)
        lbl_deck_name = QLabel("Nome mazzo filtrato (opzionale):")
        layout.addWidget(lbl_deck_name)

        self.txt_base_deck = QLineEdit()
        self.txt_base_deck.setPlaceholderText("Es. Patologia Generale")
        layout.addWidget(self.txt_base_deck)

        # 3. Lecture Text Input
        lbl_text = QLabel("Testo della lezione:")
        layout.addWidget(lbl_text)

        self.txt_lecture = QPlainTextEdit()
        self.txt_lecture.setPlaceholderText(
            "Incolla qui il testo della lezione o appunti in Markdown..."
        )
        layout.addWidget(self.txt_lecture)

        # 4. Minimal Progress Bar & Status (Hidden unless running)
        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 5. Bottom Action Buttons (Aligned Right)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_search = QPushButton("Cerca")
        self.btn_search.setDefault(True)
        self.btn_search.clicked.connect(self._start_search)
        btn_layout.addWidget(self.btn_search)

        layout.addLayout(btn_layout)

    def _on_query_text_changed(self, _text: str) -> None:
        self._query_timer.start()

    def _update_query_stats(self) -> None:
        if not self.col:
            self.lbl_query_stats.setText("Collezione Anki non disponibile.")
            return

        query = self.combo_query.currentText().strip()
        if not query:
            self.lbl_query_stats.setText("Inserisci una query (es. \"deck:AnKing Step Deck\").")
            return

        try:
            cids = self.indexer.get_cards_for_query(self.col, query)
            nids = self.indexer.get_notes_for_query(self.col, query)
            total_cards = len(cids)
            total_notes = len(nids)

            active_model = self.config.get_active_model_id()
            indexed_notes = (
                self.storage.count_embeddings_for_notes(nids, active_model) if total_notes > 0 else 0
            )
            pct = int(round((indexed_notes / total_notes) * 100)) if total_notes > 0 else 0

            self.lbl_query_stats.setText(
                f"Carte: {total_cards:,} ({total_notes:,} note) — Indicizzate: {pct}% ({indexed_notes:,} / {total_notes:,})"
            )
            # Persist current valid query so IndexDialog sees it immediately
            self.config.search_query = query
            extracted_decks = self.indexer.extract_deck_names_from_query(query)
            if extracted_decks:
                self.config.selected_decks = extracted_decks
            self.config.save()
        except Exception:
            self.lbl_query_stats.setText("Query incompleta o sintassi non valida.")

    def _on_cancel_clicked(self) -> None:
        if self.search_worker and self.search_worker.isRunning():
            self.lbl_status.setText("Interruzione in corso...")
            self.search_worker.cancel()
        else:
            self.reject()

    def _start_search(self) -> None:
        raw_text = self.txt_lecture.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Attenzione", "Inserisci il testo della lezione prima di avviare la ricerca.")
            return

        query = self.combo_query.currentText().strip()
        if not query:
            QMessageBox.warning(self, "Attenzione", "Inserisci una query di ricerca valida.")
            return

        if self.col:
            try:
                matched_nids = self.indexer.get_notes_for_query(self.col, query)
                if not matched_nids:
                    QMessageBox.warning(
                        self,
                        "Nessuna carta trovata",
                        f"La query '{query}' non restituisce alcuna nota nella tua collezione.",
                    )
                    return
            except Exception as e:
                QMessageBox.warning(self, "Query non valida", f"Errore nella query di ricerca:\n{e}")
                return

        self.config.add_query_to_history(query)
        extracted_decks = self.indexer.extract_deck_names_from_query(query)
        if extracted_decks:
            self.config.selected_decks = extracted_decks
        self.config.save()

        self.btn_search.setEnabled(False)
        self.btn_cancel.setText("Interrompi")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("Avvio elaborazione in corso...")

        # Pre-read collection notes and card mappings on the main thread before starting background thread
        presearch_data = collect_presearch_data(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            search_query=query,
        )

        self.search_worker = SearchBatchWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            raw_text=raw_text,
            base_deck_name=self.txt_base_deck.text().strip(),
            precomputed_data=presearch_data,
            profile=_get_current_profile_name(),
        )
        self.search_worker.progress_sig.connect(self._on_search_progress)
        self.search_worker.finished_sig.connect(self._on_search_finished)
        self.search_worker.error_sig.connect(self._on_search_error)
        self.search_worker.start()

    def _on_search_progress(self, current: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.lbl_status.setText(msg)

    def _on_search_finished(self, res: Dict[str, Any]) -> None:
        self.btn_search.setEnabled(True)
        self.btn_cancel.setText("Annulla")
        self.progress_bar.setVisible(False)
        self.lbl_status.setVisible(False)

        status = res.get("status", "done")
        results = res.get("results", [])

        if status == "cancelled":
            QMessageBox.information(self, "Annullato", "Ricerca interrotta dall'utente.")
            return

        total_added = sum(r.total_cards_count for r in results)
        total_unsuspended = sum(r.newly_unsuspended_count for r in results)
        total_already_active = sum(len(r.already_active_cids) for r in results)
        decks_created = len([r for r in results if r.total_cards_count > 0])

        all_coverage: List[Dict[str, Any]] = []
        for r in results:
            if getattr(r, "concept_coverage", None):
                all_coverage.extend(r.concept_coverage)

        cov_stats = compute_coverage_stats(all_coverage)
        cov_line = ""
        if cov_stats["total"] > 0:
            cov_line = (
                f"\n\nCopertura concetti: {cov_stats['covered']} / {cov_stats['total']} ({cov_stats['pct']}% coperti al 100%)\n"
                f"• Coperti: {cov_stats['covered']}\n"
                f"• Parziali: {cov_stats['partial']}\n"
                f"• Scoperti: {cov_stats['uncovered']}"
            )

        err_units = [r for r in results if getattr(r, "unit_error", "")]
        err_section = ""
        if err_units:
            err_lines = [f"• {r.unit_title}: {r.unit_error}" for r in err_units]
            err_section = (
                "⚠️ ATTENZIONE — Errori Stage 1:\n"
                "Nessuna carta riattivata per le seguenti unità a causa di errori nel testo/parsing:\n"
                + "\n".join(err_lines)
                + "\n\n"
            )

        # 1. Automatically close the SearchDialog (CollectionOp handles UI refresh)
        self.accept()

        # 2. Present clean post-search summary popup with Browser & Coverage Details buttons
        parent_widget = mw if mw else None
        msg_box = QMessageBox(parent_widget)
        if err_units:
            msg_box.setWindowTitle("Ricerca completata con errori")
            try:
                msg_box.setIcon(QMessageBox.Icon.Warning)
            except Exception:
                pass
        else:
            msg_box.setWindowTitle("Ricerca completata")

        if total_added == 0 and total_already_active > 0:
            msg_box.setText(
                f"{err_section}"
                f"Nessuna nuova carta da riattivare.\n\n"
                f"Carte correlate già attive in studio: {total_already_active}"
                f"{cov_line}"
            )
        elif total_added == 0 and not total_already_active and err_units:
            msg_box.setText(
                f"{err_section}"
                f"Nessun mazzo filtrato creato a causa degli errori sopra indicati."
                f"{cov_line}"
            )
        else:
            msg_box.setText(
                f"{err_section}"
                f"Mazzi filtrati creati: {decks_created}\n"
                f"Nuove carte riattivate: {total_unsuspended}\n"
                f"Carte correlate già attive: {total_already_active}\n"
                f"Carte totali nel mazzo filtrato: {total_added}"
                f"{cov_line}"
            )

        msg_box.addButton(QMessageBox.StandardButton.Ok)
        btn_browser = None
        btn_coverage = None
        created_names = [r.deck_name for r in results if r.filtered_did and r.total_cards_count > 0]
        if created_names:
            btn_browser = msg_box.addButton("Browser", QMessageBox.ButtonRole.ActionRole)
        if all_coverage:
            btn_coverage = msg_box.addButton("Dettagli copertura", QMessageBox.ButtonRole.ActionRole)

        msg_box.exec()
        clicked = msg_box.clickedButton()

        if btn_browser and clicked == btn_browser:
            try:
                import aqt
                if aqt.mw:
                    query = " OR ".join(f'"deck:{name}"' for name in created_names)
                    browser = aqt.dialogs.open("Browser", aqt.mw)
                    if browser and hasattr(browser, "search_for"):
                        browser.search_for(query)
            except Exception as e:
                print(f"[SmartSearch] Error opening browser: {e}")
        elif btn_coverage and clicked == btn_coverage:
            cov_dlg = CoverageDetailDialog(
                coverage_items=all_coverage,
                created_deck_names=created_names,
                parent=parent_widget,
            )
            cov_dlg.exec()

    def _on_search_error(self, err_msg: str) -> None:
        self.btn_search.setEnabled(True)
        self.btn_cancel.setText("Annulla")
        self.progress_bar.setVisible(False)
        self.lbl_status.setVisible(False)
        QMessageBox.critical(self, "Errore Ricerca", f"Si è verificato un errore:\n\n{err_msg}")


# Backwards compatibility alias for imports
AnkiHubSearcherDialog = SearchDialog


# =====================================================================
# 2. HISTORY DIALOG (Cronologia)
# =====================================================================

class HistoryDialog(QDialog):
    """Minimal dialog displaying past search runs and safe rollback controls."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.storage = SearchDatabase()
        self.current_profile = _get_current_profile_name()

        self.setWindowTitle("Smart Search — Cronologia")
        self.resize(720, 420)

        self.rollback_worker: Optional[RollbackWorker] = None
        self._init_ui()
        self._refresh_history()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_lbl = QLabel(
            "Cronologia delle ricerche eseguite. Il rollback ri-sospende esclusivamente le carte che erano "
            "sospese prima della run, senza modificare le carte già attive."
        )
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels(
            ["Data", "Mazzo Filtrato", "Unità", "Nel Mazzo", "Nuove Riattivate", "Già Attive", "Copertura", "Stato"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.history_table)

        self.chk_keep_reviewed = QCheckBox("Mantieni attive le carte già ripassate")
        self.chk_keep_reviewed.setChecked(self.config.rollback_keep_reviewed_default)
        layout.addWidget(self.chk_keep_reviewed)

        btn_layout = QHBoxLayout()

        self.btn_rollback = QPushButton("Annulla run selezionata")
        self.btn_rollback.clicked.connect(self._on_rollback_clicked)
        btn_layout.addWidget(self.btn_rollback)

        self.btn_delete_deck_only = QPushButton("Elimina solo mazzo filtrato")
        self.btn_delete_deck_only.setToolTip(
            "Svuota ed elimina il mazzo filtrato ma lascia le carte riattivate nei mazzi principali per il ripasso."
        )
        self.btn_delete_deck_only.clicked.connect(self._on_delete_deck_only_clicked)
        btn_layout.addWidget(self.btn_delete_deck_only)

        self.btn_coverage = QPushButton("Dettagli copertura")
        self.btn_coverage.clicked.connect(self._on_coverage_clicked)
        btn_layout.addWidget(self.btn_coverage)

        btn_layout.addStretch()

        btn_refresh = QPushButton("Aggiorna")
        btn_refresh.clicked.connect(self._refresh_history)
        btn_layout.addWidget(btn_refresh)

        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _refresh_history(self) -> None:
        runs = self.storage.get_runs(profile=self.current_profile)
        self.history_table.setRowCount(len(runs))

        for row_idx, r in enumerate(runs):
            t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))
            unit_err = getattr(r, "unit_error", "")
            if unit_err:
                status_str = f"Errore: {unit_err}"
            elif r.rolled_back:
                status_str = "Annullata"
            else:
                status_str = "Attiva"

            if not r.profile:
                status_str += " [⚠️ senza profilo]"

            cov_list = getattr(r, "concept_coverage", None) or []
            if cov_list:
                stats = compute_coverage_stats(cov_list)
                cov_str = stats["history_str"]
            else:
                cov_str = "-"

            item_date = QTableWidgetItem(t_str)
            item_date.setData(Qt.ItemDataRole.UserRole, r.run_id)

            self.history_table.setItem(row_idx, 0, item_date)
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(r.deck_name))
            self.history_table.setItem(row_idx, 2, QTableWidgetItem(r.unit_title))
            self.history_table.setItem(row_idx, 3, QTableWidgetItem(str(len(r.total_matched_cids))))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(str(len(r.newly_unsuspended_cids))))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(str(len(r.already_active_cids))))
            item_cov = QTableWidgetItem(cov_str)
            if cov_list:
                item_cov.setToolTip(
                    f"Coperti: {stats['covered']} | Parziali: {stats['partial']} | Scoperti: {stats['uncovered']}"
                )
            self.history_table.setItem(row_idx, 6, item_cov)
            item_status = QTableWidgetItem(status_str)
            if unit_err:
                item_status.setToolTip(unit_err)
            elif not r.profile:
                item_status.setToolTip("Run salvata senza indicazione del profilo (versione precedente).")
            self.history_table.setItem(row_idx, 7, item_status)

    def _get_selected_run_id(self) -> Optional[str]:
        selected_rows = self.history_table.selectedItems()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        item = self.history_table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_rollback_clicked(self) -> None:
        run_id = self._get_selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "Attenzione", "Seleziona prima una run dalla tabella da annullare.")
            return

        run_rec = self.storage.get_run(run_id)
        if not run_rec:
            return

        if run_rec.rolled_back:
            QMessageBox.information(self, "Info", "Questa run è già stata annullata in precedenza.")
            return

        if run_rec.profile and self.current_profile and run_rec.profile != self.current_profile:
            QMessageBox.critical(
                self,
                "Rollback Rifiutato",
                f"Impossibile annullare questa run: appartiene al profilo '{run_rec.profile}', "
                f"mentre il profilo aperto è '{self.current_profile}'.",
            )
            return

        if not run_rec.profile:
            warn_reply = QMessageBox.warning(
                self,
                "Attenzione: Run Precedente",
                "Questa run è stata salvata senza indicazione del profilo (versione precedente).\n\n"
                "Vuoi procedere comunque con il rollback sul profilo aperto?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if warn_reply != QMessageBox.StandardButton.Yes:
                return

        keep_rev = self.chk_keep_reviewed.isChecked()
        reply = QMessageBox.question(
            self,
            "Conferma Rollback",
            f"Sei sicuro di voler annullare la run per il mazzo '{run_rec.deck_name}'?\n\n"
            f"Verranno ri-sospese le carte unsospese da questa run.\n"
            f"Mantieni carte già ripassate: {'Sì' if keep_rev else 'No'}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_rollback.setEnabled(False)
        self.rollback_worker = RollbackWorker(
            col=self.col,
            storage=self.storage,
            run_id=run_id,
            keep_reviewed=keep_rev,
            current_profile=self.current_profile,
        )
        self.rollback_worker.finished_sig.connect(self._on_rollback_finished)
        self.rollback_worker.error_sig.connect(self._on_rollback_error)
        self.rollback_worker.start()

    def _on_rollback_finished(self, res: Any) -> None:
        self.btn_rollback.setEnabled(True)
        self._refresh_history()
        QMessageBox.information(
            self,
            "Rollback Eseguito",
            f"Rollback completato con successo.\n\n"
            f"Mazzo filtrato rimosso: {'Sì' if res.deck_removed else 'No/Già rimosso'}\n"
            f"Carte ri-sospese: {res.resuspended_count}\n"
            f"Carte mantenute attive perché ripassate: {res.kept_reviewed_count}",
        )

    def _on_rollback_error(self, err: str) -> None:
        self.btn_rollback.setEnabled(True)
        QMessageBox.critical(self, "Errore Rollback", f"Errore durante il rollback:\n{err}")

    def _on_delete_deck_only_clicked(self) -> None:
        run_id = self._get_selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "Attenzione", "Seleziona prima una run dalla tabella.")
            return

        run_rec = self.storage.get_run(run_id)
        if not run_rec:
            return

        if run_rec.profile and self.current_profile and run_rec.profile != self.current_profile:
            QMessageBox.critical(
                self,
                "Azione Rifiutata",
                f"Impossibile modificare questa run: appartiene al profilo '{run_rec.profile}', "
                f"mentre il profilo aperto è '{self.current_profile}'.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Elimina Solo Mazzo",
            f"Vuoi eliminare il mazzo filtrato '{run_rec.deck_name}' mantenendo tutte le carte attive nella tua collezione?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from aqt.operations import CollectionOp  # type: ignore
            except ImportError:
                CollectionOp = None

            if mw is not None and CollectionOp is not None:

                def _del_op(col: Any):
                    ok, changes = delete_filtered_deck_only_op(col, self.storage, run_id)
                    return _get_op_changes(changes)

                def _on_del_success(changes: Any):
                    self._refresh_history()
                    QMessageBox.information(self, "Fatto", "Mazzo filtrato eliminato. Le carte restano attive.")

                CollectionOp(parent=self, op=_del_op).success(_on_del_success).run_in_background()
            else:
                success = delete_filtered_deck_only(self.col, self.storage, run_id)
                self._refresh_history()
                if success:
                    QMessageBox.information(self, "Fatto", "Mazzo filtrato eliminato. Le carte restano attive.")
                else:
                    QMessageBox.warning(self, "Avviso", "Il mazzo filtrato non esiste più.")

    def _on_coverage_clicked(self) -> None:
        run_id = self._get_selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "Attenzione", "Seleziona prima una run dalla tabella.")
            return
        run_rec = self.storage.get_run(run_id)
        if not run_rec or not getattr(run_rec, "concept_coverage", None):
            QMessageBox.information(
                self,
                "Dettagli Copertura",
                "Nessun dettaglio di copertura salvato per questa run (eseguita con una versione precedente).",
            )
            return
        dlg = CoverageDetailDialog(
            coverage_items=run_rec.concept_coverage,
            title_context=run_rec.unit_title,
            created_deck_names=[run_rec.deck_name] if run_rec.filtered_did and not run_rec.rolled_back else None,
            parent=self,
        )
        dlg.exec()


# =====================================================================
# 3. INDEX DIALOG (Indicizzazione)
# =====================================================================

class IndexDialog(QDialog):
    """Minimal dialog managing vector index coverage, updates, and DB import/export."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.storage = SearchDatabase()
        self.ai_client = OpenAICompatibleClient()

        self.setWindowTitle("Smart Search — Indicizzazione")
        self.resize(620, 420)

        self.index_worker: Optional[IndexWorker] = None
        self._init_ui()
        self._refresh_status()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Status Box
        status_group = QGroupBox("Stato Modello Attivo")
        status_layout = QVBoxLayout(status_group)

        self.lbl_active_model = QLabel("Modello Attivo: -")
        status_layout.addWidget(self.lbl_active_model)

        self.lbl_coverage = QLabel("Copertura mazzi selezionati: -")
        status_layout.addWidget(self.lbl_coverage)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        self.lbl_progress_msg = QLabel("")
        self.lbl_progress_msg.setVisible(False)
        status_layout.addWidget(self.lbl_progress_msg)

        btn_box = QHBoxLayout()
        self.btn_run = QPushButton("Indicizza / Aggiorna")
        self.btn_run.clicked.connect(self._start_indexing)
        btn_box.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Interrompi")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_indexing)
        btn_box.addWidget(self.btn_cancel)
        status_layout.addLayout(btn_box)

        layout.addWidget(status_group)

        # Cached models table
        models_group = QGroupBox("Cache Modelli nel Database")
        models_layout = QVBoxLayout(models_group)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(3)
        self.models_table.setHorizontalHeaderLabels(["Modello", "Note indicizzate", "Dimensione (MB)"])
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        models_layout.addWidget(self.models_table)

        layout.addWidget(models_group)

        # Bottom actions
        bottom_layout = QHBoxLayout()

        btn_export = QPushButton("Esporta indice (.db)...")
        btn_export.clicked.connect(self._export_database)
        bottom_layout.addWidget(btn_export)

        btn_import = QPushButton("Importa / Unisci (.db)...")
        btn_import.clicked.connect(self._import_database)
        bottom_layout.addWidget(btn_import)

        btn_purge = QPushButton("Elimina cache modelli inattivi")
        btn_purge.clicked.connect(self._purge_inactive_models)
        bottom_layout.addWidget(btn_purge)

        bottom_layout.addStretch()

        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(self.accept)
        bottom_layout.addWidget(btn_close)

        layout.addLayout(bottom_layout)

    def _refresh_status(self) -> None:
        active_id = self.config.get_active_model_id()
        self.lbl_active_model.setText(f"Modello Attivo: {active_id}")

        stats = self.storage.get_indexed_models_stats()
        self.models_table.setRowCount(len(stats))
        active_count = 0
        for r_idx, s in enumerate(stats):
            self.models_table.setItem(r_idx, 0, QTableWidgetItem(s["model_id"]))
            self.models_table.setItem(r_idx, 1, QTableWidgetItem(str(s["note_count"])))
            self.models_table.setItem(r_idx, 2, QTableWidgetItem(f"{s['size_mb']} MB"))
            if s["model_id"] == active_id:
                active_count = s["note_count"]

        if self.col:
            indexer = DeckIndexer(self.config, self.storage, self.ai_client)
            active_query = self.config.get_effective_search_query()
            try:
                target_nids = indexer.get_notes_for_query(self.col, active_query)
            except Exception:
                target_nids = []
            total_notes = len(target_nids)
            deck_active_count = self.storage.count_embeddings_for_notes(target_nids, active_id) if target_nids else 0
            pct = int(deck_active_count / total_notes * 100) if total_notes > 0 else 0
            self.lbl_coverage.setText(
                f"Copertura query attiva ({active_query}): {deck_active_count} / {total_notes} note ({pct}%)"
            )

    def _start_indexing(self) -> None:
        if not self.col:
            return

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_msg.setVisible(True)
        self.lbl_progress_msg.setText("Avvio indicizzazione...")

        # Scan note metadata on the main thread before launching background embedding worker
        indexer = DeckIndexer(self.config, self.storage, self.ai_client)
        active_query = self.config.get_effective_search_query()
        if active_query:
            nids = indexer.get_notes_for_query(self.col, active_query)
        else:
            dids = indexer.find_target_deck_ids(self.col, self.config.selected_decks)
            nids = indexer.get_notes_in_decks(self.col, dids)
        precomputed_notes = indexer.scan_notes_metadata(self.col, nids)

        self.index_worker = IndexWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            selected_decks=None,
            precomputed_notes_data=precomputed_notes,
        )
        self.index_worker.progress_sig.connect(self._on_progress)
        self.index_worker.finished_sig.connect(self._on_finished)
        self.index_worker.error_sig.connect(self._on_error)
        self.index_worker.start()

    def _cancel_indexing(self) -> None:
        if self.index_worker:
            self.lbl_progress_msg.setText("Interruzione in corso...")
            self.index_worker.cancel()

    def _on_progress(self, cur: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(cur)
        self.lbl_progress_msg.setText(msg)

    def _on_finished(self, res: Dict[str, Any]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_progress_msg.setVisible(False)
        self._refresh_status()

        QMessageBox.information(
            self,
            "Indicizzazione",
            f"Operazione completata.\n\n"
            f"Note con testo indicizzato: {res.get('text_indexed', 0)}\n"
            f"Nuovi vettori calcolati: {res.get('embeddings_indexed', 0)}\n"
            f"Stato: {'Interrotta' if res.get('cancelled') else 'Completata'}",
        )

    def _on_error(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_progress_msg.setVisible(False)
        QMessageBox.critical(self, "Errore Indicizzazione", f"Errore:\n{err}")

    def _export_database(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta Indice Smart Search (.db)",
            os.path.expanduser("~/Desktop/smart_search.db"),
            "Database SQLite (*.db);;Tutti i file (*.*)",
        )
        if not path:
            return
        try:
            self.storage.export_database(path)
            QMessageBox.information(self, "Esportazione Completata", f"Indice esportato con successo in:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Esportazione", f"Impossibile esportare l'indice:\n{e}")

    def _import_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importa / Unisci Indice (.db)",
            os.path.expanduser("~/Desktop"),
            "Database SQLite (*.db);;Tutti i file (*.*)",
        )
        if not path:
            return
        try:
            res = self.storage.import_database(path)
            self._refresh_status()
            QMessageBox.information(
                self,
                "Importazione Completata",
                f"Importazione e unione completate con successo!\n\n"
                f"Note importate/aggiornate: {res.get('notes_imported', 0)}\n"
                f"Embeddings importati/aggiornati: {res.get('embeddings_imported', 0)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore Importazione", f"Impossibile importare l'indice:\n{e}")

    def _purge_inactive_models(self) -> None:
        active_id = self.config.get_active_model_id()
        reply = QMessageBox.question(
            self,
            "Conferma Pulizia",
            f"Vuoi davvero eliminare tutti i vettori di modelli diversi da '{active_id}'?\nQuesta operazione libererà spazio su disco.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            deleted = self.storage.delete_inactive_models(active_id)
            self._refresh_status()
            QMessageBox.information(self, "Pulizia Completata", f"Eliminati {deleted} vettori inattivi.")


# =====================================================================
# 4. SETTINGS DIALOG (Impostazioni)
# =====================================================================

class SettingsDialog(QDialog):
    """Minimal dialog for configuring search sensitivity, AI endpoints, and note fields."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.ai_client = OpenAICompatibleClient()

        self.setWindowTitle("Smart Search — Impostazioni")
        self.resize(580, 560)

        self._init_ui()
        self._load_config_to_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Parametri di Ricerca
        search_group = QGroupBox("Parametri di Ricerca")
        search_layout = QVBoxLayout(search_group)

        # Sensitivity slider
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensibilità ricerca:"))
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(10, 95)
        self.slider_sens.setValue(int(self.config.sensitivity * 100))
        self.slider_sens.valueChanged.connect(self._on_sens_slider_changed)
        sens_row.addWidget(self.slider_sens)

        self.lbl_sens_val = QLabel(f"{int(self.config.sensitivity * 100)}%")
        sens_row.addWidget(self.lbl_sens_val)
        search_layout.addLayout(sens_row)

        self.chk_llm_verify = QCheckBox("Verifica finale dei risultati con LLM (più accurato, più lento)")
        self.chk_llm_verify.setChecked(self.config.enable_llm_verification)
        search_layout.addWidget(self.chk_llm_verify)

        self.chk_include_active = QCheckBox("Inserisci nel mazzo filtrato anche le carte già attive in studio")
        self.chk_include_active.setChecked(self.config.include_active_cards_in_filtered_deck)
        search_layout.addWidget(self.chk_include_active)

        naming_row = QHBoxLayout()
        naming_row.addWidget(QLabel("Stile nomi mazzi multi-unità:"))
        self.combo_naming = QComboBox()
        self.combo_naming.addItem("Nome Base - Titolo Sezione", "dash")
        self.combo_naming.addItem("Nome Base::Titolo Sezione (Sottomazzo)", "subdeck")
        naming_row.addWidget(self.combo_naming)
        search_layout.addLayout(naming_row)

        layout.addWidget(search_group)

        # 2. Modello Chat (LLM)
        chat_group = QGroupBox("Modello Chat (Decomposizione Concettuale)")
        chat_layout = QVBoxLayout(chat_group)

        row_chat_preset = QHBoxLayout()
        row_chat_preset.addWidget(QLabel("Preset:"))
        self.combo_chat_preset = QComboBox()
        self.combo_chat_preset.addItems(["LM Studio Locale (http://127.0.0.1:1234/v1)", "OpenRouter", "Custom"])
        self.combo_chat_preset.currentIndexChanged.connect(self._on_chat_preset_changed)
        row_chat_preset.addWidget(self.combo_chat_preset)
        chat_layout.addLayout(row_chat_preset)

        row_chat_url = QHBoxLayout()
        row_chat_url.addWidget(QLabel("Base URL:"))
        self.txt_chat_url = QLineEdit()
        row_chat_url.addWidget(self.txt_chat_url)
        chat_layout.addLayout(row_chat_url)

        row_chat_key = QHBoxLayout()
        row_chat_key.addWidget(QLabel("API Key:"))
        self.txt_chat_key = QLineEdit()
        self.txt_chat_key.setEchoMode(QLineEdit.EchoMode.Password)
        row_chat_key.addWidget(self.txt_chat_key)
        chat_layout.addLayout(row_chat_key)

        row_chat_model = QHBoxLayout()
        row_chat_model.addWidget(QLabel("Modello:"))
        self.combo_chat_model = QComboBox()
        self.combo_chat_model.setEditable(True)
        row_chat_model.addWidget(self.combo_chat_model)

        btn_fetch_chat = QPushButton("Carica modelli")
        btn_fetch_chat.clicked.connect(self._fetch_chat_models)
        row_chat_model.addWidget(btn_fetch_chat)

        btn_test_chat = QPushButton("Test")
        btn_test_chat.clicked.connect(self._test_chat_connection)
        row_chat_model.addWidget(btn_test_chat)
        chat_layout.addLayout(row_chat_model)

        self.chk_chat_schema = QCheckBox("Forza output JSON strutturato via schema (sperimentale, disattivato di default)")
        chat_layout.addWidget(self.chk_chat_schema)

        layout.addWidget(chat_group)

        # 3. Modello Embedding
        embed_group = QGroupBox("Modello Embedding (Indice Vettoriale)")
        embed_layout = QVBoxLayout(embed_group)

        row_embed_preset = QHBoxLayout()
        row_embed_preset.addWidget(QLabel("Preset:"))
        self.combo_embed_preset = QComboBox()
        self.combo_embed_preset.addItems(["LM Studio Locale (http://127.0.0.1:1234/v1)", "OpenRouter", "Custom"])
        self.combo_embed_preset.currentIndexChanged.connect(self._on_embed_preset_changed)
        row_embed_preset.addWidget(self.combo_embed_preset)
        embed_layout.addLayout(row_embed_preset)

        row_embed_url = QHBoxLayout()
        row_embed_url.addWidget(QLabel("Base URL:"))
        self.txt_embed_url = QLineEdit()
        row_embed_url.addWidget(self.txt_embed_url)
        embed_layout.addLayout(row_embed_url)

        row_embed_key = QHBoxLayout()
        row_embed_key.addWidget(QLabel("API Key:"))
        self.txt_embed_key = QLineEdit()
        self.txt_embed_key.setEchoMode(QLineEdit.EchoMode.Password)
        row_embed_key.addWidget(self.txt_embed_key)
        embed_layout.addLayout(row_embed_key)

        row_embed_model = QHBoxLayout()
        row_embed_model.addWidget(QLabel("Modello:"))
        self.combo_embed_model = QComboBox()
        self.combo_embed_model.setEditable(True)
        row_embed_model.addWidget(self.combo_embed_model)

        btn_fetch_embed = QPushButton("Carica modelli")
        btn_fetch_embed.clicked.connect(self._fetch_embed_models)
        row_embed_model.addWidget(btn_fetch_embed)

        btn_test_embed = QPushButton("Test")
        btn_test_embed.clicked.connect(self._test_embed_connection)
        row_embed_model.addWidget(btn_test_embed)
        embed_layout.addLayout(row_embed_model)

        row_strategy = QHBoxLayout()
        row_strategy.addWidget(QLabel("Strategia linguistica:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItem("Traduzione concettuale in inglese (consigliato)", "translate_to_en")
        self.combo_strategy.addItem("Multilingua diretto", "multilingual_direct")
        row_strategy.addWidget(self.combo_strategy)
        embed_layout.addLayout(row_strategy)

        layout.addWidget(embed_group)

        # 4. Campi indicizzati per Tipo di Nota
        fields_group = QGroupBox("Campi Indicizzati per Tipo di Nota")
        fields_layout = QVBoxLayout(fields_group)

        model_select_row = QHBoxLayout()
        model_select_row.addWidget(QLabel("Tipo di Nota:"))
        self.combo_note_models = QComboBox()
        self.combo_note_models.currentIndexChanged.connect(self._on_note_model_selected)
        model_select_row.addWidget(self.combo_note_models)
        fields_layout.addLayout(model_select_row)

        self.list_fields = QListWidget()
        self.list_fields.setMaximumHeight(80)
        fields_layout.addWidget(self.list_fields)

        self.chk_index_tags = QCheckBox("Indicizza anche i tag delle note")
        self.chk_index_tags.setChecked(self.config.index_tags)
        fields_layout.addWidget(self.chk_index_tags)

        layout.addWidget(fields_group)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("Annulla")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Salva")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._save_settings)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

        self._populate_note_models()

    def _load_config_to_ui(self) -> None:
        self.slider_sens.setValue(int(self.config.sensitivity * 100))
        self.lbl_sens_val.setText(f"{int(self.config.sensitivity * 100)}%")
        self.chk_llm_verify.setChecked(self.config.enable_llm_verification)
        self.chk_include_active.setChecked(self.config.include_active_cards_in_filtered_deck)

        idx_naming = self.combo_naming.findData(self.config.batch_deck_naming_style)
        if idx_naming != -1:
            self.combo_naming.setCurrentIndex(idx_naming)

        self.txt_chat_url.setText(self.config.chat_base_url)
        self.txt_chat_key.setText(self.config.chat_api_key)
        self.combo_chat_model.setCurrentText(self.config.chat_model)
        self.chk_chat_schema.setChecked(getattr(self.config, "chat_use_json_schema", False))

        self.txt_embed_url.setText(self.config.embed_base_url)
        self.txt_embed_key.setText(self.config.embed_api_key)
        self.combo_embed_model.setCurrentText(self.config.embed_model)

        # Preset selection
        if "127.0.0.1" in self.config.chat_base_url:
            self.combo_chat_preset.setCurrentIndex(0)
        elif "openrouter" in self.config.chat_base_url:
            self.combo_chat_preset.setCurrentIndex(1)
        else:
            self.combo_chat_preset.setCurrentIndex(2)

        if "127.0.0.1" in self.config.embed_base_url:
            self.combo_embed_preset.setCurrentIndex(0)
        elif "openrouter" in self.config.embed_base_url:
            self.combo_embed_preset.setCurrentIndex(1)
        else:
            self.combo_embed_preset.setCurrentIndex(2)

        idx_strat = self.combo_strategy.findData(self.config.embed_language_strategy)
        if idx_strat != -1:
            self.combo_strategy.setCurrentIndex(idx_strat)

    def _on_sens_slider_changed(self, val: int) -> None:
        self.lbl_sens_val.setText(f"{val}%")

    def _on_chat_preset_changed(self, idx: int) -> None:
        if idx == 0:
            self.txt_chat_url.setText("http://127.0.0.1:1234/v1")
        elif idx == 1:
            self.txt_chat_url.setText("https://openrouter.ai/api/v1")

    def _on_embed_preset_changed(self, idx: int) -> None:
        if idx == 0:
            self.txt_embed_url.setText("http://127.0.0.1:1234/v1")
        elif idx == 1:
            self.txt_embed_url.setText("https://openrouter.ai/api/v1")

    def _fetch_chat_models(self) -> None:
        url = self.txt_chat_url.text().strip()
        key = self.txt_chat_key.text().strip()
        try:
            models = self.ai_client.list_models(url, key)
            self.combo_chat_model.clear()
            self.combo_chat_model.addItems(models)
            QMessageBox.information(self, "Modelli Caricati", f"Trovati {len(models)} modelli disponibili.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare i modelli:\n{e}")

    def _fetch_embed_models(self) -> None:
        url = self.txt_embed_url.text().strip()
        key = self.txt_embed_key.text().strip()
        try:
            models = self.ai_client.list_models(url, key)
            self.combo_embed_model.clear()
            self.combo_embed_model.addItems(models)
            QMessageBox.information(self, "Modelli Caricati", f"Trovati {len(models)} modelli disponibili.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare i modelli:\n{e}")

    def _test_chat_connection(self) -> None:
        url = self.txt_chat_url.text().strip()
        key = self.txt_chat_key.text().strip()
        ok = self.ai_client.test_connection(url, key)
        if ok:
            QMessageBox.information(self, "Connessione", "Connessione riuscita all'endpoint Chat LLM.")
        else:
            QMessageBox.critical(self, "Connessione", "Impossibile connettersi all'endpoint specificato.")

    def _test_embed_connection(self) -> None:
        url = self.txt_embed_url.text().strip()
        key = self.txt_embed_key.text().strip()
        ok = self.ai_client.test_connection(url, key)
        if ok:
            QMessageBox.information(self, "Connessione", "Connessione riuscita all'endpoint Embedding Model.")
        else:
            QMessageBox.critical(self, "Connessione", "Impossibile connettersi all'endpoint specificato.")

    def _populate_note_models(self) -> None:
        self.combo_note_models.clear()
        if not self.col:
            return
        for m in sorted(self.col.models.all(), key=lambda x: x.get("name", "")):
            self.combo_note_models.addItem(m.get("name", ""), m.get("id"))
        self._on_note_model_selected(0)

    def _on_note_model_selected(self, index: int) -> None:
        self.list_fields.clear()
        if not self.col or index < 0:
            return
        m_name = self.combo_note_models.currentText()
        mid = self.combo_note_models.currentData()
        model = self.col.models.get(mid)
        if not model:
            return

        available_fields = [f["name"] for f in model.get("flds", [])]
        configured_fields = set(self.config.get_indexed_fields_for_model(m_name, available_fields))

        for fld in available_fields:
            item = QListWidgetItem(fld)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if fld in configured_fields else Qt.CheckState.Unchecked)
            self.list_fields.addItem(item)

    def _save_settings(self) -> None:
        self.config.sensitivity = round(self.slider_sens.value() / 100.0, 2)
        self.config.enable_llm_verification = self.chk_llm_verify.isChecked()
        self.config.include_active_cards_in_filtered_deck = self.chk_include_active.isChecked()
        self.config.batch_deck_naming_style = self.combo_naming.currentData() or "dash"

        self.config.chat_base_url = self.txt_chat_url.text().strip()
        self.config.chat_api_key = self.txt_chat_key.text().strip()
        self.config.chat_model = self.combo_chat_model.currentText().strip()
        self.config.chat_use_json_schema = self.chk_chat_schema.isChecked()

        self.config.embed_base_url = self.txt_embed_url.text().strip()
        self.config.embed_api_key = self.txt_embed_key.text().strip()
        self.config.embed_model = self.combo_embed_model.currentText().strip()
        self.config.embed_language_strategy = self.combo_strategy.currentData() or "translate_to_en"
        self.config.index_tags = self.chk_index_tags.isChecked()

        # Save selected fields for active model
        m_name = self.combo_note_models.currentText()
        if m_name:
            checked_fields = []
            for i in range(self.list_fields.count()):
                it = self.list_fields.item(i)
                if it.checkState() == Qt.CheckState.Checked:
                    checked_fields.append(it.text())
            if checked_fields:
                self.config.indexed_fields_by_model[m_name] = checked_fields

        self.config.save()
        self.accept()
