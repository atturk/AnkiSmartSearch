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
        Qt,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt6.QtCore import Qt  # type: ignore
        from PyQt6.QtWidgets import (  # type: ignore
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
        QDialog = object  # type: ignore
        QTreeWidgetItem = object  # type: ignore
        mw = None


from ..ai_client import OpenAICompatibleClient
from ..config_manager import ConfigManager
from ..deck_ops import delete_filtered_deck_only
from ..indexer import DeckIndexer
from ..storage import SearchDatabase
from ..text_utils import parse_markdown_units
from .worker import IndexWorker, RollbackWorker, SearchBatchWorker

CURRENT_VERSION = "0.2.0"


# =====================================================================
# 1. SEARCH DIALOG (Compact Main Window)
# =====================================================================

class SearchDialog(QDialog):
    """Compact, minimal main search dialog with hierarchical deck tree."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.storage = SearchDatabase()
        self.ai_client = OpenAICompatibleClient()

        self.setWindowTitle("Smart Search")
        self.resize(520, 460)

        self.search_worker: Optional[SearchBatchWorker] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Hierarchical Deck Selection Tree
        lbl_decks = QLabel("Mazzi in cui cercare:")
        layout.addWidget(lbl_decks)

        self.deck_tree = QTreeWidget()
        self.deck_tree.setHeaderHidden(True)
        self.deck_tree.setMaximumHeight(130)
        self.deck_tree.itemChanged.connect(self._on_tree_item_changed)
        layout.addWidget(self.deck_tree)
        self._populate_deck_tree()

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

    def _populate_deck_tree(self) -> None:
        self.deck_tree.clear()
        if not self.col:
            return

        items_by_path: Dict[str, QTreeWidgetItem] = {}
        selected_set = set(self.config.selected_decks)

        # Collect and sort all normal (non-filtered) deck names from collection
        try:
            raw_decks = self.col.decks.all()
        except Exception:
            raw_decks = self.col.decks.all_names_and_ids() if hasattr(self.col.decks, "all_names_and_ids") else []

        deck_names = sorted(
            (d.get("name", "") if isinstance(d, dict) else getattr(d, "name", ""))
            for d in raw_decks
            if (d.get("name", "") if isinstance(d, dict) else getattr(d, "name", ""))
            and not (d.get("dyn", 0) if isinstance(d, dict) else getattr(d, "dyn", 0))
        )

        self.deck_tree.blockSignals(True)
        for dname in deck_names:
            parts = dname.split("::")
            parent_item = None
            curr_path = ""
            for i, part in enumerate(parts):
                curr_path = "::".join(parts[: i + 1])
                if curr_path in items_by_path:
                    parent_item = items_by_path[curr_path]
                    continue

                item = QTreeWidgetItem()
                item.setText(0, part)
                item.setData(0, Qt.ItemDataRole.UserRole, curr_path)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

                # Check if this path (or a parent prefix) was configured as selected
                is_selected = (curr_path in selected_set) or any(
                    curr_path.startswith(f"{s}::") for s in selected_set
                )
                item.setCheckState(0, Qt.CheckState.Checked if is_selected else Qt.CheckState.Unchecked)

                if parent_item is None:
                    self.deck_tree.addTopLevelItem(item)
                    item.setExpanded(False)  # Collapsed by default
                else:
                    parent_item.addChild(item)
                    item.setExpanded(False)

                items_by_path[curr_path] = item
                parent_item = item

        # Update parent states bottom-up based on initial child states
        for top_idx in range(self.deck_tree.topLevelItemCount()):
            self._sync_subtree_check_state_bottom_up(self.deck_tree.topLevelItem(top_idx))

        self.deck_tree.blockSignals(False)

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        self.deck_tree.blockSignals(True)
        state = item.checkState(column)
        if state in (Qt.CheckState.Checked, Qt.CheckState.Unchecked):
            self._set_children_check_state(item, state)
        self._update_ancestors_check_state(item)
        self.deck_tree.blockSignals(False)
        self.config.selected_decks = self._get_selected_deck_names()
        self.config.save()

    def _set_children_check_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_children_check_state(child, state)

    def _recompute_node_check_state_from_children(self, node: QTreeWidgetItem) -> None:
        child_count = node.childCount()
        if child_count == 0:
            return
        checked_count = 0
        partially_count = 0
        for i in range(child_count):
            c_state = node.child(i).checkState(0)
            if c_state == Qt.CheckState.Checked:
                checked_count += 1
            elif c_state == Qt.CheckState.PartiallyChecked:
                partially_count += 1

        if checked_count == child_count:
            node.setCheckState(0, Qt.CheckState.Checked)
        elif checked_count > 0 or partially_count > 0:
            node.setCheckState(0, Qt.CheckState.PartiallyChecked)
        else:
            node.setCheckState(0, Qt.CheckState.Unchecked)

    def _sync_subtree_check_state_bottom_up(self, item: QTreeWidgetItem) -> None:
        for i in range(item.childCount()):
            self._sync_subtree_check_state_bottom_up(item.child(i))
        if item.childCount() > 0:
            self._recompute_node_check_state_from_children(item)

    def _update_ancestors_check_state(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        while parent is not None:
            self._recompute_node_check_state_from_children(parent)
            parent = parent.parent()

    def _get_selected_deck_names(self) -> List[str]:
        selected: List[str] = []

        def _traverse(item: QTreeWidgetItem) -> None:
            state = item.checkState(0)
            full_name = item.data(0, Qt.ItemDataRole.UserRole) or item.text(0)

            if state == Qt.CheckState.Checked:
                # The entire deck (and its subdecks, if any) is selected
                selected.append(full_name)
            elif state == Qt.CheckState.PartiallyChecked:
                # Only some children are selected, recurse
                for i in range(item.childCount()):
                    _traverse(item.child(i))

        for idx in range(self.deck_tree.topLevelItemCount()):
            _traverse(self.deck_tree.topLevelItem(idx))

        return selected

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

        selected_decks = self._get_selected_deck_names()
        if not selected_decks:
            QMessageBox.warning(self, "Attenzione", "Seleziona almeno un mazzo in cui cercare.")
            return

        self.config.selected_decks = selected_decks
        self.config.save()

        self.btn_search.setEnabled(False)
        self.btn_cancel.setText("Interrompi")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("Avvio elaborazione in corso...")

        self.search_worker = SearchBatchWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            raw_text=raw_text,
            base_deck_name=self.txt_base_deck.text().strip(),
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

        # 1. Reset Anki window so created filtered decks are immediately visible
        try:
            from aqt import mw
            if mw:
                mw.reset()
        except Exception:
            pass

        # 2. Automatically close the SearchDialog
        self.accept()

        # 3. Present clean post-search summary popup with Browser shortcut
        parent_widget = mw if mw else None
        msg_box = QMessageBox(parent_widget)
        msg_box.setWindowTitle("Ricerca completata")

        if total_added == 0 and total_already_active > 0:
            msg_box.setText(
                f"Nessuna nuova carta da riattivare.\n\n"
                f"Carte correlate già attive in studio: {total_already_active}"
            )
        else:
            msg_box.setText(
                f"Mazzi filtrati creati: {decks_created}\n"
                f"Nuove carte riattivate: {total_unsuspended}\n"
                f"Carte correlate già attive: {total_already_active}\n"
                f"Carte totali nel mazzo filtrato: {total_added}"
            )

        msg_box.addButton(QMessageBox.StandardButton.Ok)
        btn_browser = None
        created_names = [r.deck_name for r in results if r.filtered_did and r.total_cards_count > 0]
        if created_names:
            btn_browser = msg_box.addButton("Browser", QMessageBox.ButtonRole.ActionRole)

        msg_box.exec()

        if btn_browser and msg_box.clickedButton() == btn_browser:
            try:
                import aqt
                if aqt.mw:
                    query = " OR ".join(f'"deck:{name}"' for name in created_names)
                    browser = aqt.dialogs.open("Browser", aqt.mw)
                    if browser and hasattr(browser, "search_for"):
                        browser.search_for(query)
            except Exception as e:
                print(f"[SmartSearch] Error opening browser: {e}")

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
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(
            ["Data", "Mazzo Filtrato", "Unità", "Nel Mazzo", "Nuove Riattivate", "Già Attive", "Stato"]
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

        btn_layout.addStretch()

        btn_refresh = QPushButton("Aggiorna")
        btn_refresh.clicked.connect(self._refresh_history)
        btn_layout.addWidget(btn_refresh)

        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _refresh_history(self) -> None:
        runs = self.storage.get_runs()
        self.history_table.setRowCount(len(runs))

        for row_idx, r in enumerate(runs):
            t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))
            status_str = "Annullata" if r.rolled_back else "Attiva"

            item_date = QTableWidgetItem(t_str)
            item_date.setData(Qt.ItemDataRole.UserRole, r.run_id)

            self.history_table.setItem(row_idx, 0, item_date)
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(r.deck_name))
            self.history_table.setItem(row_idx, 2, QTableWidgetItem(r.unit_title))
            self.history_table.setItem(row_idx, 3, QTableWidgetItem(str(len(r.total_matched_cids))))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(str(len(r.newly_unsuspended_cids))))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(str(len(r.already_active_cids))))
            self.history_table.setItem(row_idx, 6, QTableWidgetItem(status_str))

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
        )
        self.rollback_worker.finished_sig.connect(self._on_rollback_finished)
        self.rollback_worker.error_sig.connect(self._on_rollback_error)
        self.rollback_worker.start()

    def _on_rollback_finished(self, res: Any) -> None:
        self.btn_rollback.setEnabled(True)
        if mw:
            mw.reset()
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

        reply = QMessageBox.question(
            self,
            "Elimina Solo Mazzo",
            f"Vuoi eliminare il mazzo filtrato '{run_rec.deck_name}' mantenendo tutte le carte attive nella tua collezione?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            success = delete_filtered_deck_only(self.col, self.storage, run_id)
            if success:
                QMessageBox.information(self, "Fatto", "Mazzo filtrato eliminato. Le carte restano attive.")
            else:
                QMessageBox.warning(self, "Avviso", "Il mazzo filtrato non esiste più.")


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
            target_dids = indexer.find_target_deck_ids(self.col, self.config.selected_decks)
            target_nids = indexer.get_notes_in_decks(self.col, target_dids)
            total_notes = len(target_nids)
            deck_active_count = self.storage.count_embeddings_for_notes(target_nids, active_id) if target_nids else 0
            pct = int(deck_active_count / total_notes * 100) if total_notes > 0 else 0
            self.lbl_coverage.setText(
                f"Copertura mazzi selezionati: {deck_active_count} / {total_notes} note ({pct}%)"
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

        self.index_worker = IndexWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            selected_decks=self.config.selected_decks,
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
