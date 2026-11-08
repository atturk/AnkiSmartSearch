"""Main PyQt6 Dialog for AnkiHubSearcher.

Organized into 4 tabs:
1. 🔍 Smart Search (Main Workflow)
2. 🕒 Cronologia & Rollback (Safe Undo)
3. 🗂️ Gestione Indice & Embedding Cache
4. ⚙️ Impostazioni (AI Providers, Campi & Avanzate)
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
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
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
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        mw = None
    except ImportError:
        # Running headless
        QDialog = object  # type: ignore
        mw = None


from .. import __version__
from ..ai_client import OpenAICompatibleClient
from ..config_manager import ConfigManager
from ..deck_ops import delete_filtered_deck_only
from ..indexer import DeckIndexer
from ..storage import SearchDatabase
from ..text_utils import parse_markdown_units
from .worker import IndexWorker, RollbackWorker, SearchBatchWorker


class AnkiHubSearcherDialog(QDialog):
    """Main dialog window for AnkiHubSearcher."""

    def __init__(self, col: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.col = col or (mw.col if mw else None)
        self.config = ConfigManager.load()
        self.storage = SearchDatabase()
        self.ai_client = OpenAICompatibleClient()

        self.setWindowTitle(f"AnkiHub Smart Searcher v{__version__} — Med School AI Bridge")
        self.resize(920, 680)

        self.search_worker: Optional[SearchBatchWorker] = None
        self.index_worker: Optional[IndexWorker] = None
        self.rollback_worker: Optional[RollbackWorker] = None

        self._init_ui()
        self._load_config_to_ui()
        self._refresh_history_tab()
        self._refresh_index_tab()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Smart Search
        self.tab_search = QWidget()
        self._build_search_tab(self.tab_search)
        self.tab_widget.addTab(self.tab_search, "🔍 Smart Search")

        # Tab 2: Cronologia & Rollback
        self.tab_history = QWidget()
        self._build_history_tab(self.tab_history)
        self.tab_widget.addTab(self.tab_history, "🕒 Cronologia & Rollback")

        # Tab 3: Gestione Indice
        self.tab_index = QWidget()
        self._build_index_tab(self.tab_index)
        self.tab_widget.addTab(self.tab_index, "🗂️ Gestione Indice")

        # Tab 4: Impostazioni
        self.tab_settings = QWidget()
        self._build_settings_tab(self.tab_settings)
        self.tab_widget.addTab(self.tab_settings, "⚙️ Impostazioni")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._refresh_history_tab()
        elif idx == 2:
            self._refresh_index_tab()

    # ---------------- TAB 1: SMART SEARCH ----------------

    def _build_search_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)

        # Top section: Source decks selection + Base deck name
        top_group = QGroupBox("1. Selezione Mazzi di Origine e Mazzo Filtrato")
        top_layout = QVBoxLayout(top_group)

        # Decks selection
        deck_filter_layout = QHBoxLayout()
        self.deck_filter_input = QLineEdit()
        self.deck_filter_input.setPlaceholderText("Filtra mazzi...")
        self.deck_filter_input.textChanged.connect(self._filter_deck_list)
        deck_filter_layout.addWidget(self.deck_filter_input)

        btn_sel_all = QPushButton("Tutti")
        btn_sel_all.clicked.connect(self._select_all_decks)
        deck_filter_layout.addWidget(btn_sel_all)

        btn_desel_all = QPushButton("Nessuno")
        btn_desel_all.clicked.connect(self._deselect_all_decks)
        deck_filter_layout.addWidget(btn_desel_all)
        top_layout.addLayout(deck_filter_layout)

        self.deck_list_widget = QListWidget()
        self.deck_list_widget.setMaximumHeight(100)
        self.deck_list_widget.itemChanged.connect(self._on_deck_selection_changed)
        top_layout.addWidget(self.deck_list_widget)
        self._populate_deck_list()

        # Base Deck Name
        base_deck_layout = QHBoxLayout()
        base_deck_layout.addWidget(QLabel("Nome Mazzo Base:"))
        self.base_deck_input = QLineEdit()
        self.base_deck_input.setPlaceholderText(
            "Es. Patologia Generale (opzionale se usi header Markdown # Capitolo 2 ...)"
        )
        base_deck_layout.addWidget(self.base_deck_input)
        top_layout.addLayout(base_deck_layout)
        layout.addWidget(top_group)

        # Middle section: Text Input
        mid_group = QGroupBox("2. Appunti o Trascrizione della Lezione (Italiano / Markdown)")
        mid_layout = QVBoxLayout(mid_group)

        self.lecture_text_input = QPlainTextEdit()
        self.lecture_text_input.setPlaceholderText(
            "Incolla qui gli appunti della lezione in italiano...\n"
            "Supporta sia testo semplice che dispense Markdown con capitoli e sottocapitoli (#, ##, ###)."
        )
        self.lecture_text_input.textChanged.connect(self._on_lecture_text_changed)
        mid_layout.addWidget(self.lecture_text_input)

        self.lbl_units_status = QLabel("Nessun testo inserito.")
        self.lbl_units_status.setStyleSheet("color: #7f8c8d; font-style: italic;")
        mid_layout.addWidget(self.lbl_units_status)
        layout.addWidget(mid_group)

        # Quick Options Bar
        opt_group = QGroupBox("3. Opzioni di Ricerca Rapide")
        opt_layout = QHBoxLayout(opt_group)

        # Sensitivity slider
        sens_layout = QVBoxLayout()
        self.lbl_sensitivity = QLabel(f"Sensibilità Ricerca: {int(self.config.sensitivity * 100)}%")
        sens_layout.addWidget(self.lbl_sensitivity)

        slider_box = QHBoxLayout()
        slider_box.addWidget(QLabel("Preciso"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(10, 95)
        self.sensitivity_slider.setValue(int(self.config.sensitivity * 100))
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_slider_changed)
        slider_box.addWidget(self.sensitivity_slider)
        slider_box.addWidget(QLabel("Ampio"))
        sens_layout.addLayout(slider_box)
        opt_layout.addLayout(sens_layout)

        # Checkboxes
        chk_layout = QVBoxLayout()
        self.chk_llm_verify = QCheckBox("Verifica finale con LLM (più lento)")
        self.chk_llm_verify.setChecked(self.config.enable_llm_verification)
        chk_layout.addWidget(self.chk_llm_verify)

        self.chk_include_active = QCheckBox("Inserisci nel mazzo filtrato anche le carte già attive")
        self.chk_include_active.setToolTip(
            "Se disattivato (default), nel mazzo filtrato finiscono solo le carte nuove appena unsospese, "
            "mentre il contatore ti mostra comunque quante carte correlate erano già attive nella tua collezione."
        )
        self.chk_include_active.setChecked(self.config.include_active_cards_in_filtered_deck)
        chk_layout.addWidget(self.chk_include_active)
        opt_layout.addLayout(chk_layout)
        layout.addWidget(opt_group)

        # Progress & Actions
        bottom_layout = QVBoxLayout()
        self.search_progress_bar = QProgressBar()
        self.search_progress_bar.setValue(0)
        self.search_progress_bar.setVisible(False)
        bottom_layout.addWidget(self.search_progress_bar)

        self.lbl_search_status = QLabel("")
        self.lbl_search_status.setStyleSheet("font-weight: bold;")
        bottom_layout.addWidget(self.lbl_search_status)

        action_btn_layout = QHBoxLayout()
        self.btn_run_search = QPushButton("🚀 Cerca Carte e Crea Mazzo Filtrato")
        self.btn_run_search.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; font-size: 14px; font-weight: bold; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #3498db; }"
        )
        self.btn_run_search.clicked.connect(self._start_search)
        action_btn_layout.addWidget(self.btn_run_search)

        self.btn_cancel_search = QPushButton("⏹️ Annulla")
        self.btn_cancel_search.setEnabled(False)
        self.btn_cancel_search.clicked.connect(self._cancel_search)
        action_btn_layout.addWidget(self.btn_cancel_search)

        bottom_layout.addLayout(action_btn_layout)
        layout.addLayout(bottom_layout)

    def _populate_deck_list(self) -> None:
        self.deck_list_widget.blockSignals(True)
        self.deck_list_widget.clear()
        if not self.col:
            self.deck_list_widget.blockSignals(False)
            return

        selected_set = set(self.config.selected_decks)
        try:
            all_decks = self.col.decks.all()
        except Exception:
            all_decks = []
        if not all_decks and hasattr(self.col.decks, "all_names_and_ids"):
            all_decks = self.col.decks.all_names_and_ids()

        deck_names = []
        for deck in all_decks:
            dname = deck.get("name", "") if isinstance(deck, dict) else getattr(deck, "name", "")
            if dname:
                deck_names.append(dname)

        for dname in sorted(deck_names):
            item = QListWidgetItem(dname)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if dname in selected_set or any(dname.startswith(f"{s}::") for s in selected_set):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.deck_list_widget.addItem(item)
        self.deck_list_widget.blockSignals(False)

    def _on_deck_selection_changed(self, _item: Optional[QListWidgetItem] = None) -> None:
        self.config.selected_decks = self._get_selected_deck_names()
        self.config.save()

    def _filter_deck_list(self, filter_text: str) -> None:
        filter_lower = filter_text.strip().lower()
        for idx in range(self.deck_list_widget.count()):
            item = self.deck_list_widget.item(idx)
            item.setHidden(filter_lower not in item.text().lower())

    def _select_all_decks(self) -> None:
        self.deck_list_widget.blockSignals(True)
        for idx in range(self.deck_list_widget.count()):
            item = self.deck_list_widget.item(idx)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)
        self.deck_list_widget.blockSignals(False)
        self._on_deck_selection_changed()

    def _deselect_all_decks(self) -> None:
        self.deck_list_widget.blockSignals(True)
        for idx in range(self.deck_list_widget.count()):
            item = self.deck_list_widget.item(idx)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)
        self.deck_list_widget.blockSignals(False)
        self._on_deck_selection_changed()

    def _get_selected_deck_names(self) -> List[str]:
        names = []
        for idx in range(self.deck_list_widget.count()):
            item = self.deck_list_widget.item(idx)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text())
        return names

    def _on_lecture_text_changed(self) -> None:
        text = self.lecture_text_input.toPlainText()
        if not text.strip():
            self.lbl_units_status.setText("Nessun testo inserito.")
            return

        units = parse_markdown_units(
            raw_text=text,
            base_deck_name=self.base_deck_input.text().strip(),
            naming_style=self.config.batch_deck_naming_style,
        )
        if not units:
            self.lbl_units_status.setText("⚠️ Nessuna unità con contenuto testuale rilevata.")
        elif len(units) == 1:
            self.lbl_units_status.setText("✅ 1 singola unità rilevata (testo unico).")
        else:
            self.lbl_units_status.setText(
                f"✅ Rilevate {len(units)} unità di studio distinte (header vuoti ignorati come mazzo)."
            )

    def _on_sensitivity_slider_changed(self, val: int) -> None:
        self.config.sensitivity = round(val / 100.0, 2)
        self.lbl_sensitivity.setText(f"Sensibilità Ricerca: {val}%")

    def _start_search(self) -> None:
        raw_text = self.lecture_text_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Attenzione", "Inserisci il testo della lezione prima di avviare la ricerca.")
            return

        selected_decks = self._get_selected_deck_names()
        if not selected_decks:
            QMessageBox.warning(self, "Attenzione", "Seleziona almeno un mazzo di origine in cui cercare le carte.")
            return

        # Save active options
        self.config.selected_decks = selected_decks
        self.config.enable_llm_verification = self.chk_llm_verify.isChecked()
        self.config.include_active_cards_in_filtered_deck = self.chk_include_active.isChecked()
        self.config.save()

        self.btn_run_search.setEnabled(False)
        self.btn_cancel_search.setEnabled(True)
        self.search_progress_bar.setVisible(True)
        self.search_progress_bar.setRange(0, 100)
        self.search_progress_bar.setValue(0)
        self.lbl_search_status.setText("Avvio elaborazione in corso...")

        self.search_worker = SearchBatchWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            raw_text=raw_text,
            base_deck_name=self.base_deck_input.text().strip(),
        )
        self.search_worker.progress_sig.connect(self._on_search_progress)
        self.search_worker.finished_sig.connect(self._on_search_finished)
        self.search_worker.error_sig.connect(self._on_search_error)
        self.search_worker.start()

    def _cancel_search(self) -> None:
        if self.search_worker:
            self.lbl_search_status.setText("Interruzione in corso...")
            self.search_worker.cancel()

    def _on_search_progress(self, current: int, total: int, msg: str) -> None:
        if total > 0:
            self.search_progress_bar.setMaximum(total)
            self.search_progress_bar.setValue(current)
        self.lbl_search_status.setText(msg)

    def _on_search_finished(self, res: Dict[str, Any]) -> None:
        self.btn_run_search.setEnabled(True)
        self.btn_cancel_search.setEnabled(False)
        self.search_progress_bar.setVisible(False)

        status = res.get("status", "done")
        results = res.get("results", [])

        if status == "cancelled":
            self.lbl_search_status.setText("Operazione annullata.")
            QMessageBox.information(self, "Annullato", "Ricerca interrotta dall'utente.")
        else:
            if mw:
                try:
                    mw.reset()
                except Exception:
                    pass
            total_added = sum(r.total_cards_count for r in results)
            total_unsuspended = sum(r.newly_unsuspended_count for r in results)
            total_already_active = sum(getattr(r, "already_active_count", 0) for r in results)
            decks_created = sum(1 for r in results if r.filtered_did)

            if total_added == 0 and total_already_active > 0:
                self.lbl_search_status.setText(
                    f"ℹ️ 0 nuove carte da unsospendere — {total_already_active} carte correlate erano GIÀ attive!"
                )
                QMessageBox.information(
                    self,
                    "Argomenti Già Attivi in Studio!",
                    f"ℹ️ Nessuna nuova carta da unsospendere per questa lezione!\n\n"
                    f"✅ Carte correlate che avevi GIÀ attive (non-sospese): {total_already_active}\n"
                    f"🆕 Nuove carte unsospese: 0\n\n"
                    f"Tutte le carte rilevanti trovate per questi argomenti fanno già parte del tuo studio attivo.",
                )
            else:
                self.lbl_search_status.setText(
                    f"✅ Completato! {decks_created} mazzi filtrati ({total_unsuspended} nuove unsospese, {total_already_active} già attive)."
                )
                QMessageBox.information(
                    self,
                    "Ricerca Completata!",
                    f"Creati {decks_created} mazzi filtrati con successo!\n\n"
                    f"🆕 Nuove carte unsospese: {total_unsuspended}\n"
                    f"✅ Carte correlate GIÀ attive in studio: {total_already_active}\n"
                    f"📦 Carte totali inserite nel Mazzo Filtrato: {total_added}\n\n"
                    f"Puoi iniziare subito il ripasso su Anki!",
                )

        self._refresh_history_tab()
        self._refresh_index_tab()

    def _on_search_error(self, err_msg: str) -> None:
        self.btn_run_search.setEnabled(True)
        self.btn_cancel_search.setEnabled(False)
        self.search_progress_bar.setVisible(False)
        self.lbl_search_status.setText(f"❌ Errore: {err_msg}")
        QMessageBox.critical(self, "Errore Ricerca", f"Si è verificato un errore:\n\n{err_msg}")

    # ---------------- TAB 2: HISTORY & ROLLBACK ----------------

    def _build_history_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)

        info_lbl = QLabel(
            "Cronologia delle ricerche eseguite. Il rollback ri-sospende ESCLUSIVAMENTE le carte che erano "
            "sospese prima della run, senza toccare MAI le carte già attive nella tua collezione."
        )
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels(
            ["Run ID", "Data/Ora", "Mazzo Filtrato", "Unità", "Nel Mazzo", "Nuove Unsospese", "Già Attive", "Stato"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.history_table)

        # Rollback options and buttons
        btn_layout = QHBoxLayout()
        self.chk_keep_reviewed = QCheckBox("Mantieni attive le carte che ho già ripassato (consigliato)")
        self.chk_keep_reviewed.setChecked(self.config.rollback_keep_reviewed_default)
        btn_layout.addWidget(self.chk_keep_reviewed)

        btn_refresh = QPushButton("🔄 Aggiorna")
        btn_refresh.clicked.connect(self._refresh_history_tab)
        btn_layout.addWidget(btn_refresh)
        layout.addLayout(btn_layout)

        actions_layout = QHBoxLayout()
        self.btn_rollback = QPushButton("⏪ Annulla Run Selezionata (Ri-sospendi carte e rimuovi mazzo)")
        self.btn_rollback.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
        self.btn_rollback.clicked.connect(self._on_rollback_clicked)
        actions_layout.addWidget(self.btn_rollback)

        self.btn_delete_deck_only = QPushButton("🗑️ Elimina Solo Mazzo Filtrato (Lascia carte attive)")
        self.btn_delete_deck_only.setToolTip(
            "Svuota ed elimina il mazzo filtrato ma lascia le carte unsospese nei mazzi principali per il ripasso futuro."
        )
        self.btn_delete_deck_only.clicked.connect(self._on_delete_filtered_deck_only_clicked)
        actions_layout.addWidget(self.btn_delete_deck_only)
        layout.addLayout(actions_layout)

    def _refresh_history_tab(self) -> None:
        runs = self.storage.get_runs()
        self.history_table.setRowCount(len(runs))

        for row_idx, r in enumerate(runs):
            t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))
            status_str = "❌ Annullata" if r.rolled_back else "✅ Attiva"

            self.history_table.setItem(row_idx, 0, QTableWidgetItem(r.run_id))
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(t_str))
            self.history_table.setItem(row_idx, 2, QTableWidgetItem(r.deck_name))
            self.history_table.setItem(row_idx, 3, QTableWidgetItem(r.unit_title))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(str(len(r.total_matched_cids))))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(str(len(r.newly_unsuspended_cids))))
            self.history_table.setItem(row_idx, 6, QTableWidgetItem(str(len(getattr(r, "already_active_cids", [])))))
            self.history_table.setItem(row_idx, 7, QTableWidgetItem(status_str))

    def _get_selected_run_id(self) -> Optional[str]:
        selected_rows = self.history_table.selectedItems()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        item = self.history_table.item(row, 0)
        return item.text() if item else None

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
            try:
                mw.reset()
            except Exception:
                pass
        self._refresh_history_tab()
        QMessageBox.information(
            self,
            "Rollback Eseguito",
            f"Rollback completato con successo!\n\n"
            f"Mazzo filtrato rimosso: {'Sì' if res.deck_removed else 'No/Già rimosso'}\n"
            f"Carte ri-sospese: {res.resuspended_count}\n"
            f"Carte mantenute attive perché ripassate: {res.kept_reviewed_count}",
        )

    def _on_rollback_error(self, err: str) -> None:
        self.btn_rollback.setEnabled(True)
        QMessageBox.critical(self, "Errore Rollback", f"Errore durante il rollback:\n{err}")

    def _on_delete_filtered_deck_only_clicked(self) -> None:
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
            f"Vuoi eliminare il mazzo filtrato '{run_rec.deck_name}' mantenendo tutte le carte unsospese nella tua collezione?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            success = delete_filtered_deck_only(self.col, self.storage, run_id)
            if success:
                QMessageBox.information(self, "Fatto", "Mazzo filtrato eliminato. Le carte restano attive!")
            else:
                QMessageBox.warning(self, "Avviso", "Il mazzo filtrato non esiste più.")

    # ---------------- TAB 3: INDEX MANAGEMENT ----------------

    def _build_index_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)

        info_lbl = QLabel(
            "Gestione dell'indice vettoriale e FTS5. Gli embeddings vengono calcolati per il modello attivo "
            "e salvati localmente in SQLite. Cambiare modello non cancella le cache precedenti a meno che non le elimini esplicitamente."
        )
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        # Status Box
        status_group = QGroupBox("Stato Modello Attivo")
        status_layout = QVBoxLayout(status_group)
        self.lbl_active_model_badge = QLabel("Modello Attivo: -")
        self.lbl_active_model_badge.setStyleSheet("font-size: 13px; font-weight: bold;")
        status_layout.addWidget(self.lbl_active_model_badge)

        self.lbl_indexing_coverage = QLabel("Copertura Mazzi Selezionati: -")
        status_layout.addWidget(self.lbl_indexing_coverage)

        self.index_progress_bar = QProgressBar()
        self.index_progress_bar.setVisible(False)
        status_layout.addWidget(self.index_progress_bar)

        self.lbl_index_status = QLabel("")
        status_layout.addWidget(self.lbl_index_status)

        btn_box = QHBoxLayout()
        self.btn_run_indexing = QPushButton("▶️ Indicizza / Aggiorna Ora")
        self.btn_run_indexing.clicked.connect(self._start_indexing)
        btn_box.addWidget(self.btn_run_indexing)

        self.btn_cancel_indexing = QPushButton("⏸️ Interrompi")
        self.btn_cancel_indexing.setEnabled(False)
        self.btn_cancel_indexing.clicked.connect(self._cancel_indexing)
        btn_box.addWidget(self.btn_cancel_indexing)
        status_layout.addLayout(btn_box)
        layout.addWidget(status_group)

        # Cached models table
        models_group = QGroupBox("Cache Modelli nel Database Locale")
        models_layout = QVBoxLayout(models_group)
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(3)
        self.models_table.setHorizontalHeaderLabels(["Modello ID", "Note con Embedding", "Spazio su Disco (MB)"])
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        models_layout.addWidget(self.models_table)

        purge_layout = QHBoxLayout()
        btn_export_db = QPushButton("📤 Esporta Indice (.db)...")
        btn_export_db.clicked.connect(self._export_db)
        purge_layout.addWidget(btn_export_db)

        btn_import_db = QPushButton("📥 Importa / Unisci da altro .db...")
        btn_import_db.clicked.connect(self._import_external_db)
        purge_layout.addWidget(btn_import_db)

        btn_purge_inactive = QPushButton("🧹 Elimina Cache Modelli Non Attivi")
        btn_purge_inactive.clicked.connect(self._purge_inactive_models)
        purge_layout.addWidget(btn_purge_inactive)

        btn_refresh_idx = QPushButton("🔄 Aggiorna")
        btn_refresh_idx.clicked.connect(self._refresh_index_tab)
        purge_layout.addWidget(btn_refresh_idx)
        models_layout.addLayout(purge_layout)
        layout.addWidget(models_group)

    def _export_db(self) -> None:
        default_path = os.path.expanduser("~/Desktop/ankihub_searcher_export.db")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta Indice AnkiHubSearcher (.db)",
            default_path,
            "SQLite Database (*.db);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.endswith(".db"):
            file_path += ".db"
        try:
            size_mb = self.storage.export_db(file_path)
            QMessageBox.information(
                self,
                "Esportazione Completata",
                f"Indice esportato con successo!\n\n"
                f"File: {file_path}\n"
                f"Dimensione: {size_mb} MB\n\n"
                f"Ora puoi inviarlo via AirDrop sull'altro Mac e importarlo con '📥 Importa / Unisci da altro .db...'.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore Esportazione", f"Impossibile esportare il database:\n{e}")

    def _import_external_db(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona il file ankihub_searcher.db da importare/unire",
            "",
            "SQLite Database (*.db);;All Files (*)",
        )
        if not file_path:
            return
        try:
            res = self.storage.merge_external_db(file_path)
            self._refresh_index_tab()
            QMessageBox.information(
                self,
                "Unione Indice Completata",
                f"Indice unito con successo!\n\n"
                f"Vettori importati/aggiornati: {res['embeddings_merged']}\n"
                f"Note importate: {res['notes_merged']}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore Importazione", f"Impossibile unire il database:\n{e}")

    def _refresh_index_tab(self) -> None:
        self.config.selected_decks = self._get_selected_deck_names()
        active_id = self.config.get_active_model_id()
        self.lbl_active_model_badge.setText(f"Modello Attivo: {active_id}")

        stats = self.storage.get_indexed_models_stats()
        self.models_table.setRowCount(len(stats))
        for r_idx, s in enumerate(stats):
            self.models_table.setItem(r_idx, 0, QTableWidgetItem(s["model_id"]))
            self.models_table.setItem(r_idx, 1, QTableWidgetItem(str(s["note_count"])))
            self.models_table.setItem(r_idx, 2, QTableWidgetItem(f"{s['size_mb']} MB"))

        # Calculate coverage of currently checked decks in Tab 1
        if self.col:
            indexer = DeckIndexer(self.config, self.storage, self.ai_client)
            target_dids = indexer.find_target_deck_ids(self.col, self.config.selected_decks)
            target_nids = indexer.get_notes_in_decks(self.col, target_dids)
            total_notes = len(target_nids)
            covered_notes = self.storage.count_embeddings_for_notes(target_nids, active_id)
            pct = int(covered_notes / total_notes * 100) if total_notes > 0 else 0
            decks_summary = ", ".join(self.config.selected_decks[:3])
            if len(self.config.selected_decks) > 3:
                decks_summary += f" (+{len(self.config.selected_decks) - 3})"
            if not decks_summary:
                decks_summary = "Nessun mazzo selezionato"
            self.lbl_indexing_coverage.setText(
                f"Copertura Mazzi Selezionati [{decks_summary}]: {covered_notes} / {total_notes} note ({pct}%)"
            )

    def _start_indexing(self) -> None:
        if not self.col:
            return

        self.config.selected_decks = self._get_selected_deck_names()
        if not self.config.selected_decks:
            QMessageBox.warning(self, "Attenzione", "Seleziona almeno un mazzo nella tab Smart Search.")
            return
        self.config.save()

        self.btn_run_indexing.setEnabled(False)
        self.btn_cancel_indexing.setEnabled(True)
        self.index_progress_bar.setVisible(True)
        self.index_progress_bar.setValue(0)
        self.lbl_index_status.setText("Avvio indicizzazione...")

        self.index_worker = IndexWorker(
            col=self.col,
            config=self.config,
            storage=self.storage,
            ai_client=self.ai_client,
            selected_decks=self.config.selected_decks,
        )
        self.index_worker.progress_sig.connect(self._on_index_progress)
        self.index_worker.finished_sig.connect(self._on_index_finished)
        self.index_worker.error_sig.connect(self._on_index_error)
        self.index_worker.start()

    def _cancel_indexing(self) -> None:
        if self.index_worker:
            self.lbl_index_status.setText("Interruzione in corso...")
            self.index_worker.cancel()

    def _on_index_progress(self, cur: int, total: int, msg: str) -> None:
        if total > 0:
            self.index_progress_bar.setMaximum(total)
            self.index_progress_bar.setValue(cur)
        self.lbl_index_status.setText(msg)

    def _on_index_finished(self, res: Dict[str, Any]) -> None:
        self.btn_run_indexing.setEnabled(True)
        self.btn_cancel_indexing.setEnabled(False)
        self.index_progress_bar.setVisible(False)
        self.lbl_index_status.setText("Indicizzazione completata / in pausa.")
        self._refresh_index_tab()
        QMessageBox.information(
            self,
            "Indicizzazione",
            f"Operazione completata!\n\n"
            f"Note con testo indicizzato: {res.get('text_indexed', 0)}\n"
            f"Nuovi vettori calcolati: {res.get('embeddings_indexed', 0)}\n"
            f"Stato: {'Interrotta' if res.get('cancelled') else 'Completata'}",
        )

    def _on_index_error(self, err: str) -> None:
        self.btn_run_indexing.setEnabled(True)
        self.btn_cancel_indexing.setEnabled(False)
        self.index_progress_bar.setVisible(False)
        self.lbl_index_status.setText(f"Errore: {err}")
        QMessageBox.critical(self, "Errore Indicizzazione", f"Errore:\n{err}")

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
            self._refresh_index_tab()
            QMessageBox.information(self, "Pulizia Completata", f"Eliminati {deleted} vettori inattivi.")

    # ---------------- TAB 4: SETTINGS ----------------

    def _build_settings_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)

        # 1. Chat LLM
        chat_group = QGroupBox("Chat / Reasoning LLM (Decomposizione Concettuale Stage 1)")
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

        btn_fetch_chat = QPushButton("🔄 Carica Modelli")
        btn_fetch_chat.clicked.connect(self._fetch_chat_models)
        row_chat_model.addWidget(btn_fetch_chat)

        btn_test_chat = QPushButton("Test Connessione")
        btn_test_chat.clicked.connect(self._test_chat_connection)
        row_chat_model.addWidget(btn_test_chat)
        chat_layout.addLayout(row_chat_model)
        layout.addWidget(chat_group)

        # 2. Embedding Model
        embed_group = QGroupBox("Embedding Model (Indice Vettoriale Stage 2)")
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

        btn_fetch_embed = QPushButton("🔄 Carica Modelli")
        btn_fetch_embed.clicked.connect(self._fetch_embed_models)
        row_embed_model.addWidget(btn_fetch_embed)

        btn_test_embed = QPushButton("Test Connessione")
        btn_test_embed.clicked.connect(self._test_embed_connection)
        row_embed_model.addWidget(btn_test_embed)
        embed_layout.addLayout(row_embed_model)

        row_strategy = QHBoxLayout()
        row_strategy.addWidget(QLabel("Strategia Linguistica:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItem("Traduzione concettuale in Inglese (Consigliato per AnKing)", "translate_to_en")
        self.combo_strategy.addItem("Multilingua Diretto (Italiano + Inglese)", "multilingual_direct")
        row_strategy.addWidget(self.combo_strategy)
        embed_layout.addLayout(row_strategy)
        layout.addWidget(embed_group)

        # 3. Naming & Advanced
        adv_group = QGroupBox("Stile Mazzi Filtrati Multi-Unità, Campi Indicizzati & Opzioni Avanzate")
        adv_layout = QVBoxLayout(adv_group)

        row_naming = QHBoxLayout()
        row_naming.addWidget(QLabel("Stile Nomi Mazzi Batch:"))
        self.combo_naming = QComboBox()
        self.combo_naming.addItem("Opzione A: Nome Base - Titolo Header", "dash")
        self.combo_naming.addItem("Opzione B: Nome Base::Titolo Header (Sottomazzi)", "subdeck")
        row_naming.addWidget(self.combo_naming)
        adv_layout.addLayout(row_naming)

        self.chk_index_tags = QCheckBox("Indicizza anche i Tag delle note (filtrando il boilerplate)")
        self.chk_index_tags.setChecked(self.config.index_tags)
        adv_layout.addWidget(self.chk_index_tags)

        # Note Type -> Field selector
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Tipo di Nota (Campi da indicizzare):"))
        self.combo_note_model = QComboBox()
        self.combo_note_model.currentIndexChanged.connect(self._on_note_model_selected)
        fields_row.addWidget(self.combo_note_model, stretch=1)
        adv_layout.addLayout(fields_row)

        self.fields_list_widget = QListWidget()
        self.fields_list_widget.setMaximumHeight(95)
        self.fields_list_widget.itemChanged.connect(self._on_field_check_changed)
        adv_layout.addWidget(self.fields_list_widget)
        self._populate_note_models_combo()

        layout.addWidget(adv_group)

        # Save Button
        btn_save = QPushButton("💾 Salva Impostazioni")
        btn_save.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self._save_settings)
        layout.addWidget(btn_save)

    def _populate_note_models_combo(self) -> None:
        self._model_fields_map: Dict[str, List[str]] = {}
        self.combo_note_model.blockSignals(True)
        self.combo_note_model.clear()
        if self.col:
            try:
                for item in self.col.models.all_names_and_ids():
                    mid_val = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                    if mid_val is not None:
                        m = self.col.models.get(mid_val)
                        if m and "name" in m and "flds" in m:
                            mname = m["name"]
                            flds = [f["name"] for f in m["flds"]]
                            self._model_fields_map[mname] = flds
            except Exception:
                pass
        for mname in sorted(self._model_fields_map.keys()):
            self.combo_note_model.addItem(mname)
        self.combo_note_model.blockSignals(False)
        if self.combo_note_model.count() > 0:
            self._on_note_model_selected(0)

    def _on_note_model_selected(self, _idx: int) -> None:
        mname = self.combo_note_model.currentText()
        if not mname or mname not in self._model_fields_map:
            return
        avail_fields = self._model_fields_map[mname]
        active_fields = set(self.config.get_indexed_fields_for_model(mname, avail_fields))

        self.fields_list_widget.blockSignals(True)
        self.fields_list_widget.clear()
        for fname in avail_fields:
            item = QListWidgetItem(fname)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if fname in active_fields else Qt.CheckState.Unchecked)
            self.fields_list_widget.addItem(item)
        self.fields_list_widget.blockSignals(False)

    def _on_field_check_changed(self, _item: QListWidgetItem) -> None:
        mname = self.combo_note_model.currentText()
        if not mname:
            return
        checked = []
        for idx in range(self.fields_list_widget.count()):
            it = self.fields_list_widget.item(idx)
            if it.checkState() == Qt.CheckState.Checked:
                checked.append(it.text())
        if checked:
            self.config.indexed_fields_by_model[mname] = checked

    def _load_config_to_ui(self) -> None:
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

        # Strategy
        idx = self.combo_strategy.findData(self.config.embed_language_strategy)
        if idx != -1:
            self.combo_strategy.setCurrentIndex(idx)

        # Naming style
        idx_naming = self.combo_naming.findData(self.config.batch_deck_naming_style)
        if idx_naming != -1:
            self.combo_naming.setCurrentIndex(idx_naming)

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
            QMessageBox.information(self, "Modelli Caricati", f"Trovati {len(models)} modelli disponibili!")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare i modelli:\n{e}")

    def _fetch_embed_models(self) -> None:
        url = self.txt_embed_url.text().strip()
        key = self.txt_embed_key.text().strip()
        try:
            models = self.ai_client.list_models(url, key)
            self.combo_embed_model.clear()
            self.combo_embed_model.addItems(models)
            QMessageBox.information(self, "Modelli Caricati", f"Trovati {len(models)} modelli disponibili!")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare i modelli:\n{e}")

    def _test_chat_connection(self) -> None:
        url = self.txt_chat_url.text().strip()
        key = self.txt_chat_key.text().strip()
        ok = self.ai_client.test_connection(url, key)
        if ok:
            QMessageBox.information(self, "Connessione OK", "Connessione riuscita all'endpoint Chat LLM!")
        else:
            QMessageBox.critical(self, "Errore Connessione", "Impossibile connettersi all'endpoint specificato.")

    def _test_embed_connection(self) -> None:
        url = self.txt_embed_url.text().strip()
        key = self.txt_embed_key.text().strip()
        ok = self.ai_client.test_connection(url, key)
        if ok:
            QMessageBox.information(self, "Connessione OK", "Connessione riuscita all'endpoint Embedding Model!")
        else:
            QMessageBox.critical(self, "Errore Connessione", "Impossibile connettersi all'endpoint specificato.")

    def _save_settings(self) -> None:
        preset_map = {0: "lm_studio", 1: "openrouter", 2: "custom"}
        self.config.chat_provider = preset_map.get(self.combo_chat_preset.currentIndex(), "custom")
        self.config.chat_base_url = self.txt_chat_url.text().strip()
        self.config.chat_api_key = self.txt_chat_key.text().strip()
        self.config.chat_model = self.combo_chat_model.currentText().strip()

        self.config.embed_provider = preset_map.get(self.combo_embed_preset.currentIndex(), "custom")
        self.config.embed_base_url = self.txt_embed_url.text().strip()
        self.config.embed_api_key = self.txt_embed_key.text().strip()
        self.config.embed_model = self.combo_embed_model.currentText().strip()

        self.config.embed_language_strategy = self.combo_strategy.currentData() or "translate_to_en"
        self.config.batch_deck_naming_style = self.combo_naming.currentData() or "dash"
        self.config.index_tags = self.chk_index_tags.isChecked()

        self.config.save()
        self._refresh_index_tab()
        QMessageBox.information(self, "Salvataggio", "Impostazioni salvate con successo!")
