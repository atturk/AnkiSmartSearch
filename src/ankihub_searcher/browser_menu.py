"""Browser integration for Smart Search.

Registers the "Smart Search" menu inside Anki's Browser window:
1. "Cerca con la query corrente…": opens SearchDialog pre-filled with the current browser search query.
2. "Trova carte simili" (Ctrl+Alt+M): enabled only when at least 1 card is selected;
   computes cosine similarity across indexed notes using the normalized mean vector of selected notes,
   and displays matching note IDs in the browser search bar.
3. Custom browser column "Smart Search: similarità" showing similarity scores of the last search.
4. Non-modal panel "Carte simili" listing ranked notes with double-click selection in the browser.
5. Context menu item in the browser table rows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from aqt import gui_hooks, mw  # type: ignore
    from aqt.operations import QueryOp  # type: ignore
    from aqt.qt import (  # type: ignore
        QAction,
        QDialog,
        QHeaderView,
        QLabel,
        QMenu,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        Qt,
        QVBoxLayout,
    )
    from aqt.utils import showInfo, tooltip  # type: ignore
except ImportError:
    gui_hooks = None
    mw = None
    QueryOp = None
    QAction = None
    QDialog = object  # type: ignore
    QHeaderView = None
    QLabel = None
    QMenu = None
    QPushButton = None
    QTableWidget = None
    QTableWidgetItem = None
    Qt = None
    QVBoxLayout = None
    showInfo = None
    tooltip = None

from .config_manager import ConfigManager
from .search_pipeline import find_similar_notes
from .storage import SearchDatabase

# Global caches for the similarity column in Anki Browser
_last_similar_scores: Dict[int, float] = {}
_cid_to_nid_cache: Dict[int, int] = {}


def get_browser_search_text(browser: Any) -> str:
    """Safely retrieves the current search query string from Anki's Browser."""
    if hasattr(browser, "current_search"):
        try:
            val = browser.current_search()
            if val:
                return str(val).strip()
        except Exception:
            pass

    for form_attr in ("form", "searchEdit"):
        obj = getattr(browser, form_attr, None)
        if obj and hasattr(obj, "searchEdit"):
            try:
                edit = obj.searchEdit
                if hasattr(edit, "lineEdit") and edit.lineEdit():
                    text = edit.lineEdit().text()
                    if text:
                        return str(text).strip()
            except Exception:
                pass
        elif obj and hasattr(obj, "lineEdit") and obj.lineEdit():
            try:
                text = obj.lineEdit().text()
                if text:
                    return str(text).strip()
            except Exception:
                pass
    return ""


def set_browser_search(browser: Any, query: str) -> None:
    """Sets the search query in the browser search bar and triggers execution."""
    if hasattr(browser, "search_for"):
        browser.search_for(query)
    elif hasattr(browser, "set_filter"):
        browser.set_filter(query)
    else:
        try:
            if hasattr(browser, "form") and hasattr(browser.form, "searchEdit"):
                edit = browser.form.searchEdit
                if hasattr(edit, "lineEdit") and edit.lineEdit():
                    edit.lineEdit().setText(query)
            if hasattr(browser, "onSearchActivated"):
                browser.onSearchActivated()
            elif hasattr(browser, "onSearch"):
                browser.onSearch(query)
        except Exception as e:
            print(f"[SmartSearch] Error setting browser search: {e}")


def show_browser_tooltip(browser: Any, text: str) -> None:
    """Displays a native Anki tooltip or logs fallback."""
    if tooltip is not None:
        try:
            tooltip(text, parent=browser)
            return
        except Exception:
            pass
    print(f"[SmartSearch] Tooltip: {text}")


def show_browser_info(browser: Any, text: str) -> None:
    """Displays an informative modal message in the browser window."""
    if showInfo is not None:
        try:
            showInfo(text, parent=browser)
            return
        except Exception:
            pass
    print(f"[SmartSearch] Info: {text}")


def get_deck_name_for_note(col: Any, nid: int) -> str:
    """Determines the deck name containing the given note ID."""
    if not col:
        return ""
    try:
        note = col.get_note(nid)
        cids = note.card_ids()
        if cids:
            card = col.get_card(cids[0])
            return str(col.decks.name(card.did))
    except Exception:
        pass
    return ""


def select_note_in_browser(browser: Any, nid: int) -> None:
    """Attempts to select and scroll to the note in the Browser table, or searches for it."""
    selected = False
    try:
        col = getattr(browser, "col", None) or (mw.col if mw else None)
        if col:
            note = col.get_note(nid)
            cids = note.card_ids()
            if cids and hasattr(browser, "table") and hasattr(browser.table, "select_single_card"):
                browser.table.select_single_card(cids[0])
                selected = True
    except Exception:
        selected = False

    if not selected:
        set_browser_search(browser, f"nid:{nid}")


class SimilarNotesDialog(QDialog):
    """Small non-modal panel displaying ranked similar notes.

    Double-clicking an item selects the corresponding note in the Anki Browser.
    """

    def __init__(
        self,
        browser: Any,
        matched: List[Tuple[int, float]],
        min_score: float,
        note_previews: Dict[int, str],
        parent: Optional[Any] = None,
    ):
        if QDialog is not object:
            super().__init__(parent or browser)
        self.browser = browser
        self.matched = matched
        self.min_score = min_score
        self.note_previews = note_previews

        if hasattr(self, "setWindowTitle"):
            self.setWindowTitle("Carte simili")
            self.resize(540, 360)
            if Qt is not None and hasattr(Qt, "WindowModality"):
                self.setWindowModality(Qt.WindowModality.NonModal)
            self._init_ui()

    def _init_ui(self) -> None:
        if QVBoxLayout is None or QTableWidget is None:
            return

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        lbl_info = QLabel(
            f"Trovate {len(self.matched)} note simili (soglia minima: {self.min_score:.2f}).\n"
            "Doppio clic su una riga per selezionare la nota nel Browser:"
        )
        layout.addWidget(lbl_info)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Similarità", "Prima riga della nota"])
        if hasattr(self.table, "horizontalHeader") and self.table.horizontalHeader():
            try:
                self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
                self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            except Exception:
                pass
        try:
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        except Exception:
            pass
        self.table.setRowCount(len(self.matched))

        for row_idx, (nid, score) in enumerate(self.matched):
            preview_text = self.note_previews.get(nid, "")
            lines = [l.strip() for l in preview_text.splitlines() if l.strip()]
            first_line = lines[0] if lines else f"Nota {nid}"

            item_score = QTableWidgetItem(f"{score:.2f}")
            if hasattr(item_score, "setTextAlignment") and Qt is not None:
                item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if hasattr(item_score, "setData") and Qt is not None:
                item_score.setData(Qt.ItemDataRole.UserRole, nid)

            item_text = QTableWidgetItem(first_line)
            if hasattr(item_text, "setData") and Qt is not None:
                item_text.setData(Qt.ItemDataRole.UserRole, nid)

            self.table.setItem(row_idx, 0, item_score)
            self.table.setItem(row_idx, 1, item_text)

        if hasattr(self.table, "itemDoubleClicked"):
            self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.table)

    def _on_item_double_clicked(self, item: Any) -> None:
        if Qt is not None and hasattr(item, "data"):
            nid = item.data(Qt.ItemDataRole.UserRole)
            if nid:
                select_note_in_browser(self.browser, int(nid))


def on_find_similar_cards_clicked(browser: Any) -> None:
    """Executes similarity retrieval for the selected cards in background via QueryOp."""
    selected_nids: List[int] = []
    if hasattr(browser, "selected_notes"):
        try:
            selected_nids = list(browser.selected_notes() or [])
        except Exception:
            selected_nids = []

    if not selected_nids:
        return

    config = ConfigManager.load()
    active_mid = config.get_active_model_id()
    if not active_mid:
        show_browser_info(
            browser,
            "Nessun modello di embedding configurato nelle Impostazioni di Smart Search.",
        )
        return

    def _background_op(col: Any) -> Dict[str, Any]:
        storage = SearchDatabase()
        scope = getattr(config, "similar_scope", "all")
        candidate_nids: Optional[Set[int]] = None

        if scope == "query":
            q = config.get_effective_search_query()
            if q and col and hasattr(col, "find_notes"):
                candidate_nids = set(col.find_notes(q))
        elif scope == "custom":
            q = getattr(config, "similar_custom_query", "").strip()
            if q and col and hasattr(col, "find_notes"):
                candidate_nids = set(col.find_notes(q))
        # if scope == "all": candidate_nids remains None (all indexed notes for active model)

        matched, missing = find_similar_notes(
            storage=storage,
            selected_nids=selected_nids,
            active_model_id=active_mid,
            candidate_nids=candidate_nids,
            min_score=getattr(config, "similar_min_score", 0.65),
            max_notes=getattr(config, "similar_max_notes", 20),
        )

        if missing:
            deck_name = get_deck_name_for_note(col, missing[0]) if col else ""
            if deck_name:
                msg = f"Nota non indicizzata: indicizza il mazzo '{deck_name}'."
            else:
                msg = "Nota non indicizzata: indicizza prima le note nel database di Smart Search."
            return {"error": msg}

        # Fetch preview text for matched notes
        note_previews: Dict[int, str] = {}
        if matched:
            content_map = storage.get_notes_content([n for n, _ in matched])
            for n, _ in matched:
                note_previews[n] = content_map.get(n, {}).get("clean_text", "")

        return {
            "matched": matched,
            "min_score": getattr(config, "similar_min_score", 0.65),
            "note_previews": note_previews,
        }

    def _on_success(res: Dict[str, Any]) -> None:
        if "error" in res:
            show_browser_info(browser, res["error"])
            return

        matched = res.get("matched", [])
        min_score = res.get("min_score", 0.65)
        note_previews = res.get("note_previews", {})

        if not matched:
            show_browser_tooltip(
                browser, f"Nessuna carta simile trovata con similarità ≥ {min_score:.2f}"
            )
            return

        # Update similarity column caches
        _last_similar_scores.clear()
        _last_similar_scores.update({nid: score for nid, score in matched})

        col = getattr(browser, "col", None) or (mw.col if mw else None)
        if col and matched:
            try:
                nids_tuple = tuple(nid for nid, _ in matched)
                if len(nids_tuple) == 1:
                    q = f"SELECT id, nid FROM cards WHERE nid = {nids_tuple[0]}"
                else:
                    q = f"SELECT id, nid FROM cards WHERE nid IN {nids_tuple}"
                for cid, nid in col.db.execute(q):
                    _cid_to_nid_cache[cid] = nid
            except Exception:
                pass

        nids_csv = ",".join(str(nid) for nid, _ in matched)
        set_browser_search(browser, f"nid:{nids_csv}")
        show_browser_tooltip(browser, f"Trovate {len(matched)} note simili")

        # Redraw browser table cells to refresh similarity column
        if hasattr(browser, "table") and hasattr(browser.table, "redraw_cells"):
            try:
                browser.table.redraw_cells()
            except Exception:
                pass

        # Show non-modal panel "Carte simili"
        try:
            panel = SimilarNotesDialog(
                browser=browser,
                matched=matched,
                min_score=min_score,
                note_previews=note_previews,
            )
            if hasattr(panel, "show"):
                panel.show()
                if hasattr(panel, "raise_"):
                    panel.raise_()
                if hasattr(panel, "activateWindow"):
                    panel.activateWindow()
            setattr(browser, "_smart_search_similar_panel", panel)
        except Exception as e:
            print(f"[SmartSearch] Error presenting SimilarNotesDialog: {e}")

    # In Anki: use QueryOp with background thread and progress spinner
    if QueryOp is not None:
        QueryOp(
            parent=browser,
            op=_background_op,
            success=_on_success,
        ).with_progress(label="Ricerca carte simili...").run_in_background()
    else:
        # Standalone / test fallback
        col = getattr(browser, "col", None) or (mw.col if mw else None)
        res = _background_op(col)
        _on_success(res)


def _open_search_from_browser(browser: Any) -> None:
    """Opens SearchDialog pre-filled with the browser's current search query."""
    query = get_browser_search_text(browser)
    try:
        from .ui.main_dialog import SearchDialog

        col = getattr(browser, "col", None) or (mw.col if mw else None)
        dialog = SearchDialog(col=col, parent=browser, initial_query=query)
        dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening SearchDialog from browser: {e}")


def _update_similar_action_state(browser: Any) -> None:
    """Enables 'Trova carte simili' only when at least 1 card is selected in the browser."""
    act = getattr(browser, "_smart_search_similar_action", None)
    if act is not None:
        try:
            has_sel = (
                bool(browser.selected_notes())
                if hasattr(browser, "selected_notes")
                else False
            )
            act.setEnabled(has_sel)
        except Exception:
            pass


def on_browser_menus_did_init(browser: Any) -> None:
    """Adds the Smart Search menu to the Browser menubar."""
    try:
        menubar = getattr(browser.form, "menubar", None) or getattr(browser, "menubar", None)
        if not menubar:
            return

        menu = menubar.addMenu("Smart Search")

        act_search = menu.addAction("Cerca con la query corrente…")
        act_search.triggered.connect(lambda _checked=False: _open_search_from_browser(browser))

        act_similar = menu.addAction("Trova carte simili")
        if hasattr(act_similar, "setShortcut"):
            # Scorciatoia Ctrl+Alt+M: verificato sul form del Browser di Anki che Ctrl+Shift+S
            # è già occupata da "Riposiziona" e Ctrl+Alt+S è occupata da "Inverti selezione".
            act_similar.setShortcut("Ctrl+Alt+M")
        act_similar.triggered.connect(lambda _checked=False: on_find_similar_cards_clicked(browser))
        setattr(browser, "_smart_search_similar_action", act_similar)

        if hasattr(menu, "aboutToShow") and hasattr(menu.aboutToShow, "connect"):
            menu.aboutToShow.connect(lambda: _update_similar_action_state(browser))

        _update_similar_action_state(browser)
    except Exception as e:
        print(f"[SmartSearch] Error initializing browser menu: {e}")


def on_browser_did_change_row(browser: Any) -> None:
    """Updates action enabled state when the selected row changes."""
    _update_similar_action_state(browser)


def on_browser_will_show_context_menu(browser: Any, menu: Any) -> None:
    """Adds 'Trova carte simili' to the Browser row context menu."""
    try:
        act = menu.addAction("Smart Search: Trova carte simili")
        if hasattr(act, "setShortcut"):
            act.setShortcut("Ctrl+Alt+M")
        has_sel = bool(browser.selected_notes()) if hasattr(browser, "selected_notes") else False
        act.setEnabled(has_sel)
        act.triggered.connect(lambda _checked=False: on_find_similar_cards_clicked(browser))
    except Exception as e:
        print(f"[SmartSearch] Error in browser_will_show_context_menu: {e}")


def on_browser_did_fetch_columns(columns: dict[str, Any]) -> None:
    """Registers the 'Smart Search: similarità' custom column in Anki Browser."""
    try:
        from aqt.browser import Column

        columns["smart_search_similarity"] = Column(
            key="smart_search_similarity",
            cards_mode_label="Smart Search: similarità",
            notes_mode_label="Smart Search: similarità",
        )
    except Exception:
        pass


def on_browser_did_fetch_row(
    item: Any, is_notes_mode: bool, row: Any, active_columns: Sequence[str]
) -> None:
    """Populates the similarity cell for a row if the similarity column is active."""
    if "smart_search_similarity" not in active_columns:
        return
    idx = active_columns.index("smart_search_similarity")
    if not hasattr(row, "cells") or idx >= len(row.cells):
        return

    score: Optional[float] = None
    if is_notes_mode:
        score = _last_similar_scores.get(item)
    else:
        if item in _cid_to_nid_cache:
            nid = _cid_to_nid_cache[item]
            score = _last_similar_scores.get(nid)
        else:
            try:
                from aqt import mw

                if mw and mw.col:
                    card = mw.col.get_card(item)
                    if card:
                        _cid_to_nid_cache[item] = card.nid
                        score = _last_similar_scores.get(card.nid)
            except Exception:
                pass

    score_str = f"{score:.2f}" if score is not None else ""
    try:
        from aqt.browser import Cell

        row.cells[idx] = Cell(text=score_str)
    except Exception:
        if hasattr(row.cells[idx], "text"):
            row.cells[idx].text = score_str


def setup_browser_menu() -> None:
    """Hooks the Smart Search menu, context menu, and similarity column into Anki's browser window."""
    if gui_hooks is None:
        return
    try:
        gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
        gui_hooks.browser_did_change_row.append(on_browser_did_change_row)
        if hasattr(gui_hooks, "browser_will_show_context_menu"):
            gui_hooks.browser_will_show_context_menu.append(on_browser_will_show_context_menu)
        if hasattr(gui_hooks, "browser_did_fetch_columns"):
            gui_hooks.browser_did_fetch_columns.append(on_browser_did_fetch_columns)
        if hasattr(gui_hooks, "browser_did_fetch_row"):
            gui_hooks.browser_did_fetch_row.append(on_browser_did_fetch_row)
    except Exception as e:
        print(f"[SmartSearch] Error registering browser menu hooks: {e}")
