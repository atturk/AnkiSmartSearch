"""Browser integration for Smart Search.

Registers the "Smart Search" menu inside Anki's Browser window:
1. "Cerca con la query corrente…": opens SearchDialog pre-filled with the current browser search query.
2. "Trova carte simili" (Ctrl+Alt+M): enabled only when at least 1 card is selected;
   computes cosine similarity across indexed notes using the normalized mean vector of selected notes,
   and displays matching note IDs in the browser search bar.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    from aqt import gui_hooks, mw  # type: ignore
    from aqt.operations import QueryOp  # type: ignore
    from aqt.qt import QAction, QMenu  # type: ignore
    from aqt.utils import showInfo, tooltip  # type: ignore
except ImportError:
    gui_hooks = None
    mw = None
    QueryOp = None
    QAction = None
    QMenu = None
    showInfo = None
    tooltip = None

from .config_manager import ConfigManager
from .search_pipeline import find_similar_notes
from .storage import SearchDatabase


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
        scope = getattr(config, "similar_scope", "query")
        candidate_nids: Optional[Set[int]] = None

        if scope == "query":
            q = config.get_effective_search_query()
            if q and col and hasattr(col, "find_notes"):
                candidate_nids = set(col.find_notes(q))
        elif scope == "custom":
            q = getattr(config, "similar_custom_query", "").strip()
            if q and col and hasattr(col, "find_notes"):
                candidate_nids = set(col.find_notes(q))

        matched, missing = find_similar_notes(
            storage=storage,
            selected_nids=selected_nids,
            active_model_id=active_mid,
            candidate_nids=candidate_nids,
            min_score=getattr(config, "similar_min_score", 0.70),
            max_notes=getattr(config, "similar_max_notes", 20),
        )

        if missing:
            deck_name = get_deck_name_for_note(col, missing[0]) if col else ""
            if deck_name:
                msg = f"Nota non indicizzata: indicizza il mazzo '{deck_name}'."
            else:
                msg = "Nota non indicizzata: indicizza prima le note nel database di Smart Search."
            return {"error": msg}

        return {
            "matched": matched,
            "min_score": getattr(config, "similar_min_score", 0.70),
        }

    def _on_success(res: Dict[str, Any]) -> None:
        if "error" in res:
            show_browser_info(browser, res["error"])
            return

        matched = res.get("matched", [])
        min_score = res.get("min_score", 0.70)
        if not matched:
            show_browser_tooltip(
                browser, f"Nessuna carta simile trovata con similarità ≥ {min_score:.2f}"
            )
            return

        nids_csv = ",".join(str(nid) for nid, _ in matched)
        set_browser_search(browser, f"nid:{nids_csv}")
        show_browser_tooltip(browser, f"Trovate {len(matched)} note simili")

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


def setup_browser_menu() -> None:
    """Hooks the Smart Search menu into Anki's browser window."""
    if gui_hooks is None:
        return
    try:
        gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
        gui_hooks.browser_did_change_row.append(on_browser_did_change_row)
    except Exception as e:
        print(f"[SmartSearch] Error registering browser menu hooks: {e}")
