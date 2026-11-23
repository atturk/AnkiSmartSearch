"""Browser integration for Smart Search.

Registers the "Smart Search" menu inside Anki's Browser window:
1. "Cerca con la query corrente…": opens SearchDialog pre-filled with the current browser search query.
2. "Trova carte simili (stessa finestra)" (Ctrl+Alt+M) and "Trova carte simili (nuova finestra)" (Ctrl+Shift+Alt+M):
   enabled only when at least 1 card is selected; supports multi-criteria search across
   semantic embeddings, hierarchical IDF-weighted tags, and syntactic/structural sentence similarity.
3. Submenu "Trova carte simili per criterio" for dedicated Embedding / Tag / Structure searches.
4. Custom browser column "Smart Search: similarità" showing similarity scores and active criterion breakdown.
5. Context menu items in the browser table rows.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
from .debug_log import log_error, log_info, log_warning
from .search_pipeline import find_similar_notes
from .storage import SearchDatabase

# Global caches for the similarity column in Anki Browser
_last_similar_scores: Dict[int, float] = {}
_last_similar_details: Dict[int, Dict[str, float]] = {}
_last_similar_query: str = ""
_similar_custom_order_active: bool = False
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


def is_notes_mode(browser: Any) -> bool:
    """Checks whether the Anki browser is currently in Notes mode instead of Cards mode."""
    if not browser:
        return False
    if hasattr(browser, "table") and hasattr(browser.table, "is_notes_mode"):
        try:
            return bool(browser.table.is_notes_mode())
        except Exception:
            pass
    if hasattr(browser, "is_notes_mode"):
        try:
            return bool(browser.is_notes_mode())
        except Exception:
            pass
    return False


def get_cids_to_nids(col: Any, cids: Sequence[int]) -> Dict[int, int]:
    """Retrieves mapping from card ID to note ID for a list of card IDs using SQL."""
    cid_to_nid: Dict[int, int] = {}
    if not cids:
        return cid_to_nid

    missing = [cid for cid in cids if cid not in _cid_to_nid_cache]
    if missing and col and hasattr(col, "db"):
        try:
            for i in range(0, len(missing), 900):
                chunk = tuple(missing[i : i + 900])
                if len(chunk) == 1:
                    q = f"SELECT id, nid FROM cards WHERE id = {chunk[0]}"
                else:
                    q = f"SELECT id, nid FROM cards WHERE id IN {chunk}"
                if hasattr(col.db, "all"):
                    rows = col.db.all(q)
                else:
                    rows = col.db.execute(q).fetchall()
                for row in rows:
                    _cid_to_nid_cache[row[0]] = row[1]
        except Exception as e:
            print(f"[SmartSearch] Error querying card nids: {e}")

    for cid in cids:
        if cid in _cid_to_nid_cache:
            cid_to_nid[cid] = _cid_to_nid_cache[cid]
        elif col and hasattr(col, "get_card"):
            try:
                card = col.get_card(cid)
                if card:
                    _cid_to_nid_cache[cid] = card.nid
                    cid_to_nid[cid] = card.nid
            except Exception:
                pass
    return cid_to_nid


def on_browser_will_search(context: Any) -> None:
    """Disables backend ordering when the search query was triggered by find_similar_notes."""
    global _similar_custom_order_active
    if not _similar_custom_order_active or not _last_similar_query:
        return
    search_str = getattr(context, "search", "")
    if search_str == _last_similar_query:
        try:
            context.order = False
        except Exception:
            pass


def on_browser_did_search(context: Any) -> None:
    """Sorts browser search results in descending order of similarity score.

    Applies only when context.search exactly matches the last similarity query
    and similarity ordering is currently active. If the user subsequently sorts
    by another column, this hook does not interfere.
    """
    global _similar_custom_order_active
    if not _similar_custom_order_active or not _last_similar_query:
        return
    search_str = getattr(context, "search", "")
    if search_str != _last_similar_query:
        return

    # Deactivate so subsequent user column sorts are not overridden
    _similar_custom_order_active = False

    ids = getattr(context, "ids", None)
    if not ids:
        return

    try:
        context.order = False
    except Exception:
        pass

    browser = getattr(context, "browser", None)
    if is_notes_mode(browser):
        # Notes mode: context.ids contains note IDs
        indexed = list(enumerate(ids))
        indexed.sort(
            key=lambda pair: (
                _last_similar_scores.get(pair[1], -1.0),
                -pair[0],
            ),
            reverse=True,
        )
        context.ids = [nid for _, nid in indexed]
    else:
        # Cards mode: context.ids contains card IDs
        col = getattr(browser, "col", None) or (mw.col if mw else None)
        cid_to_nid = get_cids_to_nids(col, ids)
        indexed = list(enumerate(ids))
        indexed.sort(
            key=lambda pair: (
                _last_similar_scores.get(cid_to_nid.get(pair[1], -1), -1.0),
                -pair[0],
            ),
            reverse=True,
        )
        context.ids = [cid for _, cid in indexed]


def show_scored_notes_in_browser(
    browser: Any,
    matched: Sequence[Tuple[int, float]],
    message: str = "",
    parent: Any = None,
    details: Optional[Dict[int, Dict[str, float]]] = None,
) -> None:
    """Shows the given notes in the Browser, ordered by descending score.

    The scores feed the similarity column and the one-shot custom ordering applied
    by on_browser_did_search.
    """
    global _last_similar_query, _similar_custom_order_active
    if not browser or not matched:
        return

    _last_similar_scores.clear()
    _last_similar_scores.update({nid: score for nid, score in matched})
    _last_similar_details.clear()
    if details:
        _last_similar_details.update(details)

    col = getattr(browser, "col", None) or getattr(parent, "col", None) or (mw.col if mw else None)
    if col:
        try:
            nids_tuple = tuple(int(nid) for nid, _ in matched)
            if len(nids_tuple) == 1:
                q = f"SELECT id, nid FROM cards WHERE nid = {nids_tuple[0]}"
            else:
                q = f"SELECT id, nid FROM cards WHERE nid IN {nids_tuple}"
            if hasattr(col.db, "all"):
                rows = col.db.all(q)
            else:
                rows = col.db.execute(q).fetchall()
            for row in rows:
                _cid_to_nid_cache[row[0]] = row[1]
        except Exception:
            pass

    nids_csv = ",".join(str(nid) for nid, _ in matched)
    _last_similar_query = f"nid:{nids_csv}"
    _similar_custom_order_active = True

    set_browser_search(browser, _last_similar_query)
    if message:
        show_browser_tooltip(browser, message)

    if hasattr(browser, "table") and hasattr(browser.table, "redraw_cells"):
        try:
            browser.table.redraw_cells()
        except Exception:
            pass


def run_find_similar(
    selected_nids: Sequence[int],
    parent: Any,
    get_browser: Callable[[], Any],
    mode: str = "combined",
) -> None:
    """Executes similarity retrieval for selected notes and applies results to a browser window."""
    if not selected_nids:
        return

    config = ConfigManager.load()
    active_mid = config.get_active_model_id()
    eff_mode = (mode or "combined").strip().lower()
    run_in_bg = bool(getattr(config, "run_search_in_background", True))

    if not active_mid and eff_mode == "embedding":
        show_browser_info(
            parent,
            "Nessun modello di embedding configurato nelle Impostazioni di Smart Search.",
        )
        return

    log_info(
        "browser_menu",
        f"run_find_similar start selected={list(selected_nids)} mode={eff_mode} background={run_in_bg} scope={getattr(config, 'similar_scope', 'all')}",
    )

    def _safe_parent() -> Any:
        from .browser_windows import _is_alive

        if _is_alive(parent):
            return parent
        return mw

    def _background_op(col: Any) -> Dict[str, Any]:
        t0 = time.monotonic()
        c = col or getattr(parent, "col", None) or (mw.col if mw else None)
        storage = SearchDatabase()
        scope = getattr(config, "similar_scope", "all")
        candidate_nids: Optional[Set[int]] = None

        if scope == "query":
            q = config.get_effective_search_query()
            if q and c and hasattr(c, "find_notes"):
                candidate_nids = set(c.find_notes(q))
        elif scope == "custom":
            q = getattr(config, "similar_custom_query", "").strip()
            if q and c and hasattr(c, "find_notes"):
                candidate_nids = set(c.find_notes(q))
        # if scope == "all": candidate_nids remains None (all collection / indexed notes)

        details: Dict[int, Dict[str, float]] = {}
        matched, missing = find_similar_notes(
            storage=storage,
            selected_nids=list(selected_nids),
            active_model_id=active_mid or "",
            candidate_nids=candidate_nids,
            min_score=getattr(config, "similar_min_score", 0.65),
            max_notes=getattr(config, "similar_max_notes", 20),
            mode=eff_mode,
            col=c,
            use_embeddings=getattr(config, "similar_use_embeddings", True),
            use_tags=getattr(config, "similar_use_tags", True),
            use_structure=getattr(config, "similar_use_structure", True),
            weight_embedding=getattr(config, "similar_weight_embedding", 0.55),
            weight_tags=getattr(config, "similar_weight_tags", 0.25),
            weight_structure=getattr(config, "similar_weight_structure", 0.20),
            tag_max_df=getattr(config, "similar_tag_max_df", 250),
            details_out=details,
        )

        if missing:
            deck_name = get_deck_name_for_note(c, missing[0]) if c else ""
            if deck_name:
                msg = f"Nota non indicizzata: indicizza il mazzo '{deck_name}'."
            else:
                msg = "Nota non indicizzata: indicizza prima le note nel database di Smart Search."
            log_warning("browser_menu", f"run_find_similar missing embeddings: {missing} ({time.monotonic() - t0:.2f}s)")
            return {"error": msg}

        # Fetch preview text for matched notes
        note_previews: Dict[int, str] = {}
        if matched:
            content_map = storage.get_notes_content([n for n, _ in matched])
            for n, _ in matched:
                note_previews[n] = content_map.get(n, {}).get("clean_text", "")

        log_info(
            "browser_menu",
            f"run_find_similar finished mode={eff_mode} matched={len(matched)} in {time.monotonic() - t0:.2f}s",
        )
        return {
            "matched": matched,
            "details": details,
            "mode": eff_mode,
            "min_score": getattr(config, "similar_min_score", 0.65),
            "note_previews": note_previews,
        }

    def _on_success(res: Dict[str, Any]) -> None:
        safe_p = _safe_parent()
        if "error" in res:
            show_browser_info(safe_p, res["error"])
            return

        matched = res.get("matched", [])
        details = res.get("details", {})
        res_mode = res.get("mode", eff_mode)
        min_score = res.get("min_score", 0.65)

        if not matched:
            mode_label = {
                "embedding": " (Embedding)",
                "tags": " (Tag)",
                "structure": " (Struttura)",
            }.get(res_mode, "")
            show_browser_tooltip(
                safe_p, f"Nessuna carta simile trovata{mode_label} con similarità ≥ {min_score:.2f}"
            )
            if run_in_bg:
                show_browser_info(
                    safe_p,
                    f"Ricerca carte simili completata: sono state trovate 0 note{mode_label} (soglia ≥ {min_score:.2f}).",
                )
            return

        from .browser_windows import _is_alive, get_browser_for_results

        browser = get_browser()
        if not browser or not _is_alive(browser):
            target_mw = mw or safe_p
            if target_mw:
                browser = get_browser_for_results(target_mw)
        if not browser or not _is_alive(browser):
            return

        mode_tag = {
            "embedding": " [Embedding]",
            "tags": " [Tag]",
            "structure": " [Struttura]",
        }.get(res_mode, "")
        show_scored_notes_in_browser(
            browser,
            matched,
            f"Trovate {len(matched)} note simili{mode_tag}",
            parent=safe_p,
            details=details,
        )

    def _on_failure(exc: Exception) -> None:
        log_error("browser_menu", f"run_find_similar background exception (mode={eff_mode})", exc)
        show_browser_info(_safe_parent(), f"Errore durante la ricerca di carte simili:\n{exc}")

    # In Anki: use QueryOp on background thread
    if QueryOp is not None:
        op = QueryOp(
            parent=parent,
            op=_background_op,
            success=_on_success,
        )
        if hasattr(op, "failure"):
            op = op.failure(_on_failure)
        if run_in_bg:
            show_browser_tooltip(parent, "Smart Search: ricerca carte simili in background…")
            op.run_in_background()
        else:
            op.with_progress(label="Ricerca carte simili...").run_in_background()
    else:
        # Standalone / test fallback
        try:
            col = getattr(parent, "col", None) or (mw.col if mw else None)
            res = _background_op(col)
            _on_success(res)
        except Exception as exc:
            _on_failure(exc)


def on_find_similar_cards_clicked(
    browser: Any,
    new_window: bool = True,
    mode: str = "combined",
) -> None:
    """Executes similarity retrieval for selected cards, in same window or new Browser window."""
    selected_nids: List[int] = []
    if hasattr(browser, "selected_notes"):
        try:
            selected_nids = list(browser.selected_notes() or [])
        except Exception:
            selected_nids = []

    if not selected_nids:
        return

    if new_window:
        from .browser_windows import open_new_browser

        target_mw = getattr(browser, "mw", None) or mw
        get_target_browser = lambda: open_new_browser(target_mw)
    else:
        get_target_browser = lambda: browser

    run_find_similar(
        selected_nids=selected_nids,
        parent=browser,
        get_browser=get_target_browser,
        mode=mode,
    )


def on_find_similar_cards_review_clicked(
    parent_window: Any = None,
    mode: str = "combined",
) -> None:
    """Executes similarity retrieval for the currently reviewed card in Anki reviewer."""
    target_mw = parent_window or mw
    if not target_mw:
        return
    reviewer = getattr(target_mw, "reviewer", None)
    card = getattr(reviewer, "card", None)
    nid = getattr(card, "nid", None)
    if not nid:
        return

    from .browser_windows import get_browser_for_results

    run_find_similar(
        selected_nids=[nid],
        parent=target_mw,
        get_browser=lambda: get_browser_for_results(target_mw),
        mode=mode,
    )


def _open_search_dialog(parent: Any, initial_query: Optional[str] = None) -> None:
    """Opens SearchDialog with the given parent and optional initial query."""
    try:
        from .ui.main_dialog import SearchDialog

        col = getattr(parent, "col", None) or (mw.col if mw else None)
        dialog = SearchDialog(col=col, parent=parent, initial_query=initial_query)
        dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening SearchDialog: {e}")


def _open_history_dialog(parent: Any) -> None:
    """Opens HistoryDialog with the given parent."""
    try:
        from .ui.main_dialog import HistoryDialog

        col = getattr(parent, "col", None) or (mw.col if mw else None)
        dialog = HistoryDialog(col=col, parent=parent)
        dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening HistoryDialog: {e}")


def _open_index_dialog(parent: Any) -> None:
    """Opens IndexDialog with the given parent."""
    try:
        from .ui.main_dialog import IndexDialog

        col = getattr(parent, "col", None) or (mw.col if mw else None)
        dialog = IndexDialog(col=col, parent=parent)
        dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening IndexDialog: {e}")


def _open_settings_dialog(parent: Any) -> None:
    """Opens SettingsDialog with the given parent."""
    try:
        from .ui.main_dialog import SettingsDialog

        col = getattr(parent, "col", None) or (mw.col if mw else None)
        dialog = SettingsDialog(col=col, parent=parent)
        dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening SettingsDialog: {e}")


def _open_search_from_browser(browser: Any) -> None:
    """Opens SearchDialog pre-filled with the browser's current search query."""
    query = get_browser_search_text(browser)
    _open_search_dialog(parent=browser, initial_query=query)


def _update_similar_action_state(browser: Any) -> None:
    """Enables 'Trova carte simili' actions only when at least 1 card is selected in the browser."""
    try:
        has_sel = (
            bool(browser.selected_notes())
            if hasattr(browser, "selected_notes")
            else False
        )
    except Exception:
        has_sel = False

    acts: List[Any] = []
    single_act = getattr(browser, "_smart_search_similar_action", None)
    if single_act is not None:
        acts.append(single_act)
    extra_acts = getattr(browser, "_smart_search_similar_actions", None)
    if isinstance(extra_acts, (list, tuple)):
        for a in extra_acts:
            if a is not None and a not in acts:
                acts.append(a)

    for act in acts:
        try:
            act.setEnabled(has_sel)
        except Exception:
            pass


def _populate_mode_submenu(
    sub_menu: Any,
    browser: Any,
    has_sel: bool,
    tracked_actions: Optional[List[Any]] = None,
) -> None:
    """Populates a submenu with dedicated Embedding / Tag / Structure similarity actions."""
    same_items = [
        ("Solo Embedding (stessa finestra)", "embedding", False),
        ("Solo Tag (stessa finestra)", "tags", False),
        ("Solo Struttura della frase (stessa finestra)", "structure", False),
    ]
    new_items = [
        ("Solo Embedding (nuova finestra)", "embedding", True),
        ("Solo Tag (nuova finestra)", "tags", True),
        ("Solo Struttura della frase (nuova finestra)", "structure", True),
    ]

    for label, mode_name, is_new in same_items:
        act = sub_menu.addAction(label)
        act.setEnabled(has_sel)
        act.triggered.connect(
            lambda _checked=False, m=mode_name, nw=is_new: on_find_similar_cards_clicked(
                browser, new_window=nw, mode=m
            )
        )
        if tracked_actions is not None:
            tracked_actions.append(act)

    if hasattr(sub_menu, "addSeparator"):
        sub_menu.addSeparator()

    for label, mode_name, is_new in new_items:
        act = sub_menu.addAction(label)
        act.setEnabled(has_sel)
        act.triggered.connect(
            lambda _checked=False, m=mode_name, nw=is_new: on_find_similar_cards_clicked(
                browser, new_window=nw, mode=m
            )
        )
        if tracked_actions is not None:
            tracked_actions.append(act)


def _update_mw_similar_action_state(
    target_mw: Any = None, action: Optional[Any] = None
) -> None:
    """Enables 'Trova carte simili' on the main window only when reviewing a card."""
    m = target_mw or mw
    act = action or (getattr(m, "_smart_search_similar_action_mw", None) if m else None)
    if act is None:
        return
    try:
        is_review = getattr(m, "state", None) == "review"
        has_card = bool(getattr(getattr(m, "reviewer", None), "card", None))
        act.setEnabled(is_review and has_card)
    except Exception:
        act.setEnabled(False)


def _on_state_did_change(*args: Any, **kwargs: Any) -> None:
    _update_mw_similar_action_state()


def _on_reviewer_did_show_question(*args: Any, **kwargs: Any) -> None:
    _update_mw_similar_action_state()


def setup_main_window_hooks() -> None:
    """Registers hooks to keep the main window 'Trova carte simili' action updated."""
    if gui_hooks is None:
        return
    try:
        if hasattr(gui_hooks, "state_did_change"):
            gui_hooks.state_did_change.append(_on_state_did_change)
        if hasattr(gui_hooks, "reviewer_did_show_question"):
            gui_hooks.reviewer_did_show_question.append(_on_reviewer_did_show_question)
    except Exception as e:
        print(f"[SmartSearch] Error registering main window hooks: {e}")


def populate_smart_search_menu(
    menu: Any, parent_window: Any, browser: Optional[Any] = None
) -> Dict[str, Any]:
    """Populates a QMenu with the unified Smart Search action items.

    Items and order:
    1. Nuova ricerca… (Ctrl+Shift+F)
    2. Cerca con la query corrente… (disabled outside browser)
    3. Trova carte simili (Ctrl+Alt+M, disabled outside browser or without selection)
    -- Separator --
    4. Cronologia
    5. Indicizzazione
    6. Impostazioni
    -- Separator --
    7. Verifica aggiornamenti
    """
    from .updater import run_update_check_ui

    actions: Dict[str, Any] = {}

    # 1. Nuova ricerca…
    act_new = menu.addAction("Nuova ricerca…")
    if hasattr(act_new, "setShortcut"):
        act_new.setShortcut("Ctrl+Shift+F")
    act_new.triggered.connect(
        lambda _checked=False: _open_search_dialog(browser or parent_window)
    )
    actions["new_search"] = act_new

    if hasattr(menu, "setDefaultAction"):
        menu.setDefaultAction(act_new)

    # 2. Cerca con la query corrente…
    act_current = menu.addAction("Cerca con la query corrente…")
    if browser is not None:
        act_current.setEnabled(True)
        act_current.triggered.connect(
            lambda _checked=False: _open_search_from_browser(browser)
        )
    else:
        act_current.setEnabled(False)
    actions["search_current"] = act_current

    # 3. Trova carte simili
    if browser is not None:
        has_sel = (
            bool(browser.selected_notes())
            if hasattr(browser, "selected_notes")
            else False
        )
        tracked_acts: List[Any] = []

        # Scorciatoia Ctrl+Alt+M: verificato sul form del Browser di Anki che Ctrl+Shift+S
        # è già occupata da "Riposiziona" e Ctrl+Alt+S è occupata da "Inverti selezione".
        act_similar_same = menu.addAction("Trova carte simili (stessa finestra)")
        if hasattr(act_similar_same, "setShortcut"):
            act_similar_same.setShortcut("Ctrl+Alt+M")
        act_similar_same.setEnabled(has_sel)
        act_similar_same.triggered.connect(
            lambda _checked=False: on_find_similar_cards_clicked(
                browser, new_window=False, mode="combined"
            )
        )
        tracked_acts.append(act_similar_same)
        actions["find_similar"] = act_similar_same
        actions["find_similar_same"] = act_similar_same

        act_similar_new = menu.addAction("Trova carte simili (nuova finestra)")
        if hasattr(act_similar_new, "setShortcut"):
            act_similar_new.setShortcut("Ctrl+Shift+Alt+M")
        act_similar_new.setEnabled(has_sel)
        act_similar_new.triggered.connect(
            lambda _checked=False: on_find_similar_cards_clicked(
                browser, new_window=True, mode="combined"
            )
        )
        tracked_acts.append(act_similar_new)
        actions["find_similar_new"] = act_similar_new

        if hasattr(menu, "addMenu"):
            sub_menu = menu.addMenu("Trova carte simili per criterio")
            if sub_menu is not None:
                _populate_mode_submenu(sub_menu, browser, has_sel, tracked_acts)
                actions["find_similar_by_mode"] = sub_menu

        setattr(browser, "_smart_search_similar_action", act_similar_same)
        setattr(browser, "_smart_search_similar_actions", tracked_acts)
        if hasattr(menu, "aboutToShow") and hasattr(menu.aboutToShow, "connect"):
            menu.aboutToShow.connect(lambda: _update_similar_action_state(browser))
    else:
        act_similar = menu.addAction("Trova carte simili")
        if hasattr(act_similar, "setShortcut"):
            act_similar.setShortcut("Ctrl+Alt+M")
        target_mw = parent_window or mw
        act_similar.triggered.connect(
            lambda _checked=False: on_find_similar_cards_review_clicked(target_mw)
        )
        if target_mw:
            setattr(target_mw, "_smart_search_similar_action_mw", act_similar)
        _update_mw_similar_action_state(target_mw, act_similar)
        if hasattr(menu, "aboutToShow") and hasattr(menu.aboutToShow, "connect"):
            menu.aboutToShow.connect(lambda: _update_mw_similar_action_state(target_mw, act_similar))
        actions["find_similar"] = act_similar

    menu.addSeparator()

    # 4. Cronologia
    act_history = menu.addAction("Cronologia")
    act_history.triggered.connect(
        lambda _checked=False: _open_history_dialog(browser or parent_window)
    )
    actions["history"] = act_history

    # 5. Indicizzazione
    act_index = menu.addAction("Indicizzazione")
    act_index.triggered.connect(
        lambda _checked=False: _open_index_dialog(browser or parent_window)
    )
    actions["index"] = act_index

    # 6. Impostazioni
    act_settings = menu.addAction("Impostazioni")
    act_settings.triggered.connect(
        lambda _checked=False: _open_settings_dialog(browser or parent_window)
    )
    actions["settings"] = act_settings

    menu.addSeparator()

    # 7. Verifica aggiornamenti
    act_updates = menu.addAction("Verifica aggiornamenti")
    act_updates.triggered.connect(
        lambda _checked=False: run_update_check_ui(browser or parent_window)
    )
    actions["updates"] = act_updates

    return actions


def on_browser_menus_did_init(browser: Any) -> None:
    """Adds the unified Smart Search menu to the Browser menubar."""
    try:
        menubar = (
            getattr(browser.form, "menubar", None)
            or getattr(browser, "menubar", None)
        )
        if not menubar:
            return

        menu = menubar.addMenu("Smart Search")
        populate_smart_search_menu(menu=menu, parent_window=browser, browser=browser)
    except Exception as e:
        print(f"[SmartSearch] Error initializing browser menu: {e}")


def on_browser_did_change_row(browser: Any) -> None:
    """Updates action enabled state when the selected row changes."""
    _update_similar_action_state(browser)


def on_browser_will_show_context_menu(browser: Any, menu: Any) -> None:
    """Adds 'Trova carte simili' (same/new window + by-criterion submenu) to the Browser row context menu."""
    try:
        has_sel = bool(browser.selected_notes()) if hasattr(browser, "selected_notes") else False

        act_same = menu.addAction("Smart Search: Trova carte simili (stessa finestra)")
        if hasattr(act_same, "setShortcut"):
            act_same.setShortcut("Ctrl+Alt+M")
        act_same.setEnabled(has_sel)
        act_same.triggered.connect(
            lambda _checked=False: on_find_similar_cards_clicked(
                browser, new_window=False, mode="combined"
            )
        )

        act_new = menu.addAction("Smart Search: Trova carte simili (nuova finestra)")
        if hasattr(act_new, "setShortcut"):
            act_new.setShortcut("Ctrl+Shift+Alt+M")
        act_new.setEnabled(has_sel)
        act_new.triggered.connect(
            lambda _checked=False: on_find_similar_cards_clicked(
                browser, new_window=True, mode="combined"
            )
        )

        if hasattr(menu, "addMenu"):
            sub_menu = menu.addMenu("Smart Search: Trova simili per criterio")
            if sub_menu is not None:
                _populate_mode_submenu(sub_menu, browser, has_sel)
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
    target_nid: Optional[int] = None
    if is_notes_mode:
        target_nid = item
        score = _last_similar_scores.get(item)
    else:
        if item in _cid_to_nid_cache:
            target_nid = _cid_to_nid_cache[item]
            score = _last_similar_scores.get(target_nid)
        else:
            try:
                from aqt import mw

                if mw and mw.col:
                    card = mw.col.get_card(item)
                    if card:
                        _cid_to_nid_cache[item] = card.nid
                        target_nid = card.nid
                        score = _last_similar_scores.get(card.nid)
            except Exception:
                pass

    if score is not None:
        comp = _last_similar_details.get(target_nid) if target_nid is not None else None
        if comp and len(comp) >= 2:
            parts = [f"{k}:{comp[k]:.2f}" for k in ("E", "T", "S") if k in comp and comp[k] >= 0.05]
            if parts:
                score_str = f"{score:.2f} ({' '.join(parts)})"
            else:
                score_str = f"{score:.2f}"
        else:
            score_str = f"{score:.2f}"
    else:
        score_str = ""
    try:
        from aqt.browser import Cell

        row.cells[idx] = Cell(text=score_str)
    except Exception:
        if hasattr(row.cells[idx], "text"):
            row.cells[idx].text = score_str


def setup_browser_menu() -> None:
    """Hooks the Smart Search menu, context menu, and similarity column into Anki's browser window."""
    setup_main_window_hooks()
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
        if hasattr(gui_hooks, "browser_will_search"):
            gui_hooks.browser_will_search.append(on_browser_will_search)
        if hasattr(gui_hooks, "browser_did_search"):
            gui_hooks.browser_did_search.append(on_browser_did_search)
    except Exception as e:
        print(f"[SmartSearch] Error registering browser menu hooks: {e}")
