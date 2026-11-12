"""Smart Search: Smart Lecture-to-Filtered-Deck Anki Add-on.

Decomposes Italian lecture notes via LLM into atomic medical concepts,
queries English flashcard collections (AnKing / Mastering Radiology) via
hybrid dense vector + FTS5 BM25 search, and generates clean Filtered Decks
using safe cid: queries without temporary tags or note pollution.
"""

from __future__ import annotations

import os
import sys

__version__ = "0.5.3"

from .ai_client import OpenAICompatibleClient
from .config_manager import ConfigManager
from .deck_ops import (
    execute_unit_deck_creation,
    resolve_notes_to_cards,
    rollback_run,
)
from .indexer import DeckIndexer
from .search_pipeline import SmartSearchPipeline, find_similar_notes
from .storage import SearchDatabase
from .text_utils import clean_html_and_cloze, clean_tags, parse_markdown_units
from .ui.main_dialog import (
    AnkiHubSearcherDialog,
    HistoryDialog,
    IndexDialog,
    SearchDialog,
    SettingsDialog,
    show_search_summary,
)
from .updater import run_update_check_ui


def open_search_dialog(initial_query: Any = None, parent: Optional[Any] = None) -> None:
    """Instantiates and presents the SearchDialog inside Anki."""
    if not isinstance(initial_query, str):
        initial_query = None
    try:
        from aqt import mw  # type: ignore

        p = parent or mw
        if p:
            col = getattr(p, "col", None) or (mw.col if mw else None)
            dialog = SearchDialog(col=col, parent=p, initial_query=initial_query)
            dialog.exec()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[SmartSearch] Error opening search dialog: {e}")


def open_history_dialog(parent: Optional[Any] = None) -> None:
    """Instantiates and presents the HistoryDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        p = parent or mw
        if p:
            col = getattr(p, "col", None) or (mw.col if mw else None)
            dialog = HistoryDialog(col=col, parent=p)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening history dialog: {e}")


def open_index_dialog(parent: Optional[Any] = None) -> None:
    """Instantiates and presents the IndexDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        p = parent or mw
        if p:
            col = getattr(p, "col", None) or (mw.col if mw else None)
            dialog = IndexDialog(col=col, parent=p)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening index dialog: {e}")


def open_settings_dialog(parent: Optional[Any] = None) -> None:
    """Instantiates and presents the SettingsDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        p = parent or mw
        if p:
            col = getattr(p, "col", None) or (mw.col if mw else None)
            dialog = SettingsDialog(col=col, parent=p)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening settings dialog: {e}")


def init_addon() -> None:
    """Initializes the Smart Search menu in Anki's menubar or Tools menu."""
    try:
        from aqt import mw  # type: ignore
        from aqt.qt import QAction, QMenu  # type: ignore

        if not mw or not hasattr(mw, "form") or not mw.form:
            return

        from .browser_menu import (
            populate_smart_search_menu,
            run_find_similar,
            setup_browser_menu,
        )

        config = ConfigManager.load()
        menu_loc = getattr(config, "menu_location", "tools")

        if menu_loc == "menubar" and hasattr(mw.form, "menubar") and mw.form.menubar:
            menu = mw.form.menubar.addMenu("Smart Search")
        elif hasattr(mw.form, "menuTools") and mw.form.menuTools:
            menu = QMenu("Smart Search", mw)
            mw.form.menuTools.addMenu(menu)
        else:
            return

        populate_smart_search_menu(menu=menu, parent_window=mw, browser=None)

        # Register Smart Search Browser menu hooks
        setup_browser_menu()

        try:
            from .browser_windows import install as install_browser_windows
            install_browser_windows()
        except Exception as e:
            print(f"[SmartSearch] Error installing browser window management: {e}")

    except ImportError:
        # Running in standalone test environment outside Anki
        pass
    except Exception as e:
        print(f"[SmartSearch] Error initializing menu: {e}")


# Automatically register menu action on Anki startup
init_addon()
