"""Smart Search: Smart Lecture-to-Filtered-Deck Anki Add-on.

Decomposes Italian lecture notes via LLM into atomic medical concepts,
queries English flashcard collections (AnKing / Mastering Radiology) via
hybrid dense vector + FTS5 BM25 search, and generates clean Filtered Decks
using safe cid: queries without temporary tags or note pollution.
"""

from __future__ import annotations

import os
import sys

__version__ = "0.3.0"

from .ai_client import OpenAICompatibleClient
from .config_manager import ConfigManager
from .deck_ops import (
    execute_unit_deck_creation,
    resolve_notes_to_cards,
    rollback_run,
)
from .indexer import DeckIndexer
from .search_pipeline import SmartSearchPipeline
from .storage import SearchDatabase
from .text_utils import clean_html_and_cloze, clean_tags, parse_markdown_units
from .ui.main_dialog import (
    AnkiHubSearcherDialog,
    HistoryDialog,
    IndexDialog,
    SearchDialog,
    SettingsDialog,
)
from .updater import run_update_check_ui


def open_search_dialog() -> None:
    """Instantiates and presents the SearchDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        if mw:
            dialog = SearchDialog(col=mw.col, parent=mw)
            dialog.exec()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[SmartSearch] Error opening search dialog: {e}")


def open_history_dialog() -> None:
    """Instantiates and presents the HistoryDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        if mw:
            dialog = HistoryDialog(col=mw.col, parent=mw)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening history dialog: {e}")


def open_index_dialog() -> None:
    """Instantiates and presents the IndexDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        if mw:
            dialog = IndexDialog(col=mw.col, parent=mw)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening index dialog: {e}")


def open_settings_dialog() -> None:
    """Instantiates and presents the SettingsDialog inside Anki."""
    try:
        from aqt import mw  # type: ignore

        if mw:
            dialog = SettingsDialog(col=mw.col, parent=mw)
            dialog.exec()
    except Exception as e:
        print(f"[SmartSearch] Error opening settings dialog: {e}")


def init_addon() -> None:
    """Initializes the Smart Search menu and submenus in Anki's Tools menu."""
    try:
        from aqt import mw  # type: ignore
        from aqt.qt import QAction, QMenu  # type: ignore

        if not mw or not hasattr(mw, "form") or not mw.form or not hasattr(mw.form, "menuTools"):
            return

        # Create Smart Search submenu in Tools menu
        menu = QMenu("Smart Search", mw)
        mw.form.menuTools.addMenu(menu)

        # 1. Cerca... (Shortcut: Ctrl+Shift+F)
        action_search = QAction("Cerca...", mw)
        action_search.setShortcut("Ctrl+Shift+F")
        action_search.triggered.connect(open_search_dialog)
        menu.addAction(action_search)

        # Connect default action if platform supports clicking parent menu
        if hasattr(menu, "setDefaultAction"):
            menu.setDefaultAction(action_search)

        menu.addSeparator()

        # 2. Cronologia
        action_history = QAction("Cronologia", mw)
        action_history.triggered.connect(open_history_dialog)
        menu.addAction(action_history)

        # 3. Indicizzazione
        action_index = QAction("Indicizzazione", mw)
        action_index.triggered.connect(open_index_dialog)
        menu.addAction(action_index)

        # 4. Impostazioni
        action_settings = QAction("Impostazioni", mw)
        action_settings.triggered.connect(open_settings_dialog)
        menu.addAction(action_settings)

        menu.addSeparator()

        # 5. Verifica aggiornamenti
        action_update = QAction("Verifica aggiornamenti", mw)
        action_update.triggered.connect(lambda: run_update_check_ui(mw))
        menu.addAction(action_update)

    except ImportError:
        # Running in standalone test environment outside Anki
        pass
    except Exception as e:
        print(f"[SmartSearch] Error initializing menu: {e}")


# Automatically register menu action on Anki startup
init_addon()
