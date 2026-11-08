"""AnkiHubSearcher: Smart Lecture-to-Filtered-Deck Anki Add-on.

Decomposes Italian lecture notes via LLM into atomic medical concepts,
queries English flashcard collections (AnKing / Mastering Radiology) via
hybrid dense vector + FTS5 BM25 search, and generates clean Filtered Decks
using safe cid: queries without temporary tags or note pollution.
"""

from __future__ import annotations

import os
import sys

__version__ = "0.1.1"

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


def _show_main_dialog() -> None:
    """Instantiates and presents the AnkiHubSearcher main dialog inside Anki."""
    try:
        from aqt import mw  # type: ignore
        from .ui.main_dialog import AnkiHubSearcherDialog

        if mw:
            dialog = AnkiHubSearcherDialog(col=mw.col, parent=mw)
            dialog.exec()
    except Exception as e:
        print(f"[AnkiHubSearcher] Error launching dialog: {e}")


def init_addon() -> None:
    """Initializes menu actions inside Anki if aqt / mw are present."""
    try:
        from aqt import mw  # type: ignore
        from aqt.qt import QAction  # type: ignore

        if mw and hasattr(mw, "form") and mw.form and hasattr(mw.form, "menuTools"):
            action = QAction("AnkiHub Smart Searcher...", mw)
            action.setShortcut("Ctrl+Shift+F")
            action.triggered.connect(_show_main_dialog)
            mw.form.menuTools.addAction(action)
    except Exception:
        # Running in headless / test environment outside Anki
        pass


# Automatically register menu action on Anki startup
init_addon()
