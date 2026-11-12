"""Tests for multi-window Browser management (browser_windows.py)."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher import browser_menu, browser_windows


class FakePoint:
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y


class MockBrowserWindow:
    def __init__(self, mw=None):
        self.mw = mw
        self.moved_to = None
        self.activated = False
        self.raised = False
        self.closed = False
        self._pos = FakePoint(100, 100)
        self.close_with_cb_called = False
        self.executed_searches = []
        self._selected_notes = []
        self.search_query = ""

    def selected_notes(self):
        return list(self._selected_notes)

    def search_for(self, query: str):
        self.executed_searches.append(query)
        self.search_query = query

    def pos(self):
        return self._pos

    def move(self, x: int, y: int):
        self.moved_to = (x, y)
        self._pos = FakePoint(x, y)

    def activateWindow(self):
        self.activated = True

    def raise_(self):
        self.raised = True

    def _closeWindow(self):
        self.closed = True
        # Anki native Browser._closeWindow always calls markClosed("Browser")
        import aqt

        if hasattr(aqt, "dialogs") and hasattr(aqt.dialogs, "markClosed"):
            aqt.dialogs.markClosed("Browser")

    def closeWithCallback(self, cb):
        self.close_with_cb_called = True
        self._pending_close_cb = cb


class MockDialogManager:
    def __init__(self):
        self._dialogs = {"Browser": [MockBrowserWindow, None]}
        self.close_all_called = False
        self.all_closed_called = False

    def markClosed(self, name: str):
        if name in self._dialogs:
            self._dialogs[name][1] = None

    def open(self, name: str, *args, **kwargs):
        if name in self._dialogs:
            if self._dialogs[name][1] is None:
                inst = self._dialogs[name][0](*args, **kwargs)
                self._dialogs[name][1] = inst
            return self._dialogs[name][1]
        return None

    def closeAll(self, onsuccess=None):
        self.close_all_called = True
        if onsuccess:
            onsuccess()
        return True

    def allClosed(self):
        self.all_closed_called = True
        return self._dialogs.get("Browser", [None, None])[1] is None


class TestBrowserWindows(unittest.TestCase):
    def setUp(self):
        browser_windows._extra_browsers.clear()
        browser_windows._installed = False

        self.fake_dialogs = MockDialogManager()
        self.fake_mw = MagicMock()
        self.fake_aqt = types.ModuleType("aqt")
        self.fake_aqt.dialogs = self.fake_dialogs
        self.fake_aqt.mw = self.fake_mw

        self.fake_browser_mod = types.ModuleType("aqt.browser")
        self.fake_browser_mod.Browser = MockBrowserWindow
        self.fake_aqt.browser = self.fake_browser_mod

        self.fake_qt_mod = types.ModuleType("aqt.qt")
        self.fake_qt_mod.QApplication = MagicMock()
        self.fake_qt_mod.QApplication.activeWindow.return_value = None
        self.fake_aqt.qt = self.fake_qt_mod

        self.patches = patch.dict(
            sys.modules,
            {
                "aqt": self.fake_aqt,
                "aqt.browser": self.fake_browser_mod,
                "aqt.qt": self.fake_qt_mod,
            },
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()
        browser_windows._extra_browsers.clear()

    def test_open_new_browser_adds_to_extra_and_positions(self):
        # 1. Main browser exists at (100, 100)
        main_browser = MockBrowserWindow(self.fake_mw)
        self.fake_dialogs._dialogs["Browser"][1] = main_browser

        new_b = browser_windows.open_new_browser(self.fake_mw)
        self.assertIsNotNone(new_b)
        self.assertIn(new_b, browser_windows._extra_browsers)
        self.assertEqual(len(browser_windows._extra_browsers), 1)

        # Position shifted +30/+30 from main browser (100, 100) -> (130, 130)
        self.assertEqual(new_b.moved_to, (130, 130))
        self.assertTrue(new_b.activated)
        self.assertTrue(new_b.raised)

    def test_close_extra_browser_does_not_clear_primary_slot(self):
        main_browser = MockBrowserWindow(self.fake_mw)
        self.fake_dialogs._dialogs["Browser"][1] = main_browser

        extra_b = browser_windows.open_new_browser(self.fake_mw)
        self.assertIn(extra_b, browser_windows._extra_browsers)

        # Closing extra browser triggers markClosed("Browser") inside orig _closeWindow
        extra_b._closeWindow()

        # Slot for primary browser must NOT be cleared!
        self.assertIs(self.fake_dialogs._dialogs["Browser"][1], main_browser)
        # Extra browser must be removed from tracking
        self.assertNotIn(extra_b, browser_windows._extra_browsers)

    def test_close_main_browser_clears_slot_normally(self):
        main_browser = MockBrowserWindow(self.fake_mw)
        self.fake_dialogs._dialogs["Browser"][1] = main_browser

        # Main browser's _closeWindow was not wrapped by open_new_browser
        main_browser._closeWindow()
        self.assertIsNone(self.fake_dialogs._dialogs["Browser"][1])

    def test_close_all_closes_extras_then_calls_original(self):
        browser_windows.install()

        extra1 = browser_windows.open_new_browser(self.fake_mw)
        extra2 = browser_windows.open_new_browser(self.fake_mw)
        self.assertEqual(len(browser_windows._extra_browsers), 2)

        # allClosed must return False when extras are alive
        self.assertFalse(self.fake_dialogs.allClosed())

        on_success = MagicMock()
        res = self.fake_dialogs.closeAll(onsuccess=on_success)
        self.assertTrue(res)

        # Original closeAll not called yet because callbacks have not completed
        self.assertFalse(self.fake_dialogs.close_all_called)
        on_success.assert_not_called()
        self.assertTrue(extra1.close_with_cb_called)
        self.assertTrue(extra2.close_with_cb_called)

        # Fire first callback
        extra1._pending_close_cb()
        self.assertFalse(self.fake_dialogs.close_all_called)
        on_success.assert_not_called()

        # Fire second callback
        extra2._pending_close_cb()
        self.assertTrue(self.fake_dialogs.close_all_called)
        on_success.assert_called_once()

    def test_close_all_empty_delegates_immediately(self):
        browser_windows.install()
        self.assertEqual(len(browser_windows._extra_browsers), 0)

        on_success = MagicMock()
        self.fake_dialogs.closeAll(onsuccess=on_success)
        self.assertTrue(self.fake_dialogs.close_all_called)
        on_success.assert_called_once()
        self.assertTrue(self.fake_dialogs.allClosed())

    def test_get_browser_for_results(self):
        # Case A: primary browser is open -> opens new extra browser
        main_browser = MockBrowserWindow(self.fake_mw)
        self.fake_dialogs._dialogs["Browser"][1] = main_browser

        b1 = browser_windows.get_browser_for_results(self.fake_mw)
        self.assertIsNot(b1, main_browser)
        self.assertIn(b1, browser_windows._extra_browsers)

        # Case B: primary browser slot is None, but an extra browser is alive -> opens new extra browser
        self.fake_dialogs._dialogs["Browser"][1] = None
        self.assertEqual(len(browser_windows._extra_browsers), 1)
        b2 = browser_windows.get_browser_for_results(self.fake_mw)
        self.assertIsNotNone(b2)
        self.assertEqual(len(browser_windows._extra_browsers), 2)

        # Case C: no primary browser and no extra browsers -> opens primary via dialogs.open
        browser_windows._extra_browsers.clear()
        self.fake_dialogs._dialogs["Browser"][1] = None
        b3 = browser_windows.get_browser_for_results(self.fake_mw)
        self.assertIs(b3, self.fake_dialogs._dialogs["Browser"][1])
        self.assertEqual(len(browser_windows._extra_browsers), 0)

    def test_install_idempotent(self):
        orig_close_all = self.fake_dialogs.closeAll
        self.assertTrue(browser_windows.install())
        wrapped_close_all = self.fake_dialogs.closeAll
        self.assertIsNot(orig_close_all, wrapped_close_all)

        # Second call must not wrap again
        self.assertTrue(browser_windows.install())
        self.assertIs(self.fake_dialogs.closeAll, wrapped_close_all)

    def test_find_similar_from_browser_opens_new_window_only_on_results(self):
        from ankihub_searcher import browser_menu
        from ankihub_searcher.config_manager import ConfigManager

        starting_browser = MockBrowserWindow(self.fake_mw)
        starting_browser.search_query = "deck:OriginalDeck"
        starting_browser._selected_notes = [101]

        cfg = ConfigManager(
            embed_model="test-embed-model",
            embed_provider="test",
            similar_scope="all",
            similar_min_score=0.70,
            similar_max_notes=10,
        )
        cfg.get_active_model_id = lambda: "test-mid"

        tooltips = []
        infos = []

        with patch.object(ConfigManager, "load", return_value=cfg), \
             patch("ankihub_searcher.browser_menu.find_similar_notes", return_value=([(201, 0.95), (202, 0.85)], [])), \
             patch("ankihub_searcher.browser_menu.show_browser_tooltip", side_effect=lambda w, txt: tooltips.append((w, txt))), \
             patch("ankihub_searcher.browser_menu.show_browser_info", side_effect=lambda w, txt: infos.append((w, txt))):

            browser_menu.on_find_similar_cards_clicked(starting_browser)

        # 1. Starting browser untouched
        self.assertEqual(starting_browser.executed_searches, [])
        self.assertEqual(starting_browser.search_query, "deck:OriginalDeck")
        self.assertEqual(starting_browser.selected_notes(), [101])

        # 2. A new browser window was created and added to extra browsers
        self.assertEqual(len(browser_windows._extra_browsers), 1)
        new_b = browser_windows._extra_browsers[0]
        self.assertIsNot(new_b, starting_browser)
        self.assertIn("nid:201,202", new_b.executed_searches)
        self.assertTrue(any("Trovate 2 note simili" in t[1] for t in tooltips))
        # Tooltip is shown on the new browser
        self.assertEqual(tooltips[0][0], new_b)

    def test_find_similar_no_results_or_unindexed_does_not_open_window(self):
        from ankihub_searcher import browser_menu
        from ankihub_searcher.config_manager import ConfigManager

        starting_browser = MockBrowserWindow(self.fake_mw)
        starting_browser.search_query = "deck:OriginalDeck"
        starting_browser._selected_notes = [101]

        cfg = ConfigManager(
            embed_model="test-embed-model",
            embed_provider="test",
            similar_scope="all",
            similar_min_score=0.70,
            similar_max_notes=10,
        )
        cfg.get_active_model_id = lambda: "test-mid"

        tooltips = []
        infos = []

        # A: No results
        with patch.object(ConfigManager, "load", return_value=cfg), \
             patch("ankihub_searcher.browser_menu.find_similar_notes", return_value=([], [])), \
             patch("ankihub_searcher.browser_menu.show_browser_tooltip", side_effect=lambda w, txt: tooltips.append((w, txt))), \
             patch("ankihub_searcher.browser_menu.show_browser_info", side_effect=lambda w, txt: infos.append((w, txt))):

            browser_menu.on_find_similar_cards_clicked(starting_browser)

        self.assertEqual(len(browser_windows._extra_browsers), 0)
        self.assertEqual(starting_browser.executed_searches, [])
        self.assertTrue(any("Nessuna carta simile" in t[1] for t in tooltips))
        # Tooltip on starting window
        self.assertEqual(tooltips[0][0], starting_browser)

        # B: Unindexed note
        tooltips.clear()
        infos.clear()
        with patch.object(ConfigManager, "load", return_value=cfg), \
             patch("ankihub_searcher.browser_menu.find_similar_notes", return_value=([], [101])), \
             patch("ankihub_searcher.browser_menu.show_browser_tooltip", side_effect=lambda w, txt: tooltips.append((w, txt))), \
             patch("ankihub_searcher.browser_menu.show_browser_info", side_effect=lambda w, txt: infos.append((w, txt))):

            browser_menu.on_find_similar_cards_clicked(starting_browser)

        self.assertEqual(len(browser_windows._extra_browsers), 0)
        self.assertEqual(starting_browser.executed_searches, [])
        self.assertTrue(any("Nota non indicizzata" in m[1] for m in infos))
        # Info modal on starting window
        self.assertEqual(infos[0][0], starting_browser)

    def test_initial_search_does_not_consume_similarity_sort_order(self):
        from ankihub_searcher import browser_menu

        browser = MockBrowserWindow(self.fake_mw)
        matched = [(301, 0.99), (302, 0.75)]
        browser_menu.show_scored_notes_in_browser(browser, matched)

        self.assertTrue(browser_menu._similar_custom_order_active)
        self.assertEqual(browser_menu._last_similar_query, "nid:301,302")

        # Simulate Browser.__init__ running an initial search (e.g. deck:current or empty)
        ctx_init = types.SimpleNamespace(search="deck:current", ids=[999, 888], order=True, browser=browser)
        browser_menu.on_browser_will_search(ctx_init)
        browser_menu.on_browser_did_search(ctx_init)

        # Ordering must still be active and order flag not modified
        self.assertTrue(browser_menu._similar_custom_order_active)
        self.assertTrue(ctx_init.order)

        # Now simulate the similarity search query
        ctx_sim = types.SimpleNamespace(search="nid:301,302", ids=[302, 301], order=True, browser=browser)
        # Pretend browser is in notes mode
        browser.is_notes_mode = lambda: True
        browser_menu.on_browser_will_search(ctx_sim)
        browser_menu.on_browser_did_search(ctx_sim)

        # Custom ordering applied (301 with 0.99 before 302 with 0.75) and flag deactivated
        self.assertEqual(ctx_sim.ids, [301, 302])
        self.assertFalse(browser_menu._similar_custom_order_active)

    def test_find_similar_from_extra_browser_opens_another_browser(self):
        from ankihub_searcher import browser_menu
        from ankihub_searcher.config_manager import ConfigManager

        extra1 = browser_windows.open_new_browser(self.fake_mw)
        self.assertEqual(len(browser_windows._extra_browsers), 1)
        extra1.search_query = "nid:101,102"
        extra1._selected_notes = [102]

        cfg = ConfigManager(
            embed_model="test-embed-model",
            embed_provider="test",
            similar_scope="all",
            similar_min_score=0.70,
            similar_max_notes=10,
        )
        cfg.get_active_model_id = lambda: "test-mid"

        with patch.object(ConfigManager, "load", return_value=cfg), \
             patch("ankihub_searcher.browser_menu.find_similar_notes", return_value=([(401, 0.91)], [])), \
             patch("ankihub_searcher.browser_menu.show_browser_tooltip"):

            browser_menu.on_find_similar_cards_clicked(extra1)

        # A second extra browser is opened
        self.assertEqual(len(browser_windows._extra_browsers), 2)
        extra2 = browser_windows._extra_browsers[1]
        self.assertIsNot(extra2, extra1)
        self.assertIn("nid:401", extra2.executed_searches)
        # extra1 search remains unchanged
        self.assertEqual(extra1.search_query, "nid:101,102")


if __name__ == "__main__":
    unittest.main()
