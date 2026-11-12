"""Multi-window Browser management for Smart Search.

Provides helpers to open additional Anki Browser windows without disturbing
existing searches or overwriting the primary Browser registered in
aqt.dialogs.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, List, Optional

_extra_browsers: List[Any] = []
_installed: bool = False


def _is_alive(obj: Any) -> bool:
    """Checks whether a Qt widget / Python object is alive and not deleted."""
    if obj is None:
        return False
    # Check explicit mock/test deletion flag if present
    if getattr(obj, "_is_deleted", False):
        return False

    # Check PyQt/sip deletion
    for mod_name in ("aqt.qt", "PyQt6.sip", "PyQt5.sip", "sip"):
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = __import__(mod_name, fromlist=["sip"])
                except Exception:
                    continue
            sip_mod = getattr(mod, "sip", mod)
            if hasattr(sip_mod, "isdeleted"):
                if sip_mod.isdeleted(obj):
                    return False
        except Exception:
            continue
    return True


def _clean_extra_browsers() -> None:
    """Removes destroyed or garbage-collected browser instances from tracking."""
    global _extra_browsers
    _extra_browsers = [b for b in _extra_browsers if _is_alive(b)]


def _get_main_browser() -> Any:
    """Retrieves the primary Browser instance registered with Anki DialogManager."""
    try:
        import aqt

        if hasattr(aqt, "dialogs") and hasattr(aqt.dialogs, "_dialogs"):
            entry = aqt.dialogs._dialogs.get("Browser")
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                inst = entry[1]
                if _is_alive(inst):
                    return inst
    except Exception:
        pass
    return None


def open_new_browser(mw: Any) -> Any:
    """Creates a new native Anki Browser window and tracks it.

    Positions the new window +30/+30 px relative to the active Browser window
    (if one exists), activates and raises it.
    Wraps its instance _closeWindow method so closing this extra window does not
    clear the primary Browser registered in aqt.dialogs._dialogs.
    """
    _clean_extra_browsers()

    try:
        import aqt.browser

        browser_cls = getattr(aqt.browser, "Browser", None)
        if not browser_cls:
            return None
        instance = browser_cls(mw)
    except Exception as e:
        print(f"[SmartSearch] Error creating new browser: {e}")
        return None

    if instance is None:
        return None

    # Identify existing active or previous browser for offset positioning
    active_b = None
    try:
        import aqt.qt

        qapp = getattr(aqt.qt, "QApplication", None)
        if qapp and hasattr(qapp, "activeWindow"):
            app_active = qapp.activeWindow()
            if (
                app_active
                and app_active is not instance
                and isinstance(app_active, browser_cls)
                and _is_alive(app_active)
            ):
                active_b = app_active
    except Exception:
        pass

    if not active_b:
        for b in reversed(_extra_browsers):
            if b is not instance and _is_alive(b):
                active_b = b
                break
        if not active_b:
            main_b = _get_main_browser()
            if main_b is not instance and _is_alive(main_b):
                active_b = main_b

    if active_b and hasattr(active_b, "pos") and hasattr(instance, "move"):
        try:
            p = active_b.pos()
            x = p.x() if callable(getattr(p, "x", None)) else getattr(p, "x", 0)
            y = p.y() if callable(getattr(p, "y", None)) else getattr(p, "y", 0)
            instance.move(x + 30, y + 30)
        except Exception:
            pass

    if hasattr(instance, "activateWindow"):
        try:
            instance.activateWindow()
        except Exception:
            pass
    if hasattr(instance, "raise_"):
        try:
            instance.raise_()
        except Exception:
            pass

    _extra_browsers.append(instance)

    # Wrap instance _closeWindow
    orig_close_window = getattr(instance, "_closeWindow", None)

    def _wrapped_close_window(*args: Any, **kwargs: Any) -> Any:
        main_instance = None
        dialogs_dict = None
        try:
            import aqt

            if hasattr(aqt, "dialogs") and hasattr(aqt.dialogs, "_dialogs"):
                dialogs_dict = aqt.dialogs._dialogs
                entry = dialogs_dict.get("Browser")
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    main_instance = entry[1]
        except Exception:
            pass

        res = None
        if callable(orig_close_window):
            res = orig_close_window(*args, **kwargs)

        try:
            if (
                main_instance is not None
                and main_instance is not instance
                and _is_alive(main_instance)
                and dialogs_dict is not None
            ):
                entry = dialogs_dict.get("Browser")
                if isinstance(entry, list) and len(entry) >= 2:
                    entry[1] = main_instance
        except Exception:
            pass

        try:
            if instance in _extra_browsers:
                _extra_browsers.remove(instance)
        except Exception:
            pass

        return res

    instance._closeWindow = _wrapped_close_window
    return instance


def get_browser_for_results(mw: Any) -> Any:
    """Returns a Browser to show results in.

    If a primary Browser is already open or any extra Browser exists, opens a new
    Browser instance so the existing search is preserved.
    Otherwise opens and returns the primary Browser via aqt.dialogs.open.
    """
    _clean_extra_browsers()
    main_b = _get_main_browser()
    if main_b is not None or len(_extra_browsers) > 0:
        return open_new_browser(mw)

    try:
        import aqt

        if hasattr(aqt, "dialogs") and hasattr(aqt.dialogs, "open"):
            return aqt.dialogs.open("Browser", mw)
    except Exception as e:
        print(f"[SmartSearch] Error opening browser dialog: {e}")
    return None


def install() -> bool:
    """Idempotently patches aqt.dialogs to cleanly manage extra Browser windows.

    Wraps closeAll and allClosed on the aqt.dialogs instance so all extra windows
    are closed cleanly before collection/profile shutdown or sync, composing
    cleanly with third-party multi-window addons.
    """
    global _installed
    try:
        import aqt

        if not hasattr(aqt, "dialogs") or aqt.dialogs is None:
            return False
        dialogs = aqt.dialogs
    except Exception:
        return False

    if getattr(dialogs, "_smart_search_installed", False):
        return True

    orig_close_all = getattr(dialogs, "closeAll", None)
    orig_all_closed = getattr(dialogs, "allClosed", None)

    def wrapped_close_all(onsuccess: Optional[Callable[[], None]] = None) -> bool:
        _clean_extra_browsers()
        alive_extras = [b for b in _extra_browsers if _is_alive(b)]
        if not alive_extras:
            if callable(orig_close_all):
                return orig_close_all(onsuccess)
            if callable(onsuccess):
                onsuccess()
            return True

        remaining = list(alive_extras)

        def make_cb(target_browser: Any) -> Callable[[], None]:
            def cb() -> None:
                if target_browser in remaining:
                    remaining.remove(target_browser)
                if not remaining:
                    _clean_extra_browsers()
                    if callable(orig_close_all):
                        orig_close_all(onsuccess)
                    elif callable(onsuccess):
                        onsuccess()

            return cb

        for b in list(alive_extras):
            if hasattr(b, "closeWithCallback"):
                try:
                    b.closeWithCallback(make_cb(b))
                except Exception:
                    make_cb(b)()
            else:
                if hasattr(b, "close"):
                    try:
                        b.close()
                    except Exception:
                        pass
                make_cb(b)()

        return True

    def wrapped_all_closed() -> bool:
        _clean_extra_browsers()
        if len(_extra_browsers) > 0:
            return False
        if callable(orig_all_closed):
            return orig_all_closed()
        return True

    dialogs.closeAll = wrapped_close_all
    dialogs.allClosed = wrapped_all_closed
    dialogs._smart_search_installed = True
    _installed = True
    return True
