"""Thread-safe rotating debug logger for Smart Search.

Stores diagnostic logs inside <addon_dir>/user_files/smart_search.log so logs
are preserved across addon updates and excluded from .ankiaddon packages.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_LOCK = threading.Lock()
_DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_DEFAULT_BACKUP_COUNT = 2
_DEFAULT_MAX_READ_CHARS = 200_000

_LOGGER_NAME = "ankihub_searcher.debug"
_configured_path: Optional[str] = None
_active_handler: Optional[RotatingFileHandler] = None


def get_log_file_path(log_path: Optional[str] = None) -> str:
    """Returns the resolved path to the debug log file, creating parent dirs if needed."""
    if log_path:
        resolved = os.path.abspath(log_path)
    elif _configured_path:
        resolved = os.path.abspath(_configured_path)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        user_files = os.path.join(base_dir, "user_files")
        resolved = os.path.join(user_files, "smart_search.log")

    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    return resolved


def _ensure_logger(log_path: Optional[str] = None) -> logging.Logger:
    """Returns the configured logger, (re)attaching the rotating file handler if needed."""
    global _active_handler, _configured_path

    target_path = get_log_file_path(log_path)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    current_base = getattr(_active_handler, "baseFilename", None) if _active_handler else None
    if _active_handler is None or (current_base and os.path.abspath(current_base) != target_path):
        if _active_handler is not None:
            try:
                logger.removeHandler(_active_handler)
                _active_handler.close()
            except Exception:
                pass
            _active_handler = None

        handler = RotatingFileHandler(
            target_path,
            maxBytes=_DEFAULT_MAX_BYTES,
            backupCount=_DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _active_handler = handler

    return logger


def configure_log_file(log_path: Optional[str] = None) -> str:
    """Sets or resets the active log file path (useful for tests)."""
    global _configured_path, _active_handler
    with _LOG_LOCK:
        logger = logging.getLogger(_LOGGER_NAME)
        if _active_handler is not None:
            try:
                logger.removeHandler(_active_handler)
                _active_handler.close()
            except Exception:
                pass
            _active_handler = None
        _configured_path = os.path.abspath(log_path) if log_path else None
        return get_log_file_path()


def _write_record(level: int, component: str, message: str) -> None:
    try:
        with _LOG_LOCK:
            _ensure_logger()
            comp = (component or "SmartSearch").strip()
            sub_logger = logging.getLogger(f"{_LOGGER_NAME}.{comp}")
            sub_logger.setLevel(logging.DEBUG)
            sub_logger.propagate = True
            sub_logger.log(level, "%s", message)
    except Exception:
        # Logging must never break addon execution
        pass


def log_info(component: str, message: str) -> None:
    """Writes an INFO entry to the rotating debug log."""
    _write_record(logging.INFO, component, message)


def log_warning(component: str, message: str) -> None:
    """Writes a WARNING entry to the rotating debug log."""
    _write_record(logging.WARNING, component, message)


def log_error(component: str, message: str, exc: Optional[BaseException] = None) -> None:
    """Writes an ERROR entry (with optional exception traceback) to the rotating debug log."""
    full_msg = message
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        if tb:
            full_msg = f"{message}\n{tb}"
        else:
            full_msg = f"{message}: {exc}"
    _write_record(logging.ERROR, component, full_msg)


def read_log_text(max_chars: int = _DEFAULT_MAX_READ_CHARS, log_path: Optional[str] = None) -> str:
    """Reads up to the last max_chars characters from the debug log file."""
    try:
        with _LOG_LOCK:
            path = get_log_file_path(log_path)
            if not os.path.exists(path):
                return ""
            if _active_handler is not None:
                try:
                    _active_handler.flush()
                except Exception:
                    pass
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if max_chars > 0 and len(content) > max_chars:
                return "... [log troncato all'inizio] ...\n" + content[-max_chars:]
            return content
    except Exception as e:
        return f"Impossibile leggere il file di log: {e}"


def clear_log_file(log_path: Optional[str] = None) -> None:
    """Truncates the active debug log file safely."""
    global _active_handler
    try:
        with _LOG_LOCK:
            path = get_log_file_path(log_path)
            logger = logging.getLogger(_LOGGER_NAME)
            if _active_handler is not None:
                try:
                    logger.removeHandler(_active_handler)
                    _active_handler.close()
                except Exception:
                    pass
                _active_handler = None
            with open(path, "w", encoding="utf-8"):
                pass
    except Exception:
        pass


def get_log_file_size_bytes(log_path: Optional[str] = None) -> int:
    """Returns the current size of the log file in bytes."""
    try:
        with _LOG_LOCK:
            path = get_log_file_path(log_path)
            if not os.path.exists(path):
                return 0
            return os.path.getsize(path)
    except Exception:
        return 0
