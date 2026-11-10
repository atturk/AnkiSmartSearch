"""GitHub Auto-Updater for Smart Search.

Checks releases on https://github.com/atturk/AnkiSmartSearch,
downloads the latest .ankiaddon asset, and safely extracts it while
strictly preserving user_files/ (SQLite databases) and meta.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from typing import Any, Dict, Optional, Tuple

try:
    from aqt import mw  # type: ignore
    from aqt.qt import QMessageBox, QWidget  # type: ignore
except ImportError:
    try:
        from PyQt6.QtWidgets import QMessageBox, QWidget  # type: ignore
        mw = None
    except ImportError:
        QMessageBox = None  # type: ignore
        QWidget = None  # type: ignore
        mw = None


GITHUB_REPO = "atturk/AnkiSmartSearch"
try:
    from . import __version__ as CURRENT_VERSION
except Exception:
    CURRENT_VERSION = "0.5.1"


def parse_version(v: str) -> Tuple[int, ...]:
    """Parses a version string like 'v0.2.0' or '0.1.1' into an integer tuple."""
    clean = str(v).strip().lstrip("vV")
    parts = []
    for piece in re.split(r"[-+.]", clean):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_for_updates(current_version: str = CURRENT_VERSION) -> Dict[str, Any]:
    """Queries the GitHub API for the latest release and compares versions."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AnkiSmartSearch-Updater",
            "Accept": "application/vnd.github.v3+json",
        },
        method="GET",
    )

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Errore durante la connessione a GitHub Releases: {e}") from e

    raw_tag = data.get("tag_name", "")
    latest_version = raw_tag.lstrip("vV").strip()
    html_url = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")
    release_notes = data.get("body", "")

    # Locate .ankiaddon asset or zip asset
    download_url = ""
    for asset in data.get("assets", []):
        name = asset.get("name", "").lower()
        if name.endswith(".ankiaddon"):
            download_url = asset.get("browser_download_url", "")
            break

    # If no .ankiaddon asset found, fallback to zipball_url
    if not download_url:
        download_url = data.get("zipball_url", "")

    curr_tuple = parse_version(current_version)
    latest_tuple = parse_version(latest_version)
    update_available = latest_tuple > curr_tuple and bool(download_url)

    return {
        "update_available": update_available,
        "latest_version": latest_version,
        "current_version": current_version,
        "download_url": download_url,
        "html_url": html_url,
        "release_notes": release_notes,
    }


def download_and_install_update(download_url: str, addon_dir: Optional[str] = None) -> None:
    """Downloads an .ankiaddon update and extracts it into addon_dir,

    strictly skipping and preserving user_files/ and meta.json.
    """
    if not download_url:
        raise ValueError("Download URL non valido.")

    target_dir = addon_dir or os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(target_dir)

    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "AnkiSmartSearch-Updater"},
        method="GET",
    )

    with tempfile.NamedTemporaryFile(suffix=".ankiaddon", delete=False) as temp_archive:
        temp_path = temp_archive.name
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                shutil.copyfileobj(resp, temp_archive)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Impossibile scaricare l'aggiornamento: {e}") from e

    try:
        if not zipfile.is_zipfile(temp_path):
            raise ValueError("Il pacchetto scaricato non è un archivio zip valido.")

        with zipfile.ZipFile(temp_path, "r") as zf:
            for info in zf.infolist():
                norm_name = os.path.normpath(info.filename)

                # Skip root dotfiles or relative references
                if norm_name.startswith("..") or os.path.isabs(norm_name):
                    continue

                # STRICT SAFETY RULE: Never overwrite user_files/ or meta.json!
                first_part = norm_name.split(os.sep)[0]
                if first_part == "user_files" or norm_name == "meta.json":
                    continue

                dest_path = os.path.abspath(os.path.join(target_dir, norm_name))
                if not dest_path.startswith(target_dir + os.sep) and dest_path != target_dir:
                    # Prevent Zip Slip directory traversal
                    continue

                if info.is_dir():
                    os.makedirs(dest_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with zf.open(info) as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_update_check_ui(parent: Optional[Any] = None) -> None:
    """Checks for updates and displays standard native QMessageBox dialogs."""
    if QMessageBox is None:
        return

    parent_widget = parent or (mw if mw else None)

    try:
        res = check_for_updates(CURRENT_VERSION)
    except Exception as e:
        QMessageBox.warning(
            parent_widget,
            "Verifica aggiornamenti",
            f"Impossibile verificare gli aggiornamenti:\n{e}",
        )
        return

    if not res["update_available"]:
        QMessageBox.information(
            parent_widget,
            "Verifica aggiornamenti",
            f"Smart Search è già aggiornato all'ultima versione (v{CURRENT_VERSION}).",
        )
        return

    latest_ver = res["latest_version"]
    curr_ver = res["current_version"]
    notes = res.get("release_notes", "").strip()

    prompt = (
        f"È disponibile una nuova versione di Smart Search (v{latest_ver}, attuale: v{curr_ver}).\n\n"
        f"Vuoi scaricare e installare l'aggiornamento ora?"
    )
    if notes:
        preview = notes[:300] + ("..." if len(notes) > 300 else "")
        prompt += f"\n\nNote di rilascio:\n{preview}"

    reply = QMessageBox.question(
        parent_widget,
        "Aggiornamento disponibile",
        prompt,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )

    if reply == QMessageBox.StandardButton.Yes:
        try:
            download_and_install_update(res["download_url"])
            QMessageBox.information(
                parent_widget,
                "Aggiornamento completato",
                f"Aggiornamento a v{latest_ver} installato con successo.\nRiavvia Anki per applicare le modifiche.",
            )
        except Exception as err:
            QMessageBox.critical(
                parent_widget,
                "Errore aggiornamento",
                f"Si è verificato un errore durante l'installazione:\n{err}",
            )
