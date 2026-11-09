#!/usr/bin/env python3
"""Build & Versioning script for AnkiHubSearcher (.ankiaddon packager).

Usage:
    python3 scripts/build_addon.py                  # Build current version in VERSION
    python3 scripts/build_addon.py --set-version 0.2.0  # Bump version & build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERSION_FILE = os.path.join(ROOT_DIR, "VERSION")
SRC_DIR = os.path.join(ROOT_DIR, "src", "ankihub_searcher")
INIT_FILE = os.path.join(SRC_DIR, "__init__.py")
MANIFEST_FILE = os.path.join(SRC_DIR, "manifest.json")
DIST_DIR = os.path.join(ROOT_DIR, "dist")


def get_version() -> str:
    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def set_version(version: str) -> None:
    if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", version):
        raise ValueError(f"Invalid semantic version: {version}")

    # 1. Update VERSION
    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        f.write(f"{version}\n")

    # 2. Update __init__.py
    with open(INIT_FILE, "r", encoding="utf-8") as f:
        init_text = f.read()
    init_text = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', init_text)
    with open(INIT_FILE, "w", encoding="utf-8") as f:
        f.write(init_text)

    # 3. Update manifest.json
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["version"] = version
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 4. Update main_dialog.py
    main_dlg_file = os.path.join(SRC_DIR, "ui", "main_dialog.py")
    if os.path.exists(main_dlg_file):
        with open(main_dlg_file, "r", encoding="utf-8") as f:
            dlg_text = f.read()
        dlg_text = re.sub(r'CURRENT_VERSION\s*=\s*"[^"]+"', f'CURRENT_VERSION = "{version}"', dlg_text)
        with open(main_dlg_file, "w", encoding="utf-8") as f:
            f.write(dlg_text)

    # 5. Update updater.py
    updater_file = os.path.join(SRC_DIR, "updater.py")
    if os.path.exists(updater_file):
        with open(updater_file, "r", encoding="utf-8") as f:
            up_text = f.read()
        up_text = re.sub(r'CURRENT_VERSION\s*=\s*"[^"]+"', f'CURRENT_VERSION = "{version}"', up_text)
        with open(updater_file, "w", encoding="utf-8") as f:
            f.write(up_text)


def package_addon(version: str) -> tuple[str, str]:
    os.makedirs(DIST_DIR, exist_ok=True)
    versioned_out = os.path.join(DIST_DIR, f"AnkiHubSearcher-v{version}.ankiaddon")
    smart_out = os.path.join(DIST_DIR, f"SmartSearch-v{version}.ankiaddon")
    latest_out = os.path.join(ROOT_DIR, "AnkiHubSearcher.ankiaddon")

    for out_path in (versioned_out, smart_out, latest_out):
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(SRC_DIR):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "user_files", ".git")]
                for fname in sorted(files):
                    if fname.endswith((".pyc", ".pyo", ".DS_Store", ".db", "meta.json")):
                        continue
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, SRC_DIR)
                    zf.write(full_path, rel_path)

    return latest_out, versioned_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AnkiHubSearcher .ankiaddon package")
    parser.add_argument("--set-version", type=str, help="Set new semantic version (e.g. 0.1.0)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running unit tests before packaging")
    args = parser.parse_args()

    if args.set_version:
        set_version(args.set_version)

    version = get_version()
    set_version(version)  # Ensure all files are synced with VERSION

    if not args.skip_tests:
        print(f"Running unit tests for v{version}...")
        res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", os.path.join(ROOT_DIR, "tests"), "-v"],
            cwd=ROOT_DIR,
        )
        if res.returncode != 0:
            sys.exit(res.returncode)

    latest_out, versioned_out = package_addon(version)
    print(f"\nSuccessfully built AnkiHubSearcher v{version}:")
    print(f"  - {latest_out}")
    print(f"  - {versioned_out}")


if __name__ == "__main__":
    main()
