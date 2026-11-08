"""Unit tests for updater.py (GitHub Auto-Updater)."""

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher.updater import (
    CURRENT_VERSION,
    check_for_updates,
    download_and_install_update,
    parse_version,
)


class TestUpdater(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(parse_version("0.1.1"), (0, 1, 1))
        self.assertEqual(parse_version("v0.2.0"), (0, 2, 0))
        self.assertEqual(parse_version("V1.0"), (1, 0, 0))
        self.assertTrue(parse_version("0.2.0") > parse_version("0.1.1"))
        self.assertTrue(parse_version("1.0.0") > parse_version("0.9.9"))
        self.assertFalse(parse_version("0.2.0") > parse_version("0.2.0"))
        self.assertFalse(parse_version("0.1.9") > parse_version("0.2.0"))

    @patch("urllib.request.urlopen")
    def test_check_for_updates_available(self, mock_urlopen):
        mock_response = io.BytesIO(
            json.dumps(
                {
                    "tag_name": "v0.3.0",
                    "html_url": "https://github.com/atturk/AnkiSmartSearch/releases/tag/v0.3.0",
                    "body": "New feature release notes",
                    "assets": [
                        {
                            "name": "AnkiSmartSearch-v0.3.0.ankiaddon",
                            "browser_download_url": "https://github.com/atturk/AnkiSmartSearch/releases/download/v0.3.0/AnkiSmartSearch.ankiaddon",
                        }
                    ],
                }
            ).encode("utf-8")
        )
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_cm

        res = check_for_updates(current_version="0.2.0")
        self.assertTrue(res["update_available"])
        self.assertEqual(res["latest_version"], "0.3.0")
        self.assertEqual(res["current_version"], "0.2.0")
        self.assertEqual(
            res["download_url"],
            "https://github.com/atturk/AnkiSmartSearch/releases/download/v0.3.0/AnkiSmartSearch.ankiaddon",
        )
        self.assertEqual(res["release_notes"], "New feature release notes")

    @patch("urllib.request.urlopen")
    def test_check_for_updates_up_to_date(self, mock_urlopen):
        mock_response = io.BytesIO(
            json.dumps(
                {
                    "tag_name": "v0.2.0",
                    "html_url": "https://github.com/atturk/AnkiSmartSearch/releases/tag/v0.2.0",
                    "body": "Current release notes",
                    "assets": [
                        {
                            "name": "AnkiSmartSearch.ankiaddon",
                            "browser_download_url": "https://download/test.ankiaddon",
                        }
                    ],
                }
            ).encode("utf-8")
        )
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_cm

        res = check_for_updates(current_version="0.2.0")
        self.assertFalse(res["update_available"])

    @patch("urllib.request.urlopen")
    def test_download_and_install_preserves_user_files_and_meta(self, mock_urlopen):
        # Create a mock target add-on directory
        with tempfile.TemporaryDirectory() as addon_dir:
            user_files_dir = os.path.join(addon_dir, "user_files")
            os.makedirs(user_files_dir, exist_ok=True)
            db_path = os.path.join(user_files_dir, "smart_search.db")
            with open(db_path, "w") as f:
                f.write("ORIGINAL_USER_DATABASE_CONTENT")

            meta_path = os.path.join(addon_dir, "meta.json")
            with open(meta_path, "w") as f:
                f.write('{"user_config": "custom"}')

            existing_code_path = os.path.join(addon_dir, "manifest.json")
            with open(existing_code_path, "w") as f:
                f.write('{"name": "Old Version"}')

            # Create an in-memory zip archive representing the update package
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("manifest.json", '{"name": "Smart Search v0.3.0"}')
                zf.writestr("new_module.py", "# New code")
                # Attempt to overwrite user_files and meta.json in archive
                zf.writestr("user_files/smart_search.db", "OVERWRITTEN_MALICIOUS_DB")
                zf.writestr("meta.json", "OVERWRITTEN_META")
            zip_bytes = zip_buffer.getvalue()

            # Mock download response
            mock_cm = MagicMock()
            mock_cm.__enter__.return_value = io.BytesIO(zip_bytes)
            mock_urlopen.return_value = mock_cm

            # Execute update
            download_and_install_update("https://dummy/test.ankiaddon", addon_dir=addon_dir)

            # Verify updated files
            with open(existing_code_path, "r") as f:
                self.assertIn("Smart Search v0.3.0", f.read())
            self.assertTrue(os.path.exists(os.path.join(addon_dir, "new_module.py")))

            # Verify STRICT PRESERVATION of user_files and meta.json!
            with open(db_path, "r") as f:
                self.assertEqual(f.read(), "ORIGINAL_USER_DATABASE_CONTENT")
            with open(meta_path, "r") as f:
                self.assertEqual(f.read(), '{"user_config": "custom"}')


if __name__ == "__main__":
    unittest.main()
