"""Read-only integration test against real Anki notes via AnkiConnect (port 3343).

Strictly read-only: only calls version, notesInfo, and findNotes.
Skips gracefully if AnkiConnect is not running on port 3343.
"""

import json
import os
import sys
import tempfile
import unittest
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher.storage import SearchDatabase
from ankihub_searcher.text_utils import clean_html_and_cloze, clean_tags, parse_markdown_units


def _ac_readonly(action: str, **params):
    assert action in {"version", "notesInfo", "findNotes"}, f"Forbidden non-readonly action: {action}"
    req = urllib.request.Request(
        "http://127.0.0.1:3343",
        json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestLiveAnkiConnectReadOnly(unittest.TestCase):
    def test_real_anking_notes_cleaning_and_fts_matching(self):
        try:
            ver = _ac_readonly("version")
            if not ver or ver.get("error"):
                self.skipTest("AnkiConnect not available")
        except Exception:
            self.skipTest("AnkiConnect not reachable on 127.0.0.1:3343")

        # Fetch the real Factor XII (1487640225601) and Gout/Pseudogout NLRP3 (1497310326771) notes
        info = _ac_readonly("notesInfo", notes=[1487640225601, 1497310326771])["result"]
        self.assertEqual(len(info), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SearchDatabase(os.path.join(tmpdir, "live_readonly_test.db"))
            batch = []
            for n in info:
                nid = n["noteId"]
                mid = 1659130414530
                text_clean = clean_html_and_cloze(n["fields"]["Text"]["value"])
                extra_clean = clean_html_and_cloze(n["fields"]["Extra"]["value"])
                combined_text = f"{text_clean}\n{extra_clean}".strip()
                tags_clean = clean_tags(n["tags"])
                batch.append((nid, mid, 1, "hash", combined_text, tags_clean))

                # Ensure HTML/images were stripped and cloze was unwrapped
                self.assertNotIn("{{c1::", combined_text)
                self.assertNotIn("<img", combined_text)
                self.assertNotIn("#AK_Step1_v12", tags_clean)

            db.upsert_notes_content_batch(batch)

            # Query FTS with concepts from the user's Italian lecture (translated to USMLE terms)
            hits_hageman = db.search_fts(["Hageman", "Factor XII", "subendothelial"])
            self.assertTrue(any(nid == 1487640225601 for nid, _ in hits_hageman))

            hits_nlrp3 = db.search_fts(["uric acid", "calcium pyrophosphate", "inflammasome"])
            self.assertTrue(any(nid == 1497310326771 for nid, _ in hits_nlrp3))


if __name__ == "__main__":
    unittest.main()
