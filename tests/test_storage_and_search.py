"""Comprehensive unit tests for storage, vector math, search pipeline, and deck operations."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher.ai_client import OpenAICompatibleClient
from ankihub_searcher.config_manager import ConfigManager
from ankihub_searcher.deck_ops import (
    execute_unit_deck_creation,
    resolve_notes_to_cards,
    rollback_run,
)
from ankihub_searcher.search_pipeline import (
    ExtractedConcept,
    ScoredNote,
    SmartSearchPipeline,
)
from ankihub_searcher.storage import (
    RunRecord,
    SearchDatabase,
    normalize_l2,
    pack_vector,
    unpack_vector,
)
from ankihub_searcher.text_utils import LectureUnit


class MockAnkiCol:
    """Mock for Anki Collection (col) with isolated in-memory or SQLite database."""

    def __init__(self):
        import sqlite3

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

        # Mock sub-objects
        self.sched = MagicMock()
        self.decks = MagicMock()
        self.models = MagicMock()

        self.decks_dict = {
            1: {"id": 1, "name": "Default"},
            100: {"id": 100, "name": "AnKing Step Deck"},
            101: {"id": 101, "name": "AnKing Step Deck::Pathology"},
            200: {"id": 200, "name": "Mastering Radiology"},
            300: {"id": 300, "name": "Other Unrelated Deck"},
        }
        self.decks.all.side_effect = lambda: list(self.decks_dict.values())
        self.decks.get.side_effect = lambda did: self.decks_dict.get(did, {})
        self.decks.id_for_name.side_effect = lambda name: next(
            (d["id"] for d in self.decks_dict.values() if d["name"] == name), None
        )
        self.next_did = 500

        def mock_new_filtered(name):
            did = self.next_did
            self.next_did += 1
            d = {"id": did, "name": name, "terms": [], "resched": True, "dyn": 1}
            self.decks_dict[did] = d
            return did

        self.decks.new_filtered.side_effect = mock_new_filtered
        self.decks.have.side_effect = lambda did: did in self.decks_dict
        self.decks.remove.side_effect = lambda dids: [self.decks_dict.pop(d, None) for d in dids]

    def _init_schema(self):
        self.conn.execute(
            """
            CREATE TABLE cards (
                id INTEGER PRIMARY KEY,
                nid INTEGER NOT NULL,
                did INTEGER NOT NULL,
                odid INTEGER DEFAULT 0,
                queue INTEGER DEFAULT 0,
                type INTEGER DEFAULT 0,
                reps INTEGER DEFAULT 0
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE revlog (
                id INTEGER PRIMARY KEY,
                cid INTEGER NOT NULL,
                usn INTEGER DEFAULT 0,
                ease INTEGER DEFAULT 0,
                ivl INTEGER DEFAULT 0,
                lastIvl INTEGER DEFAULT 0,
                factor INTEGER DEFAULT 0,
                time INTEGER DEFAULT 0,
                type INTEGER DEFAULT 0
            )
            """
        )
        self.conn.commit()

    @property
    def db(self):
        return self.conn


class TestStorageAndSearch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_search.db")
        self.storage = SearchDatabase(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Vector Normalization, Packing & Cosine Similarity
    def test_vector_packing_and_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [0.7071, 0.7071, 0.0]

        blob1 = pack_vector(v1)
        unpacked1 = unpack_vector(blob1, 3)
        self.assertAlmostEqual(unpacked1[0], 1.0, places=5)
        self.assertAlmostEqual(unpacked1[1], 0.0, places=5)

        # Store in database
        model_a = "model_a"
        self.storage.save_embeddings_batch(
            [
                (1, model_a, "h1", 3, v1),
                (2, model_a, "h2", 3, v2),
                (3, model_a, "h3", 3, v3),
            ]
        )

        # Query vector close to v1
        q = [0.99, 0.01, 0.0]

        # Test with numpy (if installed)
        sims_np = self.storage.batch_cosine_similarity([1, 2, 3], model_a, q, force_pure_python=False)
        self.assertEqual(sims_np[0][0], 1)  # v1 should be highest
        self.assertGreater(sims_np[0][1], 0.95)

        # Test with pure Python fallback
        sims_py = self.storage.batch_cosine_similarity([1, 2, 3], model_a, q, force_pure_python=True)
        self.assertEqual(sims_py[0][0], 1)
        self.assertAlmostEqual(sims_np[0][1], sims_py[0][1], places=4)

        # Test large candidate list (> 900 nids) to ensure no SQLite variable limit crash
        large_nids = list(range(1, 1500))
        sims_large = self.storage.batch_cosine_similarity(large_nids, model_a, q)
        self.assertEqual(len(sims_large), 3)
        self.assertEqual(sims_large[0][0], 1)

    # 2. Model Scoped Embedding Isolation
    def test_model_isolation_and_stats(self):
        model_1 = "provider::model_1"
        model_2 = "provider::model_2"

        self.storage.save_embeddings_batch([(101, model_1, "hash1", 2, [1.0, 0.0])])
        self.storage.save_embeddings_batch([(101, model_2, "hash1", 2, [0.0, 1.0])])

        # Query model_1
        hits_1 = self.storage.batch_cosine_similarity([101], model_1, [1.0, 0.0])
        self.assertAlmostEqual(hits_1[0][1], 1.0, places=4)

        hits_2 = self.storage.batch_cosine_similarity([101], model_2, [1.0, 0.0])
        self.assertAlmostEqual(hits_2[0][1], 0.0, places=4)

        stats = self.storage.get_indexed_models_stats()
        self.assertEqual(len(stats), 2)

        # Delete unused model
        deleted = self.storage.delete_embeddings_for_model(model_1)
        self.assertEqual(deleted, 1)
        new_stats = self.storage.get_indexed_models_stats()
        self.assertEqual(len(new_stats), 1)
        self.assertEqual(new_stats[0]["model_id"], model_2)

    # 3. Notes Content and FTS5 Lexical BM25 Search
    def test_fts5_lexical_search(self):
        notes = [
            (10, 1, 1000, "h10", "Gout is characterized by monosodium urate crystal deposition.", "Pathology, Joints"),
            (20, 1, 1000, "h20", "Pseudogout shows calcium pyrophosphate dihydrate CPPD crystals.", "Pathology, Cartilage"),
            (30, 1, 1000, "h30", "Acute pancreatitis with Cullen sign.", "GI, Pancreas"),
        ]
        self.storage.upsert_notes_content_batch(notes)

        # Search for urate crystals
        hits = self.storage.search_fts(["urate", "crystals"])
        self.assertTrue(len(hits) >= 1)
        self.assertEqual(hits[0][0], 10)  # Note 10 should be top result

        # Search for pseudogout
        hits_pseudo = self.storage.search_fts(["calcium", "pyrophosphate", "CPPD"])
        self.assertEqual(hits_pseudo[0][0], 20)

    # 4. Resilient JSON extraction in AI Client
    def test_ai_client_resilient_json(self):
        # Case 1: wrapped in <think> tags + json markdown fences
        raw_output = """
<think>
The student is asking about NLRP3 and gout.
I should decompose this into atomic concepts.
</think>
```json
{
  "concepts": [
    {
      "topic_it": "Inflammasoma NLRP3",
      "claim_en": "NLRP3 inflammasome is activated by uric acid crystals.",
      "keywords_en": ["NLRP3", "uric acid", "inflammasome"],
      "keywords_it": ["acido urico", "inflammasoma"]
    }
  ]
}
```
"""
        parsed = OpenAICompatibleClient.extract_clean_json(raw_output)
        self.assertIn("concepts", parsed)
        self.assertEqual(len(parsed["concepts"]), 1)
        self.assertEqual(parsed["concepts"][0]["keywords_en"][0], "NLRP3")

        # Case 2: Array wrapped in plain text
        raw_array = "Here is the result:\n[101, 102, 103]\nHope this helps!"
        parsed_arr = OpenAICompatibleClient.extract_clean_json(raw_array)
        self.assertEqual(parsed_arr, [101, 102, 103])

    # 5. Hybrid Search Pipeline (Stage 1 + 2 + 3)
    def test_search_pipeline_fusion(self):
        config = ConfigManager(
            sensitivity=0.55,
            hybrid_dense_weight=0.78,
            top_k_per_concept=5,
            max_cards_per_unit=10,
        )

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        # Mock chat completion returning concepts
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "topic_it": "Inflammasoma",
                    "claim_en": "NLRP3 inflammasome cleaves pro-IL-1beta into IL-1beta.",
                    "keywords_en": ["NLRP3", "inflammasome", "IL-1beta"],
                    "keywords_it": ["inflammasoma"],
                }
            ]
        }
        # Mock embeddings (2D vectors for simplicity)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        # Populate storage
        model_id = config.get_active_model_id()
        self.storage.upsert_notes_content_batch(
            [
                (1, 1, 100, "h1", "NLRP3 inflammasome activates IL-1beta.", "Pathology"),
                (2, 1, 100, "h2", "Unrelated card on Cardiology ECG.", "Cardio"),
            ]
        )
        self.storage.save_embeddings_batch(
            [
                (1, model_id, "h1", 2, [0.99, 0.01]),  # High similarity
                (2, model_id, "h2", 2, [0.05, 0.95]),  # Low similarity
            ]
        )

        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit("Inflammasoma", "Patologia - Inflammasoma", "L'inflammasoma NLRP3 attiva IL-1...")

        results = pipeline.search_unit(unit, candidate_nids=[1, 2], active_model_id=model_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].nid, 1)
        self.assertGreater(results[0].score, 0.70)

    # 6. Deck Operations: Safe Unsuspend & Cid Query Engine
    def test_deck_ops_creation_and_safe_rollback(self):
        col = MockAnkiCol()

        # Seed mock cards:
        # Notes 10 & 20 belong to AnKing (did 100 or 101)
        # Note 10: cid 1001 (suspended: queue=-1), cid 1002 (active: queue=1)
        # Note 20: cid 2001 (suspended: queue=-1)
        # Note 30: cid 3001 (in unrelated deck 300, suspended)
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (1001, 10, 100, 0, -1, 0, 0)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (1002, 10, 101, 0, 1, 1, 5)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (2001, 20, 100, 0, -1, 0, 0)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (3001, 30, 300, 0, -1, 0, 0)")
        col.conn.commit()

        config = ConfigManager(
            selected_decks=["AnKing Step Deck"],
            include_active_cards_in_filtered_deck=False,  # Default safety rule
        )

        # 1. Test resolve_notes_to_cards
        suspended_cids, active_cids = resolve_notes_to_cards(col, [10, 20, 30], config.selected_decks)
        # Sibling cards in AnKing: 1001 (suspended), 1002 (active), 2001 (suspended). Card 3001 is excluded!
        self.assertEqual(set(suspended_cids), {1001, 2001})
        self.assertEqual(set(active_cids), {1002})

        # 2. Test execute_unit_deck_creation
        unit = LectureUnit("Inflammation Unit", "Patologia - Infiammazione", "Body text")
        run_res = execute_unit_deck_creation(col, unit, [10, 20], config, self.storage)

        self.assertEqual(run_res.total_cards_count, 2)  # only suspended cards included by default
        self.assertEqual(run_res.newly_unsuspended_count, 2)
        self.assertEqual(set(run_res.newly_unsuspended_cids), {1001, 2001})
        self.assertEqual(run_res.already_active_count, 1)
        self.assertEqual(run_res.already_active_cids, [1002])

        # Verify unsuspend was called strictly on {1001, 2001}
        col.sched.unsuspend_cards.assert_called_once()
        unsuspended_arg = col.sched.unsuspend_cards.call_args[0][0]
        self.assertEqual(set(unsuspended_arg), {1001, 2001})

        # Verify filtered deck terms was set to exact cid: query
        created_did = run_res.filtered_did
        created_deck = col.decks_dict[created_did]
        self.assertTrue(created_deck["resched"])
        search_term = created_deck["terms"][0][0]
        self.assertTrue(search_term.startswith("cid:"))
        self.assertIn("1001", search_term)
        self.assertIn("2001", search_term)
        self.assertNotIn("1002", search_term)

        # Verify col.sched.rebuild_filtered_deck was called
        col.sched.rebuild_filtered_deck.assert_called_with(created_did)

        # 3. Test Rollback with keep_reviewed=True
        # Simulate that card 1001 was reviewed after the run (reps = 1)
        col.conn.execute("UPDATE cards SET reps = 1 WHERE id = 1001")
        col.conn.commit()

        # Update card queue state in DB: 1001 is unsuspended (queue=0), 2001 is unsuspended (queue=0), 1002 was already active
        col.conn.execute("UPDATE cards SET queue = 0 WHERE id IN (1001, 2001)")
        col.conn.commit()

        rollback_res = rollback_run(col, self.storage, run_res.run_id, keep_reviewed=True)
        self.assertTrue(rollback_res.deck_removed)
        self.assertEqual(rollback_res.kept_reviewed_count, 1)  # Card 1001 was reviewed, kept active!
        self.assertEqual(rollback_res.resuspended_count, 1)  # Card 2001 re-suspended

        # Verify suspend was called only on card 2001, NEVER on card 1002 (active before) or 1001 (reviewed)
        col.sched.suspend_cards.assert_called_once_with([2001])
        col.conn.close()

    def test_deck_ops_include_active_cards_in_filtered_deck(self):
        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (501, 50, 100, 0, -1, 0, 0)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (502, 50, 100, 0, 1, 1, 3)")
        col.conn.commit()

        config = ConfigManager(
            selected_decks=["AnKing Step Deck"],
            include_active_cards_in_filtered_deck=True,  # User explicitly included active cards
        )

        unit = LectureUnit("Unit Active Test", "Test - Active", "Body")
        run_res = execute_unit_deck_creation(col, unit, [50], config, self.storage)

        # Total cards in deck should be 2 (501 + 502), but newly_unsuspended strictly 1 (501)!
        self.assertEqual(run_res.total_cards_count, 2)
        self.assertEqual(run_res.newly_unsuspended_count, 1)
        self.assertEqual(run_res.newly_unsuspended_cids, [501])

        # unsuspend_cards should ONLY have been called with [501]
        col.sched.unsuspend_cards.assert_called_once_with([501])

        # Filtered deck query includes both 501 and 502
        created_deck = col.decks_dict[run_res.filtered_did]
        self.assertIn("501", created_deck["terms"][0][0])
        self.assertIn("502", created_deck["terms"][0][0])
        col.conn.close()

    def test_deck_ops_rollback_without_keep_reviewed(self):
        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (601, 60, 100, 0, -1, 0, 0)")
        col.conn.commit()

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        unit = LectureUnit("Unit Rollback All", "Test - Rollback All", "Body")
        run_res = execute_unit_deck_creation(col, unit, [60], config, self.storage)

        # Simulate card was reviewed
        col.conn.execute("UPDATE cards SET reps = 5, queue = 0 WHERE id = 601")
        col.conn.commit()

        # Rollback with keep_reviewed=False
        rollback_res = rollback_run(col, self.storage, run_res.run_id, keep_reviewed=False)
        self.assertEqual(rollback_res.resuspended_count, 1)
        self.assertEqual(rollback_res.kept_reviewed_count, 0)
        col.sched.suspend_cards.assert_called_once_with([601])
        col.conn.close()

    def test_config_manager_helpers(self):
        cfg = ConfigManager(
            embed_provider="openrouter",
            embed_model="openai/text-embedding-3-small",
            sensitivity=0.55,
            top_k_per_concept=10,
        )
        self.assertEqual(cfg.get_active_model_id(), "openrouter::openai/text-embedding-3-small")
        self.assertAlmostEqual(cfg.get_min_similarity_threshold(), 0.515, places=3)
        self.assertEqual(cfg.get_effective_top_k(), 10)

        # Field resolution
        fields = cfg.get_indexed_fields_for_model(
            "AnKingOverhaul (AnKing Step Deck / AnKingMed)",
            ["Text", "Extra", "Sketchy", "Lecture Notes"],
        )
        self.assertEqual(fields, ["Text", "Extra"])

        # Unknown model fallback to default fields
        fields_def = cfg.get_indexed_fields_for_model("Custom Model", ["Front", "Back", "RandomFld"])
        self.assertEqual(fields_def, ["Front", "Back"])

    def test_incremental_indexer(self):
        from ankihub_searcher.indexer import DeckIndexer

        col = MockAnkiCol()
        # Seed notes table
        col.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                mid INTEGER NOT NULL,
                mod INTEGER NOT NULL,
                flds TEXT NOT NULL,
                tags TEXT NOT NULL
            )
            """
        )
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (7001, 70, 100, 0, -1)")
        col.conn.execute(
            "INSERT INTO notes (id, mid, mod, flds, tags) VALUES (70, 1, 1234, 'Gout note text\x1fExtra info', '#AK_Step1_v12::Pathology')"
        )
        col.conn.commit()

        # Mock model structure in col.models
        mock_mid = MagicMock()
        mock_mid.id = 1
        col.models.all_names_and_ids.return_value = [mock_mid]
        col.models.get.return_value = {
            "id": 1,
            "name": "Custom Model",
            "flds": [{"name": "Text"}, {"name": "Extra"}],
        }

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[0.8, 0.6]]

        indexer = DeckIndexer(config, self.storage, mock_ai)
        summary = indexer.index_selected_decks(col)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["text_indexed"], 1)
        self.assertEqual(summary["embeddings_indexed"], 1)
        self.assertFalse(summary["cancelled"])

        # Second indexing run should detect nothing changed!
        summary2 = indexer.index_selected_decks(col)
        self.assertEqual(summary2["text_indexed"], 0)
        self.assertEqual(summary2["embeddings_indexed"], 0)

        col.conn.close()

    def test_merge_external_db_and_slot_separation(self):
        # 1. Test merging an external DB with another model's embeddings
        ext_db_path = os.path.join(self.temp_dir.name, "external_air.db")
        ext_db = SearchDatabase(ext_db_path)
        ext_db.upsert_notes_content_batch([(500, 1, 1000, "h500", "External note text", "Pathology")])
        ext_db.save_embeddings_batch([(500, "openrouter::text-embedding-3-small", "h500", 3, [0.1, 0.2, 0.9])])

        merge_res = self.storage.merge_external_db(ext_db_path)
        self.assertGreaterEqual(merge_res["embeddings_merged"], 1)
        stats = {s["model_id"]: s["note_count"] for s in self.storage.get_indexed_models_stats()}
        self.assertIn("openrouter::text-embedding-3-small", stats)
        self.assertEqual(stats["openrouter::text-embedding-3-small"], 1)

        # 2. Test slot separation: active notes don't steal top_k_per_concept from suspended notes
        config = ConfigManager(
            sensitivity=0.55,
            top_k_per_concept=1,
            max_cards_per_unit=5,
            include_active_cards_in_filtered_deck=False,
        )
        model_id = config.get_active_model_id()
        self.storage.upsert_notes_content_batch(
            [
                (1, 1, 100, "h1", "Active note with highest similarity.", "Pathology"),
                (2, 1, 100, "h2", "Suspended note with second highest similarity.", "Pathology"),
            ]
        )
        self.storage.save_embeddings_batch(
            [
                (1, model_id, "h1", 2, [1.0, 0.0]),
                (2, model_id, "h2", 2, [0.95, 0.05]),
            ]
        )
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [{"topic_it": "Test", "claim_en": "Test claim", "keywords_en": ["note"], "keywords_it": []}]
        }
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit("Test", "Test Deck", "Test body")
        # Note 1 is already active, Note 2 is suspended (suspended_nids={2})
        # Even with top_k_per_concept=1, Note 2 must NOT be dropped because Note 1 is active!
        results = pipeline.search_unit(unit, candidate_nids=[1, 2], active_model_id=model_id, suspended_nids={2})
        result_nids = {r.nid for r in results}
        self.assertEqual(result_nids, {1, 2})


if __name__ == "__main__":
    unittest.main()
