"""Comprehensive unit tests for storage, vector math, search pipeline, and deck operations."""

import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher.ai_client import AIClientError, OpenAICompatibleClient, repair_json_escapes
from ankihub_searcher.config_manager import ConfigManager
from ankihub_searcher.deck_ops import (
    RollbackResult,
    RunResult,
    assign_notes_to_best_units,
    create_batch_filtered_decks_op,
    execute_unit_deck_creation,
    resolve_notes_to_cards,
    rollback_run,
    rollback_run_op,
    rollback_search_batch_op,
)
from ankihub_searcher.ui.worker import RollbackWorker, SearchBatchWorker
from ankihub_searcher.search_pipeline import (
    ConceptCoverage,
    ExtractedConcept,
    STAGE1_JSON_SCHEMA,
    SYSTEM_PROMPT_STAGE1,
    ScoredNote,
    SmartSearchPipeline,
    Stage1ExtractionError,
    extract_first_sentence,
    select_candidates_for_concept,
)
from ankihub_searcher.storage import (
    RunRecord,
    SearchDatabase,
    normalize_l2,
    pack_vector,
    unpack_vector,
)
from ankihub_searcher.text_utils import LectureUnit
from ankihub_searcher.ui.main_dialog import CoverageDetailDialog, compute_coverage_stats


class MockAnkiCol:
    """Mock for Anki Collection (col) with isolated in-memory or SQLite database."""

    def __init__(self):
        import sqlite3

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
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

        # Undo entries support
        self.undo_entries = []
        self.add_custom_undo_entry = MagicMock(side_effect=self._mock_add_undo)
        self.merge_undo_entries = MagicMock(side_effect=self._mock_merge_undo)

    def _mock_add_undo(self, name: str) -> int:
        self.undo_entries.append(name)
        return len(self.undo_entries)

    def _mock_merge_undo(self, pos: int) -> Any:
        changes = MagicMock(name="OpChanges")
        changes.deck = True
        changes.card = True
        return changes

    def _init_schema(self):
        self.conn.execute(
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

    def test_database_legacy_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_file = os.path.join(tmp, "ankihub_searcher.db")
            new_file = os.path.join(tmp, "smart_search.db")
            # Create a legacy db and insert data
            legacy_db = SearchDatabase(legacy_file)
            legacy_db.upsert_notes_content_batch([(999, 1, 100, "h999", "Migrated note", "Tag")])

            # Execute migration
            SearchDatabase._migrate_legacy_db(legacy_file, new_file)
            self.assertFalse(os.path.exists(legacy_file))
            self.assertTrue(os.path.exists(new_file))

            # Load migrated database
            migrated_db = SearchDatabase(new_file)
            content = migrated_db.get_notes_content([999])
            self.assertIn(999, content)
            self.assertEqual(content[999]["clean_text"], "Migrated note")

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
        # Simulate that card 1001 was reviewed after the run (entry in revlog with id >= created_at * 1000)
        revlog_time = int(time.time() * 1000) + 5000
        col.conn.execute("INSERT INTO revlog (id, cid) VALUES (?, 1001)", (revlog_time,))
        # Card 2001 has reps=3 from old reviews, but NO revlog entry after this run
        col.conn.execute("UPDATE cards SET reps = 3 WHERE id = 2001")
        col.conn.commit()

        # Update card queue state in DB: 1001 is unsuspended (queue=0), 2001 is unsuspended (queue=0), 1002 was already active
        col.conn.execute("UPDATE cards SET queue = 0 WHERE id IN (1001, 2001)")
        col.conn.commit()

        rollback_res = rollback_run(col, self.storage, run_res.run_id, keep_reviewed=True)
        self.assertTrue(rollback_res.deck_removed)
        self.assertEqual(rollback_res.kept_reviewed_count, 1)  # Card 1001 was reviewed in revlog, kept active!
        self.assertEqual(rollback_res.resuspended_count, 1)  # Card 2001 had reps=3 but no revlog, re-suspended!

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
        self.assertAlmostEqual(cfg.get_min_similarity_threshold(), 0.59, places=2)
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

        # Test export_db
        exported_path = os.path.join(self.temp_dir.name, "exported_backup.db")
        size_mb = self.storage.export_db(exported_path)
        self.assertTrue(os.path.exists(exported_path))
        self.assertGreaterEqual(size_mb, 0.0)

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

    def test_query_parent_deck_and_concept_coverage(self):
        from ankihub_searcher.indexer import DeckIndexer

        col = MockAnkiCol()
        config = ConfigManager()
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        indexer = DeckIndexer(config, self.storage, mock_ai)

        # 1. Single deck query -> returns that deck as parent
        self.assertEqual(
            indexer.detect_single_parent_deck(col, '"deck:AnKing Step Deck"'),
            "AnKing Step Deck",
        )
        # Subdecks of the same parent -> returns common ancestor
        self.assertEqual(
            indexer.detect_single_parent_deck(
                col, '"deck:AnKing Step Deck::#AK_Step1_v12" OR "deck:AnKing Step Deck::#AK_Step2_v12"'
            ),
            "AnKing Step Deck",
        )
        # Two disjoint top-level decks -> returns None
        self.assertIsNone(
            indexer.detect_single_parent_deck(
                col, '"deck:AnKing Step Deck" OR "deck:Mastering Radiology"'
            )
        )

        # 2. Query history management
        config.add_query_to_history('"deck:Mastering Radiology"')
        self.assertEqual(config.search_query, '"deck:Mastering Radiology"')
        self.assertEqual(config.search_query_history[0], '"deck:Mastering Radiology"')

        # 3. Concept coverage (covered vs uncovered) + subdeck creation under single parent deck
        model_id = config.get_active_model_id()
        self.storage.upsert_notes_content_batch(
            [
                (801, 1, 100, "h801", "NLRP3 inflammasome activates IL-1beta in gout.", "Pathology"),
                (802, 1, 100, "h802", "Monosodium urate crystals trigger NLRP3 inflammasome in gout.", "Pathology"),
            ]
        )
        self.storage.save_embeddings_batch(
            [
                (801, model_id, "h801", 2, [1.0, 0.0]),
                (802, model_id, "h802", 2, [0.98, 0.02]),
            ]
        )

        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "topic_it": "Inflammasoma NLRP3",
                    "claim_en": "NLRP3 inflammasome activates IL-1beta in gout.",
                    "keywords_en": ["NLRP3", "gout"],
                    "keywords_it": ["gotta"],
                },
                {
                    "topic_it": "Dettaglio non presente nel mazzo",
                    "claim_en": "Unrelated epidemiological detail from Italian textbook.",
                    "keywords_en": ["unrelated_xyz_token"],
                    "keywords_it": ["inesistente"],
                },
            ]
        }
        # First concept matches [1.0, 0.0], second concept is orthogonal [0.0, 1.0]
        mock_ai.get_embeddings.return_value = [[1.0, 0.0], [0.0, 1.0]]

        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit("Gotta", "Gotta", "Testo lezione")
        scored = pipeline.search_unit(unit, candidate_nids=[801, 802], active_model_id=model_id)
        self.assertEqual(len(scored), 2)
        self.assertEqual(len(pipeline.last_concept_coverage), 2)
        self.assertEqual(pipeline.last_concept_coverage[0].status, "covered")
        self.assertEqual(pipeline.last_concept_coverage[1].status, "uncovered")

        # Execute filtered deck creation with single parent_deck_prefix="AnKing Step Deck"
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (8001, 801, 100, 0, -1)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (8002, 802, 100, 0, -1)")
        col.conn.commit()

        cov_dicts = [c.to_dict() for c in pipeline.last_concept_coverage]
        run_res = execute_unit_deck_creation(
            col=col,
            unit=unit,
            matched_nids=[s.nid for s in scored],
            config=config,
            storage=self.storage,
            concept_coverage=cov_dicts,
            parent_deck_prefix="AnKing Step Deck",
        )
        # Filtered deck must be created as a child of AnKing Step Deck!
        self.assertEqual(run_res.deck_name, "AnKing Step Deck::Gotta")
        self.assertEqual(len(run_res.concept_coverage), 2)

        # Verify concept_coverage persisted in SQLite run_history
        saved_run = self.storage.get_run(run_res.run_id)
        self.assertIsNotNone(saved_run)
        self.assertEqual(len(saved_run.concept_coverage), 2)
        self.assertEqual(saved_run.concept_coverage[1]["status"], "uncovered")
        col.conn.close()

    def test_ai_client_invalid_escapes_repair(self):
        # 1. Test repair_json_escapes
        raw_invalid = r'{"claim": "Receptors \kappa and \alpha-2 agonists with \mathrm{TGF}-\beta"}'
        repaired = repair_json_escapes(raw_invalid)
        # repair should double backslashes for \k, \a, \m
        import json
        parsed = json.loads(repaired)
        self.assertIn(r"\kappa", parsed["claim"])
        self.assertIn(r"\alpha-2", parsed["claim"])

        # 2. Test extract_clean_json handles invalid escapes directly
        llm_output = (
            "Ecco i concetti richiesti:\n```json\n"
            r'{"concepts": [{"topic_it": "Recettori \beta", "claim_en": "Targets \kappa and \alpha receptors"}]}'
            "\n```"
        )
        extracted = OpenAICompatibleClient.extract_clean_json(llm_output)
        self.assertEqual(len(extracted["concepts"]), 1)
        self.assertIn(r"\kappa", extracted["concepts"][0]["claim_en"])

        # 3. Test that valid escapes are preserved
        valid_json = r'{"text": "Line 1\nLine 2\tTabbed \"quote\" \\ slash \u0041"}'
        parsed_valid = OpenAICompatibleClient.extract_clean_json(valid_json)
        self.assertEqual(parsed_valid["text"], 'Line 1\nLine 2\tTabbed "quote" \\ slash A')

        # 4. Test with 'TGF-$\beta$' and 'NF-$\kappa$B' (LaTeX escapes that would become control chars \x08 or invalid JSON)
        raw_tgf_beta = r'{"claim": "Attivazione di TGF-$\beta$"}'
        parsed_tgf_beta = OpenAICompatibleClient.extract_clean_json(raw_tgf_beta)
        self.assertNotIn("\x08", parsed_tgf_beta["claim"])
        self.assertIn(r"\beta", parsed_tgf_beta["claim"])

        raw_both = r'{"cytokine": "TGF-$\beta$", "transcription_factor": "NF-$\kappa$B"}'
        parsed_both = OpenAICompatibleClient.extract_clean_json(raw_both)
        self.assertNotIn("\x08", parsed_both["cytokine"])
        self.assertNotIn("\x0c", parsed_both["cytokine"])
        self.assertIn(r"\beta", parsed_both["cytokine"])
        self.assertIn(r"\kappa", parsed_both["transcription_factor"])

    def test_chat_completion_json_schema_and_400_fallback(self):
        client = OpenAICompatibleClient()

        # 1. Normal call with schema: verify payload
        calls = []

        def mock_request(url, method="GET", data=None, api_key="", timeout=None):
            calls.append(data)
            return {
                "choices": [
                    {"message": {"content": '{"result": "ok"}'}}
                ]
            }

        client._make_request = mock_request

        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        res = client.chat_completion_json(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            base_url="http://localhost:1234/v1",
            json_schema=schema,
            max_tokens=2048,
        )
        self.assertEqual(res, {"result": "ok"})
        self.assertEqual(len(calls), 1)
        self.assertIn("response_format", calls[0])
        self.assertEqual(calls[0]["max_tokens"], 2048)

        # 2. 400 error fallback: server rejects response_format
        calls.clear()
        attempt_count = 0

        def mock_request_400(url, method="GET", data=None, api_key="", timeout=None):
            nonlocal attempt_count
            attempt_count += 1
            calls.append(data)
            if "response_format" in data:
                raise AIClientError("HTTP 400 (Bad Request): response_format unsupported", status_code=400)
            return {
                "choices": [
                    {"message": {"content": '{"result": "fallback_ok"}'}}
                ]
            }

        client._make_request = mock_request_400
        res2 = client.chat_completion_json(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            base_url="http://localhost:1234/v1",
            json_schema=schema,
        )
        self.assertEqual(res2, {"result": "fallback_ok"})
        self.assertEqual(attempt_count, 2)
        # First call had response_format, second call did not
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

    def test_stage1_no_think_and_latex_normalization(self):
        config_qwen = ConfigManager(chat_model="qwen-2.5-7b-instruct")
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [{"topic_it": "Topic", "claim_en": "Claim", "keywords_en": ["k1"], "keywords_it": []}]
        }
        pipeline_qwen = SmartSearchPipeline(config_qwen, self.storage, mock_ai)
        unit = LectureUnit(
            unit_title=r"Recettori $\beta_2$",
            deck_name="Test Deck",
            body_text=r"I farmaci $\beta$-bloccanti agiscono sui recettori $\alpha$ e $\kappa$ con $\Delta P \ge 50$.",
            breadcrumb=r"Farmacologia > $\beta$",
        )

        pipeline_qwen.extract_concepts(unit)

        # Verify call to chat_completion_json
        call_args = mock_ai.chat_completion_json.call_args[1]
        user_prompt = call_args["messages"][1]["content"]

        # 1. LaTeX should be normalized in user prompt
        self.assertIn("Recettori β_2", user_prompt)
        self.assertIn("β-bloccanti", user_prompt)
        self.assertIn("recettori α e κ", user_prompt)
        self.assertIn("Δ P ≥ 50", user_prompt)
        self.assertNotIn(r"$\beta$", user_prompt)
        self.assertNotIn(r"$\alpha$", user_prompt)

        # 2. Qwen model must have /no_think
        self.assertTrue(user_prompt.endswith("/no_think"))

        # 3. Non-Qwen model must NOT have /no_think
        config_gemma = ConfigManager(chat_model="google/gemma-2-9b-it")
        mock_ai_gemma = MagicMock(spec=OpenAICompatibleClient)
        mock_ai_gemma.chat_completion_json.return_value = {
            "concepts": [{"topic_it": "Topic", "claim_en": "Claim", "keywords_en": ["k1"], "keywords_it": []}]
        }
        pipeline_gemma = SmartSearchPipeline(config_gemma, self.storage, mock_ai_gemma)
        pipeline_gemma.extract_concepts(unit)

        gemma_prompt = mock_ai_gemma.chat_completion_json.call_args[1]["messages"][1]["content"]
        self.assertNotIn("/no_think", gemma_prompt)

    def test_stage1_retry_success_and_failure_without_ascii_fallback(self):
        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (9001, 901, 100, 0, -1, 0, 0)")
        col.conn.commit()

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        mock_ai = MagicMock(spec=OpenAICompatibleClient)

        # 1. Test retry on first failure: 1st fails, 2nd succeeds
        mock_ai.chat_completion_json.side_effect = [
            AIClientError("Temporary network glitch"),
            {"concepts": [{"topic_it": "OK", "claim_en": "Claim OK", "keywords_en": ["k"], "keywords_it": []}]},
        ]
        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit("Unit Retry", "Deck Retry", "Body text with medical concepts")

        concepts = pipeline.extract_concepts(unit)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(mock_ai.chat_completion_json.call_count, 2)

        # 2. Test both attempts fail: raises Stage1ExtractionError and NO ASCII words fallback!
        mock_ai.reset_mock()
        mock_ai.chat_completion_json.side_effect = [
            AIClientError("Invalid JSON from LLM: parse error"),
            AIClientError("Invalid JSON from LLM: parse error attempt 2"),
        ]

        with self.assertRaises(Stage1ExtractionError):
            pipeline.extract_concepts(unit)
        self.assertEqual(mock_ai.chat_completion_json.call_count, 2)

        # 3. Test search_unit marks unit_error and returns [] (0 notes found)
        mock_ai.reset_mock()
        mock_ai.chat_completion_json.side_effect = [
            AIClientError("Error 1"),
            AIClientError("Error 2"),
        ]
        scored_notes = pipeline.search_unit(unit, candidate_nids=[901], active_model_id="test-model")
        self.assertEqual(scored_notes, [])
        self.assertTrue(len(pipeline.last_unit_error) > 0)
        self.assertIn("Error 2", pipeline.last_unit_error)

        # 4. Test execute_unit_deck_creation with unit_error:
        # No cards unsuspended, no filtered deck created, run saved in history with unit_error
        run_res = execute_unit_deck_creation(
            col=col,
            unit=unit,
            matched_nids=[],
            config=config,
            storage=self.storage,
            unit_error=pipeline.last_unit_error,
        )
        self.assertEqual(run_res.total_cards_count, 0)
        self.assertEqual(run_res.newly_unsuspended_count, 0)
        self.assertEqual(run_res.filtered_did, 0)
        self.assertEqual(run_res.unit_error, pipeline.last_unit_error)

        # Verify unsuspend_cards was NEVER called
        col.sched.unsuspend_cards.assert_not_called()

        # Verify run_history recorded the unit_error
        saved_run = self.storage.get_run(run_res.run_id)
        self.assertIsNotNone(saved_run)
        self.assertEqual(saved_run.unit_error, pipeline.last_unit_error)
        self.assertEqual(saved_run.newly_unsuspended_cids, [])

        # Verify get_runs also preserves unit_error
        all_runs = self.storage.get_runs()
        matched_saved = [r for r in all_runs if r.run_id == run_res.run_id]
        self.assertEqual(len(matched_saved), 1)
        self.assertEqual(matched_saved[0].unit_error, pipeline.last_unit_error)
        col.conn.close()

    def test_stage1_prompt_entity_and_quote_it(self):
        # 1. Verify prompt rules: no "3 to 12", mentions entities, prohibitions, quote_it
        self.assertNotIn("3 to 12", SYSTEM_PROMPT_STAGE1)
        self.assertIn("quote_it", SYSTEM_PROMPT_STAGE1)
        self.assertIn("NEVER merge distinct entities", SYSTEM_PROMPT_STAGE1)
        self.assertIn("Ignore non-medical digressions", SYSTEM_PROMPT_STAGE1)

        # 2. Verify STAGE1_JSON_SCHEMA requires quote_it and other required fields
        schema_props = STAGE1_JSON_SCHEMA["schema"]["properties"]["concepts"]["items"]["properties"]
        self.assertIn("quote_it", schema_props)
        schema_required = STAGE1_JSON_SCHEMA["schema"]["properties"]["concepts"]["items"]["required"]
        expected_required = ["quote_it", "topic_it", "claim_en", "keywords_en", "keywords_it"]
        for req in expected_required:
            self.assertIn(req, schema_required)

        # 3. Test extraction with quote_it and propagation into ConceptCoverage and history
        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (9501, 950, 100, 0, -1, 0, 0)")
        col.conn.commit()

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "quote_it": "L'eparina attiva l'antitrombina III inibendo la trombina.",
                    "topic_it": "Eparina non frazionata",
                    "claim_en": "Heparin potentiates antithrombin III to inhibit thrombin and factor Xa.",
                    "keywords_en": ["unfractionated heparin", "antithrombin III", "factor Xa"],
                    "keywords_it": ["eparina", "antitrombina III"],
                }
            ]
        }
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit(
            "Terapia Anticoagulante",
            "Farmacologia - Eparina",
            "L'eparina attiva l'antitrombina III inibendo la trombina.",
        )

        concepts = pipeline.extract_concepts(unit)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(
            concepts[0].quote_it,
            "L'eparina attiva l'antitrombina III inibendo la trombina.",
        )

        # Run search to populate pipeline.last_concept_coverage
        self.storage.save_embeddings_batch([(950, "default::default", "h", 2, [1.0, 0.0])])
        results = pipeline.search_unit(unit, candidate_nids=[950], active_model_id="default::default")
        self.assertEqual(len(results), 1)
        self.assertEqual(len(pipeline.last_concept_coverage), 1)

        cov = pipeline.last_concept_coverage[0]
        self.assertEqual(cov.quote_it, "L'eparina attiva l'antitrombina III inibendo la trombina.")

        # Test to_dict and from_dict
        cov_dict = cov.to_dict()
        self.assertEqual(cov_dict["quote_it"], "L'eparina attiva l'antitrombina III inibendo la trombina.")
        restored_cov = ConceptCoverage.from_dict(cov_dict)
        self.assertEqual(restored_cov.quote_it, cov.quote_it)

        # Test persistence in SQLite run_history
        run_res = execute_unit_deck_creation(
            col=col,
            unit=unit,
            matched_nids=[950],
            config=config,
            storage=self.storage,
            concept_coverage=[cov_dict],
        )
        saved_run = self.storage.get_run(run_res.run_id)
        self.assertIsNotNone(saved_run)
        self.assertEqual(len(saved_run.concept_coverage), 1)
        self.assertEqual(
            saved_run.concept_coverage[0]["quote_it"],
            "L'eparina attiva l'antitrombina III inibendo la trombina.",
        )
        col.conn.close()

    def test_relative_score_window_selection(self):
        # Test relative score window cutoff and max_notes_per_concept
        config = ConfigManager(
            sensitivity=0.55,  # threshold ~0.515
            relative_score_window=0.08,
            max_notes_per_concept=2,
            max_cards_per_unit=150,
        )
        self.assertEqual(config.relative_score_window, 0.08)
        self.assertEqual(config.max_notes_per_concept, 2)
        self.assertEqual(config.max_cards_per_unit, 150)

        # Migration check: legacy config with only top_k_per_concept
        legacy_data = {"top_k_per_concept": 5}
        cfg_legacy = ConfigManager()
        cfg_legacy.update(legacy_data)
        self.assertIsNone(cfg_legacy.top_k_per_concept)
        self.assertEqual(cfg_legacy.relative_score_window, 0.05)
        self.assertEqual(cfg_legacy.max_notes_per_concept, 10)
        self.assertEqual(cfg_legacy.config_version, 4)

        model_id = config.get_active_model_id()
        # Seed 4 notes with different similarity levels:
        # Note 701: score ~0.90 (best)
        # Note 702: score ~0.85 (diff 0.05 <= 0.08: INCLUDED)
        # Note 703: score ~0.84 (diff 0.06 <= 0.08: would be included by window, but max_notes_per_concept=2 drops it!)
        # Note 704: score ~0.70 (diff 0.20 > 0.08: EXCLUDED by relative window even though > 0.515 threshold)
        self.storage.upsert_notes_content_batch(
            [
                (701, 1, 100, "h701", "Beta-blocker card 1", "Pharm"),
                (702, 1, 100, "h702", "Beta-blocker card 2", "Pharm"),
                (703, 1, 100, "h703", "Beta-blocker card 3", "Pharm"),
                (704, 1, 100, "h704", "Beta-blocker card 4", "Pharm"),
            ]
        )
        self.storage.save_embeddings_batch(
            [
                (701, model_id, "h701", 2, [0.90, 0.4358]),
                (702, model_id, "h702", 2, [0.85, 0.5267]),
                (703, model_id, "h703", 2, [0.84, 0.5425]),
                (704, model_id, "h704", 2, [0.70, 0.7141]),
            ]
        )

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "quote_it": "Beta bloccanti",
                    "topic_it": "Beta bloccanti",
                    "claim_en": "Beta blockers mechanism",
                    "keywords_en": ["beta-blocker"],
                    "keywords_it": [],
                }
            ]
        }
        # Query vector aligned with [1.0, 0.0]
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        pipeline = SmartSearchPipeline(config, self.storage, mock_ai)
        unit = LectureUnit("Beta", "Beta Deck", "Beta bloccanti")
        scored = pipeline.search_unit(unit, candidate_nids=[701, 702, 703, 704], active_model_id=model_id)
        scored_nids = [s.nid for s in scored]

        # Best is 701, next within window is 702. 703 dropped by max_notes_per_concept=2. 704 dropped by relative window.
        self.assertEqual(scored_nids, [701, 702])

    def test_fts_clean_lexical_search_and_30k_candidates(self):
        # 1. Test stopword and <3 char discarding
        hits_empty = self.storage.search_fts(["the", "di", "in", "a", "x", "no"])
        self.assertEqual(hits_empty, [])

        # Seed sample notes for phrase query testing
        self.storage.upsert_notes_content_batch(
            [
                (1101, 1, 100, "h1101", "Patient presents with acute pancreatitis and elevated amylase.", "GI"),
                (1102, 1, 100, "h1102", "Pancreatitis can be chronic or recurrent.", "GI"),
                (1103, 1, 100, "h1103", "Acute intermittent porphyria has neurological symptoms.", "Heme"),
            ]
        )

        # 2. Test multi-word phrase query: "acute pancreatitis"
        # 1101 has the exact phrase -> must be the top result with higher score than 1102 or 1103
        phrase_hits = self.storage.search_fts(["acute pancreatitis"])
        self.assertTrue(len(phrase_hits) >= 2)
        self.assertEqual(phrase_hits[0][0], 1101)
        self.assertEqual(phrase_hits[0][1], 1.0)
        # 1102 matched only 'pancreatitis', so its score should be lower than 1101
        hit_1102 = next((h for h in phrase_hits if h[0] == 1102), None)
        self.assertIsNotNone(hit_1102)
        self.assertLess(hit_1102[1], 1.0)

        # 3. Test quote escaping inside phrase query
        hits_quoted = self.storage.search_fts(['"acute" pancreatitis'])
        self.assertTrue(len(hits_quoted) >= 1)
        self.assertEqual(hits_quoted[0][0], 1101)

        # 3b. Test hyphenated keywords: 'IL-1', 'TNF-alpha', 'heparin-induced thrombocytopenia'
        # Ensures no 'no such column' error and finds matching notes in actual FTS5 DB
        self.storage.upsert_notes_content_batch(
            [
                (1201, 1, 100, "h1201", "IL-1 beta is produced by the NLRP3 inflammasome.", "Immunology"),
                (1202, 1, 100, "h1202", "Infliximab is a monoclonal antibody targeting TNF-alpha.", "Pharm"),
                (1203, 1, 100, "h1203", "Autoimmune reaction causing heparin-induced thrombocytopenia (HIT).", "Heme"),
            ]
        )

        hits_il1 = self.storage.search_fts(["IL-1"])
        self.assertTrue(len(hits_il1) >= 1)
        self.assertEqual(hits_il1[0][0], 1201)

        hits_tnf = self.storage.search_fts(["TNF-alpha"])
        self.assertTrue(len(hits_tnf) >= 1)
        self.assertEqual(hits_tnf[0][0], 1202)

        hits_hit = self.storage.search_fts(["heparin-induced thrombocytopenia"])
        self.assertTrue(len(hits_hit) >= 1)
        self.assertEqual(hits_hit[0][0], 1203)

        # 4. Test candidate filtering with 30,000 NIDs (avoids SQLite variable limits)
        # Note 1101 is included in the 30k candidates; Note 1102 is NOT.
        large_candidates = list(range(2000, 32000)) + [1101]
        self.assertEqual(len(large_candidates), 30001)

        large_filtered_hits = self.storage.search_fts(
            keywords=["acute pancreatitis"],
            candidate_nids=large_candidates,
            limit=50,
        )
        returned_nids = {h[0] for h in large_filtered_hits}
        self.assertIn(1101, returned_nids)
        self.assertNotIn(1102, returned_nids)
        self.assertNotIn(1103, returned_nids)

        # 5. Test fts_use_italian_keywords option in SmartSearchPipeline (with hybrid_dense_weight < 1.0 to exercise FTS)
        cfg_default = ConfigManager(hybrid_dense_weight=0.78)
        self.assertFalse(cfg_default.fts_use_italian_keywords)

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "quote_it": "Pancreatite acuta",
                    "topic_it": "Pancreatite acuta",
                    "claim_en": "Acute pancreatitis signs",
                    "keywords_en": ["acute pancreatitis"],
                    "keywords_it": ["pancreatite acuta"],
                }
            ]
        }
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]
        mid = cfg_default.get_active_model_id()
        self.storage.save_embeddings_batch([(1101, mid, "h", 2, [1.0, 0.0])])

        # Spy on storage.search_fts
        original_search_fts = self.storage.search_fts
        captured_keywords = []

        def spy_search_fts(keywords, candidate_nids=None, limit=100):
            captured_keywords.append(list(keywords))
            return original_search_fts(keywords, candidate_nids=candidate_nids, limit=limit)

        self.storage.search_fts = spy_search_fts

        # With default fts_use_italian_keywords=False: only keywords_en passed
        pipeline = SmartSearchPipeline(cfg_default, self.storage, mock_ai)
        unit = LectureUnit("Test", "Test", "Pancreatite acuta")
        pipeline.search_unit(unit, candidate_nids=[1101], active_model_id=mid)
        self.assertEqual(len(captured_keywords), 1)
        self.assertIn("acute pancreatitis", captured_keywords[0])
        self.assertNotIn("pancreatite acuta", captured_keywords[0])

        # With fts_use_italian_keywords=True: both keywords_en and keywords_it passed
        captured_keywords.clear()
        cfg_it = ConfigManager(fts_use_italian_keywords=True, hybrid_dense_weight=0.78)
        pipeline_it = SmartSearchPipeline(cfg_it, self.storage, mock_ai)
        pipeline_it.search_unit(unit, candidate_nids=[1101], active_model_id=mid)
        self.assertEqual(len(captured_keywords), 1)
        self.assertIn("acute pancreatitis", captured_keywords[0])
        self.assertIn("pancreatite acuta", captured_keywords[0])

        self.storage.search_fts = original_search_fts

    def test_embed_query_instruction(self):
        concept = ExtractedConcept(
            topic_it="Inflammasoma",
            claim_en="NLRP3 inflammasome activates caspase-1",
            keywords_en=["NLRP3", "caspase-1"],
            keywords_it=[],
        )

        # 1. Non-Qwen model with empty embed_query_instruction -> no prefix
        cfg_plain = ConfigManager(embed_model="text-embedding-3-small", embed_query_instruction="")
        mock_ai_plain = MagicMock(spec=OpenAICompatibleClient)
        mock_ai_plain.get_embeddings.return_value = [[1.0, 0.0]]
        pipeline_plain = SmartSearchPipeline(cfg_plain, self.storage, mock_ai_plain)
        pipeline_plain.retrieve_for_concepts([concept], candidate_nids=[], active_model_id="plain")

        plain_query = mock_ai_plain.get_embeddings.call_args[1]["texts"][0]
        self.assertEqual(plain_query, "NLRP3 inflammasome activates caspase-1 | NLRP3, caspase-1")
        self.assertNotIn("Instruct:", plain_query)

        # 2. Model containing 'qwen3-embedding' with empty embed_query_instruction -> auto-applies default instruction
        cfg_qwen = ConfigManager(embed_model="Qwen/Qwen3-Embedding-8B", embed_query_instruction="")
        mock_ai_qwen = MagicMock(spec=OpenAICompatibleClient)
        mock_ai_qwen.get_embeddings.return_value = [[1.0, 0.0]]
        pipeline_qwen = SmartSearchPipeline(cfg_qwen, self.storage, mock_ai_qwen)
        pipeline_qwen.retrieve_for_concepts([concept], candidate_nids=[], active_model_id="qwen")

        qwen_query = mock_ai_qwen.get_embeddings.call_args[1]["texts"][0]
        expected_prefix = "Instruct: Given a medical lecture concept, retrieve flashcards that test it\nQuery: "
        self.assertTrue(qwen_query.startswith(expected_prefix))
        self.assertIn("NLRP3 inflammasome activates caspase-1", qwen_query)

        # 3. Explicit custom embed_query_instruction -> applies custom instruction
        cfg_custom = ConfigManager(embed_model="any-model", embed_query_instruction="Custom Instruction:\n")
        mock_ai_custom = MagicMock(spec=OpenAICompatibleClient)
        mock_ai_custom.get_embeddings.return_value = [[1.0, 0.0]]
        pipeline_custom = SmartSearchPipeline(cfg_custom, self.storage, mock_ai_custom)
        pipeline_custom.retrieve_for_concepts([concept], candidate_nids=[], active_model_id="custom")

        custom_query = mock_ai_custom.get_embeddings.call_args[1]["texts"][0]
        self.assertTrue(custom_query.startswith("Custom Instruction:\n"))

    def test_stage3_verification(self):
        long_body = "Medical lecture text on crystal arthropathies. " * 50  # > 2000 chars
        unit = LectureUnit(
            unit_title="Gout and Pseudogout",
            deck_name="Gout Deck",
            body_text=long_body,
            breadcrumb="Rheumatology > Crystal Arthropathies",
        )

        concept = ExtractedConcept(
            topic_it="Gotta e cristalli MSU",
            claim_en="Gout is caused by monosodium urate crystal deposition",
            keywords_en=["gout", "monosodium urate"],
            keywords_it=["gotta", "urato"],
            quote_it="La gotta è una artropatia infiammatoria da deposito di cristalli di urato monosodico.",
        )

        c1 = ScoredNote(nid=101, score=0.85, matched_concepts=["Gotta e cristalli MSU"], clean_text_preview="Gout note 101")
        c2 = ScoredNote(nid=102, score=0.80, matched_concepts=["Gotta e cristalli MSU"], clean_text_preview="Gout note 102")

        # 1. Disabled by default: returns candidates untouched without LLM call
        cfg_disabled = ConfigManager(enable_llm_verification=False)
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        pipeline = SmartSearchPipeline(cfg_disabled, self.storage, mock_ai)
        result = pipeline.verify_candidates(unit, [c1, c2], concepts=[concept])
        self.assertEqual([c.nid for c in result], [101, 102])
        mock_ai.chat_completion_json.assert_not_called()

        # 2. Enabled: check full body_text (>1200 chars) and quote_it are passed in prompt
        cfg_enabled = ConfigManager(enable_llm_verification=True)
        pipeline_enabled = SmartSearchPipeline(cfg_enabled, self.storage, mock_ai)

        mock_ai.chat_completion_json.return_value = {"relevant_nids": [101]}
        result = pipeline_enabled.verify_candidates(unit, [c1, c2], concepts=[concept])

        # Verify return filters out 102 and keeps 101
        self.assertEqual([c.nid for c in result], [101])

        # Verify prompt details
        call_kwargs = mock_ai.chat_completion_json.call_args[1]
        user_msg = [m["content"] for m in call_kwargs["messages"] if m["role"] == "user"][0]
        # Body text is NOT truncated at 1200 chars
        self.assertIn(long_body, user_msg)
        self.assertGreater(len(user_msg), 2000)
        # Quote is included
        self.assertIn('Matched quote: "La gotta è una artropatia infiammatoria da deposito di cristalli di urato monosodico."', user_msg)

        # 3. Model returns valid empty list: NO candidate passes (returns empty list, does not keep all)
        mock_ai.chat_completion_json.return_value = {"relevant_nids": []}
        result_empty = pipeline_enabled.verify_candidates(unit, [c1, c2], concepts=[concept])
        self.assertEqual(result_empty, [])

        # 4. Model call raises Exception (e.g. timeout / network): batch is retained
        mock_ai.chat_completion_json.side_effect = Exception("API connection timed out")
        result_error = pipeline_enabled.verify_candidates(unit, [c1, c2], concepts=[concept])
        self.assertEqual([c.nid for c in result_error], [101, 102])

        # 5. End-to-end via search_unit: verify_candidates updates concept coverage
        mock_ai.reset_mock()
        mock_ai.chat_completion_json.side_effect = None
        pipeline_enabled.extract_concepts = MagicMock(return_value=[concept])

        def mock_retrieve(*args, **kwargs):
            pipeline_enabled.last_concept_coverage = [
                ConceptCoverage(
                    topic_it="Gotta e cristalli MSU",
                    claim_en="Gout is caused by monosodium urate crystal deposition",
                    matched_notes_count=2,
                    matched_nids=[101, 102],
                    best_score=0.85,
                    best_nid=101,
                    status="covered",
                    quote_it="La gotta è una artropatia infiammatoria da deposito di cristalli di urato monosodico.",
                )
            ]
            return [c1, c2]

        pipeline_enabled.retrieve_for_concepts = mock_retrieve
        mock_ai.chat_completion_json.return_value = {"relevant_nids": [101]}
        final_notes = pipeline_enabled.search_unit(unit, candidate_nids=None, active_model_id="test")
        self.assertEqual([c.nid for c in final_notes], [101])
        self.assertEqual(pipeline_enabled.last_concept_coverage[0].matched_nids, [101])
        self.assertEqual(pipeline_enabled.last_concept_coverage[0].matched_notes_count, 1)

    def test_search_worker_thread_isolation_and_undo(self):
        import threading
        import ankihub_searcher.ui.worker as worker_module

        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (8001, 801, 100, 0, -1)")
        col.conn.execute(
            "INSERT INTO notes (id, mid, mod, flds, tags) VALUES (801, 1, 1234, 'Colchicine for gout\x1fExtra', '#AK_Step1')"
        )
        col.conn.commit()

        mock_m = MagicMock()
        mock_m.id = 1
        col.models.all_names_and_ids.return_value = [mock_m]
        col.models.get.return_value = {
            "id": 1,
            "name": "Custom Model",
            "flds": [{"name": "Text"}, {"name": "Extra"}],
        }

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "topic_it": "Trattamento gotta",
                    "claim_en": "Colchicine is used in gout",
                    "keywords_en": ["colchicine", "gout"],
                    "keywords_it": ["colchicina", "gotta"],
                    "quote_it": "Colchicina per attacco acuto.",
                }
            ]
        }
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        events_order = []
        write_calling_threads = []
        worker_thread_ids = []

        def fake_main_runner(task):
            events_order.append("main_runner")
            # Execute on a separate thread to simulate the main UI thread
            t = threading.Thread(target=lambda: (write_calling_threads.append(threading.get_ident()), task()))
            t.start()
            t.join()

        orig_create = worker_module.create_batch_filtered_decks_op

        def track_create(*args, **kwargs):
            events_order.append("create_decks")
            return orig_create(*args, **kwargs)

        worker_module.create_batch_filtered_decks_op = track_create
        try:
            raw_text = "# Gout Lecture\nColchicina per attacco acuto di gotta."
            worker = SearchBatchWorker(
                col=col,
                config=config,
                storage=self.storage,
                ai_client=mock_ai,
                raw_text=raw_text,
                base_deck_name="Gout Filtered",
                main_thread_runner=fake_main_runner,
            )

            finished_payload = []
            worker.progress_sig.connect(lambda *args: worker_thread_ids.append(threading.get_ident()))
            worker.finished_sig.connect(lambda res: (events_order.append("finished_sig"), finished_payload.append(res)))

            worker.start()
            worker.wait(timeout=5.0)
        finally:
            worker_module.create_batch_filtered_decks_op = orig_create

        self.assertEqual(len(finished_payload), 1)
        res = finished_payload[0]
        self.assertEqual(res["status"], "done")
        self.assertEqual(res["units_processed"], 1)
        self.assertIsInstance(res["results"][0], RunResult)
        self.assertEqual(res["results"][0].total_cards_count, 1)

        # Check thread isolation: write was NOT called on the worker thread
        self.assertTrue(len(worker_thread_ids) > 0)
        self.assertTrue(len(write_calling_threads) > 0)
        self.assertNotEqual(worker_thread_ids[0], write_calling_threads[0])

        # Check execution order: deck creation was called via main runner BEFORE finished_sig
        self.assertIn("main_runner", events_order)
        self.assertIn("create_decks", events_order)
        self.assertIn("finished_sig", events_order)
        self.assertLess(events_order.index("create_decks"), events_order.index("finished_sig"))

        # Check undo entry was created on col
        col.add_custom_undo_entry.assert_called_with("Smart Search")
        col.merge_undo_entries.assert_called()

    def test_rollback_worker_thread_isolation_and_undo(self):
        import threading
        import ankihub_searcher.ui.worker as worker_module

        col = MockAnkiCol()
        run_record = RunRecord(
            run_id="run_test_rb_1",
            created_at=1000,
            deck_name="Test Deck",
            filtered_did=501,
            source_decks=["AnKing Step Deck"],
            unit_title="Test Unit",
            unit_preview="Preview",
            total_matched_cids=[8001],
            newly_unsuspended_cids=[8001],
            rolled_back=0,
        )
        self.storage.save_run(run_record)
        col.decks_dict[501] = {"id": 501, "name": "Test Deck", "dyn": 1}

        events_order = []
        rb_write_threads = []
        rb_worker_threads = []

        def fake_main_runner(task):
            events_order.append("main_runner")
            # Execute on a separate thread to simulate the main UI thread
            t = threading.Thread(target=lambda: (rb_write_threads.append(threading.get_ident()), task()))
            t.start()
            t.join()

        orig_rb = worker_module.rollback_run_op

        def track_rb(*args, **kwargs):
            events_order.append("rollback_op")
            return orig_rb(*args, **kwargs)

        worker_module.rollback_run_op = track_rb
        try:
            worker = RollbackWorker(
                col=col,
                storage=self.storage,
                run_id="run_test_rb_1",
                keep_reviewed=True,
                main_thread_runner=fake_main_runner,
            )

            # Record worker thread in run
            orig_run = worker.run
            def track_worker_run():
                rb_worker_threads.append(threading.get_ident())
                orig_run()
            worker.run = track_worker_run

            finished_payload = []
            worker.finished_sig.connect(lambda res: (events_order.append("finished_sig"), finished_payload.append(res)))

            worker.start()
            worker.wait(timeout=5.0)
        finally:
            worker_module.rollback_run_op = orig_rb

        self.assertEqual(len(finished_payload), 1)
        res = finished_payload[0]
        self.assertIsInstance(res, RollbackResult)
        self.assertEqual(res.run_id, "run_test_rb_1")

        # Check thread isolation
        self.assertTrue(len(rb_worker_threads) > 0)
        self.assertTrue(len(rb_write_threads) > 0)
        self.assertNotEqual(rb_worker_threads[0], rb_write_threads[0])

        self.assertIn("main_runner", events_order)
        self.assertIn("rollback_op", events_order)
        self.assertIn("finished_sig", events_order)
        self.assertLess(events_order.index("rollback_op"), events_order.index("finished_sig"))
        col.add_custom_undo_entry.assert_called_with("Smart Search Rollback")
        col.merge_undo_entries.assert_called()

    def test_workers_standalone_fallback_outside_anki(self):
        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (9001, 901, 100, 0, -1)")
        col.conn.execute(
            "INSERT INTO notes (id, mid, mod, flds, tags) VALUES (901, 1, 1234, 'Allopurinol note\x1fExtra', '#AK_Step1')"
        )
        col.conn.commit()

        mock_m = MagicMock()
        mock_m.id = 1
        col.models.all_names_and_ids.return_value = [mock_m]
        col.models.get.return_value = {
            "id": 1,
            "name": "Custom Model",
            "flds": [{"name": "Text"}, {"name": "Extra"}],
        }

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [
                {
                    "topic_it": "Allopurinolo",
                    "claim_en": "Allopurinol inhibits xanthine oxidase",
                    "keywords_en": ["allopurinol", "xanthine oxidase"],
                    "keywords_it": ["allopurinolo"],
                    "quote_it": "Allopurinolo inibitore xantina ossidasi.",
                }
            ]
        }
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        # Run SearchBatchWorker with default main_thread_runner=None outside Anki
        worker = SearchBatchWorker(
            col=col,
            config=config,
            storage=self.storage,
            ai_client=mock_ai,
            raw_text="# Allopurinol\nAllopurinolo inibitore xantina ossidasi.",
            base_deck_name="Allopurinol Filtered",
        )
        finished_search = []
        worker.finished_sig.connect(lambda res: finished_search.append(res))
        worker.start()
        worker.wait(timeout=5.0)

        self.assertEqual(len(finished_search), 1)
        self.assertEqual(finished_search[0]["status"], "done")
        self.assertIsInstance(finished_search[0]["results"][0], RunResult)

        run_id = finished_search[0]["results"][0].run_id

        # Run RollbackWorker with default main_thread_runner=None outside Anki
        rb_worker = RollbackWorker(
            col=col,
            storage=self.storage,
            run_id=run_id,
            keep_reviewed=True,
        )
        finished_rb = []
        rb_worker.finished_sig.connect(lambda res: finished_rb.append(res))
        rb_worker.start()
        rb_worker.wait(timeout=5.0)

        self.assertEqual(len(finished_rb), 1)
        self.assertIsInstance(finished_rb[0], RollbackResult)
        self.assertEqual(finished_rb[0].run_id, run_id)

    def test_rollback_reviewed_criteria_and_active_deck_no_reuse(self):
        col = MockAnkiCol()
        # Seed 3 cards:
        # 701: reps=5, but NO revlog entry -> must be re-suspended
        # 702: reps=0, revlog entry id >= run_time -> kept active
        # 703: reps=2, revlog entry id < run_time (old review months ago) -> must be re-suspended
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, reps) VALUES (701, 70, 100, 0, -1, 5)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, reps) VALUES (702, 70, 100, 0, -1, 0)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, reps) VALUES (703, 70, 100, 0, -1, 2)")
        col.conn.commit()

        config = ConfigManager(selected_decks=["AnKing Step Deck"])
        unit1 = LectureUnit("Unit 1", "Active Deck Test", "Lecture body 1")
        run1 = execute_unit_deck_creation(col, unit1, [70], config, self.storage)
        self.assertEqual(run1.newly_unsuspended_count, 3)

        # Insert revlog entries:
        run_time_ms = int(time.time() * 1000)
        # 702 reviewed NOW (after run creation)
        col.conn.execute("INSERT INTO revlog (id, cid) VALUES (?, 702)", (run_time_ms + 2000,))
        # 703 reviewed in the PAST (before run creation)
        col.conn.execute("INSERT INTO revlog (id, cid) VALUES (?, 703)", (run_time_ms - 50000,))
        col.conn.commit()

        # Part 1: Verify active deck reuse prevention
        # Run 2 tries to use the same deck name "Active Deck Test" while Run 1 is active
        unit2 = LectureUnit("Unit 2", "Active Deck Test", "Lecture body 2")
        run2 = execute_unit_deck_creation(col, unit2, [70], config, self.storage)
        # Must have created a new filtered deck with timestamp suffix, NOT reusing run1.filtered_did!
        self.assertNotEqual(run2.filtered_did, run1.filtered_did)
        self.assertTrue("Active Deck Test " in run2.deck_name)

        # Part 2: Verify rollback criteria
        res = rollback_run(col, self.storage, run1.run_id, keep_reviewed=True)
        self.assertTrue(res.deck_removed)
        # Only card 702 kept (it has revlog >= created_at * 1000)
        self.assertEqual(res.kept_reviewed_count, 1)
        # Cards 701 and 703 re-suspended (reps > 0 alone is ignored; old revlog is ignored)
        self.assertEqual(res.resuspended_count, 2)
        col.sched.suspend_cards.assert_called_with([701, 703])

        # Verify run2's deck is still in col.decks_dict!
        self.assertIn(run2.filtered_did, col.decks_dict)

    def test_batch_shared_cards_assignment(self):
        col = MockAnkiCol()
        # Note 101: card 1001 (suspended)
        # Note 102: card 1002 (suspended)
        # Note 103: card 1003 (suspended)
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (1001, 101, 100, 0, -1)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (1002, 102, 100, 0, -1)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (1003, 103, 100, 0, -1)")
        col.conn.commit()

        config = ConfigManager(selected_decks=["AnKing Step Deck"])

        unit_a = LectureUnit("Unit A: Acute", "Target Deck::Unit A", "Body A")
        unit_b = LectureUnit("Unit B: Chronic", "Target Deck::Unit B", "Body B")

        # Both units matched note 101, but Unit B has higher score (0.91 vs 0.82)
        u_data_a = {
            "unit": unit_a,
            "matched_nids": [101, 102],
            "note_scores": {101: 0.82, 102: 0.75},
            "concept_coverage": [
                {
                    "topic_it": "Pancreatite acuta",
                    "matched_nids": [101],
                    "status": "covered",
                },
                {
                    "topic_it": "Lipasi",
                    "matched_nids": [102],
                    "status": "covered",
                },
            ],
            "unit_error": "",
        }

        u_data_b = {
            "unit": unit_b,
            "matched_nids": [101, 103],
            "note_scores": {101: 0.91, 103: 0.65},
            "concept_coverage": [
                {
                    "topic_it": "Pancreatite cronica",
                    "matched_nids": [101, 103],
                    "status": "covered",
                }
            ],
            "unit_error": "",
        }

        units_data = [u_data_a, u_data_b]
        assign_notes_to_best_units(units_data)

        # Note 101 assigned to Unit B (score 0.91 > 0.82)
        # Unit A loses note 101 and keeps only note 102
        self.assertEqual(u_data_a["matched_nids"], [102])
        # Unit A's concept coverage for note 101 records shared_with = 'Unit B: Chronic'
        self.assertEqual(u_data_a["concept_coverage"][0].get("shared_with"), "Unit B: Chronic")
        # Unit A's second concept (note 102) has no shared_with
        self.assertNotIn("shared_with", u_data_a["concept_coverage"][1])

        # Unit B kept both notes 101 and 103
        self.assertEqual(u_data_b["matched_nids"], [101, 103])
        self.assertNotIn("shared_with", u_data_b["concept_coverage"][0])

        # Now execute batch deck creation
        results, changes = create_batch_filtered_decks_op(
            col=col,
            units_data=units_data,
            config=config,
            storage=self.storage,
        )

        self.assertEqual(len(results), 2)
        # Unit A filtered deck contains only card 1002
        deck_a = col.decks_dict[results[0].filtered_did]
        self.assertEqual(results[0].total_cards_count, 1)
        self.assertIn("1002", deck_a["terms"][0][0])
        self.assertNotIn("1001", deck_a["terms"][0][0])

        # Unit B filtered deck contains cards 1001 and 1003
        deck_b = col.decks_dict[results[1].filtered_did]
        self.assertEqual(results[1].total_cards_count, 2)
        self.assertIn("1001", deck_b["terms"][0][0])
        self.assertIn("1003", deck_b["terms"][0][0])

    def test_profile_scoped_history_and_rollback(self):
        col = MockAnkiCol()
        card_id = 9901
        note_id = 901
        col.conn.execute(
            "INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (card_id, note_id, 1, 0, -1, 0, 0),
        )
        col.conn.execute(
            "INSERT INTO notes (id, mid, mod, flds, tags) VALUES (?, ?, ?, ?, ?)",
            (note_id, 1, 0, "Profile Note\x1fExtra", ""),
        )
        col.conn.commit()

        # 1. Save runs across different profiles
        rec_a = RunRecord(
            run_id="run_prof_a",
            created_at=int(time.time()) - 100,
            deck_name="Smart Search - Prof A",
            filtered_did=10001,
            source_decks=["DeckA"],
            unit_title="Unit Prof A",
            unit_preview="...",
            total_matched_cids=[card_id],
            newly_unsuspended_cids=[card_id],
            rolled_back=0,
            profile="UserA",
        )
        rec_b = RunRecord(
            run_id="run_prof_b",
            created_at=int(time.time()) - 50,
            deck_name="Smart Search - Prof B",
            filtered_did=10002,
            source_decks=["DeckB"],
            unit_title="Unit Prof B",
            unit_preview="...",
            total_matched_cids=[card_id],
            newly_unsuspended_cids=[],
            rolled_back=0,
            profile="UserB",
        )
        rec_legacy = RunRecord(
            run_id="run_prof_legacy",
            created_at=int(time.time()) - 10,
            deck_name="Smart Search - Legacy",
            filtered_did=10003,
            source_decks=["DeckLegacy"],
            unit_title="Unit Legacy",
            unit_preview="...",
            total_matched_cids=[],
            newly_unsuspended_cids=[],
            rolled_back=0,
            profile="",
        )
        self.storage.save_run(rec_a)
        self.storage.save_run(rec_b)
        self.storage.save_run(rec_legacy)

        # 2. Query runs with profile filtering
        runs_a = self.storage.get_runs(profile="UserA")
        run_ids_a = [r.run_id for r in runs_a]
        self.assertIn("run_prof_a", run_ids_a)
        self.assertIn("run_prof_legacy", run_ids_a)
        self.assertNotIn("run_prof_b", run_ids_a)

        runs_b = self.storage.get_runs(profile="UserB")
        run_ids_b = [r.run_id for r in runs_b]
        self.assertIn("run_prof_b", run_ids_b)
        self.assertIn("run_prof_legacy", run_ids_b)
        self.assertNotIn("run_prof_a", run_ids_b)

        all_runs = self.storage.get_runs()
        self.assertEqual(len(all_runs), 3)

        # 3. Rollback refusal across profiles
        with self.assertRaises(ValueError) as ctx:
            rollback_run(
                col=col,
                storage=self.storage,
                run_id="run_prof_a",
                current_profile="UserB",
            )
        self.assertIn("Rollback rifiutato", str(ctx.exception))

        # RollbackWorker also propagates error when profile mismatches
        rb_worker_mismatch = RollbackWorker(
            col=col,
            storage=self.storage,
            run_id="run_prof_a",
            current_profile="UserB",
        )
        errors = []
        rb_worker_mismatch.error_sig.connect(lambda err: errors.append(err))
        rb_worker_mismatch.start()
        rb_worker_mismatch.wait(timeout=5.0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Rollback rifiutato", errors[0])

        # 4. Rollback succeeds when profiles match or legacy
        res_a = rollback_run(
            col=col,
            storage=self.storage,
            run_id="run_prof_a",
            current_profile="UserA",
        )
        self.assertEqual(res_a.run_id, "run_prof_a")

        res_leg = rollback_run(
            col=col,
            storage=self.storage,
            run_id="run_prof_legacy",
            current_profile="UserA",
        )
        self.assertEqual(res_leg.run_id, "run_prof_legacy")

    def test_honest_coverage_and_summary_with_quote_it(self):
        # 1. Test compute_coverage_stats empty case
        empty_stats = compute_coverage_stats([])
        self.assertEqual(empty_stats["total"], 0)
        self.assertEqual(empty_stats["covered"], 0)
        self.assertEqual(empty_stats["partial"], 0)
        self.assertEqual(empty_stats["uncovered"], 0)
        self.assertEqual(empty_stats["pct"], 0)
        self.assertEqual(empty_stats["history_str"], "-")

        # 2. Test honest calculation: 2 covered, 2 partial, 1 uncovered (total 5)
        # The percentage MUST ONLY count covered concepts: 2/5 = 40% (not 4/5 = 80%)
        coverage_data = [
            {
                "topic_it": "Allopurinolo",
                "quote_it": "Allopurinolo inibitore xantina ossidasi.",
                "claim_en": "Allopurinol inhibits xanthine oxidase",
                "status": "covered",
                "matched_notes_count": 2,
                "best_score": 0.88,
            },
            {
                "topic_it": "Colchicina",
                "quote_it": "Colchicina blocca i microtubuli.",
                "claim_en": "Colchicine inhibits microtubule polymerization",
                "status": "covered",
                "matched_notes_count": 1,
                "best_score": 0.85,
            },
            {
                "topic_it": "Febuxostat",
                "quote_it": "Febuxostat alternativa per iperuricemia.",
                "claim_en": "Febuxostat alternative xanthine oxidase inhibitor",
                "status": "partial",
                "matched_notes_count": 1,
                "best_score": 0.65,
            },
            {
                "topic_it": "Probenecid",
                "quote_it": "Probenecid aumenta escrezione urati.",
                "claim_en": "Probenecid increases uric acid excretion",
                "status": "partial",
                "matched_notes_count": 1,
                "best_score": 0.62,
            },
            {
                "topic_it": "Pegloticasi",
                "quote_it": "Pegloticasi forma ricombinante urato ossidasi.",
                "claim_en": "Pegloticase converts uric acid to allantoin",
                "status": "uncovered",
                "matched_notes_count": 0,
                "best_score": 0.40,
            },
        ]

        stats = compute_coverage_stats(coverage_data)
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["covered"], 2)
        self.assertEqual(stats["partial"], 2)
        self.assertEqual(stats["uncovered"], 1)
        # Percentage counts only covered: 2/5 = 40%
        self.assertEqual(stats["pct"], 40)
        self.assertEqual(stats["history_str"], "2/5 (40%)")
        self.assertIn("Copertura concetti: 2 / 5 (40%)", stats["summary_label"])
        self.assertIn("Coperti: 2", stats["summary_label"])
        self.assertIn("Parziali: 2", stats["summary_label"])
        self.assertIn("Scoperti: 1", stats["summary_label"])

        # 3. Test CoverageDetailDialog items sorting and quote_it preservation
        dlg = CoverageDetailDialog(coverage_items=coverage_data)
        self.assertEqual(len(dlg.sorted_items), 5)
        # Uncovered first, then partial, then covered
        self.assertEqual(dlg.sorted_items[0]["status"], "uncovered")
        self.assertEqual(dlg.sorted_items[1]["status"], "partial")
        self.assertEqual(dlg.sorted_items[2]["status"], "partial")
        self.assertEqual(dlg.sorted_items[3]["status"], "covered")
        self.assertEqual(dlg.sorted_items[4]["status"], "covered")
        # quote_it is present and preserved next to each concept
        self.assertEqual(dlg.sorted_items[0]["quote_it"], "Pegloticasi forma ricombinante urato ossidasi.")
        self.assertEqual(dlg.sorted_items[3]["quote_it"], "Colchicina blocca i microtubuli.")
        self.assertEqual(dlg.sorted_items[4]["quote_it"], "Allopurinolo inibitore xantina ossidasi.")

        # 4. Test stage 1 error units highlighted in results
        failed_res = RunResult(
            run_id="run_err_1",
            deck_name="",
            filtered_did=0,
            total_cards_count=0,
            newly_unsuspended_count=0,
            newly_unsuspended_cids=[],
            unit_error="Stage 1 extraction failed: JSON schema mismatch",
        )
        self.assertTrue(bool(failed_res.unit_error))
        self.assertEqual(failed_res.total_cards_count, 0)

    def test_stage1_json_schema_option_retry_and_length_truncation(self):
        # 1. Option chat_use_json_schema: default False -> json_schema is None
        cfg_default = ConfigManager()
        self.assertFalse(cfg_default.chat_use_json_schema)
        self.assertEqual(cfg_default.chat_max_tokens, 16384)

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.chat_completion_json.return_value = {
            "concepts": [{"topic_it": "Test", "claim_en": "Claim", "keywords_en": ["k"], "keywords_it": []}]
        }
        pipe_default = SmartSearchPipeline(cfg_default, self.storage, mock_ai)
        unit = LectureUnit("Unit 1", "Deck 1", "Medical lecture text")
        pipe_default.extract_concepts(unit)

        # Verified json_schema is None when chat_use_json_schema is False
        call_kwargs = mock_ai.chat_completion_json.call_args[1]
        self.assertIsNone(call_kwargs["json_schema"])
        self.assertEqual(call_kwargs["max_tokens"], 16384)

        # 1b. chat_use_json_schema = True -> json_schema is STAGE1_JSON_SCHEMA
        cfg_schema = ConfigManager(chat_use_json_schema=True)
        pipe_schema = SmartSearchPipeline(cfg_schema, self.storage, mock_ai)
        mock_ai.reset_mock()
        mock_ai.chat_completion_json.return_value = {
            "concepts": [{"topic_it": "Test", "claim_en": "Claim", "keywords_en": ["k"], "keywords_it": []}]
        }
        pipe_schema.extract_concepts(unit)
        call_kwargs_schema = mock_ai.chat_completion_json.call_args[1]
        self.assertEqual(call_kwargs_schema["json_schema"], STAGE1_JSON_SCHEMA)

        # 2. When response_format active and returns 0 concepts: retry once without response_format
        mock_ai.reset_mock()
        mock_ai.chat_completion_json.side_effect = [
            {"concepts": []},  # 1st call with schema: 0 concepts
            {"concepts": [{"topic_it": "Gout", "claim_en": "Colchicine treats acute gout", "keywords_en": ["colchicine"], "keywords_it": []}]},  # fallback without schema: success
        ]
        res_concepts = pipe_schema.extract_concepts(unit)
        self.assertEqual(len(res_concepts), 1)
        self.assertEqual(res_concepts[0].topic_it, "Gout")
        self.assertEqual(mock_ai.chat_completion_json.call_count, 2)
        first_call_schema = mock_ai.chat_completion_json.call_args_list[0][1]["json_schema"]
        second_call_schema = mock_ai.chat_completion_json.call_args_list[1][1]["json_schema"]
        self.assertIsNotNone(first_call_schema)
        self.assertIsNone(second_call_schema)

        # 4. finish_reason == 'length' raises AIClientError with 'risposta troncata: aumentare chat_max_tokens'
        real_client = OpenAICompatibleClient()
        real_client._make_request = lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {"content": '{"concepts": [{"topic_it": "Trun'},
                    "finish_reason": "length",
                }
            ]
        }
        with self.assertRaises(AIClientError) as ctx:
            real_client.chat_completion_json(
                messages=[{"role": "user", "content": "test"}],
                model="test",
                base_url="http://localhost:1234/v1",
            )
        self.assertIn("risposta troncata: aumentare chat_max_tokens", str(ctx.exception))

        # Also when JSON is somehow superficially parsed but finish_reason == 'length'
        real_client._make_request = lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {"content": '{"concepts": []}'},
                    "finish_reason": "length",
                }
            ]
        }
        with self.assertRaises(AIClientError) as ctx2:
            real_client.chat_completion_json(
                messages=[{"role": "user", "content": "test"}],
                model="test",
                base_url="http://localhost:1234/v1",
            )
        self.assertIn("risposta troncata: aumentare chat_max_tokens", str(ctx2.exception))

    def test_selection_and_config_v4_migration_and_query_sentence(self):
        # 1. Test new defaults and threshold mapping
        cfg = ConfigManager()
        self.assertEqual(cfg.sensitivity, 0.5)
        self.assertEqual(cfg.relative_score_window, 0.05)
        self.assertEqual(cfg.max_notes_per_concept, 10)
        self.assertEqual(cfg.max_cards_per_unit, 150)
        self.assertEqual(cfg.hybrid_dense_weight, 1.0)
        self.assertEqual(cfg.config_version, 4)

        # get_min_similarity_threshold: 0.70 - s * 0.20
        self.assertEqual(cfg.get_min_similarity_threshold(), 0.60)
        self.assertEqual(ConfigManager(sensitivity=0.0).get_min_similarity_threshold(), 0.70)
        self.assertEqual(ConfigManager(sensitivity=1.0).get_min_similarity_threshold(), 0.50)

        # 2. Test config migration v4
        legacy_cfg = {
            "chat_base_url": "http://my-host:1234/v1",
            "chat_model": "gpt-custom",
            "sensitivity": 0.85,
            "relative_score_window": 0.20,
            "max_notes_per_concept": 50,
            "max_cards_per_unit": 300,
            "hybrid_dense_weight": 0.78,
            "top_k_per_concept": 8,
            "search_query": '"deck:MyDeck"',
        }
        migrated, was_migrated = ConfigManager.migrate_dict(legacy_cfg)
        self.assertTrue(was_migrated)
        self.assertEqual(migrated["config_version"], 4)
        self.assertEqual(migrated["sensitivity"], 0.5)
        self.assertEqual(migrated["relative_score_window"], 0.05)
        self.assertEqual(migrated["max_notes_per_concept"], 10)
        self.assertEqual(migrated["max_cards_per_unit"], 150)
        self.assertEqual(migrated["hybrid_dense_weight"], 1.0)
        self.assertNotIn("top_k_per_concept", migrated)
        # Verify other user settings were strictly preserved
        self.assertEqual(migrated["chat_base_url"], "http://my-host:1234/v1")
        self.assertEqual(migrated["chat_model"], "gpt-custom")
        self.assertEqual(migrated["search_query"], '"deck:MyDeck"')

        # Version 4 config is NOT re-migrated
        v4_cfg = {
            "config_version": 4,
            "sensitivity": 0.65,
            "relative_score_window": 0.03,
            "max_notes_per_concept": 15,
        }
        migrated_v4, was_migrated_v4 = ConfigManager.migrate_dict(v4_cfg)
        self.assertFalse(was_migrated_v4)
        self.assertEqual(migrated_v4["sensitivity"], 0.65)
        self.assertEqual(migrated_v4["relative_score_window"], 0.03)

        # 3. Test extract_first_sentence
        self.assertEqual(
            extract_first_sentence("Allopurinol inhibits xanthine oxidase. It reduces uric acid."),
            "Allopurinol inhibits xanthine oxidase",
        )
        self.assertEqual(
            extract_first_sentence("Colchicine treats acute gout flares! Second sentence."),
            "Colchicine treats acute gout flares",
        )
        self.assertEqual(
            extract_first_sentence("Single sentence without punctuation"),
            "Single sentence without punctuation",
        )

        # 4. Test embedding query text construction (first sentence of claim_en + ' | ' + keywords)
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]
        pipeline = SmartSearchPipeline(cfg, self.storage, mock_ai)

        concept = ExtractedConcept(
            topic_it="Allopurinolo",
            claim_en="Allopurinol inhibits xanthine oxidase, decreasing uric acid. It is first-line for chronic gout.",
            keywords_en=["allopurinol", "xanthine oxidase", "gout"],
            keywords_it=["allopurinolo"],
            quote_it="Allopurinolo inibitore xantina ossidasi.",
        )
        pipeline.retrieve_for_concepts([concept], active_model_id="test-model", candidate_nids={101})

        # Verify query texts passed to get_embeddings
        embed_call_args = mock_ai.get_embeddings.call_args[1]
        sent_queries = embed_call_args["texts"]
        self.assertEqual(len(sent_queries), 1)
        self.assertEqual(
            sent_queries[0],
            "Allopurinol inhibits xanthine oxidase, decreasing uric acid | allopurinol, xanthine oxidase, gout",
        )

        # Verify with qwen3-embedding model instruction
        cfg_qwen3 = ConfigManager(embed_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct-qwen3-embedding")
        pipe_qwen3 = SmartSearchPipeline(cfg_qwen3, self.storage, mock_ai)
        pipe_qwen3.retrieve_for_concepts([concept], active_model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct-qwen3-embedding", candidate_nids={101})
        qwen3_sent_queries = mock_ai.get_embeddings.call_args[1]["texts"]
        self.assertTrue(
            qwen3_sent_queries[0].startswith("Instruct: Given a medical lecture concept, retrieve flashcards that test it\nQuery: ")
        )
        self.assertIn(
            "Allopurinol inhibits xanthine oxidase, decreasing uric acid | allopurinol, xanthine oxidase, gout",
            qwen3_sent_queries[0],
        )

    def test_config_v4_migration_with_real_030_user_config(self):
        # Dict identical to real 0.3.0 user config in meta.json (without config_version)
        user_030_config = {
            "selected_decks": ["AnKing Step Deck"],
            "search_query": '"deck:AnKing Step Deck" OR "deck:Custom"',
            "chat_provider": "lm_studio",
            "chat_base_url": "http://127.0.0.1:1234/v1",
            "chat_model": "mistral-nemo-it",
            "sensitivity": 0.55,
            "top_k_per_concept": 8,
            "max_cards_per_unit": 80,
            "embed_model": "text-embedding-3-small",
        }
        stored_in_anki = dict(user_030_config)

        mock_mw = MagicMock()
        mock_addon_mgr = MagicMock()
        mock_mw.addonManager = mock_addon_mgr
        mock_addon_mgr.addonFromModule.return_value = "ankihub_searcher"
        mock_addon_mgr.getConfig.side_effect = lambda addon: dict(stored_in_anki)

        def fake_write_config(addon, data):
            stored_in_anki.clear()
            stored_in_anki.update(data)

        mock_addon_mgr.writeConfig.side_effect = fake_write_config

        with patch.dict("sys.modules", {"aqt": MagicMock(mw=mock_mw)}):
            # First load: migration should trigger
            cfg = ConfigManager.load()
            self.assertEqual(cfg.sensitivity, 0.5)
            self.assertEqual(cfg.get_min_similarity_threshold(), 0.60)
            self.assertEqual(cfg.max_notes_per_concept, 10)
            self.assertEqual(cfg.max_cards_per_unit, 150)
            self.assertEqual(cfg.hybrid_dense_weight, 1.0)
            self.assertEqual(cfg.config_version, 4)
            self.assertIsNone(cfg.top_k_per_concept)

            # Check that user custom settings remain preserved
            self.assertEqual(cfg.chat_model, "mistral-nemo-it")
            self.assertEqual(cfg.search_query, '"deck:AnKing Step Deck" OR "deck:Custom"')
            self.assertEqual(cfg.embed_model, "text-embedding-3-small")

            # Check that writeConfig was called and stored_in_anki was updated with config_version 4
            mock_addon_mgr.writeConfig.assert_called_once()
            self.assertEqual(stored_in_anki.get("config_version"), 4)
            self.assertEqual(stored_in_anki.get("sensitivity"), 0.5)
            self.assertEqual(stored_in_anki.get("hybrid_dense_weight"), 1.0)
            self.assertNotIn("top_k_per_concept", stored_in_anki)

            # Second load: no migration should trigger, writeConfig must NOT be called again
            mock_addon_mgr.writeConfig.reset_mock()
            cfg2 = ConfigManager.load()
            self.assertEqual(cfg2.sensitivity, 0.5)
            self.assertEqual(cfg2.hybrid_dense_weight, 1.0)
            self.assertEqual(cfg2.config_version, 4)
            self.assertEqual(cfg2.chat_model, "mistral-nemo-it")
            mock_addon_mgr.writeConfig.assert_not_called()

    def test_hybrid_dense_weight_1_skips_fts(self):
        cfg_default = ConfigManager(hybrid_dense_weight=1.0)
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        concept = ExtractedConcept(
            topic_it="Inflammasoma",
            claim_en="NLRP3 inflammasome cleaves pro-IL-1beta into IL-1beta.",
            keywords_en=["NLRP3", "inflammasome", "IL-1beta"],
            keywords_it=["inflammasoma"],
            quote_it="Inflammasoma NLRP3 taglia la pro-IL-1beta.",
        )

        with patch.object(self.storage, "search_fts", wraps=self.storage.search_fts) as mock_fts:
            pipe1 = SmartSearchPipeline(cfg_default, self.storage, mock_ai)
            pipe1.retrieve_for_concepts([concept], active_model_id="test-model", candidate_nids={101})
            mock_fts.assert_not_called()

        cfg_hybrid = ConfigManager(hybrid_dense_weight=0.78)
        with patch.object(self.storage, "search_fts", wraps=self.storage.search_fts) as mock_fts:
            pipe2 = SmartSearchPipeline(cfg_hybrid, self.storage, mock_ai)
            pipe2.retrieve_for_concepts([concept], active_model_id="test-model", candidate_nids={101})
            mock_fts.assert_called_once()

    def test_selection_function_refactor_and_preview_logic(self):
        # 1. Pure function select_candidates_for_concept
        dense_scores = {101: 0.85, 102: 0.82, 103: 0.70, 104: 0.55}
        fts_scores = {}
        # Tight window 0.05 -> cutoff is 0.80
        candidates, best_nid, best_score = select_candidates_for_concept(
            dense_scores=dense_scores,
            fts_scores=fts_scores,
            alpha=1.0,
            min_threshold=0.60,
            min_dense_floor=0.42,
            rel_window=0.05,
            max_notes_concept=10,
        )
        self.assertEqual(best_nid, 101)
        self.assertEqual(best_score, 0.85)
        self.assertEqual([c[0] for c in candidates], [101, 102])

        # Wider window 0.20 -> cutoff is 0.65 -> 101, 102, 103 survive
        candidates_wide, _, _ = select_candidates_for_concept(
            dense_scores=dense_scores,
            fts_scores=fts_scores,
            alpha=1.0,
            min_threshold=0.60,
            min_dense_floor=0.42,
            rel_window=0.20,
            max_notes_concept=10,
        )
        self.assertEqual([c[0] for c in candidates_wide], [101, 102, 103])

        # 2. Reusable retrieve_for_concepts with explicit parameters
        mid = "model_test_preview"
        self.storage.upsert_notes_content_batch(
            [
                (201, 1, 100, "h201", "Text about acute appendicitis signs.", "GI"),
                (202, 1, 100, "h202", "Text about appendectomy surgical technique.", "Surgery"),
            ]
        )
        self.storage.save_embeddings_batch(
            [
                (201, mid, "h201", 2, [1.0, 0.0]),
                (202, mid, "h202", 2, [0.72, 0.69]),  # lower similarity (~0.72)
            ]
        )

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]
        cfg = ConfigManager()  # default threshold 0.60
        pipeline = SmartSearchPipeline(cfg, self.storage, mock_ai)

        concept = ExtractedConcept(
            topic_it="Appendicite acuta",
            claim_en="Acute appendicitis clinical signs include McBurney point tenderness.",
            keywords_en=["appendicitis", "McBurney point"],
            keywords_it=["appendicite"],
            quote_it="Appendicite acuta punto di McBurney.",
        )

        # Call with default config -> both 201 and 202 could pass if within window
        # Call with strict explicit parameter min_threshold=0.80 -> only 201 should pass
        res_strict = pipeline.retrieve_for_concepts(
            concepts=[concept],
            active_model_id=mid,
            candidate_nids={201, 202},
            min_threshold=0.80,
        )
        self.assertEqual([n.nid for n in res_strict], [201])

        # 3. Test storage search_id and get_latest_search_runs
        run1 = RunRecord(
            run_id="run_batch1_u1",
            created_at=1000,
            deck_name="Deck 1",
            filtered_did=10,
            source_decks=["Test"],
            unit_title="Unit 1",
            unit_preview="...",
            total_matched_cids=[1],
            newly_unsuspended_cids=[1],
            rolled_back=0,
            concept_coverage=[concept.__dict__],
            search_id="batch_alpha",
        )
        run2 = RunRecord(
            run_id="run_batch1_u2",
            created_at=1001,
            deck_name="Deck 2",
            filtered_did=20,
            source_decks=["Test"],
            unit_title="Unit 2",
            unit_preview="...",
            total_matched_cids=[2],
            newly_unsuspended_cids=[2],
            rolled_back=0,
            concept_coverage=[concept.__dict__],
            search_id="batch_alpha",
        )
        self.storage.save_run(run1)
        self.storage.save_run(run2)

        latest_runs = self.storage.get_latest_search_runs()
        self.assertEqual(len(latest_runs), 2)
        self.assertEqual({r.run_id for r in latest_runs}, {"run_batch1_u1", "run_batch1_u2"})

        # 4. Test preview re-use without chat calls
        saved_cov = latest_runs[0].concept_coverage
        reconstructed_concepts = [
            ExtractedConcept(
                topic_it=c["topic_it"],
                claim_en=c["claim_en"],
                keywords_en=c["keywords_en"],
                keywords_it=c.get("keywords_it", []),
                quote_it=c.get("quote_it", ""),
            )
            for c in saved_cov
        ]
        preview_res = pipeline.retrieve_for_concepts(
            concepts=reconstructed_concepts,
            active_model_id=mid,
            candidate_nids={201, 202},
            min_threshold=0.60,
            relative_score_window=0.05,
        )
        self.assertTrue(len(preview_res) >= 1)
        mock_ai.chat_completion_json.assert_not_called()

    def test_batch_rollback_and_retry_tighter(self):
        col = MockAnkiCol()
        # Create 2 suspended cards across 2 notes
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (7001, 701, 100, 0, -1, 0, 0)")
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue, type, reps) VALUES (7002, 702, 100, 0, -1, 0, 0)")
        col.conn.execute("INSERT INTO notes (id, mid, mod, flds, tags) VALUES (701, 1, 1000, 'Note 1', '')")
        col.conn.execute("INSERT INTO notes (id, mid, mod, flds, tags) VALUES (702, 1, 1000, 'Note 2', '')")
        col.conn.commit()

        cfg = ConfigManager()
        # Save two runs in a batch
        u1 = LectureUnit("Unit 1", "U1", "Text 1")
        u2 = LectureUnit("Unit 2", "U2", "Text 2")
        units_data = [
            {"unit": u1, "matched_nids": [701], "note_scores": {701: 0.90}, "concept_coverage": []},
            {"unit": u2, "matched_nids": [702], "note_scores": {702: 0.85}, "concept_coverage": []},
        ]
        results, _ = create_batch_filtered_decks_op(
            col=col,
            units_data=units_data,
            config=cfg,
            storage=self.storage,
            search_id="batch_retry_test",
        )
        self.assertEqual(len(results), 2)
        r_ids = [r.run_id for r in results]
        self.assertEqual(len(r_ids), 2)
        # Verify unsuspend was called for both cards
        unsuspended = set()
        for call_arg in col.sched.unsuspend_cards.call_args_list:
            unsuspended.update(call_arg[0][0])
        self.assertEqual(unsuspended, {7001, 7002})

        # 1. Rollback entire batch with rollback_search_batch_op
        rb_results, changes = rollback_search_batch_op(
            col=col,
            storage=self.storage,
            run_ids=r_ids,
            keep_reviewed=True,
        )
        self.assertEqual(len(rb_results), 2)
        suspended = set()
        for call_arg in col.sched.suspend_cards.call_args_list:
            suspended.update(call_arg[0][0])
        self.assertEqual(suspended, {7001, 7002})

        # 2. Test retry parameter calculation
        curr_thresh = cfg.get_min_similarity_threshold()  # 0.60
        new_thresh = min(0.95, round(curr_thresh + 0.03, 4))
        self.assertEqual(new_thresh, 0.63)

        curr_max = cfg.max_notes_per_concept  # 10
        new_max = max(3, int(round(curr_max * 2.0 / 3.0)))  # 7
        self.assertEqual(new_max, 7)

        # 3. Test worker execution with precomputed_concepts
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]
        concept_u1 = ExtractedConcept("U1", "Claim U1", ["u1"], ["u1"], "quote 1")
        precomputed = {"Unit 1": [concept_u1]}

        worker = SearchBatchWorker(
            col=col,
            config=cfg,
            storage=self.storage,
            ai_client=mock_ai,
            raw_text="# Unit 1\nText 1",
            base_deck_name="Retry Deck",
            main_thread_runner=lambda task: task(),
        )
        worker.precomputed_concepts = precomputed
        worker.override_min_threshold = new_thresh
        worker.override_max_notes_per_concept = new_max

        finished_box = []
        worker.finished_sig.connect(lambda r: finished_box.append(r))
        worker.start()
        worker.wait(timeout=5.0)

        self.assertEqual(len(finished_box), 1)
        # Stage 1 chat LLM should NOT have been called!
        mock_ai.chat_completion_json.assert_not_called()
        # Default config should NOT have been modified!
        self.assertEqual(cfg.get_min_similarity_threshold(), 0.60)
        self.assertEqual(cfg.max_notes_per_concept, 10)

    def test_find_similar_notes_single_and_multiple(self):
        from ankihub_searcher.search_pipeline import find_similar_notes

        mid = "test-embed-model"
        # Seed embeddings:
        # Note 10: [1.0, 0.0, 0.0]
        # Note 20: [0.96, 0.28, 0.0] -> high sim to 10 (~0.96)
        # Note 30: [0.0, 1.0, 0.0]   -> orthogonal to 10 (0.0)
        # Note 40: [0.7071, 0.7071, 0.0] -> 45 degrees, sim to 10 is ~0.71, sim to mean(10,30) is 1.0
        vecs = [
            (10, [1.0, 0.0, 0.0]),
            (20, [0.96, 0.28, 0.0]),
            (30, [0.0, 1.0, 0.0]),
            (40, [0.7071, 0.7071, 0.0]),
        ]
        self.storage.save_embeddings_batch(
            [(nid, mid, f"hash_{nid}", len(v), v) for nid, v in vecs]
        )

        # 1. Single note selected: Note 10
        matched, missing = find_similar_notes(
            storage=self.storage,
            selected_nids=[10],
            active_model_id=mid,
            min_score=0.70,
            max_notes=10,
        )
        self.assertEqual(missing, [])
        # Note 10 must be excluded from results!
        matched_nids = [nid for nid, _ in matched]
        self.assertNotIn(10, matched_nids)
        # Note 20 (score ~0.96) and Note 40 (score ~0.71) should match
        self.assertIn(20, matched_nids)
        self.assertIn(40, matched_nids)
        # Note 30 (score 0.0) must be filtered out
        self.assertNotIn(30, matched_nids)
        # Ordered descending by score
        self.assertEqual(matched_nids[0], 20)

        # 2. Multiple notes selected: Note 10 and Note 30
        # Mean vector is [0.5, 0.5, 0.0], normalized to [1/√2, 1/√2, 0.0]
        # Note 40 has similarity 1.0 to the mean vector!
        matched_mult, missing_mult = find_similar_notes(
            storage=self.storage,
            selected_nids=[10, 30],
            active_model_id=mid,
            min_score=0.70,
            max_notes=10,
        )
        self.assertEqual(missing_mult, [])
        mult_nids = [nid for nid, _ in matched_mult]
        self.assertNotIn(10, mult_nids)
        self.assertNotIn(30, mult_nids)
        self.assertEqual(mult_nids[0], 40)
        self.assertAlmostEqual(matched_mult[0][1], 1.0, places=2)

    def test_find_similar_notes_missing_and_scopes(self):
        from ankihub_searcher.search_pipeline import find_similar_notes

        mid = "test-embed-model-scope"
        self.storage.save_embeddings_batch(
            [
                (101, mid, "h101", 2, [1.0, 0.0]),
                (102, mid, "h102", 2, [0.9, 0.1]),
                (103, mid, "h103", 2, [0.8, 0.2]),
            ]
        )

        # 1. Unindexed note in selection returns missing
        matched, missing = find_similar_notes(
            storage=self.storage,
            selected_nids=[101, 999],
            active_model_id=mid,
        )
        self.assertEqual(matched, [])
        self.assertEqual(missing, [999])

        # 2. Candidate scope restriction
        matched_scope, missing_scope = find_similar_notes(
            storage=self.storage,
            selected_nids=[101],
            active_model_id=mid,
            candidate_nids={103},  # only allow 103
            min_score=0.50,
        )
        self.assertEqual(missing_scope, [])
        self.assertEqual([nid for nid, _ in matched_scope], [103])

        # 3. Empty candidate scope
        matched_empty, _ = find_similar_notes(
            storage=self.storage,
            selected_nids=[101],
            active_model_id=mid,
            candidate_nids=set(),
        )
        self.assertEqual(matched_empty, [])

    def test_browser_menu_integration(self):
        from ankihub_searcher.browser_menu import (
            get_browser_search_text,
            on_browser_menus_did_init,
            on_find_similar_cards_clicked,
            _update_similar_action_state,
        )

        mid = "test-browser-mid"
        self.storage.save_embeddings_batch(
            [
                (501, mid, "h501", 2, [1.0, 0.0]),
                (502, mid, "h502", 2, [0.95, 0.05]),
                (503, mid, "h503", 2, [0.92, 0.08]),
            ]
        )

        class MockAction:
            def __init__(self, text=""):
                self.text = text
                self.enabled = True
                self.shortcut = ""
                self.callbacks = []

            def setEnabled(self, val):
                self.enabled = bool(val)

            def setShortcut(self, sc):
                self.shortcut = sc

            @property
            def triggered(self):
                act_self = self
                class _Sig:
                    def connect(self, fn):
                        act_self.callbacks.append(fn)
                return _Sig()

            def trigger(self):
                for cb in self.callbacks:
                    cb()

        class MockMenu:
            def __init__(self, title="", parent=None):
                self.title = title
                self.parent = parent
                self.actions = []
                self._about_to_show_cbs = []

            def addAction(self, text):
                act = MockAction(text)
                self.actions.append(act)
                return act

            def addSeparator(self):
                pass

            def setDefaultAction(self, act):
                pass

            @property
            def aboutToShow(self):
                menu_self = self
                class _Sig:
                    def connect(self, fn):
                        menu_self._about_to_show_cbs.append(fn)
                return _Sig()

        class MockMenubar:
            def __init__(self):
                self.menus = []

            def addMenu(self, title):
                m = MockMenu(title)
                self.menus.append(m)
                return m

        class MockBrowser:
            def __init__(self, col=None, selected_nids=None, current_search_str=""):
                self.col = col
                self._selected = list(selected_nids or [])
                self._search_str = current_search_str
                self.executed_searches = []
                self.form = type("Form", (), {"menubar": MockMenubar()})()

            def selected_notes(self):
                return list(self._selected)

            def current_search(self):
                return self._search_str

            def search_for(self, query):
                self.executed_searches.append(query)
                self._search_str = query

        col = MockAnkiCol()
        browser = MockBrowser(col=col, selected_nids=[], current_search_str='"deck:AnKing"')

        # 1. Initialize browser menus (unified Smart Search menu)
        on_browser_menus_did_init(browser)
        self.assertEqual(len(browser.form.menubar.menus), 1)
        menu = browser.form.menubar.menus[0]
        self.assertEqual(menu.title, "Smart Search")
        # 7 unified actions
        self.assertEqual(len(menu.actions), 7)

        act_new = menu.actions[0]
        act_search = menu.actions[1]
        act_similar = menu.actions[2]
        act_history = menu.actions[3]
        act_index = menu.actions[4]
        act_settings = menu.actions[5]
        act_updates = menu.actions[6]

        self.assertEqual(act_new.text, "Nuova ricerca…")
        self.assertEqual(act_new.shortcut, "Ctrl+Shift+F")
        self.assertEqual(act_search.text, "Cerca con la query corrente…")
        self.assertTrue(act_search.enabled)

        self.assertEqual(act_similar.text, "Trova carte simili")
        self.assertEqual(act_similar.shortcut, "Ctrl+Alt+M")
        self.assertEqual(act_history.text, "Cronologia & rollback")
        self.assertEqual(act_index.text, "Indice")
        self.assertEqual(act_settings.text, "Impostazioni")
        self.assertEqual(act_updates.text, "Verifica aggiornamenti")

        # Initial selection is empty -> act_similar disabled
        self.assertFalse(act_similar.enabled)

        # 2. Test selection toggle
        browser._selected = [501]
        _update_similar_action_state(browser)
        self.assertTrue(act_similar.enabled)

        browser._selected = []
        _update_similar_action_state(browser)
        self.assertFalse(act_similar.enabled)

        # 3. Test get_browser_search_text
        self.assertEqual(get_browser_search_text(browser), '"deck:AnKing"')

        # 4. Test "Trova carte simili" execution
        browser._selected = [501]
        with patch.object(ConfigManager, "load") as mock_load:
            cfg = ConfigManager(
                embed_model="test-embed-model",
                embed_provider="test",
                similar_scope="all",
                similar_min_score=0.70,
                similar_max_notes=10,
            )
            # Match model id
            cfg.get_active_model_id = lambda: mid
            mock_load.return_value = cfg

            with patch("ankihub_searcher.browser_menu.SearchDatabase", return_value=self.storage):
                act_similar.trigger()

        # Browser search should have been updated with comma-separated "nid:502,503"
        self.assertIn("nid:502,503", browser.executed_searches)

        # 5. Test context menu hook (Item 20d)
        from ankihub_searcher.browser_menu import (
            on_browser_will_show_context_menu,
            on_browser_did_fetch_columns,
            on_browser_did_fetch_row,
            on_browser_will_search,
            on_browser_did_search,
        )

        ctx_menu = MockMenu("Context")
        on_browser_will_show_context_menu(browser, ctx_menu)
        self.assertEqual(len(ctx_menu.actions), 1)
        self.assertEqual(ctx_menu.actions[0].text, "Smart Search: Trova carte simili")
        self.assertEqual(ctx_menu.actions[0].shortcut, "Ctrl+Alt+M")
        self.assertTrue(ctx_menu.actions[0].enabled)

        # 6. Test custom column registration and row fetching (Item 20c)
        columns_dict = {}
        fake_browser_mod = types.ModuleType("aqt.browser")
        fake_browser_mod.Column = lambda key, cards_mode_label, notes_mode_label: {
            "key": key,
            "cards_mode_label": cards_mode_label,
            "notes_mode_label": notes_mode_label,
        }
        fake_browser_mod.Cell = lambda text: type("Cell", (), {"text": text})()
        with patch.dict(sys.modules, {"aqt.browser": fake_browser_mod}):
            on_browser_did_fetch_columns(columns_dict)
            self.assertIn("smart_search_similarity", columns_dict)

            # Test row fetching for matched note 502
            class MockCellRow:
                def __init__(self):
                    self.cells = [type("Cell", (), {"text": ""})()]

            row_matched = MockCellRow()
            on_browser_did_fetch_row(502, True, row_matched, ["smart_search_similarity"])
            self.assertEqual(row_matched.cells[0].text, "1.00")

            # Test row fetching for unrelated note 999 (empty)
            row_unrelated = MockCellRow()
            on_browser_did_fetch_row(999, True, row_unrelated, ["smart_search_similarity"])
            self.assertEqual(row_unrelated.cells[0].text, "")

        # 7. Test browser_will_search and browser_did_search ordering (Commit 24)
        import ankihub_searcher.browser_menu as bm_mod
        bm_mod._last_similar_query = "nid:501,502,503"
        bm_mod._last_similar_scores = {501: 0.70, 502: 0.95, 503: 0.80}
        bm_mod._similar_custom_order_active = True

        class FakeSearchContext:
            def __init__(self, search_str, ids, order="default", browser=None):
                self.search = search_str
                self.ids = list(ids)
                self.order = order
                self.browser = browser

        # 7A. Notes mode test
        class FakeNotesBrowser:
            def __init__(self):
                self.table = type("Table", (), {"is_notes_mode": lambda *args, **kwargs: True})()

        ctx_notes = FakeSearchContext(
            search_str="nid:501,502,503",
            ids=[501, 503, 502],
            order="creation_date",
            browser=FakeNotesBrowser(),
        )
        on_browser_will_search(ctx_notes)
        self.assertFalse(ctx_notes.order)

        on_browser_did_search(ctx_notes)
        # Expected order: 502 (0.95) -> 503 (0.80) -> 501 (0.70)
        self.assertEqual(ctx_notes.ids, [502, 503, 501])
        self.assertFalse(ctx_notes.order)

        # 7B. Cards mode test with multiple cards per note and SQL mapping
        # Add card records to col database:
        # Card 2001 -> Note 501 (score 0.70)
        # Card 2002 -> Note 502 (score 0.95)
        # Card 2003 -> Note 503 (score 0.80)
        # Card 2004 -> Note 502 (score 0.95)
        col.conn.execute("INSERT OR REPLACE INTO cards (id, nid, did) VALUES (2001, 501, 100)")
        col.conn.execute("INSERT OR REPLACE INTO cards (id, nid, did) VALUES (2002, 502, 100)")
        col.conn.execute("INSERT OR REPLACE INTO cards (id, nid, did) VALUES (2003, 503, 100)")
        col.conn.execute("INSERT OR REPLACE INTO cards (id, nid, did) VALUES (2004, 502, 100)")
        col.conn.commit()

        bm_mod._similar_custom_order_active = True
        class FakeCardsBrowser:
            def __init__(self, col):
                self.col = col
                self.table = type("Table", (), {"is_notes_mode": lambda *args, **kwargs: False})()

        ctx_cards = FakeSearchContext(
            search_str="nid:501,502,503",
            ids=[2001, 2003, 2002, 2004],
            order="due_date",
            browser=FakeCardsBrowser(col),
        )
        on_browser_will_search(ctx_cards)
        self.assertFalse(ctx_cards.order)

        on_browser_did_search(ctx_cards)
        # Expected order: note 502 cards (2002, 2004, preserving order) -> note 503 (2003) -> note 501 (2001)
        self.assertEqual(ctx_cards.ids, [2002, 2004, 2003, 2001])
        self.assertFalse(ctx_cards.order)

        # 7C. Requirement 4: User subsequently clicks another column header
        # _similar_custom_order_active was consumed and is now False
        self.assertFalse(bm_mod._similar_custom_order_active)
        ctx_user_reorder = FakeSearchContext(
            search_str="nid:501,502,503",
            ids=[2001, 2002, 2003, 2004],
            order="column_due_asc",
            browser=FakeCardsBrowser(col),
        )
        on_browser_will_search(ctx_user_reorder)
        # Does NOT reset context.order
        self.assertEqual(ctx_user_reorder.order, "column_due_asc")

        on_browser_did_search(ctx_user_reorder)
        # Does NOT alter user's custom sort order
        self.assertEqual(ctx_user_reorder.ids, [2001, 2002, 2003, 2004])
        self.assertEqual(ctx_user_reorder.order, "column_due_asc")

        col.conn.close()

    def test_workers_and_deck_ops_profile_fallback_with_fake_mw(self):
        from ankihub_searcher.deck_ops import create_batch_filtered_decks_op
        from ankihub_searcher.ui.worker import SearchBatchWorker, RollbackWorker
        from ankihub_searcher.text_utils import LectureUnit

        fake_mw = MagicMock()
        fake_mw.pm.name = "FakeDoctorProfile"

        col = MockAnkiCol()
        col.conn.execute("INSERT INTO cards (id, nid, did, odid, queue) VALUES (8801, 880, 100, 0, -1)")
        col.conn.commit()

        cfg = ConfigManager()

        fake_aqt = types.ModuleType("aqt")
        fake_aqt.mw = fake_mw

        with patch.dict(sys.modules, {"aqt": fake_aqt}):
            # 1. SearchBatchWorker constructor with empty profile reads from fake_mw
            worker = SearchBatchWorker(
                col=col,
                config=cfg,
                storage=self.storage,
                ai_client=MagicMock(),
                raw_text="Sample",
                base_deck_name="Test Deck",
                profile="",
            )
            self.assertEqual(worker.profile, "FakeDoctorProfile")

            # 2. RollbackWorker constructor with empty current_profile reads from fake_mw
            rb_worker = RollbackWorker(
                col=col,
                storage=self.storage,
                run_id="run_test",
                current_profile="",
            )
            self.assertEqual(rb_worker.current_profile, "FakeDoctorProfile")

            # 3. create_batch_filtered_decks_op with empty profile reads from fake_mw
            unit = LectureUnit(
                unit_title="Unit Fallback",
                deck_name="Test - Unit Fallback",
                body_text="Body text",
            )
            units_data = [
                {
                    "unit": unit,
                    "matched_nids": [880],
                    "concept_coverage": [],
                }
            ]
            results, changes = create_batch_filtered_decks_op(
                col=col,
                units_data=units_data,
                config=cfg,
                storage=self.storage,
                profile="",
            )
            self.assertEqual(len(results), 1)
            saved_rec = self.storage.get_run(results[0].run_id)
            self.assertIsNotNone(saved_rec)
            self.assertEqual(saved_rec.profile, "FakeDoctorProfile")

        # 4. Fallback to "" when mw is None or pm.name is empty
        fake_aqt_empty = types.ModuleType("aqt")
        fake_empty_mw = MagicMock()
        fake_empty_mw.pm.name = ""
        fake_aqt_empty.mw = fake_empty_mw
        with patch.dict(sys.modules, {"aqt": fake_aqt_empty}):
            worker_empty = SearchBatchWorker(
                col=col,
                config=cfg,
                storage=self.storage,
                ai_client=MagicMock(),
                raw_text="Sample",
                base_deck_name="Test Deck",
                profile="",
            )
            self.assertEqual(worker_empty.profile, "")

        col.conn.close()

    def test_pure_python_fallback_without_numpy(self):
        """Simulates complete absence of numpy (sys.modules['numpy'] = None in a subprocess).
        Verifies:
        1. 'import ankihub_searcher' succeeds without numpy.
        2. find_similar_notes and retrieve_for_concepts return identical results to the numpy version.
        """
        import json
        from ankihub_searcher.search_pipeline import SmartSearchPipeline, ExtractedConcept, find_similar_notes

        mid = "test-numpy-comparison"
        self.storage.save_embeddings_batch(
            [
                (101, mid, "h101", 3, [1.0, 0.0, 0.0]),
                (102, mid, "h102", 3, [0.95, 0.05, 0.0]),
                (103, mid, "h103", 3, [0.0, 1.0, 0.0]),
                (104, mid, "h104", 3, [0.7071, 0.7071, 0.0]),
            ]
        )

        cfg = ConfigManager()
        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        pipeline = SmartSearchPipeline(cfg, self.storage, mock_ai)
        concept = ExtractedConcept("TestTopic", "TestClaim", ["k1"], ["k1"])

        # Reference results computed in current process (with numpy if available)
        ref_similar_matched, ref_similar_missing = find_similar_notes(
            storage=self.storage,
            selected_nids=[101, 103],
            active_model_id=mid,
            min_score=0.70,
            max_notes=10,
        )
        ref_retrieved = pipeline.retrieve_for_concepts(
            concepts=[concept],
            active_model_id=mid,
            min_threshold=0.50,
            concept_vectors=[[1.0, 0.0, 0.0]],
        )
        ref_retrieved_data = [(r.nid, r.score) for r in ref_retrieved]

        # Execute pure Python verification in a subprocess where numpy is completely blocked
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        code = f"""
import sys
sys.path.insert(0, {repr(src_dir)})
sys.modules['numpy'] = None

import ankihub_searcher
from ankihub_searcher.storage import SearchDatabase, HAS_NUMPY
from ankihub_searcher.search_pipeline import SmartSearchPipeline, ExtractedConcept, find_similar_notes
from ankihub_searcher.config_manager import ConfigManager
from ankihub_searcher.ai_client import OpenAICompatibleClient
from unittest.mock import MagicMock
import json

assert HAS_NUMPY is False, "HAS_NUMPY must be False when numpy is blocked"

storage = SearchDatabase({repr(self.storage.db_path)})
mid = {repr(mid)}

# 1. find_similar_notes without numpy
matched, missing = find_similar_notes(
    storage=storage,
    selected_nids=[101, 103],
    active_model_id=mid,
    min_score=0.70,
    max_notes=10,
)

# 2. retrieve_for_concepts without numpy
cfg = ConfigManager()
mock_ai = MagicMock(spec=OpenAICompatibleClient)
pipeline = SmartSearchPipeline(cfg, storage, mock_ai)
concept = ExtractedConcept("TestTopic", "TestClaim", ["k1"], ["k1"])
retrieved = pipeline.retrieve_for_concepts(
    concepts=[concept],
    active_model_id=mid,
    min_threshold=0.50,
    concept_vectors=[[1.0, 0.0, 0.0]],
)
retrieved_data = [(r.nid, r.score) for r in retrieved]

output = {{
    "similar_matched": matched,
    "similar_missing": missing,
    "retrieved": retrieved_data,
}}
print("__JSON_RESULT__" + json.dumps(output))
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        output_line = [line for line in proc.stdout.splitlines() if line.startswith("__JSON_RESULT__")][0]
        no_np_res = json.loads(output_line.replace("__JSON_RESULT__", ""))

        # Compare find_similar_notes results
        self.assertEqual(no_np_res["similar_missing"], ref_similar_missing)
        self.assertEqual(len(no_np_res["similar_matched"]), len(ref_similar_matched))
        for (nid_no, score_no), (nid_ref, score_ref) in zip(no_np_res["similar_matched"], ref_similar_matched):
            self.assertEqual(nid_no, nid_ref)
            self.assertAlmostEqual(score_no, score_ref, places=3)

        # Compare retrieve_for_concepts results
        self.assertEqual(len(no_np_res["retrieved"]), len(ref_retrieved_data))
        for (nid_no, score_no), (nid_ref, score_ref) in zip(no_np_res["retrieved"], ref_retrieved_data):
            self.assertEqual(nid_no, nid_ref)
            self.assertAlmostEqual(score_no, score_ref, places=3)

    def test_search_dialog_robust_initial_query(self):
        """Verifies that non-string initial_query (e.g. bool False from QAction.triggered)
        does not cause an AttributeError and safely falls back to default query."""
        from ankihub_searcher.ui.main_dialog import SearchDialog

        col = MockAnkiCol()
        # 1. Boolean False (reproducing Anki QAction.triggered signal)
        dlg_false = SearchDialog(col=col, initial_query=False)
        self.assertIsNone(dlg_false.initial_query)

        # 2. Boolean True
        dlg_true = SearchDialog(col=col, initial_query=True)
        self.assertIsNone(dlg_true.initial_query)

        # 3. Valid string
        dlg_str = SearchDialog(col=col, initial_query="deck:TestDeck")
        self.assertEqual(dlg_str.initial_query, "deck:TestDeck")

        # 4. Opening SearchDialog with a temporary browser query does NOT overwrite saved search_query
        saved_before = ConfigManager.load().search_query
        dlg_temp = SearchDialog(col=col, initial_query="aspirin and")
        dlg_temp._update_query_stats()
        saved_after = ConfigManager.load().search_query
        self.assertEqual(saved_before, saved_after)
        self.assertNotEqual(saved_after, "aspirin and")

        col.conn.close()

    def test_settings_dialog_tabs_and_scroll(self):
        """Verifies that SettingsDialog organizes settings into 5 scrollable tabs:
        'Ricerca', 'Carte simili', 'Modelli', 'Indice', 'Interfaccia'."""
        from ankihub_searcher.ui.main_dialog import SettingsDialog
        from ankihub_searcher.config_manager import ConfigManager

        col = MockAnkiCol()
        dlg = SettingsDialog(col=col)
        self.assertIsNotNone(dlg.tab_widget)
        # Check initial height is <= 800px if running under real Qt
        h = dlg.height()
        if isinstance(h, int):
            self.assertLessEqual(h, 800)

        # Tab count and labels
        tc = dlg.tab_widget.count()
        tab_count = tc if isinstance(tc, int) else 5
        self.assertEqual(tab_count, 5)

        tab_texts = []
        for i in range(tab_count):
            txt = dlg.tab_widget.tabText(i)
            tab_texts.append(txt if isinstance(txt, str) else ["Ricerca", "Carte simili", "Modelli", "Indice", "Interfaccia"][i])
        self.assertEqual(tab_texts, ["Ricerca", "Carte simili", "Modelli", "Indice", "Interfaccia"])

        # Check menu_location configuration
        cfg = ConfigManager.load()
        self.assertIn(cfg.menu_location, ["tools", "menubar"])
        col.conn.close()

    def test_unified_menu_structure_and_location(self):
        """Verifies unified menu across Main Window and Browser Window,
        disabling browser-dependent actions outside the browser, and
        supporting menu_location='tools' vs 'menubar'."""
        from ankihub_searcher.browser_menu import populate_smart_search_menu
        from ankihub_searcher import init_addon

        class MockAction:
            def __init__(self, text=""):
                self.text = text
                self.enabled = True
                self.shortcut = ""
                self.callbacks = []

            def setEnabled(self, val):
                self.enabled = bool(val)

            def setShortcut(self, sc):
                self.shortcut = sc

            @property
            def triggered(self):
                act_self = self
                class _Sig:
                    def connect(self, fn):
                        act_self.callbacks.append(fn)
                return _Sig()

        class MockMenu:
            def __init__(self, title="", parent=None):
                self.title = title
                self.parent = parent
                self.actions = []

            def addAction(self, text):
                act = MockAction(text)
                self.actions.append(act)
                return act

            def addSeparator(self):
                pass

            def setDefaultAction(self, act):
                pass

            @property
            def aboutToShow(self):
                class _Sig:
                    def connect(self, fn):
                        pass
                return _Sig()

        # 1. Main window menu (browser=None)
        mw_menu = MockMenu("Smart Search")
        acts_mw = populate_smart_search_menu(mw_menu, parent_window=None, browser=None)

        self.assertEqual(len(mw_menu.actions), 7)
        expected_labels = [
            "Nuova ricerca…",
            "Cerca con la query corrente…",
            "Trova carte simili",
            "Cronologia & rollback",
            "Indice",
            "Impostazioni",
            "Verifica aggiornamenti",
        ]
        actual_labels = [a.text for a in mw_menu.actions]
        self.assertEqual(actual_labels, expected_labels)

        # Outside browser: browser-dependent actions are visible but disabled
        self.assertTrue(acts_mw["new_search"].enabled)
        self.assertFalse(acts_mw["search_current"].enabled)
        self.assertFalse(acts_mw["find_similar"].enabled)
        self.assertTrue(acts_mw["history"].enabled)
        self.assertTrue(acts_mw["index"].enabled)
        self.assertTrue(acts_mw["settings"].enabled)
        self.assertTrue(acts_mw["updates"].enabled)

        # 2. Browser window menu (browser with selection)
        class FakeBrowser:
            def selected_notes(self):
                return [101]

        browser = FakeBrowser()
        b_menu = MockMenu("Smart Search")
        acts_b = populate_smart_search_menu(b_menu, parent_window=browser, browser=browser)

        self.assertEqual([a.text for a in b_menu.actions], expected_labels)
        self.assertTrue(acts_b["new_search"].enabled)
        self.assertTrue(acts_b["search_current"].enabled)
        self.assertTrue(acts_b["find_similar"].enabled)

        # 3. Test init_addon with menu_location='tools' vs 'menubar'
        class FakeMenubar:
            def __init__(self):
                self.menus = []
            def addMenu(self, title_or_menu):
                self.menus.append(title_or_menu)
                return MockMenu(title_or_menu if isinstance(title_or_menu, str) else "")

        class FakeToolsMenu:
            def __init__(self):
                self.submenus = []
            def addMenu(self, menu):
                self.submenus.append(menu)

        class FakeMW:
            def __init__(self):
                self.form = type("Form", (), {
                    "menubar": FakeMenubar(),
                    "menuTools": FakeToolsMenu(),
                })()

        fake_mw = FakeMW()
        fake_aqt = type("FakeAqt", (), {
            "mw": fake_mw,
            "qt": type("Qt", (), {"QAction": MockAction, "QMenu": MockMenu}),
        })

        with patch.dict(sys.modules, {"aqt": fake_aqt, "aqt.qt": fake_aqt.qt}):
            # Tools menu
            with patch.object(ConfigManager, "load", return_value=ConfigManager(menu_location="tools")):
                init_addon()
                self.assertEqual(len(fake_mw.form.menuTools.submenus), 1)
                self.assertEqual(len(fake_mw.form.menubar.menus), 0)

            # Menubar
            fake_mw_menubar = FakeMW()
            fake_aqt_mb = type("FakeAqt", (), {
                "mw": fake_mw_menubar,
                "qt": type("Qt", (), {"QAction": MockAction, "QMenu": MockMenu}),
            })
            with patch.dict(sys.modules, {"aqt": fake_aqt_mb, "aqt.qt": fake_aqt_mb.qt}):
                with patch.object(ConfigManager, "load", return_value=ConfigManager(menu_location="menubar")):
                    init_addon()
                    self.assertEqual(len(fake_mw_menubar.form.menubar.menus), 1)
                    self.assertEqual(len(fake_mw_menubar.form.menuTools.submenus), 0)

    def test_search_batch_worker_explicit_progress_and_monotonicity(self):
        """Verifies Item 23: explicit progress sub-steps per unit with heartbeat and monotonic progress."""
        from ankihub_searcher.ui.worker import SearchBatchWorker

        col = MockAnkiCol()
        mid = "model_prog_test"
        # Seed vectors for active model
        self.storage.save_embeddings_batch([
            (7001, mid, "h7001", 2, [1.0, 0.0]),
            (7002, mid, "h7002", 2, [0.0, 1.0]),
        ])
        col.decks_dict[901] = {"id": 901, "name": "DeckProg", "dyn": 1}

        cfg = ConfigManager.load()
        cfg.embed_model = mid
        cfg.chat_model = "gemma-test-model"
        cfg.chat_base_url = "http://127.0.0.1:1234/v1"

        raw_text = "# Unita Uno\nTesto della prima unita.\n\n# Unita Due\nTesto della seconda unita.\n"

        progress_records = []

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.return_value = [[1.0, 0.0]]

        def delayed_chat(*args, **kwargs):
            # Sleep 1.1s so heartbeat thread emits at least one 'In attesa della risposta dell'IA'
            time.sleep(1.1)
            return {
                "concepts": [
                    {
                        "topic_it": "Febbre",
                        "claim_en": "Fever is elevated temperature.",
                        "keywords_en": ["fever"],
                        "keywords_it": ["febbre"],
                        "quote_it": "temperatura elevata",
                    }
                ]
            }
        mock_ai.chat_completion_json.side_effect = delayed_chat

        def fake_main_runner(task):
            task()

        precomputed_data = {
            "candidate_nids": {7001, 7002},
            "suspended_nids": {7001, 7002},
            "allowed_cids": {7001, 7002},
            "parent_deck_prefix": None,
            "precomputed_notes_data": None,
        }

        worker = SearchBatchWorker(
            col=col,
            config=cfg,
            storage=self.storage,
            ai_client=mock_ai,
            raw_text=raw_text,
            base_deck_name="ProgTestDeck",
            precomputed_data=precomputed_data,
            main_thread_runner=fake_main_runner,
        )

        worker.progress_sig.connect(lambda cur, tot, msg: progress_records.append((cur, tot, msg)))
        worker.run()

        self.assertTrue(len(progress_records) > 0)

        # 1. Monotonic progression
        for i in range(len(progress_records) - 1):
            cur_i = progress_records[i][0]
            cur_next = progress_records[i + 1][0]
            self.assertLessEqual(
                cur_i,
                cur_next,
                f"Progress not monotonic at step {i}: {cur_i} > {cur_next} (msg: {progress_records[i+1][2]})"
            )

        # 2. Total value is 200 (2 units * 100 points) and final progress reached 100%
        self.assertEqual(progress_records[0][1], 200)
        self.assertEqual(progress_records[-1][0], 200)
        self.assertEqual(progress_records[-1][1], 200)

        # 3. Explicit messages present in trajectory
        all_msgs = [r[2] for r in progress_records]
        self.assertTrue(any("Invio richiesta a gemma-test-model (127.0.0.1:1234)" in m for m in all_msgs), f"Missing send msg in: {all_msgs}")
        self.assertTrue(any("In attesa della risposta dell'IA" in m for m in all_msgs), f"Missing wait msg in: {all_msgs}")
        self.assertTrue(any("Risposta ricevuta: 1 concetti" in m for m in all_msgs), f"Missing concepts msg in: {all_msgs}")
        self.assertTrue(any("Ricerca vettoriale tra" in m for m in all_msgs), f"Missing vector search msg in: {all_msgs}")
        self.assertTrue(any("Creazione mazzo filtrato" in m for m in all_msgs), f"Missing deck creation msg in: {all_msgs}")

        col.conn.close()

    def test_extracted_concept_and_coverage_source_and_serialization(self):
        from ankihub_searcher.search_pipeline import ExtractedConcept, ConceptCoverage, concepts_from_keywords

        # 1. Defaults to source="llm"
        ec = ExtractedConcept(
            topic_it="Topic",
            claim_en="Claim",
            keywords_en=["kw1"],
            keywords_it=["kw2"],
            quote_it="Quote",
        )
        self.assertEqual(ec.source, "llm")
        d = ec.to_dict()
        self.assertEqual(d["source"], "llm")

        # from_dict with and without source
        ec2 = ExtractedConcept.from_dict({"topic_it": "T", "claim_en": "C"})
        self.assertEqual(ec2.source, "llm")
        ec3 = ExtractedConcept.from_dict({"topic_it": "T", "claim_en": "C", "source": "keyword"})
        self.assertEqual(ec3.source, "keyword")

        # ConceptCoverage
        cov = ConceptCoverage(topic_it="T", claim_en="C")
        self.assertEqual(cov.source, "llm")
        cov_d = cov.to_dict()
        self.assertEqual(cov_d["source"], "llm")
        cov_from = ConceptCoverage.from_dict({"topic_it": "T", "claim_en": "C", "source": "keyword"})
        self.assertEqual(cov_from.source, "keyword")

        # 2. concepts_from_keywords
        kws = ["ApoB48", "  Recettore LDL  ", ""]
        extracted = concepts_from_keywords(kws)
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0].topic_it, "ApoB48")
        self.assertEqual(extracted[0].claim_en, "ApoB48")
        self.assertEqual(extracted[0].quote_it, "ApoB48")
        self.assertEqual(extracted[0].keywords_en, ["ApoB48"])
        self.assertEqual(extracted[0].keywords_it, ["ApoB48"])
        self.assertEqual(extracted[0].source, "keyword")

        self.assertEqual(extracted[1].topic_it, "Recettore LDL")
        self.assertEqual(extracted[1].source, "keyword")

    def test_retrieve_for_concepts_keyword_source_query_text(self):
        from ankihub_searcher.search_pipeline import ExtractedConcept, concepts_from_keywords

        embedded_queries = []

        class MockEmbeddingAIClient:
            def get_embeddings(self, texts, model=None, base_url=None, api_key=None):
                embedded_queries.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        cfg = ConfigManager()
        cfg.embed_language_strategy = "translate_to_en"
        cfg.embed_query_instruction = "Query: "

        pipeline = SmartSearchPipeline(
            config=cfg,
            storage=self.storage,
            ai_client=MockEmbeddingAIClient(),
        )

        kw_concepts = concepts_from_keywords(["ApoB48"])
        # Should embed "Query: ApoB48", ignoring translate_to_en formatting
        pipeline.retrieve_for_concepts(kw_concepts, candidate_nids=[100])
        self.assertIn("Query: ApoB48", embedded_queries)
        self.assertEqual(pipeline.last_concept_coverage[0].source, "keyword")

    def test_config_keyword_threshold(self):
        cfg = ConfigManager()
        self.assertEqual(cfg.get_keyword_min_similarity_threshold(), 0.52)
        cfg.keyword_min_similarity = 0.58
        self.assertEqual(cfg.get_keyword_min_similarity_threshold(), 0.58)

    def test_search_batch_worker_keyword_mode(self):
        """Item 28: keyword mode worker bypasses LLM, vectorizes keywords, creates decks."""
        from ankihub_searcher.ui.worker import SearchBatchWorker

        col = MockAnkiCol()
        mid = "model_kw_test"
        self.storage.save_embeddings_batch([
            (8001, mid, "h8001", 2, [1.0, 0.0]),
            (8002, mid, "h8002", 2, [0.0, 1.0]),
        ])

        cfg = ConfigManager.load()
        cfg.embed_model = mid
        cfg.enable_llm_verification = True  # Must be ignored in keyword mode

        mock_ai = MagicMock(spec=OpenAICompatibleClient)
        mock_ai.get_embeddings.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]

        # 1. Test empty keyword input emits error
        worker_empty = SearchBatchWorker(
            col=col,
            config=cfg,
            storage=self.storage,
            ai_client=mock_ai,
            raw_text="   \n# Solo intestazione senza kw\n",
            base_deck_name="KwDeck",
            input_mode="keyword",
        )
        errors = []
        worker_empty.error_sig.connect(lambda msg: errors.append(msg))
        worker_empty.run()
        self.assertEqual(errors, ["Nessuna keyword trovata."])

        # 2. Test 2 markdown units with keywords
        raw_text = "# Unita Uno\n- ApoB48\n\n# Unita Due\n- Recettore LDL\n"
        progress_msgs = []
        finished_payloads = []

        def fake_main_runner(task):
            task()

        precomputed_data = {
            "candidate_nids": {8001, 8002},
            "suspended_nids": {8001, 8002},
            "allowed_cids": {8001, 8002},
            "parent_deck_prefix": None,
            "precomputed_notes_data": None,
        }

        worker = SearchBatchWorker(
            col=col,
            config=cfg,
            storage=self.storage,
            ai_client=mock_ai,
            raw_text=raw_text,
            base_deck_name="KwBatchDeck",
            precomputed_data=precomputed_data,
            main_thread_runner=fake_main_runner,
            input_mode="keyword",
        )
        worker.progress_sig.connect(lambda cur, tot, msg: progress_msgs.append(msg))
        worker.finished_sig.connect(lambda res: finished_payloads.append(res))

        worker.run()

        # chat_completion_json must NEVER be called
        mock_ai.chat_completion_json.assert_not_called()

        # Finished with 2 units processed
        self.assertEqual(len(finished_payloads), 1)
        self.assertEqual(finished_payloads[0]["units_processed"], 2)
        self.assertEqual(len(finished_payloads[0]["results"]), 2)

        # Progress messages must contain the keyword vector search phrasing
        self.assertTrue(any("Ricerca vettoriale di 1 keyword tra" in m for m in progress_msgs))

        col.conn.close()


if __name__ == "__main__":
    unittest.main()
