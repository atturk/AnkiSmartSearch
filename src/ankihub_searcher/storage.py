"""SQLite storage engine and model-scoped vector index for Smart Search.

Manages:
1. notes_content: extracted text and tags per note.
2. notes_fts: SQLite FTS5 table for fast BM25 full-text search.
3. note_embeddings: isolated embeddings per model_id, stored as packed float32 L2-normalized vectors.
4. run_history: records runs, filtered deck IDs, and exact newly_unsuspended_cids for safe rollbacks.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RunRecord:
    """Represents a search run that created a filtered deck."""

    run_id: str
    created_at: int
    deck_name: str
    filtered_did: int
    source_decks: List[str]
    unit_title: str
    unit_preview: str
    total_matched_cids: List[int]
    newly_unsuspended_cids: List[int]
    rolled_back: int = 0
    already_active_cids: List[int] = field(default_factory=list)
    concept_coverage: List[Dict[str, Any]] = field(default_factory=list)
    unit_error: str = ""


def normalize_l2(vec: Sequence[float]) -> List[float]:
    """Normalizes vector to unit L2 norm."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return [0.0] * len(vec)
    return [float(x / norm) for x in vec]


def pack_vector(vec: Sequence[float]) -> bytes:
    """Packs a float vector into a little-endian float32 byte blob."""
    norm_vec = normalize_l2(vec)
    return struct.pack(f"<{len(norm_vec)}f", *norm_vec)


def unpack_vector(blob: bytes, dim: int) -> List[float]:
    """Unpacks a little-endian float32 byte blob into a list of floats."""
    return list(struct.unpack(f"<{dim}f", blob))


# Short list of English and Italian stopwords discarded during FTS lexical search
_FTS_STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now", "are", "is", "was", "were",
    # Italian
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "da", "con",
    "su", "per", "tra", "fra", "ed", "del", "dello", "della", "dei", "degli",
    "delle", "al", "allo", "alla", "ai", "agli", "alle", "dal", "dallo", "dalla",
    "dai", "dagli", "dalle", "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle", "che", "non", "si", "ma",
    "anche", "come", "piu", "sono", "sia", "era", "stato", "stata", "stati", "state",
}


class SearchDatabase:
    """Thread-safe SQLite database manager for Smart Search."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to <addon_dir>/user_files/smart_search.db
            base_dir = os.path.dirname(os.path.abspath(__file__))
            user_files = os.path.join(base_dir, "user_files")
            os.makedirs(user_files, exist_ok=True)
            db_path = os.path.join(user_files, "smart_search.db")

            # Automatic migration: if smart_search.db does not exist, check legacy ankihub_searcher.db
            legacy_path = os.path.join(user_files, "ankihub_searcher.db")
            if not os.path.exists(db_path) and os.path.exists(legacy_path):
                self._migrate_legacy_db(legacy_path, db_path)

        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    @staticmethod
    def _migrate_legacy_db(legacy_path: str, new_path: str) -> None:
        """Migrates legacy ankihub_searcher.db to smart_search.db preserving WAL state."""
        try:
            try:
                conn = sqlite3.connect(legacy_path)
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.close()
            except Exception:
                pass
            os.rename(legacy_path, new_path)
            for suffix in ("-wal", "-shm"):
                old_file = legacy_path + suffix
                new_file = new_path + suffix
                if os.path.exists(old_file):
                    try:
                        os.rename(old_file, new_file)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[SmartSearch] Warning: failed to migrate legacy database: {e}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            # 1. Notes content table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes_content (
                    nid INTEGER PRIMARY KEY,
                    mid INTEGER NOT NULL,
                    note_mod INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    clean_text TEXT NOT NULL,
                    clean_tags TEXT NOT NULL
                )
                """
            )

            # 2. Virtual FTS5 table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                    nid UNINDEXED,
                    clean_text,
                    clean_tags,
                    tokenize='unicode61'
                )
                """
            )

            # 3. Model-scoped embeddings table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_embeddings (
                    nid INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    PRIMARY KEY (nid, model_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON note_embeddings(model_id)"
            )

            # 4. Run history table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_history (
                    run_id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    deck_name TEXT NOT NULL,
                    filtered_did INTEGER,
                    source_decks_json TEXT NOT NULL,
                    unit_title TEXT NOT NULL,
                    unit_preview TEXT NOT NULL,
                    total_matched_cids_json TEXT NOT NULL,
                    newly_unsuspended_cids_json TEXT NOT NULL,
                    rolled_back INTEGER DEFAULT 0,
                    already_active_cids_json TEXT DEFAULT '[]',
                    concept_coverage_json TEXT DEFAULT '[]',
                    unit_error TEXT DEFAULT ''
                )
                """
            )
            # Seamless schema migration for pre-0.1.1 / pre-0.3.0 / pre-0.4.0 databases
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(run_history)")
            existing_cols = {row["name"] for row in cur.fetchall()}
            if "already_active_cids_json" not in existing_cols:
                conn.execute("ALTER TABLE run_history ADD COLUMN already_active_cids_json TEXT DEFAULT '[]'")
            if "concept_coverage_json" not in existing_cols:
                conn.execute("ALTER TABLE run_history ADD COLUMN concept_coverage_json TEXT DEFAULT '[]'")
            if "unit_error" not in existing_cols:
                conn.execute("ALTER TABLE run_history ADD COLUMN unit_error TEXT DEFAULT ''")
            conn.commit()

    # ---------------- Note Content & FTS Operations ----------------

    def upsert_notes_content_batch(
        self,
        notes_data: List[Tuple[int, int, int, str, str, str]],
    ) -> None:
        """Batch upserts (nid, mid, note_mod, content_hash, clean_text, clean_tags)

        and updates FTS5 table.
        """
        if not notes_data:
            return

        with self._get_connection() as conn:
            # Upsert into notes_content
            conn.executemany(
                """
                INSERT INTO notes_content (nid, mid, note_mod, content_hash, clean_text, clean_tags)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(nid) DO UPDATE SET
                    mid=excluded.mid,
                    note_mod=excluded.note_mod,
                    content_hash=excluded.content_hash,
                    clean_text=excluded.clean_text,
                    clean_tags=excluded.clean_tags
                """,
                notes_data,
            )

            # Delete old FTS entries then insert new
            nids = [(x[0],) for x in notes_data]
            conn.executemany("DELETE FROM notes_fts WHERE nid = ?", nids)

            fts_data = [(x[0], x[4], x[5]) for x in notes_data]
            conn.executemany(
                "INSERT INTO notes_fts (nid, clean_text, clean_tags) VALUES (?, ?, ?)",
                fts_data,
            )
            conn.commit()

    def get_stored_notes_mod_map(self) -> Dict[int, Tuple[int, str]]:
        """Returns {nid: (note_mod, content_hash)} for fast incremental checks."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT nid, note_mod, content_hash FROM notes_content")
            return {row["nid"]: (row["note_mod"], row["content_hash"]) for row in cur.fetchall()}

    def get_notes_needing_indexing(
        self,
        notes_meta: Dict[int, Tuple[int, str]],
        active_model_id: str,
    ) -> Tuple[List[int], List[int]]:
        """Compares target notes with stored DB records.

        notes_meta: {nid: (note_mod, content_hash)}
        Returns:
            (needs_text_nids, needs_embedding_nids)
        """
        if not notes_meta:
            return [], []

        with self._get_connection() as conn:
            all_target_nids = list(notes_meta.keys())
            # Read existing notes_content
            cur = conn.cursor()
            cur.execute("SELECT nid, note_mod, content_hash FROM notes_content")
            stored_content = {row["nid"]: (row["note_mod"], row["content_hash"]) for row in cur.fetchall()}

            # Read existing embeddings for active_model_id
            cur.execute(
                "SELECT nid, content_hash FROM note_embeddings WHERE model_id = ?",
                (active_model_id,),
            )
            stored_embeddings = {row["nid"]: row["content_hash"] for row in cur.fetchall()}

        needs_text: List[int] = []
        needs_embedding: List[int] = []

        for nid in all_target_nids:
            cur_mod, cur_hash = notes_meta[nid]

            # Text check
            if nid not in stored_content:
                needs_text.append(nid)
                needs_embedding.append(nid)
                continue

            saved_mod, saved_hash = stored_content[nid]
            if saved_mod != cur_mod or saved_hash != cur_hash:
                needs_text.append(nid)
                needs_embedding.append(nid)
                continue

            # Embedding check for active model
            if nid not in stored_embeddings or stored_embeddings[nid] != cur_hash:
                needs_embedding.append(nid)

        return needs_text, needs_embedding

    def get_notes_content(self, nids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieves clean_text, clean_tags, content_hash for given nids."""
        if not nids:
            return {}

        results: Dict[int, Dict[str, Any]] = {}
        with self._get_connection() as conn:
            # Query in chunks if nids list is large
            chunk_size = 900
            for i in range(0, len(nids), chunk_size):
                chunk = nids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cur = conn.cursor()
                cur.execute(
                    f"SELECT nid, mid, note_mod, content_hash, clean_text, clean_tags FROM notes_content WHERE nid IN ({placeholders})",
                    chunk,
                )
                for row in cur.fetchall():
                    results[row["nid"]] = {
                        "nid": row["nid"],
                        "mid": row["mid"],
                        "note_mod": row["note_mod"],
                        "content_hash": row["content_hash"],
                        "clean_text": row["clean_text"],
                        "clean_tags": row["clean_tags"],
                    }
        return results

    # ---------------- Embeddings Operations ----------------

    def save_embeddings_batch(
        self,
        embeddings_data: List[Tuple[int, str, str, int, Sequence[float]]],
    ) -> None:
        """Batch saves embeddings.

        Each item is: (nid, model_id, content_hash, dim, vector)
        Vectors are L2-normalized and packed into binary blobs.
        """
        if not embeddings_data:
            return

        packed_records = []
        for nid, model_id, content_hash, dim, vec in embeddings_data:
            blob = pack_vector(vec)
            packed_records.append((nid, model_id, content_hash, dim, blob))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO note_embeddings (nid, model_id, content_hash, dim, vector_blob)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(nid, model_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    dim=excluded.dim,
                    vector_blob=excluded.vector_blob
                """,
                packed_records,
            )
            conn.commit()

    def load_embeddings_for_notes(
        self,
        nids: Optional[Sequence[int]],
        model_id: str,
    ) -> Tuple[List[int], Any, int]:
        """Loads vectors for given nids (or all if nids is None) for model_id.

        Returns (nid_list, matrix_or_list, dim).
        If numpy is available, matrix_or_list is np.ndarray of shape (N, dim).
        Otherwise, it is a list of unpacked float lists.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            if nids is None:
                cur.execute(
                    "SELECT nid, dim, vector_blob FROM note_embeddings WHERE model_id = ?",
                    (model_id,),
                )
                rows = cur.fetchall()
            else:
                if not nids:
                    return [], [] if not HAS_NUMPY else np.empty((0, 0), dtype=np.float32), 0
                if len(nids) <= 900:
                    placeholders = ",".join("?" * len(nids))
                    cur.execute(
                        f"SELECT nid, dim, vector_blob FROM note_embeddings WHERE model_id = ? AND nid IN ({placeholders})",
                        [model_id, *nids],
                    )
                    rows = cur.fetchall()
                else:
                    # Avoid SQLite SQLITE_MAX_VARIABLE_NUMBER limit when nids has 20,000+ notes
                    candidate_set = set(nids)
                    cur.execute(
                        "SELECT nid, dim, vector_blob FROM note_embeddings WHERE model_id = ?",
                        (model_id,),
                    )
                    rows = [r for r in cur.fetchall() if r["nid"] in candidate_set]

        if not rows:
            return [], [] if not HAS_NUMPY else np.empty((0, 0), dtype=np.float32), 0

        nid_list: List[int] = []
        dim = rows[0]["dim"]

        if HAS_NUMPY:
            n_rows = len(rows)
            matrix = np.empty((n_rows, dim), dtype=np.float32)
            for idx, r in enumerate(rows):
                nid_list.append(r["nid"])
                matrix[idx] = np.frombuffer(r["vector_blob"], dtype=np.float32)
            return nid_list, matrix, dim
        else:
            vectors: List[List[float]] = []
            for r in rows:
                nid_list.append(r["nid"])
                vectors.append(unpack_vector(r["vector_blob"], dim))
            return nid_list, vectors, dim

    def compute_similarities_from_loaded(
        self,
        nid_list: Sequence[int],
        vecs: Any,
        query_vec: Sequence[float],
        force_pure_python: bool = False,
    ) -> List[Tuple[int, float]]:
        """Computes cosine similarity against pre-loaded vectors without re-querying SQLite."""
        if not nid_list:
            return []

        norm_q = normalize_l2(query_vec)
        use_np = HAS_NUMPY and not force_pure_python

        if use_np and isinstance(vecs, np.ndarray):
            q_arr = np.array(norm_q, dtype=np.float32)
            scores = np.dot(vecs, q_arr)
            results = [(nid_list[i], float(scores[i])) for i in range(len(nid_list))]
        else:
            if HAS_NUMPY and isinstance(vecs, np.ndarray):
                vec_list = [v.tolist() for v in vecs]
            else:
                vec_list = vecs

            results = []
            for i, vec in enumerate(vec_list):
                sim = sum(a * b for a, b in zip(vec, norm_q))
                results.append((nid_list[i], float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def batch_cosine_similarity(
        self,
        candidate_nids: Optional[Sequence[int]],
        model_id: str,
        query_vec: Sequence[float],
        force_pure_python: bool = False,
    ) -> List[Tuple[int, float]]:
        """Computes cosine similarity between query_vec and candidate notes.

        Returns list of (nid, similarity_score) sorted descending by score.
        """
        nid_list, vecs, _ = self.load_embeddings_for_notes(candidate_nids, model_id)
        return self.compute_similarities_from_loaded(
            nid_list=nid_list,
            vecs=vecs,
            query_vec=query_vec,
            force_pure_python=force_pure_python,
        )

    def count_embeddings_for_notes(self, nids: Sequence[int], model_id: str) -> int:
        """Counts how many of the given nids have embeddings stored for model_id."""
        if not nids:
            return 0
        target_set = set(nids)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT nid FROM note_embeddings WHERE model_id = ?", (model_id,))
            return sum(1 for row in cur.fetchall() if row["nid"] in target_set)

    def get_indexed_models_stats(self) -> List[Dict[str, Any]]:
        """Returns statistics of all indexed models in note_embeddings."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT model_id, COUNT(nid) as note_count, SUM(LENGTH(vector_blob)) as bytes_used
                FROM note_embeddings
                GROUP BY model_id
                ORDER BY note_count DESC
                """
            )
            stats = []
            for row in cur.fetchall():
                b = row["bytes_used"] or 0
                stats.append(
                    {
                        "model_id": row["model_id"],
                        "note_count": row["note_count"],
                        "bytes_used": b,
                        "size_mb": round(b / (1024 * 1024), 2),
                    }
                )
            return stats

    def delete_embeddings_for_model(self, model_id: str) -> int:
        """Deletes all cached embeddings for a specific model."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM note_embeddings WHERE model_id = ?", (model_id,))
            conn.commit()
            return cur.rowcount

    def delete_inactive_models(self, active_model_id: str) -> int:
        """Deletes all cached embeddings for models other than active_model_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM note_embeddings WHERE model_id != ?", (active_model_id,))
            conn.commit()
            return cur.rowcount

    def merge_external_db(self, external_db_path: str) -> Dict[str, int]:
        """Merges notes_content and note_embeddings from an external ankihub_searcher.db file."""
        if not os.path.exists(external_db_path):
            raise FileNotFoundError(f"Database file not found: {external_db_path}")

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("ATTACH DATABASE ? AS ext_db", (external_db_path,))
            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO notes_content (nid, mid, note_mod, content_hash, clean_text, clean_tags)
                    SELECT nid, mid, note_mod, content_hash, clean_text, clean_tags FROM ext_db.notes_content
                    """
                )
                notes_merged = cur.rowcount

                cur.execute(
                    """
                    INSERT INTO note_embeddings (nid, model_id, content_hash, dim, vector_blob)
                    SELECT nid, model_id, content_hash, dim, vector_blob FROM ext_db.note_embeddings
                    WHERE true
                    ON CONFLICT(nid, model_id) DO UPDATE SET
                        content_hash=excluded.content_hash,
                        dim=excluded.dim,
                        vector_blob=excluded.vector_blob
                    """
                )
                embeddings_merged = cur.rowcount
                conn.commit()
            finally:
                cur.execute("DETACH DATABASE ext_db")

        return {"notes_merged": max(0, notes_merged), "embeddings_merged": max(0, embeddings_merged)}

    def export_db(self, target_path: str) -> float:
        """Exports a clean, self-contained SQLite backup (flushing WAL) to target_path.

        Returns the size of the exported file in MB.
        """
        abs_target = os.path.abspath(target_path)
        if abs_target == os.path.abspath(self.db_path):
            raise ValueError("Il file di destinazione non può coincidere con il database attivo.")

        os.makedirs(os.path.dirname(abs_target), exist_ok=True)
        if os.path.exists(abs_target):
            os.remove(abs_target)

        with self._get_connection() as src_conn:
            src_conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            dest_conn = sqlite3.connect(abs_target)
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()

        return round(os.path.getsize(abs_target) / (1024 * 1024), 2)

    # ---------------- FTS5 Lexical Search ----------------

    def search_fts(
        self,
        keywords: Sequence[str],
        candidate_nids: Optional[Sequence[int]] = None,
        limit: int = 100,
    ) -> List[Tuple[int, float]]:
        """Searches notes_fts using BM25 ranking.

        keywords: list of terms/phrases.
        candidate_nids: optional collection of note IDs to filter within SQL.
        Returns list of (nid, normalized_bm25_score in [0, 1]).
        """
        if not keywords:
            return []

        if candidate_nids is not None and len(candidate_nids) == 0:
            return []

        # Sanitize query tokens:
        # 1. Discard bareword tokens < 3 characters and stopwords (English + Italian).
        # 2. Multi-word or hyphenated keywords -> phrase query ("il 1", "acute pancreatitis") in OR with significant individual tokens.
        # 3. Ensure no barewords contain hyphens (which FTS5 parses as operators) or characters other than [a-zA-Z0-9_].
        # 4. Escape internal double quotes in phrases.
        query_parts: List[str] = []
        seen_terms: Set[str] = set()

        for kw in keywords:
            if not kw or not kw.strip():
                continue

            # Split on whitespace, hyphens, and other punctuation
            all_words = re.findall(r"[A-Za-z0-9_]+", kw)
            if not all_words:
                continue

            # Multi-word or hyphenated phrase query in FTS5: preserves short alphanumeric parts like '1' inside the phrase
            if len(all_words) > 1:
                escaped_phrase = " ".join(all_words).replace('"', '""')
                phrase_clause = f'"{escaped_phrase}"'
                if phrase_clause not in seen_terms:
                    seen_terms.add(phrase_clause)
                    query_parts.append(phrase_clause)

            # Individual bareword tokens: strictly letters, digits, underscores, len >= 3, not in stopwords
            for w in all_words:
                if len(w) >= 3 and w.lower() not in _FTS_STOPWORDS and re.match(r"^[A-Za-z0-9_]+$", w):
                    stem = re.sub(r"(?:ies|es|s)$", "", w.lower())
                    token = f"{stem}*" if len(stem) >= 3 else f"{w.lower()}*"
                    if token not in seen_terms:
                        seen_terms.add(token)
                        query_parts.append(token)

        if not query_parts:
            return []

        fts_query = " OR ".join(query_parts)
        effective_limit = max(limit, 300)

        with self._get_connection() as conn:
            cur = conn.cursor()
            try:
                if candidate_nids is not None:
                    # Filter candidates directly inside SQL using a temp table to avoid SQLite variable limits (up to 30k+ NIDs)
                    conn.execute("CREATE TEMP TABLE temp_fts_cand (nid INTEGER PRIMARY KEY)")
                    conn.executemany("INSERT OR IGNORE INTO temp_fts_cand (nid) VALUES (?)", ((int(n),) for n in candidate_nids))
                    cur.execute(
                        """
                        SELECT f.nid, bm25(notes_fts, 1.0, 1.5) as score
                        FROM notes_fts f
                        JOIN temp_fts_cand c ON f.nid = c.nid
                        WHERE notes_fts MATCH ?
                        ORDER BY score ASC
                        LIMIT ?
                        """,
                        (fts_query, effective_limit),
                    )
                    rows = cur.fetchall()
                    conn.execute("DROP TABLE IF EXISTS temp_fts_cand")
                else:
                    cur.execute(
                        """
                        SELECT nid, bm25(notes_fts, 1.0, 1.5) as score
                        FROM notes_fts
                        WHERE notes_fts MATCH ?
                        ORDER BY score ASC
                        LIMIT ?
                        """,
                        (fts_query, effective_limit),
                    )
                    rows = cur.fetchall()
            except sqlite3.OperationalError as e:
                # Fallback if syntax error occurs
                print(f"[SmartSearch] Notice: FTS5 query operational error: {e}")
                return []

        if not rows:
            return []

        filtered: List[Tuple[int, float]] = []
        for r in rows:
            nid = int(r["nid"])
            raw_bm = float(r["score"])
            # Invert to positive magnitude: higher is better
            abs_bm = max(0.0, -raw_bm)
            filtered.append((nid, abs_bm))

        if not filtered:
            return []

        # Normalize to [0, 1] using min-max relative to best score
        max_bm = max(s for _, s in filtered)
        if max_bm < 1e-6:
            return [(nid, 0.0) for nid, _ in filtered[:limit]]

        scored = [(nid, min(1.0, score / max_bm)) for nid, score in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ---------------- Run History & Safe Rollback Operations ----------------

    def save_run(self, record: RunRecord) -> None:
        """Saves a run record to run_history."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO run_history (
                    run_id, created_at, deck_name, filtered_did, source_decks_json,
                    unit_title, unit_preview, total_matched_cids_json,
                    newly_unsuspended_cids_json, rolled_back, already_active_cids_json,
                    concept_coverage_json, unit_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    deck_name=excluded.deck_name,
                    filtered_did=excluded.filtered_did,
                    rolled_back=excluded.rolled_back,
                    already_active_cids_json=excluded.already_active_cids_json,
                    concept_coverage_json=excluded.concept_coverage_json,
                    unit_error=excluded.unit_error
                """,
                (
                    record.run_id,
                    record.created_at,
                    record.deck_name,
                    record.filtered_did,
                    json.dumps(record.source_decks),
                    record.unit_title,
                    record.unit_preview,
                    json.dumps(record.total_matched_cids),
                    json.dumps(record.newly_unsuspended_cids),
                    record.rolled_back,
                    json.dumps(record.already_active_cids),
                    json.dumps(record.concept_coverage, ensure_ascii=False),
                    record.unit_error,
                ),
            )
            conn.commit()

    def get_runs(self) -> List[RunRecord]:
        """Retrieves all past runs sorted by created_at DESC."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM run_history ORDER BY created_at DESC")
            records = []
            for row in cur.fetchall():
                row_keys = row.keys()
                active_json = row["already_active_cids_json"] if "already_active_cids_json" in row_keys else "[]"
                cov_json = row["concept_coverage_json"] if "concept_coverage_json" in row_keys else "[]"
                unit_err = row["unit_error"] if "unit_error" in row_keys else ""
                records.append(
                    RunRecord(
                        run_id=row["run_id"],
                        created_at=row["created_at"],
                        deck_name=row["deck_name"],
                        filtered_did=row["filtered_did"],
                        source_decks=json.loads(row["source_decks_json"]),
                        unit_title=row["unit_title"],
                        unit_preview=row["unit_preview"],
                        total_matched_cids=json.loads(row["total_matched_cids_json"]),
                        newly_unsuspended_cids=json.loads(row["newly_unsuspended_cids_json"]),
                        rolled_back=row["rolled_back"],
                        already_active_cids=json.loads(active_json or "[]"),
                        concept_coverage=json.loads(cov_json or "[]"),
                        unit_error=unit_err or "",
                    )
                )
            return records

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieves a single run record by run_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM run_history WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            row_keys = row.keys()
            active_json = row["already_active_cids_json"] if "already_active_cids_json" in row_keys else "[]"
            cov_json = row["concept_coverage_json"] if "concept_coverage_json" in row_keys else "[]"
            unit_err = row["unit_error"] if "unit_error" in row_keys else ""
            return RunRecord(
                run_id=row["run_id"],
                created_at=row["created_at"],
                deck_name=row["deck_name"],
                filtered_did=row["filtered_did"],
                source_decks=json.loads(row["source_decks_json"]),
                unit_title=row["unit_title"],
                unit_preview=row["unit_preview"],
                total_matched_cids=json.loads(row["total_matched_cids_json"]),
                newly_unsuspended_cids=json.loads(row["newly_unsuspended_cids_json"]),
                rolled_back=row["rolled_back"],
                already_active_cids=json.loads(active_json or "[]"),
                concept_coverage=json.loads(cov_json or "[]"),
                unit_error=unit_err or "",
            )

    def mark_run_rolled_back(self, run_id: str) -> None:
        """Marks a run as rolled back."""
        with self._get_connection() as conn:
            conn.execute("UPDATE run_history SET rolled_back = 1 WHERE run_id = ?", (run_id,))
            conn.commit()

    def delete_run(self, run_id: str) -> None:
        """Deletes a run record from history."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM run_history WHERE run_id = ?", (run_id,))
            conn.commit()

    def export_database(self, target_file_path: str) -> None:
        """Safely exports the entire SQLite database to a target file."""
        os.makedirs(os.path.dirname(os.path.abspath(target_file_path)), exist_ok=True)
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
        with self._get_connection() as src_conn:
            dst_conn = sqlite3.connect(target_file_path)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()

    def import_database(self, source_file_path: str) -> Dict[str, int]:
        """Imports and merges notes and embeddings from an external database."""
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"Source file not found: {source_file_path}")

        with self._get_connection() as conn:
            conn.execute("ATTACH DATABASE ? AS src_db", (source_file_path,))
            try:
                cur = conn.execute(
                    """
                    INSERT INTO notes_content (nid, mid, note_mod, content_hash, clean_text, clean_tags)
                    SELECT nid, mid, note_mod, content_hash, clean_text, clean_tags FROM src_db.notes_content
                    ON CONFLICT(nid) DO UPDATE SET
                        mid=excluded.mid,
                        note_mod=excluded.note_mod,
                        content_hash=excluded.content_hash,
                        clean_text=excluded.clean_text,
                        clean_tags=excluded.clean_tags
                    """
                )
                notes_imported = cur.rowcount

                conn.execute("DELETE FROM notes_fts WHERE nid IN (SELECT nid FROM src_db.notes_content)")
                conn.execute(
                    """
                    INSERT INTO notes_fts (nid, clean_text, clean_tags)
                    SELECT nid, clean_text, clean_tags FROM src_db.notes_content
                    """
                )

                cur_emb = conn.execute(
                    """
                    INSERT INTO note_embeddings (nid, model_id, content_hash, dim, vector_blob)
                    SELECT nid, model_id, content_hash, dim, vector_blob FROM src_db.note_embeddings
                    ON CONFLICT(nid, model_id) DO UPDATE SET
                        content_hash=excluded.content_hash,
                        dim=excluded.dim,
                        vector_blob=excluded.vector_blob
                    """
                )
                embeddings_imported = cur_emb.rowcount
                conn.commit()
                return {"notes_imported": notes_imported, "embeddings_imported": embeddings_imported}
            finally:
                conn.execute("DETACH DATABASE src_db")
