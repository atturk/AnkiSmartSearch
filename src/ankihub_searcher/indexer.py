"""Incremental background note indexer for Smart Search.

Indexes notes from selected decks:
- Cleans and extracts text and tags.
- Upserts SQLite content and FTS5.
- Computes dense embeddings for the active model in batches with instant commit.
- Supports pause, resume, and cancellation callbacks.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .ai_client import OpenAICompatibleClient
from .config_manager import ConfigManager
from .storage import SearchDatabase
from .text_utils import clean_html_and_cloze, clean_tags


class DeckIndexer:
    """Manages incremental indexing of Anki deck collections."""

    def __init__(
        self,
        config: ConfigManager,
        storage: SearchDatabase,
        ai_client: OpenAICompatibleClient,
    ):
        self.config = config
        self.storage = storage
        self.ai_client = ai_client

    def find_target_deck_ids(self, col: Any, selected_deck_names: List[str]) -> List[int]:
        """Finds all deck IDs matching selected deck names or their subdecks."""
        try:
            all_decks = col.decks.all()
        except Exception:
            all_decks = []
        if not all_decks and hasattr(col.decks, "all_names_and_ids"):
            all_decks = col.decks.all_names_and_ids()

        target_dids: List[int] = []

        for deck in all_decks:
            d_name = deck.get("name", "") if isinstance(deck, dict) else getattr(deck, "name", "")
            d_id = deck.get("id") if isinstance(deck, dict) else getattr(deck, "id", None)
            if d_id is None:
                continue
            for sel in selected_deck_names:
                sel_clean = sel.strip()
                if d_name == sel_clean or d_name.startswith(f"{sel_clean}::"):
                    target_dids.append(int(d_id))
                    break

        return target_dids

    def get_notes_in_decks(self, col: Any, dids: List[int]) -> List[int]:
        """Returns all distinct note IDs belonging to the given deck IDs."""
        if not dids:
            return []
        placeholders = ",".join(str(d) for d in dids)
        # Cards can have did or odid (if in filtered deck)
        query = f"SELECT DISTINCT nid FROM cards WHERE did IN ({placeholders}) OR odid IN ({placeholders})"
        cur = col.db.execute(query)
        return [row[0] for row in cur]

    def get_notes_with_suspended_cards(self, col: Any, dids: List[int]) -> Set[int]:
        """Returns note IDs that have at least one suspended card (queue == -1) in the given decks."""
        if not dids:
            return set()
        placeholders = ",".join(str(d) for d in dids)
        query = (
            f"SELECT DISTINCT nid FROM cards "
            f"WHERE queue = -1 AND (did IN ({placeholders}) OR odid IN ({placeholders}))"
        )
        cur = col.db.execute(query)
        return {int(row[0]) for row in cur}

    @staticmethod
    def extract_deck_names_from_query(query: str) -> List[str]:
        """Extracts deck names from Anki Browser query syntax (e.g., '\"deck:AnKing Step Deck\"' or 'deck:\"My Deck\"')."""
        if not query or not query.strip():
            return []
        pattern = re.compile(
            r'"deck:([^"]+)"|deck:"([^"]+)"|\'deck:([^\']+)\'|deck:\'([^\']+)\'|\bdeck:([^\s)]+)',
            re.IGNORECASE,
        )
        decks: List[str] = []
        for match in pattern.finditer(query):
            raw = next((g for g in match.groups() if g is not None), "").strip()
            while raw.endswith("::*") or raw.endswith("*"):
                if raw.endswith("::*"):
                    raw = raw[:-3].strip()
                elif raw.endswith("*"):
                    raw = raw[:-1].strip()
            if raw and raw not in decks:
                decks.append(raw)
        return decks

    @staticmethod
    def _common_deck_ancestor(deck_names: List[str]) -> Optional[str]:
        """Finds the deepest common deck ancestor across a list of deck names, or None if disjoint."""
        cleaned = [d.strip() for d in deck_names if d and d.strip()]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        split_paths = [d.split("::") for d in cleaned]
        common: List[str] = []
        for parts in zip(*split_paths):
            if len(set(parts)) == 1:
                common.append(parts[0])
            else:
                break
        return "::".join(common) if common else None

    def get_cards_for_query(self, col: Any, query: str) -> List[int]:
        """Returns card IDs matching an Anki Browser query."""
        q = (query or "").strip()
        if not q:
            return []
        if hasattr(col, "find_cards"):
            return [int(cid) for cid in col.find_cards(q)]
        # Fallback for mock collections in tests
        deck_names = self.extract_deck_names_from_query(q) or self.config.selected_decks
        dids = self.find_target_deck_ids(col, deck_names)
        if not dids:
            return []
        placeholders = ",".join(str(d) for d in dids)
        cur = col.db.execute(f"SELECT id FROM cards WHERE did IN ({placeholders}) OR odid IN ({placeholders})")
        return [int(row[0]) for row in cur]

    def get_notes_for_query(self, col: Any, query: str) -> List[int]:
        """Returns distinct note IDs matching an Anki Browser query."""
        q = (query or "").strip()
        if not q:
            return []
        if hasattr(col, "find_notes"):
            return [int(nid) for nid in col.find_notes(q)]
        # Fallback for mock collections in tests
        deck_names = self.extract_deck_names_from_query(q) or self.config.selected_decks
        dids = self.find_target_deck_ids(col, deck_names)
        return self.get_notes_in_decks(col, dids)

    def get_notes_with_suspended_cards_for_query(self, col: Any, query: str) -> Set[int]:
        """Returns note IDs matching the query that have at least one suspended card."""
        q = (query or "").strip()
        if not q:
            return set()
        if hasattr(col, "find_notes"):
            try:
                return {int(nid) for nid in col.find_notes(f"({q}) is:suspended")}
            except Exception:
                pass
        deck_names = self.extract_deck_names_from_query(q) or self.config.selected_decks
        dids = self.find_target_deck_ids(col, deck_names)
        return self.get_notes_with_suspended_cards(col, dids)

    def detect_single_parent_deck(self, col: Any, query: str) -> Optional[str]:
        """Determines if the search query targets a single source deck (or subdeck hierarchy).

        Returns the single parent deck name if all targeted cards belong to one deck tree,
        or None if multiple unrelated top-level decks are targeted.
        """
        q = (query or "").strip()
        if not q:
            return None

        explicit_decks = self.extract_deck_names_from_query(q)
        if explicit_decks:
            return self._common_deck_ancestor(explicit_decks)

        # If no explicit deck: clause in query (e.g., tag-only query), inspect matched cards' home decks
        try:
            cids = self.get_cards_for_query(col, q)
            if not cids:
                return None
            sample_cids = cids[:2000]
            placeholders = ",".join(str(c) for c in sample_cids)
            cur = col.db.execute(
                f"SELECT DISTINCT CASE WHEN odid != 0 THEN odid ELSE did END FROM cards WHERE id IN ({placeholders})"
            )
            home_dids = [int(row[0]) for row in cur if row[0]]
            if not home_dids:
                return None

            deck_names: List[str] = []
            for did in home_dids:
                d_obj = col.decks.get(did)
                if d_obj:
                    d_name = d_obj.get("name", "") if isinstance(d_obj, dict) else getattr(d_obj, "name", "")
                    if d_name:
                        deck_names.append(d_name)
            return self._common_deck_ancestor(deck_names)
        except Exception:
            return None

    def scan_notes_metadata(
        self,
        col: Any,
        nids: List[int],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[int, Tuple[int, str]], Dict[int, Tuple[int, int, str, str, str]]]:
        """Scans notes metadata and text from Anki collection (safe to run on main thread or QueryOp).

        Returns:
            (notes_meta, notes_raw_data)
        """
        if not nids:
            return {}, {}

        # Map model IDs to model structures
        models_by_mid: Dict[int, Any] = {}
        try:
            for item in col.models.all_names_and_ids():
                mid_val = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                if mid_val is not None:
                    m = col.models.get(mid_val)
                    if m:
                        models_by_mid[int(m["id"])] = m
        except Exception:
            if hasattr(col.models, "all"):
                for m in col.models.all():
                    if isinstance(m, dict) and "id" in m:
                        models_by_mid[int(m["id"])] = m

        notes_meta: Dict[int, Tuple[int, str]] = {}
        notes_raw_data: Dict[int, Tuple[int, int, str, str, str]] = {}

        # Fast-path: load stored (note_mod, content_hash) so unchanged notes skip regex HTML cleaning
        stored_mod_map = self.storage.get_stored_notes_mod_map()

        # Fetch in chunks to avoid overwhelming memory
        chunk_size = 1000
        for i in range(0, len(nids), chunk_size):
            if cancel_check and cancel_check():
                break

            chunk = nids[i : i + chunk_size]
            placeholders = ",".join(str(x) for x in chunk)

            # First check which notes in this chunk are new or have a modified `mod` timestamp
            dirty_nids: List[int] = []
            mod_query = f"SELECT id, mod FROM notes WHERE id IN ({placeholders})"
            for nid, note_mod in col.db.execute(mod_query):
                saved = stored_mod_map.get(nid)
                if saved is not None and saved[0] == note_mod:
                    notes_meta[nid] = saved
                else:
                    dirty_nids.append(nid)

            if not dirty_nids:
                continue

            dirty_placeholders = ",".join(str(x) for x in dirty_nids)
            query = f"SELECT id, mid, mod, flds, tags FROM notes WHERE id IN ({dirty_placeholders})"
            for nid, mid, note_mod, flds_str, tags_str in col.db.execute(query):
                model = models_by_mid.get(mid)
                model_name = model["name"] if model else ""
                available_field_names = [fld["name"] for fld in model["flds"]] if model else []

                # Split fields by Anki delimiter \x1f
                raw_fields = flds_str.split("\x1f") if flds_str else []
                field_dict = {
                    name: raw_fields[idx] if idx < len(raw_fields) else ""
                    for idx, name in enumerate(available_field_names)
                }

                # Select fields to index based on configuration
                fields_to_index = self.config.get_indexed_fields_for_model(model_name, available_field_names)
                cleaned_field_parts = []
                for fname in fields_to_index:
                    if fname in field_dict:
                        cleaned_val = clean_html_and_cloze(field_dict[fname])
                        if cleaned_val:
                            cleaned_field_parts.append(cleaned_val)

                clean_text = "\n".join(cleaned_field_parts).strip()

                # Clean tags
                clean_tags_str = ""
                if self.config.index_tags:
                    raw_tags = tags_str.strip().split() if tags_str else []
                    clean_tags_str = clean_tags(raw_tags)

                content_repr = f"{clean_text}\n{clean_tags_str}"
                content_hash = hashlib.sha256(content_repr.encode("utf-8")).hexdigest()

                notes_meta[nid] = (note_mod, content_hash)
                notes_raw_data[nid] = (mid, note_mod, content_hash, clean_text, clean_tags_str)

        return notes_meta, notes_raw_data

    def index_selected_decks(
        self,
        col: Any,
        selected_deck_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        batch_size: int = 32,
        search_query: Optional[str] = None,
        precomputed_notes_data: Optional[Tuple[Dict[int, Tuple[int, str]], Dict[int, Tuple[int, int, str, str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Runs incremental indexing on notes from a Browser search query or selected decks.

        Commits embeddings after every batch to guarantee instant pause/resume safety.
        """
        if precomputed_notes_data is not None:
            notes_meta, notes_raw_data = precomputed_notes_data
            nids = list(notes_meta.keys())
        else:
            if search_query and search_query.strip():
                nids = self.get_notes_for_query(col, search_query.strip())
            elif selected_deck_names is not None:
                target_dids = self.find_target_deck_ids(col, selected_deck_names)
                nids = self.get_notes_in_decks(col, target_dids)
            elif self.config.search_query and self.config.search_query.strip() and hasattr(col, "find_notes"):
                nids = self.get_notes_for_query(col, self.config.search_query.strip())
            else:
                decks_to_index = self.config.selected_decks
                target_dids = self.find_target_deck_ids(col, decks_to_index)
                nids = self.get_notes_in_decks(col, target_dids)

            if not nids:
                if progress_callback:
                    progress_callback(0, 0, "Nessuna nota trovata nei mazzi selezionati.")
                return {"total": 0, "text_indexed": 0, "embeddings_indexed": 0, "cancelled": False}

            # Step 1: Read notes metadata from Anki
            if progress_callback:
                progress_callback(0, len(nids), f"Scansione di {len(nids)} note nei mazzi selezionati...")

            notes_meta, notes_raw_data = self.scan_notes_metadata(col, nids, cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return {"total": len(nids), "text_indexed": 0, "embeddings_indexed": 0, "cancelled": True}

        active_model_id = self.config.get_active_model_id()
        needs_text_nids, needs_embed_nids = self.storage.get_notes_needing_indexing(notes_meta, active_model_id)

        # Step 2: Batch update text and FTS
        if needs_text_nids:
            if progress_callback:
                progress_callback(
                    0,
                    len(needs_text_nids),
                    f"Aggiornamento testo ed indice FTS per {len(needs_text_nids)} note...",
                )

            text_batch = []
            for nid in needs_text_nids:
                if nid in notes_raw_data:
                    mid, note_mod, content_hash, clean_text, clean_tags_str = notes_raw_data[nid]
                    text_batch.append((nid, mid, note_mod, content_hash, clean_text, clean_tags_str))

            # Commit text updates in chunks of 500
            for i in range(0, len(text_batch), 500):
                self.storage.upsert_notes_content_batch(text_batch[i : i + 500])

        # Step 3: Compute missing embeddings for active model
        total_to_embed = len(needs_embed_nids)
        embedded_count = 0

        if total_to_embed > 0:
            # We need the clean text for these notes
            notes_content = self.storage.get_notes_content(needs_embed_nids)

            for i in range(0, total_to_embed, batch_size):
                if cancel_check and cancel_check():
                    return {
                        "total": len(nids),
                        "text_indexed": len(needs_text_nids),
                        "embeddings_indexed": embedded_count,
                        "cancelled": True,
                    }

                batch_nids = needs_embed_nids[i : i + batch_size]
                texts_for_ai: List[str] = []
                valid_batch_meta: List[Tuple[int, str, str]] = []

                for nid in batch_nids:
                    note_info = notes_content.get(nid)
                    if not note_info:
                        continue
                    c_text = note_info["clean_text"]
                    c_tags = note_info["clean_tags"]
                    c_hash = note_info["content_hash"]

                    # Truncate to reasonable character limit (e.g., 2000 chars) for embeddings
                    combined = f"{c_text}\n{c_tags}".strip()
                    if len(combined) > 2000:
                        combined = combined[:2000]

                    texts_for_ai.append(combined or "empty")
                    valid_batch_meta.append((nid, active_model_id, c_hash))

                if texts_for_ai:
                    vectors = self.ai_client.get_embeddings(
                        texts=texts_for_ai,
                        model=self.config.embed_model,
                        base_url=self.config.embed_base_url,
                        api_key=self.config.embed_api_key,
                        batch_size=batch_size,
                    )

                    save_batch = []
                    for (nid, model_id, chash), vec in zip(valid_batch_meta, vectors):
                        save_batch.append((nid, model_id, chash, len(vec), vec))

                    # Instant commit to SQLite!
                    self.storage.save_embeddings_batch(save_batch)
                    embedded_count += len(save_batch)

                if progress_callback:
                    pct = int(embedded_count / total_to_embed * 100)
                    progress_callback(
                        embedded_count,
                        total_to_embed,
                        f"Indicizzazione vettori: {embedded_count}/{total_to_embed} ({pct}%)",
                    )

        return {
            "total": len(nids),
            "text_indexed": len(needs_text_nids),
            "embeddings_indexed": embedded_count,
            "cancelled": False,
        }


def collect_presearch_data(
    col: Any,
    config: ConfigManager,
    storage: SearchDatabase,
    ai_client: OpenAICompatibleClient,
    search_query: Optional[str] = None,
) -> Dict[str, Any]:
    """Reads all necessary Anki collection data on the main thread before starting background search."""
    indexer = DeckIndexer(config, storage, ai_client)
    q = search_query if search_query is not None else config.get_effective_search_query()
    candidate_nids = indexer.get_notes_for_query(col, q)
    suspended_nids = indexer.get_notes_with_suspended_cards_for_query(col, q)
    allowed_cids = set(indexer.get_cards_for_query(col, q))
    parent_deck_prefix = indexer.detect_single_parent_deck(col, q)
    notes_meta, notes_raw_data = indexer.scan_notes_metadata(col, candidate_nids)
    return {
        "candidate_nids": candidate_nids,
        "suspended_nids": suspended_nids,
        "allowed_cids": allowed_cids,
        "parent_deck_prefix": parent_deck_prefix,
        "precomputed_notes_data": (notes_meta, notes_raw_data),
    }
