"""Anki collection operations: sibling lookup, safe unsuspend, cid-based filtered deck creation, and rollback.

Adheres strictly to modern Anki (v23.10+ / v24+ / v25+ with FSRS) APIs and invariants:
- Zero temporary tags on Notes (zero note pollution, zero AnkiWeb/AnkiHub conflicts).
- Only cards suspended before the run are unsuspended.
- Rollback only re-suspends newly_unsuspended_cids and protects already-reviewed cards.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config_manager import ConfigManager
from .storage import RunRecord, SearchDatabase
from .text_utils import LectureUnit


@dataclass
class RunResult:
    """Result of creating a filtered deck from a search run."""

    run_id: str
    deck_name: str
    filtered_did: int
    total_cards_count: int
    newly_unsuspended_count: int
    newly_unsuspended_cids: List[int]
    already_active_count: int = 0
    already_active_cids: List[int] = field(default_factory=list)
    concept_coverage: List[Dict[str, Any]] = field(default_factory=list)
    unit_error: str = ""


@dataclass
class RollbackResult:
    """Result of rolling back a search run."""

    run_id: str
    deck_removed: bool
    resuspended_count: int
    kept_reviewed_count: int


def get_deck_ids_for_names(col: Any, deck_names: Sequence[str]) -> Set[int]:
    """Resolves deck IDs for selected deck names and all their subdecks."""
    target_dids: Set[int] = set()
    try:
        all_decks = col.decks.all()
    except Exception:
        all_decks = []
    if not all_decks and hasattr(col.decks, "all_names_and_ids"):
        all_decks = col.decks.all_names_and_ids()

    for deck in all_decks:
        d_name = deck.get("name", "") if isinstance(deck, dict) else getattr(deck, "name", "")
        d_id = deck.get("id") if isinstance(deck, dict) else getattr(deck, "id", None)
        if d_id is None:
            continue

        for sel in deck_names:
            sel_clean = sel.strip()
            if d_name == sel_clean or d_name.startswith(f"{sel_clean}::"):
                target_dids.add(int(d_id))
                break

    return target_dids


def resolve_notes_to_cards(
    col: Any,
    nids: Sequence[int],
    selected_deck_names: Sequence[str],
    allowed_cids: Optional[Set[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Resolves all sibling cards (c1, c2, ...) for target nids in selected decks or query.

    Partitions them by suspension status:
    Returns (suspended_cids, active_cids).
    """
    if not nids:
        return [], []

    valid_dids = get_deck_ids_for_names(col, selected_deck_names) if selected_deck_names else set()

    suspended_cids: List[int] = []
    active_cids: List[int] = []

    # Chunk queries to prevent query length limits
    chunk_size = 900
    for i in range(0, len(nids), chunk_size):
        chunk = nids[i : i + chunk_size]
        placeholders = ",".join(str(n) for n in chunk)
        # Select cards: home deck is odid if odid != 0 else did
        query = f"SELECT id, did, odid, queue FROM cards WHERE nid IN ({placeholders})"
        for cid, did, odid, queue in col.db.execute(query):
            if allowed_cids is not None:
                if int(cid) not in allowed_cids:
                    continue
            elif valid_dids:
                home_did = odid if (odid and odid != 0) else did
                if home_did not in valid_dids:
                    continue

            # queue == -1 means suspended
            if queue == -1:
                suspended_cids.append(cid)
            else:
                active_cids.append(cid)

    return suspended_cids, active_cids


def _format_target_deck_name(raw_deck_name: str, parent_deck_prefix: Optional[str]) -> str:
    """Prepends parent_deck_prefix:: to raw_deck_name if a single parent source deck is targeted."""
    clean_name = raw_deck_name.strip()
    if not parent_deck_prefix or not parent_deck_prefix.strip():
        return clean_name
    prefix = parent_deck_prefix.strip()
    if clean_name == prefix or clean_name.startswith(f"{prefix}::"):
        return clean_name
    return f"{prefix}::{clean_name}"


def execute_unit_deck_creation(
    col: Any,
    unit: LectureUnit,
    matched_nids: Sequence[int],
    config: ConfigManager,
    storage: SearchDatabase,
    concept_coverage: Optional[List[Dict[str, Any]]] = None,
    parent_deck_prefix: Optional[str] = None,
    allowed_cids: Optional[Set[int]] = None,
    unit_error: str = "",
) -> RunResult:
    """Unsuspends matching cards and builds a clean Filtered Deck with cid: search query."""
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    coverage_list = list(concept_coverage) if concept_coverage else []
    target_deck_name = _format_target_deck_name(unit.deck_name, parent_deck_prefix)

    if unit_error:
        record = RunRecord(
            run_id=run_id,
            created_at=int(time.time()),
            deck_name=target_deck_name,
            filtered_did=0,
            source_decks=[config.get_effective_search_query()] if hasattr(config, "get_effective_search_query") else list(config.selected_decks),
            unit_title=unit.unit_title,
            unit_preview=(unit.body_text[:150] + "...") if len(unit.body_text) > 150 else unit.body_text,
            total_matched_cids=[],
            newly_unsuspended_cids=[],
            rolled_back=0,
            already_active_cids=[],
            concept_coverage=[],
            unit_error=unit_error,
        )
        storage.save_run(record)
        return RunResult(
            run_id=run_id,
            deck_name=target_deck_name,
            filtered_did=0,
            total_cards_count=0,
            newly_unsuspended_count=0,
            newly_unsuspended_cids=[],
            already_active_count=0,
            already_active_cids=[],
            concept_coverage=[],
            unit_error=unit_error,
        )

    if not matched_nids:
        return RunResult(
            run_id=run_id,
            deck_name=target_deck_name,
            filtered_did=0,
            total_cards_count=0,
            newly_unsuspended_count=0,
            newly_unsuspended_cids=[],
            concept_coverage=coverage_list,
            unit_error="",
        )

    suspended_cids, active_cids = resolve_notes_to_cards(
        col,
        matched_nids,
        config.selected_decks,
        allowed_cids=allowed_cids,
    )

    # Determine cards to place in Filtered Deck
    if config.include_active_cards_in_filtered_deck:
        deck_cids = suspended_cids + active_cids
    else:
        deck_cids = list(suspended_cids)

    if not deck_cids:
        # Even if 0 cards are placed in the filtered deck (because all matched cards were already active),
        # return the count of already-active matched cards so the user gets explicit confirmation!
        if active_cids or coverage_list:
            record = RunRecord(
                run_id=run_id,
                created_at=int(time.time()),
                deck_name=target_deck_name,
                filtered_did=0,
                source_decks=[config.get_effective_search_query()] if hasattr(config, "get_effective_search_query") else list(config.selected_decks),
                unit_title=unit.unit_title,
                unit_preview=(unit.body_text[:150] + "...") if len(unit.body_text) > 150 else unit.body_text,
                total_matched_cids=[],
                newly_unsuspended_cids=[],
                rolled_back=0,
                already_active_cids=active_cids,
                concept_coverage=coverage_list,
            )
            storage.save_run(record)
        return RunResult(
            run_id=run_id,
            deck_name=target_deck_name,
            filtered_did=0,
            total_cards_count=0,
            newly_unsuspended_count=0,
            newly_unsuspended_cids=[],
            already_active_count=len(active_cids),
            already_active_cids=active_cids,
            concept_coverage=coverage_list,
        )

    # 1. Unsuspend only cards that were suspended prior to this run
    if suspended_cids:
        col.sched.unsuspend_cards(suspended_cids)

    # 2. Build or recreate the Filtered Deck with exact cid: query
    search_query = "cid:" + ",".join(str(cid) for cid in deck_cids)
    deck_name = target_deck_name

    # Safety check: if deck_name matches an existing NORMAL (non-filtered) deck
    # (e.g. user typed "AnKing Step Deck" as base deck name without markdown headers),
    # automatically append a child/unit name so we NEVER collide with a normal deck!
    existing_did = col.decks.id_for_name(deck_name)
    if existing_did:
        existing_deck = col.decks.get(existing_did)
        is_dynamic = (
            existing_deck.get("dyn", 0) == 1
            if isinstance(existing_deck, dict)
            else getattr(existing_deck, "dyn", 0) == 1
        )
        if not is_dynamic:
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            if config.batch_deck_naming_style == "subdeck" or parent_deck_prefix:
                deck_name = f"{deck_name}::Smart Search {timestamp}"
            else:
                deck_name = f"{deck_name} - Smart Search {timestamp}"
            existing_did = col.decks.id_for_name(deck_name)

    if existing_did:
        filtered_did = existing_did
    else:
        filtered_did = col.decks.new_filtered(deck_name)

    deck = col.decks.get(filtered_did)
    limit = len(deck_cids)

    # Handle modern dictionary or protobuf DeckConfig representation
    if isinstance(deck, dict):
        deck["resched"] = True
        # terms is a list of [search_string, limit, order]
        deck["terms"] = [[search_query, limit, 0]]
        col.decks.save(deck)
    elif hasattr(deck, "terms"):
        deck.resched = True
        if not deck.terms:
            term = deck.terms.add()
        else:
            term = deck.terms[0]
        term.search = search_query
        term.limit = limit
        term.order = 0
        col.decks.save(deck)

    # Rebuild the filtered deck so cards are pulled into it
    col.sched.rebuild_filtered_deck(filtered_did)

    # 3. Save run history record
    record = RunRecord(
        run_id=run_id,
        created_at=int(time.time()),
        deck_name=deck_name,
        filtered_did=filtered_did,
        source_decks=[config.get_effective_search_query()] if hasattr(config, "get_effective_search_query") else list(config.selected_decks),
        unit_title=unit.unit_title,
        unit_preview=(unit.body_text[:150] + "...") if len(unit.body_text) > 150 else unit.body_text,
        total_matched_cids=deck_cids,
        newly_unsuspended_cids=suspended_cids,
        rolled_back=0,
        already_active_cids=active_cids,
        concept_coverage=coverage_list,
    )
    storage.save_run(record)

    return RunResult(
        run_id=run_id,
        deck_name=deck_name,
        filtered_did=filtered_did,
        total_cards_count=len(deck_cids),
        newly_unsuspended_count=len(suspended_cids),
        newly_unsuspended_cids=suspended_cids,
        already_active_count=len(active_cids),
        already_active_cids=active_cids,
        concept_coverage=coverage_list,
    )


def rollback_run(
    col: Any,
    storage: SearchDatabase,
    run_id: str,
    keep_reviewed: bool = True,
) -> RollbackResult:
    """Safely undoes a run:

    1. Empties and deletes the Filtered Deck so cards return to their home decks.
    2. Re-suspends newly_unsuspended_cids that have not been reviewed (if keep_reviewed=True).
    3. Guarantees that cards active before the run are NEVER touched!
    """
    record = storage.get_run(run_id)
    if not record or record.rolled_back:
        return RollbackResult(
            run_id=run_id,
            deck_removed=False,
            resuspended_count=0,
            kept_reviewed_count=0,
        )

    # 1. Empty and remove the Filtered Deck first (strictly verify dyn == 1!)
    deck_removed = False
    if record.filtered_did and col.decks.have(record.filtered_did):
        try:
            d_obj = col.decks.get(record.filtered_did)
            is_dyn = (
                d_obj.get("dyn", 1) == 1
                if isinstance(d_obj, dict)
                else getattr(d_obj, "dyn", 1) == 1
            )
            if is_dyn:
                col.sched.empty_filtered_deck(record.filtered_did)
                col.decks.remove([record.filtered_did])
                deck_removed = True
        except Exception as e:
            print(f"[SmartSearch] Notice: could not remove filtered deck {record.filtered_did}: {e}")

    # 2. Check which cards to re-suspend
    candidate_cids = list(record.newly_unsuspended_cids)
    if not candidate_cids:
        storage.mark_run_rolled_back(run_id)
        return RollbackResult(
            run_id=run_id,
            deck_removed=deck_removed,
            resuspended_count=0,
            kept_reviewed_count=0,
        )

    cids_to_resuspend: List[int] = []
    kept_count = 0

    chunk_size = 900
    for i in range(0, len(candidate_cids), chunk_size):
        chunk = candidate_cids[i : i + chunk_size]
        placeholders = ",".join(str(c) for c in chunk)

        if keep_reviewed:
            # Check reps in cards and revlog entries since run creation in a single batch query
            run_time_ms = record.created_at * 1000
            rev_query = f"SELECT DISTINCT cid FROM revlog WHERE cid IN ({placeholders}) AND id >= {run_time_ms}"
            reviewed_in_revlog: Set[int] = {int(r_cid) for (r_cid,) in col.db.execute(rev_query)}

            query = f"""
                SELECT id, reps FROM cards
                WHERE id IN ({placeholders})
            """
            for cid, reps in col.db.execute(query):
                if reps > 0 or int(cid) in reviewed_in_revlog:
                    kept_count += 1
                    continue

                cids_to_resuspend.append(cid)
        else:
            # Re-suspend all that still exist
            query = f"SELECT id FROM cards WHERE id IN ({placeholders})"
            for (cid,) in col.db.execute(query):
                cids_to_resuspend.append(cid)

    # 3. Suspend only the verified cards
    if cids_to_resuspend:
        col.sched.suspend_cards(cids_to_resuspend)

    # 4. Mark run rolled back
    storage.mark_run_rolled_back(run_id)

    return RollbackResult(
        run_id=run_id,
        deck_removed=deck_removed,
        resuspended_count=len(cids_to_resuspend),
        kept_reviewed_count=kept_count,
    )


def delete_filtered_deck_only_op(
    col: Any,
    storage: SearchDatabase,
    run_id: str,
) -> Tuple[bool, Any]:
    """Empties and removes a filtered deck within a custom undo entry.

    Returns (deck_removed, op_changes).
    """
    undo_pos = None
    if hasattr(col, "add_custom_undo_entry"):
        try:
            undo_pos = col.add_custom_undo_entry("Smart Search")
        except Exception:
            undo_pos = None

    record = storage.get_run(run_id)
    deck_removed = False
    if record and record.filtered_did and col.decks.have(record.filtered_did):
        d_obj = col.decks.get(record.filtered_did)
        is_dyn = (
            d_obj.get("dyn", 1) == 1
            if isinstance(d_obj, dict)
            else getattr(d_obj, "dyn", 1) == 1
        )
        if is_dyn:
            col.sched.empty_filtered_deck(record.filtered_did)
            col.decks.remove([record.filtered_did])
            deck_removed = True

    changes = None
    if undo_pos is not None and hasattr(col, "merge_undo_entries"):
        try:
            changes = col.merge_undo_entries(undo_pos)
        except Exception:
            changes = None

    return deck_removed, changes


def delete_filtered_deck_only(col: Any, storage: SearchDatabase, run_id: str) -> bool:
    """Empties and removes the filtered deck without modifying card suspension states.

    Handy shortcut for students who want to keep all unsuspended cards in their main decks.
    """
    success, changes = delete_filtered_deck_only_op(col, storage, run_id)
    if success and changes is None:
        try:
            from aqt import mw  # type: ignore

            if mw:
                mw.reset()
        except Exception:
            pass
    return success


def _get_op_changes(changes: Any) -> Any:
    """Ensures an OpChanges object is returned for CollectionOp."""
    if changes is not None:
        return changes
    try:
        from anki.collection import OpChanges  # type: ignore

        ch = OpChanges()
        ch.deck = True
        ch.card = True
        return ch
    except Exception:
        pass
    return None


def create_batch_filtered_decks_op(
    col: Any,
    units_data: List[Dict[str, Any]],
    config: ConfigManager,
    storage: SearchDatabase,
    parent_deck_prefix: Optional[str] = None,
    allowed_cids: Optional[Set[int]] = None,
) -> Tuple[List[RunResult], Any]:
    """Applies deck creation and card unsuspension for multiple units in a single custom undo entry.

    Returns (results, op_changes).
    """
    undo_pos = None
    if hasattr(col, "add_custom_undo_entry"):
        try:
            undo_pos = col.add_custom_undo_entry("Smart Search")
        except Exception:
            undo_pos = None

    results: List[RunResult] = []
    for item in units_data:
        run_res = execute_unit_deck_creation(
            col=col,
            unit=item["unit"],
            matched_nids=item["matched_nids"],
            config=config,
            storage=storage,
            concept_coverage=item.get("concept_coverage"),
            parent_deck_prefix=parent_deck_prefix,
            allowed_cids=allowed_cids,
            unit_error=item.get("unit_error", ""),
        )
        results.append(run_res)

    changes = None
    if undo_pos is not None and hasattr(col, "merge_undo_entries"):
        try:
            changes = col.merge_undo_entries(undo_pos)
        except Exception:
            changes = None

    return results, changes


def rollback_run_op(
    col: Any,
    storage: SearchDatabase,
    run_id: str,
    keep_reviewed: bool = True,
) -> Tuple[RollbackResult, Any]:
    """Rolls back a run within a custom undo entry.

    Returns (rollback_result, op_changes).
    """
    undo_pos = None
    if hasattr(col, "add_custom_undo_entry"):
        try:
            undo_pos = col.add_custom_undo_entry("Smart Search Rollback")
        except Exception:
            undo_pos = None

    res = rollback_run(
        col=col,
        storage=storage,
        run_id=run_id,
        keep_reviewed=keep_reviewed,
    )

    changes = None
    if undo_pos is not None and hasattr(col, "merge_undo_entries"):
        try:
            changes = col.merge_undo_entries(undo_pos)
        except Exception:
            changes = None

    return res, changes
