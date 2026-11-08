# Changelog

All notable changes to **AnkiSmartSearch (AnkiHubSearcher)** are documented in this file.

## [0.1.0] - 2026-11-08

### Added
- **Multi-Stage AI Hybrid Search Pipeline**:
  - **Stage 1 (Cross-Lingual Concept Decomposition):** Decomposes Italian lecture transcripts/notes into atomic USMLE/AnKing-grade English claims and medical keywords via local LLMs (**LM Studio**) or cloud APIs (**OpenRouter** / OpenAI-compatible).
  - **Stage 2 (Hybrid Dense + Lexical FTS5 BM25 Retrieval):** Combines model-scoped dense cosine similarity with SQLite FTS5 BM25 keyword search and multi-concept hit boosting.
  - **Stage 3 (Optional LLM Verification):** Filters candidate flashcards via LLM verification to eliminate false positives.
- **Markdown Batch Mode:** Automatically splits multi-section Markdown notes (`#`, `##`, `###`) into separate Filtered Decks while ignoring empty parent headers and preserving breadcrumb context. Supports both `Base - Unit` (Option A) and subdeck `Base::Unit` (Option B) naming styles.
- **Safe Card Unsuspension & `cid:` Filtered Deck Engine:**
  - Unsuspends only cards that were previously suspended (`queue == -1`).
  - Excludes already-active cards from Filtered Decks by default (configurable).
  - Uses pure `cid:` search queries (`resched = True`) with zero temporary note tags (preventing AnkiHub / AnkiWeb conflicts) and normal-deck collision protection.
- **Run History & Safe Rollback ("Review Limbo" Workflow):**
  - Tracks exact `newly_unsuspended_cids` per run.
  - Supports 1-click Rollback with optional protection for cards already reviewed (`keep_reviewed`).
  - Includes a shortcut to delete only the Filtered Deck while keeping unsuspended cards active in their home decks.
- **Incremental Model-Scoped Vector Indexer:**
  - Instant-commit batch indexing (32 notes/batch) with Pause/Resume support.
  - Per-Note-Type customizable field indexing and Smart Tag Cleaner for AnKing v12 / Mastering Radiology boilerplate removal.
