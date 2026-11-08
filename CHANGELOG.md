# Changelog

All notable changes to **Smart Search (AnkiSmartSearch)** are documented in this file.

## [0.2.0] - 2026-11-08

### Added
- **Rebranding to "Smart Search":** Renamed add-on across menus, window titles, and `manifest.json`, removing legacy "AnkiHub" references while automatically migrating existing databases (`ankihub_searcher.db` -> `smart_search.db`) without data loss.
- **`Tools -> Smart Search` Submenu Architecture:**
  - `Cerca...` (`Ctrl+Shift+F` / `⇧⌘F`)
  - `Cronologia`
  - `Indicizzazione`
  - `Impostazioni`
  - `Verifica aggiornamenti`
- **Minimal, Native-Styled UI Redesign (4 Independent Dialogs):**
  - **Compact `SearchDialog` (`520x460`):** Features a collapsible hierarchical `QTreeWidget` showing parent decks collapsed by default (`▸`) with tri-state checkbox propagation to subdecks, optional base deck name input, lecture text box, and native `"Annulla"` / `"Cerca"` buttons.
  - **Auto-Close & Direct `"Browser"` Button:** Automatically closes the search window upon completion and shows a summary dialog with `"OK"` and `"Browser"` buttons (`Browser` opens Anki's Card Browser filtered directly to the newly created deck).
  - **Dedicated `HistoryDialog`, `IndexDialog`, and `SettingsDialog`:** Clean, emoji-free native Qt layouts.
- **GitHub Auto-Updater (`Verifica aggiornamenti`):**
  - Queries `https://api.github.com/repos/atturk/AnkiSmartSearch/releases/latest`, displays release notes, and downloads/installs `.ankiaddon` updates directly inside Anki while strictly preserving `user_files/` (local SQLite embeddings database) and `meta.json` (user settings).

---

## [0.1.1] - 2026-11-08

### Added
- **Already-Active Cards Counter & Report (Default ON):** Every search run now reports both **newly unsuspended cards** and **correlated cards that were already active (unsuspended)** in your collection — including a dedicated notification when 100% of relevant cards for a lecture are already active in your study queue, plus a new `"Già Attive"` column in the *Cronologia & Rollback* table.
- **Top-K Slot Protection for New Cards:** Already-active cards no longer consume `top_k_per_concept` or `max_cards_per_unit` slots from suspended cards when *"Inserisci nel mazzo filtrato anche le carte già attive"* is unchecked.
- **1-Click Multi-Mac Database Merger (`📥 Importa / Unisci Indice da altro .db...`):** Merge `notes_content` and `note_embeddings` across different machines/models (e.g., combining OpenAI embeddings from MacBook Air with Qwen3 embeddings from MacBook Pro) directly from the *Gestione Indice* tab.

### Performance
- **Ultra-Fast Pre-Search Incremental Check (`0.02s`):** `DeckIndexer` now compares `notes.mod` timestamps against stored metadata before running HTML/Cloze/Tag regex cleaning, cutting pre-search verification on 28,600+ notes from ~2.5s down to ~20ms while still automatically re-indexing and re-embedding any note modified by the user or AnkiHub.

---

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
