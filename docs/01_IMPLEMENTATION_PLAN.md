# AnkiHubSearcher — Step-by-Step Implementation Plan (For Executor LLM)

> **Instructions for Executor LLM:**
> Read `docs/00_ARCHITECTURE_AND_SPECS.md` first. Implement the project strictly following the phases below. All code must be compatible with **Python 3.9–3.12+** and **Anki (PyQt6 / `aqt`)** with **zero external `pip` dependencies** (use standard library `urllib.request`, `sqlite3`, `json`, `struct`, `math`, `re`, `html`, `hashlib`, `concurrent.futures`, and optional `numpy` with a pure-Python fallback).
>
> **CRITICAL SAFETY RULE:** Never mutate the live Anki collection on port `3343` during automated tests. Any test talking to `127.0.0.1:3343` must strictly use read-only actions (`version`, `deckNamesAndIds`, `modelNamesAndIds`, `findNotes`, `findCards`, `notesInfo`). Unit tests for `storage.py`, `text_utils.py`, `search_pipeline.py`, and `deck_ops.py` must use temporary SQLite databases and mocked `mw.col` objects.

---

## Phase 1: Add-on Skeleton, Config Manager & Text/Markdown Utilities

### Files to Create
1. `src/ankihub_searcher/manifest.json`
2. `src/ankihub_searcher/config.json`
3. `src/ankihub_searcher/config_manager.py`
4. `src/ankihub_searcher/text_utils.py`
5. `tests/test_text_utils.py`

### Detailed Requirements

#### 1.1 `config.json` & `config_manager.py`
Provide clean defaults and helper methods to load/save config (supporting both standalone execution in unit tests and `mw.addonManager.getConfig` / `writeConfig` inside Anki):
```json
{
  "selected_decks": ["AnKing Step Deck", "Mastering Radiology"],
  "chat_provider": "lm_studio",
  "chat_base_url": "http://127.0.0.1:1234/v1",
  "chat_api_key": "",
  "chat_model": "",
  "embed_provider": "lm_studio",
  "embed_base_url": "http://127.0.0.1:1234/v1",
  "embed_api_key": "",
  "embed_model": "",
  "embed_language_strategy": "translate_to_en",
  "sensitivity": 0.55,
  "top_k_per_concept": 8,
  "max_cards_per_unit": 80,
  "hybrid_dense_weight": 0.78,
  "enable_llm_verification": false,
  "include_active_cards_in_filtered_deck": false,
  "batch_deck_naming_style": "dash",
  "rollback_keep_reviewed_default": true,
  "indexed_fields_by_model": {
    "AnKingOverhaul (AnKing Step Deck / AnKingMed)": ["Text", "Extra"],
    "Mastering Radiology (AnkiCollab) (Mastering Radiology / drradiology)-a1ef2": ["Text", "Extra"],
    "Mastering Radiology (Mastering Radiology / drradiology)": ["Text", "Extra"]
  },
  "default_indexed_field_names": ["Text", "Extra", "Front", "Back", "Back Extra", "Header", "Footer", "Remarks", "Answer"],
  "index_tags": true
}
```
- Note on `sensitivity` (float `0.0` to `1.0`, default `0.55`):
  - Higher sensitivity (`-> 1.0`, "Ampio / Recall") lowers the minimum cosine similarity cutoff (e.g., from `0.62` down to `0.38`) and increases effective `top_k_per_concept`.
  - Lower sensitivity (`-> 0.0`, "Preciso / Precision") raises the minimum cosine similarity cutoff.

#### 1.2 `text_utils.py`
Implement and unit-test:
- `clean_html_and_cloze(raw_html: str) -> str`:
  - Remove `<style>...</style>`, `<script>...</script>`, `<img[^>]*>`, `<audio[^>]*>`, `[sound:[^\]]+]`.
  - Remove external promo links like `<a [^>]*>Watch associated Bootcamp video[^<]*</a>` or `"Watch Gout & Pseudogout..."`.
  - Replace `<br\s*/?>`, `</div>`, `</p>`, `</li>` with newlines/spaces so words across tags don't concatenate.
  - Unwrap cloze syntax `{{c\d+::(.*?)(?:::(.*?))?}}` → if hint exists and is non-trivial, `f"{answer} ({hint})"`, else `f"{answer}"`.
  - Strip remaining HTML tags `<[^>]+>`, unescape HTML entities (`html.unescape`), and normalize whitespace.
- `clean_tags(tags: list[str]) -> str`:
  - Split each tag by `::`.
  - Strip leading `#`, `*`, `^`, `!`.
  - Ignore boilerplate segments matching case-insensitively:
    `ak_step1_v12`, `ak_step2_v12`, `ak_step3_v12`, `ak_other`, `ak_original_decks`, `anking_image`, `only_step_1`, `only_step_2`, `expn`, `amboss`, `low/highyield`, `highyield`, `lowyield`, `contributors`, `drradiology`, `ankore`, `physeo_image_update`, `subjects`, `systems`, `textbooks`, or segments that are purely random codes / numbers.
  - Strip leading numeric prefixes like `01_`, `02_`, `18_` from segments (e.g., `04_Acute_inflammation` → `Acute inflammation`).
  - Deduplicate while preserving order, join with `, `.
- `parse_markdown_units(raw_text: str, base_deck_name: str = "", naming_style: str = "dash") -> list[LectureUnit]`:
  - `LectureUnit` dataclass: `unit_title: str`, `deck_name: str`, `body_text: str`, `breadcrumb: str`.
  - Handle single-unit text (no `#` headers) and multi-unit Markdown text (`#`, `##`, `###`...).
  - **Empty Header Handling:** If a header is followed immediately by another header (or only whitespace/separators `---`), do **not** emit a `LectureUnit` for the empty header; store its title in a hierarchy stack so child headers get `breadcrumb` context (e.g. `"Capitolo 2 > 2.3 Segnali endogeni"`), while the created deck name uses the leaf header title (`2.3 Segnali endogeni`).
  - Raise a clear validation error or return `[]` if the entire input has no body text at all (e.g. user accidentally pasted only `# Header` with zero content).

---

## Phase 2: SQLite Storage Engine & Zero-Dependency AI Client

### Files to Create
1. `src/ankihub_searcher/storage.py`
2. `src/ankihub_searcher/ai_client.py`
3. `tests/test_storage_and_search.py`

### Detailed Requirements

#### 2.1 `storage.py` (`SearchDatabase`)
- Manages SQLite connection at `<addon_dir>/user_files/ankihub_searcher.db` (or a custom path for tests).
- Enables WAL mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).
- Tables:
  1. `notes_content(nid INTEGER PRIMARY KEY, mid INTEGER, note_mod INTEGER, content_hash TEXT, clean_text TEXT, clean_tags TEXT)`
  2. `notes_fts USING fts5(nid UNINDEXED, clean_text, clean_tags, tokenize='unicode61')`
  3. `note_embeddings(nid INTEGER, model_id TEXT, content_hash TEXT, dim INTEGER, vector_blob BLOB, PRIMARY KEY (nid, model_id))`
  4. `run_history(run_id TEXT PRIMARY KEY, created_at INTEGER, deck_name TEXT, filtered_did INTEGER, source_decks_json TEXT, unit_title TEXT, unit_preview TEXT, total_matched_cids_json TEXT, newly_unsuspended_cids_json TEXT, rolled_back INTEGER DEFAULT 0)`
- Vector Math & Serialization:
  - L2-normalize every vector before storing in `vector_blob` using `struct.pack(f"<{dim}f", *normalized_vec)`.
  - Implement fast batch dot-product similarity:
    - Try `import numpy as np`: load blobs via `np.frombuffer(blob, dtype=np.float32)` into a 2D matrix `(N, dim)` and run `matrix @ query_vec` (computes 30,000 similarities in ~5ms!).
    - Provide a pure-Python `struct.iter_unpack` / `sum(a*b ...)` fallback if `numpy` is unavailable.
- Model-Scoped Cache Management Methods:
  - `get_indexed_models_stats() -> list[dict]`: returns each `model_id` in `note_embeddings`, count of indexed notes, and approximate size in MB.
  - `delete_embeddings_for_model(model_id: str)`: deletes cached embeddings for a specific model (or all inactive models).
  - `get_notes_needing_indexing(notes_meta: dict[int, tuple[int, str]], active_model_id: str) -> tuple[list[int], list[int]]`:
    - Identifies which `nid`s need text extraction/FTS update (`note_mod` or `content_hash` changed) and which `nid`s need embedding computation for `active_model_id`.

#### 2.2 `ai_client.py` (`OpenAICompatibleClient`)
- Pure `urllib.request` client supporting both **LM Studio** (`http://127.0.0.1:1234/v1`) and **OpenRouter** (`https://openrouter.ai/api/v1`) or any custom OpenAI-compatible URL.
- Methods:
  - `list_models(base_url, api_key) -> list[str]`: calls `GET /v1/models` to populate UI dropdowns.
  - `chat_completion_json(messages, temperature=0.1, timeout=90) -> dict`:
    - Calls `POST /v1/chat/completions`.
    - Includes a resilient JSON parser that extracts the JSON object even if a local model wraps it in ````json ... ```` or `<think>...</think>` tags (critical for DeepSeek/Qwen reasoning models in LM Studio!).
  - `get_embeddings(texts: list[str], batch_size: int = 32, timeout: int = 60) -> list[list[float]]`:
    - Calls `POST /v1/embeddings` in batches, returning float vectors ordered by `index`.

---

## Phase 3: Incremental Indexer & Multi-Stage Hybrid Search Pipeline

### Files to Create
1. `src/ankihub_searcher/indexer.py`
2. `src/ankihub_searcher/search_pipeline.py`

### Detailed Requirements

#### 3.1 `indexer.py` (`DeckIndexer`)
- Given a list of selected deck names (e.g., `["AnKing Step Deck", "Mastering Radiology"]`):
  1. Finds all notes belonging to those decks (and their subdecks `DeckName::*`).
  2. Extracts and cleans only the configured fields for each Note Type (`config.get_indexed_fields_for_model(model_name, available_fields)`), plus cleaned tags if `index_tags == True`.
  3. Upserts `notes_content` and `notes_fts` in SQLite for any new or modified notes.
  4. Checks `note_embeddings` for the **currently active `model_id` only** (`f"{embed_provider}::{embed_model}"`).
  5. Processes missing embeddings in batches (e.g. 32 notes per HTTP call), committing after EVERY batch!
  6. **Pause / Cancel / Resume Support:** Accepts a `cancel_check: Callable[[], bool]` and `progress_callback: Callable[[int, int, str], None]`. Because every batch is committed to SQLite immediately, pausing or cancelling at 45% and resuming later starts immediately from the remaining 55% without losing a single vector.

#### 3.2 `search_pipeline.py` (`SmartSearchPipeline`)
- **Stage 1: Concept Extraction (`extract_concepts(unit: LectureUnit) -> list[ExtractedConcept]`):**
  - System prompt instructs the LLM as a medical curriculum & USMLE/AnKing expert to break the Italian lecture text into atomic medical concepts (`topic_it`, `claim_en`, `keywords_en`, `keywords_it`).
  - Specifically instructs the LLM to resolve Italian medical eponyms, historical receptor names (e.g. `NALP3` → `NLRP3`), and pathological syndromes into standard English USMLE terminology.
  - Fallback safety: if the Chat LLM fails or returns empty concepts, automatically constructs a fallback concept from the raw text so search never crashes.
- **Stage 2: Hybrid Retrieval (`retrieve_for_concepts(concepts, candidate_nids, active_model_id) -> list[ScoredNote]`):**
  - Loads candidate vectors for `candidate_nids` and `active_model_id`.
  - Embeds all concept queries in a single batch call to the embedding endpoint.
  - For each concept:
    - Computes dense cosine similarities against `candidate_nids`.
    - Runs FTS5 BM25 query on `notes_fts` using sanitized `keywords_en` (and `keywords_it`).
    - Normalizes BM25 scores to `[0, 1]` and fuses with dense cosine score:
      `combined = alpha * dense_score + (1 - alpha) * bm25_norm` (requiring `dense_score >= min_dense_floor`).
    - Takes up to `top_k_per_concept` notes above the threshold derived from `sensitivity`.
  - Deduplicates across concepts, adds a small multi-concept match bonus (`+0.02` per extra concept matched), and caps at `max_cards_per_unit`.
- **Stage 3: Optional LLM Verification (`verify_candidates(unit, candidates) -> list[ScoredNote]`):**
  - Runs only if `config.enable_llm_verification == True`.
  - Sends candidate notes (`nid` + `clean_text[:300]`) to the Chat LLM to filter out off-topic false positives.

---

## Phase 4: Anki Collection Operations, Filtered Decks & Safe Rollback

### Files to Create
1. `src/ankihub_searcher/deck_ops.py`

### Detailed Requirements
- **`resolve_notes_to_cards(col, nids: list[int], selected_deck_names: list[str]) -> tuple[list[int], list[int]]`:**
  - Resolves all sibling cards (`cid`) for the given `nid`s that belong to `selected_deck_names` (checking `odid` when a card is currently inside another filtered deck, else `did`).
  - Splits them into:
    - `suspended_cids`: cards where `queue == -1`
    - `active_cids`: cards where `queue != -1`
- **`execute_unit_deck_creation(col, unit, matched_nids, config, storage) -> RunResult`:**
  - Calls `resolve_notes_to_cards`.
  - Determines `target_deck_cids`:
    - `suspended_cids` if `not config.include_active_cards_in_filtered_deck`
    - `suspended_cids + active_cids` if `config.include_active_cards_in_filtered_deck`
  - If `target_deck_cids` is empty, returns a `RunResult` indicating 0 cards added (does not create an empty filtered deck).
  - Unsuspends **only** `suspended_cids` via `col.sched.unsuspend_cards(suspended_cids)`.
  - Creates (or updates) Filtered Deck `unit.deck_name`:
    - Uses `cid:123,456,...` search query.
    - Sets `resched = True` and `limit = len(target_deck_cids)`.
    - Calls `col.sched.rebuild_filtered_deck(filtered_did)`.
  - Saves the `run_history` record in `storage` with `newly_unsuspended_cids = suspended_cids`.
- **`rollback_run(col, storage, run_id: str, keep_reviewed: bool = True) -> RollbackResult`:**
  - Reads `newly_unsuspended_cids` and `filtered_did` from `run_history`.
  - If `filtered_did` exists in `col.decks`, empties and removes that Filtered Deck first so cards return to their original decks.
  - Filters `newly_unsuspended_cids`:
    - Only cards that currently exist in `cards`.
    - If `keep_reviewed == True`: excludes any card where `reps > 0` or that has an entry in `revlog` with `id >= run.created_at * 1000`.
  - Calls `col.sched.suspend_cards(cids_to_resuspend)`.
  - Marks `rolled_back = 1` in `run_history`.

---

## Phase 5: PyQt6 User Interface & Add-on Entry Point

### Files to Create
1. `src/ankihub_searcher/ui/__init__.py`
2. `src/ankihub_searcher/ui/worker.py`
3. `src/ankihub_searcher/ui/main_dialog.py`
4. `src/ankihub_searcher/__init__.py`

### Detailed UI Design (`main_dialog.py`)
A clean, modern PyQt6 dialog (`QDialog`) accessible from `Tools -> AnkiHub Smart Searcher...` (shortcut `Ctrl+Shift+F` / `Cmd+Shift+F`), organized into **4 Tabs**:

1. **Tab 1: 🔍 Smart Search (Main Workflow)**
   - **Source Decks Selector:** Checkable list/tree of top-level & sub-decks (with quick filter box and saved selection). Shows badge with how many notes are indexed for the active Embedder.
   - **Filtered Deck Base Name (`QLineEdit`):** Placeholder `"Es. Patologia Generale (opzionale se usi header Markdown # 2.3 ...)"`.
   - **Lecture Text Input (`QPlainTextEdit`):** Large area to paste one unit or a multi-header Markdown file. Live status indicator below showing detected Markdown units (e.g. *"Rilevate 3 unità valide (1 header vuoto ignorato)"*).
   - **Quick Options Bar:**
     - Sensitivity Slider (`Preciso` ↔ `Ampio`)
     - Checkbox: *"Verifica finale con LLM (più lento ma più preciso)"*
     - Checkbox: *"Includi nel mazzo filtrato anche carte già non-sospese"*
   - **Progress Bar + Cancel/Pause Button + Status Log:** Shows live progress during auto-indexing (if new cards need embeddings) and during Stage 1/2/3 search.
   - **Primary Button:** `"🚀 Cerca Carte e Crea Mazzo Filtrato"`.

2. **Tab 2: 🕒 Cronologia & Rollback (Safe Undo)**
   - Table of past runs (`Data/Ora`, `Nome Mazzo Filtrato`, `Unità`, `Carte nel Mazzo`, `Carte Unsospese dalla Run`, `Stato`).
   - Checkbox: *"Mantieni attive le carte che ho già ripassato (`keep_reviewed`)"* (checked by default).
   - Buttons:
     - `"⏪ Annulla Run Selezionata (Ri-sospendi carte e rimuovi mazzo filtrato)"`
     - `"🗑️ Elimina solo il Mazzo Filtrato (Lascia le carte unsospese nel mazzo principale)"` — **Super handy shortcut for the user's "Review Limbo" workflow!**

3. **Tab 3: 🗂️ Gestione Indice & Embedding Cache**
   - Displays active Embedding Model (`provider::model`) and indexing coverage for selected decks (`X / Y note indicizzate`).
   - Buttons: `"▶️ Indicizza / Aggiorna Ora"`, `"⏸️ Pausa"`, `"🧹 Elimina Cache Modelli Non Attivi"`.
   - Table showing all cached embedding models in SQLite and their disk usage.

4. **Tab 4: ⚙️ Impostazioni (AI Providers, Campi & Avanzate)**
   - **Chat LLM Section:** Preset (`LM Studio Locale`, `OpenRouter`, `Custom`), Base URL, API Key, Model combo box + `"🔄 Carica Modelli"` button + `"Test Connessione"`.
   - **Embedding Model Section:** Preset (`LM Studio Locale`, `OpenRouter`, `Custom`), Base URL, API Key, Model combo box + `"🔄 Carica Modelli"` button, Language Strategy (`Traduzione concettuale in Inglese (Consigliato per AnKing)` vs `Multilingua Diretto`).
   - **Field Selection per Note Type:** Select Note Type from dropdown → check/uncheck which fields to index (defaults to `Text`, `Extra`, `Front`, `Back`, etc.) + checkbox `"Indicizza anche i Tag (puliti)"`.
   - **Batch Naming Style:** Radio/Combo between Option A (`Nome Base - Titolo Header`) and Option B (`Nome Base::Titolo Header`).
