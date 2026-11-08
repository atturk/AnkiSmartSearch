# AnkiHubSearcher (Smart Lecture-to-Filtered-Deck Add-on) — Architecture & Technical Specifications

## 1. Project Overview & Goal
**AnkiHubSearcher** is an Anki desktop add-on (Python / PyQt6 / `aqt`) designed for medical students studying from Italian lecture transcripts/notes while using massive English pre-made decks (such as **AnKing Step Deck v12** with ~28,660 notes / ~35,080 cards and **Mastering Radiology** with ~23,750 notes / ~25,820 cards).

### Core Workflow
1. The user copies one or more lecture units (in Italian, optionally structured with Markdown headers `#`, `##`, `###`).
2. The user opens **AnkiHubSearcher** inside Anki, selects the target deck pool to search within (e.g., `AnKing Step Deck`, `Mastering Radiology`), specifies a base Filtered Deck name, and pastes the text.
3. The add-on runs a **Multi-Stage AI Hybrid Retrieval Pipeline**:
   - **Stage 0 (Markdown Batch Parser):** Splits the input into individual lecture units if Markdown headers with body text are present (ignoring empty parent headers that only wrap subheaders).
   - **Stage 1 (LLM Concept Decomposition & Cross-Lingual Medical Query Expansion):** Uses a local LLM (via **LM Studio**) or remote LLM (via **OpenRouter / OpenAI-compatible API**) to decompose each Italian lecture unit into atomic medical concepts translated into USMLE/AnKing-grade English claims + medical synonyms/eponyms/acronyms (e.g., `NALP3` → `NLRP3 inflammasome`, `pirofosfato di calcio` → `calcium pyrophosphate dihydrate`, `CPPD`, `pseudogout`, `fattore di Hageman` → `Factor XII`, `Hageman factor`).
   - **Stage 2 (Hybrid Search: Dense Embeddings + Lexical BM25):** Queries a local SQLite index of the selected decks using both cosine similarity on embeddings (scoped to the active embedding model) and tokenized BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF) and filtered by a configurable similarity threshold + per-concept Top-K.
   - **Stage 3 (Optional LLM Verification / Reranking):** When enabled in Settings, passes candidate notes back to the LLM to discard false positives before touching the user's collection.
4. **Card State & Filtered Deck Creation ("Review Limbo"):**
   - For every matched **Note (`nid`)**, all of its sibling **Cards (`cid`)** within the selected decks are inspected.
   - **Safety Rule 1 (Default):** Only cards that are currently **suspended (`queue == -1`)** are unsuspended and placed into the new Filtered Deck. Cards that the user is already actively studying (`queue != -1`) are untouched and excluded from the Filtered Deck by default (unless the setting *"Include already-active cards in filtered deck"* is explicitly enabled).
   - **Safety Rule 2 (`cid:` Query Engine — Zero Note Pollution):** The Filtered Deck is built using an exact `cid:12345,67890,...` search query with `reschedule = True` and `limit = len(cids)`. **No temporary tags are ever written to Notes**, avoiding AnkiWeb sync bloat and preventing conflicts with AnkiHub / AnkiCollab.
   - **Safety Rule 3 (Run History & Safe Rollback):** Every run records the exact `newly_unsuspended_cids` (cards that transitioned from `suspended -> unsuspended` during that specific run). The user can roll back any run at any time from the History tab:
     - Re-suspends **only** `newly_unsuspended_cids` (never touching cards the user had manually unsuspended previously).
     - Includes a checkbox option: *"Keep cards active if I have already reviewed them (`reps > 0` or reviewed since run timestamp)"*.
     - Deletes the corresponding Filtered Deck if it still exists.

---

## 2. Real Collection Insights (Verified via AnkiConnect Inspection)
Inspection of the target Anki environment (`Anki` with FSRS / RWKV and `AnkiConnect Extended` on port `3343`) revealed critical structural facts that the implementation **must** handle:

1. **Massive Note Counts & Card Ratios:**
   - `AnKing Step Deck`: **28,662 notes** / **35,082 cards** (~34,871 suspended, ~211 active). Primary Note Type: `AnKingOverhaul (AnKing Step Deck / AnKingMed)`.
   - `Mastering Radiology`: **23,751 notes** / **25,819 cards** (100% suspended). Primary Note Types: `Mastering Radiology (AnkiCollab) (Mastering Radiology / drradiology)-a1ef2` (20,702 notes), `Test Anatomy ...` (1,911 notes), `Image Occlusion Enhanced ...` (~979 notes).
2. **Field Noise vs. Signal:**
   - In `AnKingOverhaul`, clinical content lives in `Text` and `Extra`. Other fields (`Sketchy`, `Sketchy 2`, `Sketchy Extra`, `Physeo`, `Bootcamp`, `Picmonic`, `Pixorize`) contain mostly `<img>` tags and external video `<a>` links (`"Watch associated Bootcamp video..."`).
   - In `Mastering Radiology`, clinical text lives in `Text` and `Extra` (and optionally `Radiopaedia Text`), while image occlusion notes use `Header`, `Footer`, `Remarks`, `Extra 1`, `Extra 2`, or `Answer`.
   - **Default Field Selection Heuristic:** By default, the indexer automatically selects primary text/extra fields per Note Type (`Text`, `Extra`, `Front`, `Back`, `Back Extra`, `Header`, `Footer`, `Remarks`, `Answer`) while allowing the user to customize indexed fields per Note Type in Settings.
3. **AnKing & Mastering Radiology Tag Boilerplate:**
   - Notes carry 15–30 hierarchical tags like `#AK_Step1_v12::#FirstAid::04_Pathology::02_Inflammation::04_Acute_inflammation::03_Components::*Hageman_Factor` and noise tags like `#AK_Other::AnKing_Image::Marcos::...` or `#AK_Step1_v12::#AMBOSS::9OaNuk`.
   - Dumping raw tags into embeddings pollutes the vector with repetitive boilerplate (`#AK_Step1_v12`, `FirstAid`, `Bootcamp`, `AnKing_Image`).
   - **Smart Tag Cleaner:** Before indexing tags, strip boilerplate prefixes/segments (`#AK_Step1_v12`, `#AK_Step2_v12`, `#AK_Other`, `AnKing_Image`, `^EXPN`, `#AMBOSS`, `#Low/HighYield`, contributor names, random IDs) and extract clean, deduplicated topic keywords (e.g., `Pathology > Inflammation > Acute inflammation > Hageman Factor`).

---

## 3. System Architecture & Module Layout

All add-on code will live inside a clean package directory (which can be symlinked or copied into `~/Library/Application Support/Anki2/addons21/ankihub_searcher`):

```text
AnkiHubSearcher/
├── docs/
│   ├── 00_ARCHITECTURE_AND_SPECS.md
│   └── 01_IMPLEMENTATION_PLAN.md
├── src/
│   └── ankihub_searcher/
│       ├── __init__.py            # Entry point for Anki (registers Tools menu action & hooks)
│       ├── manifest.json          # Anki add-on manifest
│       ├── config.json            # Default Anki add-on configuration
│       ├── config_manager.py      # Typed settings wrapper around mw.addonManager
│       ├── text_utils.py          # HTML/Cloze/MathJax cleaner, Smart Tag Cleaner, Markdown Batch Parser
│       ├── storage.py             # SQLite manager: Note text, BM25 FTS5 table, Model-scoped Embeddings, Run History
│       ├── ai_client.py           # Zero-dependency HTTP client (urllib) for LM Studio & OpenRouter (/v1/chat/completions & /v1/embeddings)
│       ├── indexer.py             # Incremental background note indexer with Pause/Resume & progress callbacks
│       ├── search_pipeline.py     # Stage 1 (LLM Concept Extraction), Stage 2 (Hybrid Dense + FTS5 RRF), Stage 3 (LLM Reranker)
│       ├── deck_ops.py            # Anki collection operations: sibling lookup, safe unsuspend, cid: Filtered Deck creation, Rollback
│       └── ui/
│           ├── __init__.py
│           ├── main_dialog.py     # Main PyQt6 window (Tabs: Search, Index Manager, Run History & Rollback, Settings)
│           └── worker.py          # QThread / QueryOp background workers with progress & cancellation
└── tests/
    ├── test_text_utils.py         # Unit tests for Markdown parser, Cloze cleaner, Smart Tag cleaner
    ├── test_storage_and_search.py # Unit tests for SQLite FTS5 + Cosine Similarity + RRF fusion + Model isolation
    └── test_live_ankiconnect.py   # Read-only integration test against port 3343 verifying real AnKing note preprocessing
```

---

## 4. Detailed Component Specifications

### 4.1 Zero-Dependency AI Client (`ai_client.py`)
Anki bundles its own closed Python environment. Do **not** depend on `openai`, `requests`, `torch`, or `sentence-transformers`.
- Use Python's built-in `urllib.request`, `json`, and `ssl` (plus `numpy` if present in Anki's bundle, with a pure-Python `struct`/`math` fallback so it never fails even if `numpy` is absent!).
- **Dual Independent Endpoints:**
  1. **Chat / Reasoning LLM Configuration:**
     - `chat_provider_preset`: `"lm_studio"` | `"openrouter"` | `"custom"`
     - `chat_base_url`: e.g., `http://127.0.0.1:1234/v1` (LM Studio) or `https://openrouter.ai/api/v1`
     - `chat_api_key`: string (optional for LM Studio)
     - `chat_model`: model identifier string (with a "Fetch Models" `/v1/models` button in UI!)
  2. **Embedding Model Configuration:**
     - `embed_provider_preset`: `"lm_studio"` | `"openrouter"` | `"custom"`
     - `embed_base_url`: e.g., `http://127.0.0.1:1234/v1`
     - `embed_api_key`: string
     - `embed_model`: e.g., `text-embedding-nomic-embed-text-v1.5` or `baai/bge-m3`
     - `embed_language_strategy`: `"translate_to_en"` (default, best for English decks like AnKing) | `"multilingual_direct"` (uses both Italian & English representations for multilingual embedders).

### 4.2 Text Processing & Markdown Batch Parser (`text_utils.py`)
1. **HTML, Cloze & Media Cleaner (`clean_note_field(raw_html: str) -> str`):**
   - Strips `<style>...</style>`, `<script>...</script>`, `<img ...>`, `[sound:...]`, and HTML tags.
   - Unwraps Anki cloze deletions: `{{c1::Factor XII (Hageman factor)::coagulation factor}}` → `Factor XII (Hageman factor) (coagulation factor)` or `{{c1::Split S1}}` → `Split S1`.
   - Unescapes HTML entities (`&gt;` → `>`, `&amp;` → `&`).
   - Strips boilerplate links like `"Watch associated Bootcamp video..."`.
2. **Smart Tag Cleaner (`clean_note_tags(tags: list[str]) -> str`):**
   - Splits hierarchical tags on `::`.
   - Filters out known noisy segments (`#AK_Step1_v12`, `#AK_Step2_v12`, `#AK_Other`, `#AK_Original_Decks`, `AnKing_Image`, `!Subjects`, `^EXPN`, `#AMBOSS`, `#Low/HighYield`, `Only_Step_1`, `Only_Step_2`, `Contributors`, `DrRadiology`, `AnKore`, purely numeric prefixes like `01_`, `02_`).
   - Replaces underscores with spaces, deduplicates tokens while preserving order, and returns a compact string of meaningful medical taxonomy terms.
3. **Markdown Batch Parser (`parse_markdown_units(raw_text: str, base_deck_name: str, naming_style: str) -> list[LectureUnit]`):**
   - Detects ATX Markdown headers (`^(#{1,6})\s+(.+)$`).
   - If **no** headers exist, treats the entire non-empty text as a single `LectureUnit` named `base_deck_name` (or `"Smart Search YYYY-MM-DD HH:MM"` if `base_deck_name` is empty).
   - If headers exist:
     - Splits text into sections by header.
     - **Empty Header Rule:** Any header whose body (before the next header) contains no substantive text (e.g., `# Capitolo 2` followed immediately by `## 2.3 Segnali endogeni...`) is **ignored as a deck** (not emitted as an empty unit), though its title can optionally be kept as breadcrumb context for the LLM.
     - **Deck Naming Rule:**
       - `naming_style == "dash"` (Option A, **Default**):
         - If `base_deck_name` is provided: `"{base_deck_name} - {header_title}"` (for single headerless unit: `"{base_deck_name}"`).
         - If `base_deck_name` is empty: `"{header_title}"`.
       - `naming_style == "subdeck"` (Option B, configurable in Settings):
         - If `base_deck_name` is provided: `"{base_deck_name}::{header_title}"`.
         - If `base_deck_name` is empty: `"{header_title}"`.

### 4.3 Storage & Model-Scoped Vector Index (`storage.py`)
Stored in the add-on's `user_files/ankihub_searcher.db` (SQLite3, standard library):
- **Table `notes_content`:**
  - `nid INTEGER PRIMARY KEY`
  - `mid INTEGER NOT NULL` (Note Type ID)
  - `note_mod INTEGER NOT NULL` (Anki `notes.mod` timestamp to detect changes incrementally)
  - `content_hash TEXT NOT NULL` (SHA-256 of cleaned fields + cleaned tags + field config hash)
  - `clean_text TEXT NOT NULL` (Combined primary text + extra)
  - `clean_tags TEXT NOT NULL`
- **Virtual Table `notes_fts` (SQLite FTS5):**
  - `CREATE VIRTUAL TABLE notes_fts USING fts5(nid UNINDEXED, clean_text, clean_tags, tokenize='unicode61');`
  - Enables ultra-fast BM25 lexical search across 50,000+ notes in <15ms.
- **Table `note_embeddings` (Scoped by `model_id`!):**
  - `nid INTEGER NOT NULL`
  - `model_id TEXT NOT NULL` (e.g., `"lm_studio::text-embedding-nomic-embed-text-v1.5"`)
  - `content_hash TEXT NOT NULL`
  - `dim INTEGER NOT NULL`
  - `vector_blob BLOB NOT NULL` (Packed `float32` array via `struct.pack(f'<{dim}f', *vec)`, pre-normalized to unit L2 norm so cosine similarity is a simple dot product!)
  - `PRIMARY KEY (nid, model_id)`
  - **Lazy Per-Model Indexing Guarantee:** Embeddings are only computed for the currently active `model_id` and only for notes belonging to the user's selected target decks whose `content_hash` is missing or outdated for that `model_id`. Switching to another embedder does not delete previous models' caches unless the user clicks *"Purge Unused Model Caches"*.
- **Table `run_history`:**
  - `run_id TEXT PRIMARY KEY` (UUID / timestamp ID)
  - `created_at INTEGER NOT NULL`
  - `deck_name TEXT NOT NULL`
  - `filtered_did INTEGER` (Deck ID of the created Filtered Deck)
  - `source_decks_json TEXT NOT NULL`
  - `unit_title TEXT NOT NULL`
  - `unit_preview TEXT NOT NULL`
  - `total_matched_cids_json TEXT NOT NULL` (All cards included in the filtered deck)
  - `newly_unsuspended_cids_json TEXT NOT NULL` (**CRITICAL**: Strictly the subset of `cid`s that had `queue == -1` immediately prior to this run and were unsuspended by this run)
  - `rolled_back INTEGER DEFAULT 0`

### 4.4 Multi-Stage Search Pipeline (`search_pipeline.py`)

#### Stage 1: LLM Concept Decomposition & Cross-Lingual Expansion
Prompt sends the Italian lecture unit to the Chat LLM and requests a JSON object:
```json
{
  "concepts": [
    {
      "topic_it": "Attivazione dell'inflammasoma NLRP3 da acido urico e pirofosfato di calcio",
      "claim_en": "Uric acid (monosodium urate in gout) and calcium pyrophosphate dihydrate (CPPD in pseudogout) crystals activate the NLRP3 (NALP3) inflammasome to cleave IL-1beta.",
      "keywords_en": ["NLRP3", "inflammasome", "uric acid", "calcium pyrophosphate", "CPPD", "gout", "pseudogout", "IL-1"],
      "keywords_it": ["acido urico", "pirofosfato di calcio", "inflammasoma", "gotta", "pseudogotta"]
    }
  ]
}
```
- Why this works so well:
  1. Converts a 300-word multi-topic paragraph into 5–12 focused, single-concept micro-queries that match the atomic granularity of AnKing cards.
  2. Automatically bridges terminology gaps (`NALP3` → `NLRP3`, `fattore di Hageman` → `Factor XII`).
  3. Provides both a rich semantic sentence (`claim_en` / `topic_it`) for embedding similarity and exact medical terms (`keywords_en` / `keywords_it`) for FTS5 BM25 matching.

#### Stage 2: Hybrid Retrieval (Dense Cosine + FTS5 BM25) per Concept
For each extracted concept $c_i$:
1. **Dense Vector Search:**
   - Depending on `embed_language_strategy`:
     - `"translate_to_en"`: embeds `f"{c_i.claim_en} | {', '.join(c_i.keywords_en)}"`.
     - `"multilingual_direct"`: embeds `f"{c_i.topic_it} | {c_i.claim_en} | {', '.join(c_i.keywords_en)}"`.
   - Computes cosine similarity against the in-memory/SQLite `float32` vectors of candidate notes in the selected decks (`nid IN target_nids`).
   - Filters out notes with `cosine_sim < min_similarity_threshold`.
2. **Lexical FTS5 Search (BM25):**
   - Queries `notes_fts` with `OR`-joined tokenized `keywords_en` (and `keywords_it`), restricted to `target_nids`.
3. **Hybrid Score Fusion:**
   - Combines normalized dense cosine score ($S_{\text{dense}} \in [0,1]$) and normalized BM25 boost ($S_{\text{bm25}} \in [0,1]$):
     $$S_{\text{concept}}(n, c_i) = \alpha \cdot S_{\text{dense}}(n, c_i) + (1 - \alpha) \cdot S_{\text{bm25}}(n, c_i)$$
     (with default $\alpha = 0.78$, plus a require-minimum-dense-floor so pure accidental keyword overlap without semantic relevance cannot pass).
   - Selects up to `top_k_per_concept` (e.g., default `8`, adjustable via the Sensitivity slider) notes for concept $c_i$ that exceed the combined threshold.
4. **Cross-Concept Aggregation:**
   - Deduplicates notes across all concepts in the lecture unit, keeping `max_score` + a small multi-concept hit bonus (`+0.02` per additional concept matched) and recording which concept(s) matched each note.

#### Stage 3: Optional LLM Verification (`enable_llm_verification: bool = False`)
- Disabled by default for fast results (3–8s).
- When enabled in Settings, sends the lecture unit text + the list of candidate notes (`nid` + truncated `clean_text`) in batches of ~25 to the Chat LLM, asking it to return only the `nid`s that are genuinely relevant to the lecture concepts.

### 4.5 Safe Collection Operations & Rollback (`deck_ops.py`)

#### Resolving Notes to Cards (`nids -> cids`)
Given the winning `nid`s for a lecture unit and the user-selected source deck IDs (including their subdecks):
1. Query Anki's `cards` table:
   `SELECT id, nid, did, odid, queue, type, reps FROM cards WHERE nid IN (...)`
2. Filter to cards whose actual home deck (`odid if odid != 0 else did`) is within the selected source decks.
3. All sibling cards (`c1, c2, c3...`) of a relevant Note are included (as confirmed by user requirements).
4. **Partition by Suspension State (`queue == -1`):**
   - `suspended_cids = [cid for cid in matched_cids if queue == -1]`
   - `active_cids = [cid for cid in matched_cids if queue != -1]`
5. **Determine Cards for Filtered Deck:**
   - If `include_active_cards_in_filtered_deck == False` (Default):
     - `deck_cids = suspended_cids`
   - Else:
     - `deck_cids = suspended_cids + active_cids`
6. **Unsuspend Only `suspended_cids`:**
   - Call `col.sched.unsuspend_cards(suspended_cids)` (only if `suspended_cids` is non-empty).
   - Record `newly_unsuspended_cids = suspended_cids` in `run_history` **before/alongside** creating the Filtered Deck.
7. **Create the Filtered Deck (`cid:` Query):**
   - Build query: `search_query = "cid:" + ",".join(str(cid) for cid in deck_cids)`
   - Create or update Filtered Deck via `did = col.decks.new_filtered(deck_name)`:
     ```python
     deck = col.decks.get(did)
     deck["resched"] = True
     deck["terms"] = [[search_query, len(deck_cids), 0]]  # or protobuf/dict format matching Anki's DeckConfig
     col.decks.save(deck)
     col.sched.rebuild_filtered_deck(did)
     ```
   - Note: In modern Anki (v23.10+ / v24+ / v25+), verify both dict-style `deck["terms"]` and `col.decks.add_normal_deck_with_name` / `new_filtered` compatibility (handled cleanly in `deck_ops.py`).

#### Safe Rollback (`rollback_run(run_id: str, keep_reviewed: bool = True)`)
1. Load `run_record` from `run_history` table in SQLite.
2. Start with `cids_to_resuspend = list(run_record.newly_unsuspended_cids)`.
3. Verify each `cid` still exists in `col.db`.
4. If `keep_reviewed == True`:
   - Query `SELECT id FROM cards WHERE id IN (...) AND reps == 0 AND id NOT IN (SELECT cid FROM revlog WHERE id > :run_created_at_ms)`.
   - Only re-suspend cards from `newly_unsuspended_cids` that have **not** been reviewed by the user!
5. If `run_record.filtered_did` still exists in `col.decks`:
   - Empty and remove the filtered deck (`col.sched.empty_filtered_deck(did)` + `col.decks.remove([did])`) **first**, so cards return to their home decks, and then call `col.sched.suspend_cards(cids_to_resuspend)`.
6. Mark `rolled_back = 1` in `run_history` and refresh Anki's main window (`mw.reset()`).
7. **Absolute Safety Invariant:** Because `cids_to_resuspend` is strictly a subset of `newly_unsuspended_cids`, a card that was already unsuspended prior to the run can **never** be suspended by a rollback.
