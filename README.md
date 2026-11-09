# AnkiSmartSearch (`AnkiHubSearcher`) 🩺🧠

**AnkiSmartSearch** is an AI-powered Anki desktop add-on designed for medical students who study from **Italian lecture notes/transcripts** while using comprehensive English pre-made decks such as **AnKing Step Deck v12** and **Mastering Radiology**.

Paste your Italian lecture unit (or an entire Markdown document with multiple units), and the add-on automatically decomposes the clinical concepts, bridges language and eponym gaps (`NALP3` → `NLRP3 inflammasome`, `pirofosfato di calcio` → `CPPD / pseudogout`), unsuspends the relevant flashcards, and places them into a **Filtered Deck ("Review Limbo")** ready for study.

---

## ✨ Key Features

- **⚡ Modalità "Keyword" (Lite, senza LLM):**
  - Cerca rapidamente inserendo a mano una lista di keyword mediche, **una per riga**.
  - **Zero token chat:** Ogni keyword viene vettorizzata da sola con il modello di embedding locale o remoto e cercata direttamente tra le note senza richiedere chiamate LLM di decomposizione.
  - **Parsing flessibile:** Riconosce ed elimina automaticamente marcatori di elenco (`-`, `*`, `+`, `•`, `1.`, `1)`), spazi e punteggiatura finale; deduplica preservando la prima grafia; normalizza il LaTeX inline.
  - **Multi-unità Markdown:** Usa le intestazioni Markdown (`# Unità 1`, `## Sezione 2`) sopra le keyword per generare mazzi filtrati distinti in un unico passaggio batch.
  - 💡 *Consiglio:* Le keyword in **inglese** (es. `chylomicrons`, `LDL receptor`, `PCSK9 inhibitor`) ottengono punteggi di similarità più alti (top 0.60–0.74) rispetto all'italiano (0.45–0.57). Soglia keyword predefinita: `0.52`.
- **🔬 3-Stage Hybrid Medical Search (Modalità "Testo"):**
  1. **Stage 1 — LLM Concept Decomposition:** Breaks down dense Italian paragraphs into atomic English USMLE claims & medical keywords (supports **LM Studio** locally and **OpenRouter** / OpenAI-compatible APIs remotely).
  2. **Stage 2 — Dense Embeddings + SQLite FTS5 BM25:** Combines vector cosine similarity with tokenized BM25 keyword ranking.
  3. **Stage 3 — Optional LLM Verification:** Reranks and filters candidate cards to remove false positives.
- **📑 Ghost Batch Mode (Markdown):** Paste a Markdown document with `#` / `##` / `###` headers to automatically generate a separate Filtered Deck for each substantive section (empty parent headers are automatically skipped as decks and used as context breadcrumbs).
- **🛡️ 100% Safe for Your Collection & AnkiHub:**
  - **Zero Note Tag Pollution:** Builds Filtered Decks using native `cid:` queries (`resched = True`), never touching Note tags or triggering unnecessary AnkiHub syncs.
  - **Active Card Protection:** Only unsuspends cards that were suspended prior to the run; cards you are already studying are never accidentally suspended.
  - **1-Click Safe Rollback:** Undo any run from the *Cronologia & Rollback* tab (with an option to keep cards active if you already reviewed them), or delete just the Filtered Deck once you finish reviewing so kept cards return to their main deck.
- **⚡ Incremental Model-Scoped Indexer:**
  - Caches embeddings in a local WAL SQLite database scoped by `provider::model`.
  - Instant commit after every 32-note batch — pause and resume anytime with 0% progress lost.
  - Includes a **Smart Tag Cleaner** (strips `#AK_Step1_v12`, `AnKing_Image`, `AMBOSS` hashes, etc.) and per-Note-Type field selection.

---

## 📦 Installation

### Option A: Install from Release `.ankiaddon`
1. Download the latest **`AnkiHubSearcher.ankiaddon`** (or `dist/AnkiHubSearcher-v0.1.0.ankiaddon`) from the [Releases page](https://github.com/atturk/AnkiSmartSearch/releases).
2. In Anki, go to **Tools → Add-ons → Install from file...** and select the `.ankiaddon` file.
3. Restart Anki and open **Tools → AnkiHub Smart Searcher...** (`Ctrl+Shift+F` / `Cmd+Shift+F`).

### Option B: Development Symlink
```bash
git clone https://github.com/atturk/AnkiSmartSearch.git
ln -s "$(pwd)/AnkiSmartSearch/src/ankihub_searcher" "$HOME/Library/Application Support/Anki2/addons21/ankihub_searcher"
```

---

## 🛠️ Building & Versioning

Use `scripts/build_addon.py` to run the test suite, bump versions, and build the `.ankiaddon` packages:

```bash
# Run tests and build current version
python3 scripts/build_addon.py

# Bump version and build
python3 scripts/build_addon.py --set-version 0.2.0
```
