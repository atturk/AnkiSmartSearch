# Changelog

All notable changes to **Smart Search (AnkiSmartSearch)** are documented in this file.

## [0.4.0] - 2026-11-09

### Novità
- **Prompt Stage 1 atomico con citazione (`quote_it`):** Estrazione di un concetto distinto per ciascuna specifica entità medica (malattia, meccanismo, molecola, farmaco, segno clinico, valore numerico/soglia), con divieto di accorpamenti e ancoraggio testuale esplicito tramite il campo `quote_it` visualizzato nella finestra dei dettagli di copertura.
- **Budget token e schema JSON configurabile:** `chat_max_tokens` elevato a 16.384 token di default (usato anche in Stage 3) per supportare modelli con catene di ragionamento estese; nuova opzione `chat_use_json_schema` (default false) con fallback automatico senza schema se il modello restituisce 0 concetti con schema attivo.
- **Selezione focalizzata e calibrazione pesi (migrazione v4):** Punteggio basato interamente sulla similarità semantica densa (`hybrid_dense_weight: 1.0`, con salto a tempo zero della ricerca lessicale FTS), soglia minima 0.60 (sensibilità default 0.5), finestra relativa 0.05 (`relative_score_window`) e massimo 10 note per concetto (`max_notes_per_concept`). Query di embedding ottimizzata con la prima frase del claim concatenata alle parole chiave. Migrazione automatica effettiva della configurazione a `config_version = 4`.
- **Istruzione per query degli embedding:** Supporto all'opzione `embed_query_instruction` per guidare i modelli asimmetrici (come `qwen3-embedding`), applicata in modo trasparente solo alle query di ricerca e non all'indicizzazione delle note.
- **Scritture sicure sul thread principale con Undo di Anki:** Spostamento di tutte le scritture alla collezione (riattivazione carte, creazione/ricostruzione mazzi filtrati, cancellazione e rollback) sul main thread tramite `CollectionOp` con punto di ripristino nativo "Smart Search", preservando la massima reattività dell'interfaccia.
- **Risoluzione note condivise tra unità (Batch):** Assegnazione ottimale di ogni carta contesa all'unità didattica con il punteggio di similarità più alto, con registrazione della condivisione (`shared_with`) nella copertura delle altre unità.
- **Isolamento cronologia per profilo Anki:** Associazione di ogni run al profilo utente aperto (`mw.pm.name`), con filtraggio della cronologia e protezione dal rollback accidentale di ricerche create in profili differenti.
- **Riepilogo e copertura onesti:** Calcolo rigoroso della percentuale di copertura basato unicamente sui concetti coperti al 100% (con conteggio separato per parziali e scoperti), ed evidenziazione immediata in testa al riepilogo di eventuali unità con errori Stage 1.
- **Impostazioni a schede con scroll:** Riorganizzazione completa della finestra impostazioni in 5 schede scrollabili (`QTabWidget` con `QScrollArea`): "Ricerca", "Carte simili", "Modelli", "Indice", "Interfaccia", con dimensione iniziale compatta (≤ 800 px di altezza).
- **Impostazioni "Selezione carte" con anteprima a costo zero:** Slider continuo per la soglia minima effettiva (0.50 - 0.70 da "Ampio" a "Preciso"), configurazione di `max_notes_per_concept`, `relative_score_window`, `max_cards_per_unit` e switch ricerca FTS5; pulsante "Anteprima sull'ultima ricerca" per ricalcolare istantaneamente lo Stage 2 sui concetti salvati senza chiamate LLM né scritture sul database.
- **Annulla creazione e Riprova più stretto:** Nuovi pulsanti nel popup di riepilogo: "Annulla creazione" per il rollback atomico di tutte le unità del batch in un'unica CollectionOp di Anki (preservando le carte studiate), e "Riprova più stretto" per rilanciare la ricerca con soglia più selettiva (+0.03) e cap note ridotto di un terzo riusando i concetti già estratti a zero token di chat. Identificatore `search_id` comune per correlare tutti i run del batch.
- **Menu unificato (Finestra principale e Browser):** Stesso menu "Smart Search" con identico ordine di voci nella finestra principale di Anki e nel Browser: "Nuova ricerca…", "Cerca con la query corrente…", "Trova carte simili", separatore, "Cronologia & rollback", "Indice", "Impostazioni", con disabilitazione automatica delle voci dipendenti dal browser se usate fuori da esso o in assenza di selezione. Nuova opzione `menu_location` ("tools" di default sotto Strumenti, o "menubar" nella barra principale stile AnkiHub).
- **Browser, Carte simili e colonna similarità:** Colonna personalizzata "Smart Search: similarità" nel Browser di Anki con azione "Trova carte simili" (`Ctrl+Alt+M`) nel menu e nel menu contestuale delle righe. Ambito predefinito `similar_scope: "all"`.
- **Ordinamento nativo per carte simili nel Browser:** Utilizzo degli hook nativi `browser_will_search` e `browser_did_search` di Anki per ordinare i risultati della ricerca per similarità decrescente direttamente nella tabella del Browser (supporta sia la modalità Note che la modalità Carte con mappatura SQL ottimizzata), disattivando l'ordinamento predefinito del backend per quella query senza interferire con i successivi clic dell'utente sulle intestazioni delle colonne. Rimosso il pannello separato ridondante.
- **Avanzamento esplicito:** la barra avanza a blocchi per ogni unità (invio richiesta, risposta ricevuta, ricerca vettoriale, creazione mazzo) con messaggi che indicano modello e host; durante l'attesa della risposta dell'IA la barra resta ferma e un cronometro nel messaggio mostra i secondi trascorsi.
- **Funzionamento completo senza NumPy:** Ricerca per similarità vettoriale, media normalizzata e recupero ibrido implementati con percorsi matematici in puro Python standard library, garantendo compatibilità universale anche in assenza di NumPy nell'ambiente Python di Anki.

### Correzioni
- **Apertura ricerca dal menu:** Risolto l'errore `AttributeError 'bool' object has no attribute 'strip'` scatenato dal parametro booleano `checked` di Qt su `QAction.triggered`.
- **Preservazione query salvata:** Apertura del dialogo di ricerca da "Cerca con la query corrente…" non sovrascrive più la query di default di Smart Search fino all'effettiva conferma dell'utente.
- **Scorciatoia sicura per carte simili:** Assegnata la scorciatoia `Ctrl+Alt+M` evitando conflitti con le scorciatoie native del Browser Anki (`Ctrl+Shift+S` Riposiziona, `Ctrl+Alt+S` Inverti selezione).
- **Robustezza Stage 1, Troncamento & Escape LaTeX in JSON:** Normalizzazione del LaTeX inline prima dell'invio e riparazione automatica degli escape non validi o comandi greci/LaTeX (es. `\beta`, `\nu`, `\theta`) sia prima che dopo il parsing JSON, eliminando i fallback silenziosi a 0 carte. Diagnosi esplicita dell'esaurimento del budget token (`finish_reason == 'length'`) con invito ad aumentare `chat_max_tokens`.
- **FTS5 con keyword composite e trattini:** Gestione corretta dei token con trattino (es. `IL-1`, `TNF-alpha`, `heparin-induced`) come frasi quotate per evitare errori di sintassi SQLite FTS5, con scarto delle stopword (italiano/inglese) e opzione per escludere keyword italiane su mazzi inglesi.
- **Verifica Stage 3:** Invio del testo completo dell'unità e di `quote_it` al verificatore LLM, con scarto integrale del batch quando il modello restituisce una lista vuota valida.
- **Rollback basato su `revlog`:** Riconoscimento delle carte ripassate tramite timestamp di revisione (`revlog.id >= created_at * 1000`) invece del generico contatore `reps > 0`, e prevenzione del riuso/sovrascrittura di mazzi filtrati appartenenti a ricerche attive.

---

## [0.3.0] - 2026-11-08

### Added
- **Browser-Style Search Query Bar with History & Live Counter (`SearchDialog`):**
  - Replaced the deck checkbox tree with an editable `QComboBox` supporting full Anki Browser query syntax (e.g. `"deck:AnKing Step Deck"`, `"deck:AnKing Step Deck" OR "deck:Mastering Radiology"`, tag filters, etc.).
  - Automatically saves and restores the last used query and maintains a dropdown history of the last 20 queries.
  - Live debounced counter below the query bar displays the exact number of matching cards, notes, and active-model indexing coverage percentage (`Carte: X (Y note) — Indicizzate: Z%`).
- **Automatic Child Filtered Deck under Single Source Deck:**
  - When the search query targets a single source deck (or subdecks of a single parent deck, e.g. `"deck:AnKing Step Deck"`), the created Filtered Deck is automatically placed as a child subdeck of that deck (`AnKing Step Deck::Nome Mazzo Filtrato`).
- **Per-Claim Concept Coverage & `"Dettagli copertura"` Dialog:**
  - Computes zero-overhead coverage metrics for every LLM-extracted key point (`Coperto`, `Parziale`, `Scoperto`), showing overall coverage in the post-search summary popup (`Copertura punti chiave: X / Y (Z%) — K scoperti`).
  - Added `"Dettagli copertura"` button in both the post-search summary popup and the `Cronologia` window to inspect every claim, its matched note count, max similarity score, closest card preview, and a 1-click `"Copia punti scoperti / parziali"` button to create new flashcards for uncovered lecture material.

---

## [0.2.1] - 2026-11-08

### Fixed
- **Deck Tree Hierarchy & Protobuf Compatibility (`SearchDialog`):**
  - Fixed `RecursionError` during initial bottom-up tri-state checkbox computation (`_sync_subtree_check_state_bottom_up` vs `_update_ancestors_check_state`).
  - Fixed `AttributeError` when filtering dynamic decks on Anki 24+/26+ `DeckNameId` protobuf objects.

---

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
