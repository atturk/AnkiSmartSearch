# Piano v0.5.0 — modalità lite (Keyword) senza LLM

Richiesta dell'utente (2026-11-09): nuova modalità "lite", predefinita, senza LLM per estrazione e rielaborazione dei
concetti. Il pannello di ricerca mantiene la query dei mazzi e il nome del mazzo filtrato, poi due schede:
"Keyword" (nuova, selezionata di default) e "Testo" (la ricerca attuale con LLM). Le keyword sono scritte a mano
dall'utente, una per riga; ognuna viene vettorizzata da sola e usata esattamente come un concetto estratto dall'LLM
(stessa selezione: soglia, finestra relativa, max note/concetto, max carte/unità, assegnazione tra unità, mazzi
filtrati, cronologia, copertura, "Riprova più stretto").

Calibrazione (indice reale, qwen3-embedding-0.6b, istruzione attuale): una keyword singola ha punteggi più bassi di
un concetto LLM. Keyword inglesi: top 0.60-0.74; keyword italiane: spesso 0.45-0.57 e a volte rumore sotto 0.52.
Default della soglia keyword: 0.52 (separata dalla soglia della modalità Testo).

## Regole per chi implementa

- Lavora SOLO dentro questa repo, sul branch `agy/v0.5.0`. Non toccare file fuori dalla repo (niente profili o
  cartelle di Anki, niente `pip install` globali).
- Una voce = un commit, messaggio `N. <titolo della voce>` (numerazione da 26). Non fare push, non aprire PR.
- Dopo ogni voce: `python3 -m pytest -q tests` deve passare. Aggiungi test per ogni comportamento nuovo.
- Codice e commenti in inglese; stringhe UI in italiano. Nell'interfaccia niente frasi che spiegano l'interfaccia:
  solo etichette e dati.
- Compatibilità con config e database esistenti. Nessuna dipendenza nuova.

## Voci

### 26. Parsing delle keyword (text_utils.py)
- `LectureUnit` riceve il campo `keywords: List[str] = field(default_factory=list)`.
- `parse_keyword_lines(text) -> List[str]`: una keyword per riga; togli spazi, marcatori di elenco iniziali
  (`-`, `*`, `+`, `•`, `1.`, `1)`), virgole/punti e virgola finali; salta righe vuote e righe di header; dedup
  case-insensitive mantenendo l'ordine e la prima grafia. Applica `normalize_inline_latex`.
- `parse_keyword_units(raw_text, base_deck_name="", naming_style="dash") -> List[LectureUnit]`: usa
  `parse_markdown_units` (stessa logica di header, breadcrumb e nomi dei mazzi) e riempie `keywords` dal body di
  ogni unità; scarta le unità senza keyword. Senza header: una sola unità come oggi.

### 27. Keyword come concetti (search_pipeline.py)
- `ExtractedConcept` e `ConceptCoverage` ricevono `source: str = "llm"` ("llm" | "keyword"), incluso in
  `to_dict`/`from_dict` (default "llm" per la cronologia esistente).
- `concepts_from_keywords(keywords) -> List[ExtractedConcept]`: per ogni keyword `topic_it=claim_en=quote_it=kw`,
  `keywords_en=keywords_it=[kw]`, `source="keyword"`.
- In `retrieve_for_concepts`, per i concetti `source == "keyword"` il testo da vettorizzare è la sola keyword
  (con l'istruzione della query come oggi), qualunque sia `embed_language_strategy`.
- `ConfigManager.get_keyword_min_similarity_threshold()` restituisce `keyword_min_similarity` (voce 29).

### 28. Worker in modalità keyword (ui/worker.py)
- `SearchBatchWorker(..., input_mode="text")`, valori `"text"` | `"keyword"`.
- In modalità keyword: `parse_keyword_units`; se nessuna unità ha keyword → `error_sig` "Nessuna keyword trovata.".
  Per ogni unità i concetti sono `concepts_from_keywords(unit.keywords)`; nessuna chiamata al modello chat, nessun
  heartbeat, nessuna verifica Stage 3 anche se `enable_llm_verification` è attivo. Soglia: `override_min_threshold`
  se presente, altrimenti `get_keyword_min_similarity_threshold()`. Messaggi di avanzamento:
  `"{prefix}Ricerca vettoriale di {n} keyword tra {num_notes} note…"`.
- I concetti precomputati (Riprova più stretto) restano validi in entrambe le modalità.
- Test: un worker keyword con ai_client finto non chiama mai `chat_completion_json`, vettorizza una query per
  keyword e crea il mazzo; con due unità markdown crea due unità.

### 29. Configurazione e impostazioni (config_manager.py, ui/main_dialog.py SettingsDialog)
- Nuove chiavi: `search_input_mode: "keyword"` (ultima scheda usata), `keyword_min_similarity: 0.52`.
  `config_version` 5; i config esistenti ricevono i default (nessun altro valore cambia). Aggiorna anche
  `src/ankihub_searcher/config.json`.
- Scheda "Ricerca" delle impostazioni: campo "Soglia keyword" (QDoubleSpinBox 0.30-0.90, passo 0.01), salvato e
  caricato come gli altri. L'anteprima "ultima ricerca", se esiste, usa la soglia keyword quando i concetti sono
  keyword.

### 30. Pannello di ricerca con due schede (ui/main_dialog.py SearchDialog)
- Restano in alto: query dei mazzi con statistiche, nome mazzo filtrato.
- Sotto: `QTabWidget` con "Keyword" (QPlainTextEdit, placeholder `ApoB48\nChilomicroni\nRecettore LDL`) e
  "Testo" (il campo attuale del testo della lezione). All'apertura si seleziona la scheda di
  `config.search_input_mode`; all'avvio della ricerca si salva la scheda usata.
- Validazione per scheda: testo vuoto → avviso ("Inserisci almeno una keyword." / "Inserisci il testo della
  lezione."). Il worker riceve `input_mode` e il testo della scheda attiva.
- Correggi i bug esistenti: `SearchDialog` usa `self.profile` (mai definito) in rollback/riprova e
  `self.txt_deck_name` (si chiama `txt_base_deck`) in `_do_retry_tighter`. Definisci `self.profile` nel
  costruttore con `_get_current_profile_name()`. "Riprova più stretto" deve usare la stessa modalità e lo stesso
  testo della ricerca appena fatta, partendo dalla soglia della modalità (keyword o testo) + 0.03, e ripristinare
  il precomputed per unità come oggi (conservando `source`).
- Il riepilogo finale resta uguale; con keyword la copertura conta le keyword ("Copertura keyword: …" al posto di
  "Copertura concetti: …").

### 31. Versione 0.5.0
- `python3 scripts/build_addon.py --set-version 0.5.0`; CHANGELOG (sezione 0.5.0 in italiano) e README (modalità
  Keyword: formato, una per riga, header markdown per più unità; keyword in inglese danno risultati migliori).
