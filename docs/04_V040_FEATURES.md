# v0.4.0 — funzioni aggiuntive (richieste dell'utente, 2026-11-09)

Stesse "Regole per chi implementa" di `03_V040_PLAN.md` (solo dentro la repo, una voce = un commit, test, niente push,
nessuna dipendenza nuova, scritture sulla collezione con CollectionOp come nella voce 7).

### 15. Impostazioni "Selezione carte" con anteprima
- Nella finestra impostazioni, sezione "Selezione carte": slider soglia (etichette "Ampio" ↔ "Preciso", mostra il valore
  di soglia effettivo), `max_notes_per_concept`, `relative_score_window`, `max_cards_per_unit`, `fts` sì/no (peso lessicale).
- Pulsante "Anteprima sull'ultima ricerca": ricalcola solo lo Stage 2 sui concetti salvati dell'ultimo run del profilo
  (servono claim_en e keywords_en nella copertura: salvali se mancano) con i valori correnti e mostra per unità quante
  note/carte uscirebbero. Nessuna chiamata al modello di chat, nessuna scrittura sulla collezione.
- Refactor necessario: la funzione di selezione per concetto deve essere riusabile con parametri espliciti.

### 16. Riepilogo: "Annulla creazione" e "Riprova più stretto"
- Nel popup di riepilogo: "Annulla creazione" esegue il rollback (keep_reviewed=True) di TUTTI i run della ricerca appena
  conclusa (tutte le unità del batch) in un'unica CollectionOp con voce di annulla, poi aggiorna la UI.
- "Riprova più stretto": annulla come sopra, poi rilancia la stessa ricerca riusando i concetti già estratti (niente Stage 1)
  con soglia +0.03 (o sensitivity −0.15) e max_notes_per_concept ridotto di un terzo (min 3), mostrando i nuovi valori;
  i valori usati restano solo per quel tentativo, non vengono salvati come default.
- Serve un identificatore di ricerca (`search_id`) comune ai run dello stesso batch, salvato in run_history (migrazione).

### 17. Menu "Smart Search" nel Browser
- Hook `browser_menus_did_init`: menu "Smart Search" nella barra del Browser con:
  - "Cerca con la query corrente…": apre il SearchDialog con la query della barra del browser preimpostata come ambito.
  - "Trova carte simili" (Ctrl/Cmd+Shift+S): abilitata solo con almeno una carta selezionata.
- Trova carte simili: note delle carte selezionate → vettori dall'indice (modello di embedding attivo); se più note, media
  normalizzata; similarità coseno con le note indicizzate dell'ambito scelto; escludi le note selezionate; tieni quelle
  con similarità ≥ `similar_min_score` (default 0.70) fino a `similar_max_notes` (default 20), ordinate per similarità.
  Mostra nel browser `nid:a,b,c` (sintassi Anki con virgole) e un tooltip col numero trovato. Le carte restano come sono
  (sospese o no): è l'utente a decidere dal browser cosa riattivare. Nessuna scrittura sulla collezione.
- Ambito configurabile nelle impostazioni (`similar_scope`): "query" (default, la query Smart Search salvata),
  "all" (tutte le note indicizzate del modello attivo), "custom" (query Anki in `similar_custom_query`).
- `similar_min_score` e `similar_max_notes` sono impostazioni separate da quelle della ricerca da lezione (scala diversa:
  nota-nota vs concetto-nota), in un riquadro "Carte simili" delle impostazioni.
- Se una nota selezionata non è indicizzata: messaggio chiaro ("nota non indicizzata: indicizza il mazzo X").
- Calcolo in background (QueryOp), niente chiamate al modello di chat. Senza numpy deve restare sotto ~2 s su 30k note.

### 18. Test, CHANGELOG, pacchetto
- Test per selezione con parametri, anteprima, search_id/rollback di batch, similarità (media vettori, esclusione selezionate,
  soglia e cap), costruzione della query `nid:`.
- CHANGELOG 0.4.0 aggiornato, pacchetto ricostruito (versione invariata).
