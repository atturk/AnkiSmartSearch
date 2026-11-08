# Smart Search (v0.2.0) — Minimal UI Redesign, Submenus, Browser Integration & GitHub Auto-Updater Plan

> **Instructions for Executor LLM:**
> Read this specification carefully and implement all changes across the repository. Run the unit test suite (`python3 -m unittest discover -s tests -v`) at the end and ensure 100% of tests pass.

---

## 1. Rebranding to "Smart Search" (Zero "AnkiHub" References in UI/Docs + Seamless DB Migration)
1. Remove all user-facing references to `"AnkiHub"` or `"AnkiHub Smart Searcher"` across UI titles, labels, messages, `manifest.json`, and documentation.
   - Add-on name in `manifest.json`: `"name": "Smart Search"`.
   - Dialog titles:
     - Search Dialog: `"Smart Search"`
     - History Dialog: `"Smart Search — Cronologia"`
     - Index Manager Dialog: `"Smart Search — Indicizzazione"`
     - Settings Dialog: `"Smart Search — Impostazioni"`
2. **Seamless Database Filename Migration (`storage.py`):**
   - Default database file in `user_files/` is now `smart_search.db`.
   - In `SearchDatabase.__init__`, if `db_path is None`:
     - Check if `user_files/smart_search.db` exists.
     - If it does **not** exist, check if legacy `user_files/ankihub_searcher.db` exists; if so, automatically rename `ankihub_searcher.db` (and any `-wal`/`-shm` files, or checkpoint first) to `smart_search.db` so the user's existing 305 MB embedding index is preserved without re-indexing!

---

## 2. Tools Menu & Submenu Architecture (`__init__.py`)
In `Tools` (`mw.form.menuTools`), create a `QMenu("Smart Search", mw)`:
- Install a click/action handler or event filter so that:
  1. Clicking the `"Smart Search"` menu item directly (on platforms/configurations where `QMenu` parent click triggers) opens the minimal **Search Dialog**.
  2. Inside the `"Smart Search"` submenu, add the following clean, emoji-free items:
     - **`Cerca...`** (`Shortcut: Ctrl+Shift+F`, which maps to `⇧⌘F` on macOS) → opens `SearchDialog`
     - *(Separator)*
     - **`Cronologia`** → opens `HistoryDialog`
     - **`Indicizzazione`** → opens `IndexDialog`
     - **`Impostazioni`** → opens `SettingsDialog`
     - *(Separator)*
     - **`Verifica aggiornamenti`** → runs the GitHub Release updater check (`updater.py`)

---

## 3. Minimal, Native-Styled Separate Dialogs (`ui/main_dialog.py`)
**Design Philosophy:**
- **NO emojis** (`🔍`, `🚀`, `✅`, `🗑️`, etc. must all be removed from UI labels, buttons, table cells, and titles).
- **NO chunky custom button stylesheets** (remove `background-color: #2980b9`, `#c0392b`, `#27ae60`, large padding, etc.). Use standard native Qt/macOS controls (`QPushButton`, `QTreeWidget`, `QLineEdit`, `QPlainTextEdit`, `QTableWidget`, `QComboBox`, `QCheckBox`).
- Split the old 4-tab monolithic window into **4 lightweight, standalone dialogs** (each callable directly from the submenu, while keeping `AnkiHubSearcherDialog = SearchDialog` alias for backwards compatibility in imports).

### 3.1 `SearchDialog` (Compact, Minimal Main Search Window)
- Compact window size: `resize(520, 460)`.
- Layout (`QVBoxLayout` with clean spacing):
  1. **Selezione Mazzi (`QTreeWidget` Gerarchico):**
     - Label: `"Mazzi in cui cercare:"`
     - Uses a `QTreeWidget` with header hidden (`setHeaderHidden(True)`), maximum height `~130px`.
     - Builds a proper deck tree from Anki deck names (`Parent::Child::Grandchild`):
       - **Only top-level parent decks are shown at the root level (collapsed by default)**.
       - Users can click the expand arrow (`▸`) on any parent deck to view and individually check/uncheck its subdecks.
       - Checking/unchecking a parent item propagates the check state to its children (and saves `selected_decks` to config).
       - Note on `_get_selected_deck_names()`: if a parent deck and all its children are checked, or if specific child decks are checked, return the checked deck names accurately so `find_target_deck_ids` matches only the checked decks/subdecks!
         *(Specifically: in `indexer.py` and `deck_ops.py`, when a user unchecks a specific subdeck `Parent::Child2` under `Parent`, only checked items in the tree should be searched. The cleanest way: if all children of `Parent` are checked, or `Parent` has no children, include `Parent`; if only some children of `Parent` are checked, set `Parent` to `Qt.CheckState.PartiallyChecked` and return only the checked child deck names!)*
  2. **Nome Mazzo Filtrato (Facoltativo) (`QLineEdit`):**
     - Label: `"Nome mazzo filtrato (opzionale):"`
     - Placeholder: `"Es. Patologia Generale"`
  3. **Testo della Lezione (`QPlainTextEdit`):**
     - Label: `"Testo della lezione:"`
     - Placeholder: `"Incolla qui il testo della lezione o appunti in Markdown..."`
  4. **Barra di Progresso Minimale (`QProgressBar` + `QLabel`):**
     - Hidden by default (`setVisible(False)`), shown only while indexing/searching is running.
  5. **Pulsanti Standard in Basso (`QHBoxLayout` aligned right):**
     - `btn_cancel = QPushButton("Annulla")`:
       - If search is **not** running, clicking `"Annulla"` closes the dialog (`self.reject()`).
       - If search **is** running, clicking `"Annulla"` cancels the running worker (`self.search_worker.cancel()`).
     - `btn_search = QPushButton("Cerca")`:
       - Set as default button (`btn_search.setDefault(True)`).
       - Starts the search.

### 3.2 Automatic Close + Post-Search Summary Popup with `"Browser"` Button
When `SearchBatchWorker` finishes successfully (`status == "done"`):
1. Call `mw.reset()` if `mw` is available.
2. **Automatically close `SearchDialog`** (`self.accept()`).
3. Show a clean, minimal `QMessageBox` summary dialog (parent `mw` or `self`):
   - Title: `"Ricerca completata"`
   - Text:
     - If `total_added == 0 and total_already_active > 0`:
       `"Nessuna nuova carta da riattivare.\n\nCarte correlate già attive in studio: {total_already_active}"`
     - Else:
       `"Mazzi filtrati creati: {decks_created}\nNuove carte riattivate: {total_unsuspended}\nCarte correlate già attive: {total_already_active}\nCarte totali nel mazzo filtrato: {total_added}"`
   - Buttons:
     - Standard **`OK`** button (`QMessageBox.StandardButton.Ok`).
     - Custom **`Browser`** button (`msg_box.addButton("Browser", QMessageBox.ButtonRole.ActionRole)` — only enabled/added if at least one deck was created or cards were matched).
   - If the user clicks **`Browser`**:
     - Open Anki's Card Browser focused on the newly created filtered deck(s):
       ```python
       import aqt
       created_names = [r.deck_name for r in results if r.filtered_did]
       if created_names and aqt.mw:
           query = " OR ".join(f'"deck:{name}"' for name in created_names)
           browser = aqt.dialogs.open("Browser", aqt.mw)
           if browser and hasattr(browser, "search_for"):
               browser.search_for(query)
       ```

### 3.3 `HistoryDialog` (`Cronologia`)
- Clean, minimal dialog (`resize(720, 420)`), zero emojis.
- Table columns: `["Data", "Mazzo Filtrato", "Unità", "Nel Mazzo", "Nuove Riattivate", "Già Attive", "Stato"]` (store `run_id` in `Qt.ItemDataRole.UserRole` of column 0 so `Run ID` doesn't clutter the visual columns, or keep it clean).
- Checkbox: `"Mantieni attive le carte già ripassate"`
- Standard buttons:
  - `"Annulla run selezionata"` (executes rollback)
  - `"Elimina solo mazzo filtrato"` (deletes filtered deck, leaves cards active)
  - `"Chiudi"`

### 3.4 `IndexDialog` (`Indicizzazione`)
- Clean, minimal dialog (`resize(620, 420)`), zero emojis.
- Displays:
  - Active embedding model and coverage of selected decks (`"Copertura mazzi selezionati: X / Y note (Z%)"`).
  - Standard buttons: `"Indicizza / Aggiorna"`, `"Interrompi"`.
  - Table of cached models (`["Modello", "Note indicizzate", "Dimensione (MB)"]`).
  - Bottom buttons: `"Esporta indice (.db)..."`, `"Importa / Unisci (.db)..."`, `"Elimina cache modelli inattivi"`, `"Chiudi"`.

### 3.5 `SettingsDialog` (`Impostazioni`)
- Clean, minimal dialog (`resize(580, 560)`), zero emojis.
- Contains:
  1. **Parametri di Ricerca** (moved from the old main search window!):
     - Slider `Sensibilità ricerca` (`Preciso` ↔ `Ampio`)
     - Checkbox `"Verifica finale dei risultati con LLM"`
     - Checkbox `"Inserisci nel mazzo filtrato anche le carte già attive"`
     - Combo `"Stile nomi mazzi multi-unità"`: `"Nome Base - Titolo Sezione"` vs `"Nome Base::Titolo Sezione (Sottomazzo)"`
  2. **Modello Chat (LLM)**:
     - Preset (`LM Studio Locale`, `OpenRouter`, `Custom`), Base URL, API Key, Modello + `"Carica modelli"`, `"Test"`.
  3. **Modello Embedding**:
     - Preset (`LM Studio Locale`, `OpenRouter`, `Custom`), Base URL, API Key, Modello + `"Carica modelli"`, `"Test"`, Strategia linguistica (`"Traduzione concettuale in inglese"` vs `"Multilingua diretto"`).
  4. **Campi indicizzati per Tipo di Nota**:
     - Combo Tipo di Nota + lista spuntabile dei campi + checkbox `"Indicizza anche i tag delle note"`.
  5. Bottom buttons: `"Annulla"`, `"Salva"`.

---

## 4. GitHub Auto-Updater (`src/ankihub_searcher/updater.py`)
Create a new module `src/ankihub_searcher/updater.py` using only standard library (`urllib.request`, `json`, `zipfile`, `tempfile`, `os`, `shutil`):
1. **Repository Constant:** `GITHUB_REPO = "atturk/AnkiSmartSearch"`
2. **`check_for_updates(current_version: str) -> dict`:**
   - Calls `GET https://api.github.com/repos/atturk/AnkiSmartSearch/releases/latest` (with `User-Agent: AnkiSmartSearch-Updater`).
   - Extracts `tag_name` (strips leading `"v"`, e.g. `"0.2.0"`), `html_url`, `body` (release notes), and finds the `.ankiaddon` asset download URL (`browser_download_url` for asset ending with `.ankiaddon`).
   - Compares semantic version tuples `tuple(int(x) for x in tag.split(".")) > tuple(int(x) for x in current_version.split("."))`.
   - Returns `{"update_available": bool, "latest_version": str, "current_version": str, "download_url": str, "release_notes": str}`.
3. **`download_and_install_update(download_url: str, addon_dir: Optional[str] = None) -> None`:**
   - Downloads the `.ankiaddon` ZIP archive to a temporary file.
   - Validates it is a valid ZIP archive (`zipfile.is_zipfile`).
   - Extracts files into `addon_dir` (defaults to `os.path.dirname(os.path.abspath(__file__))`), **strictly skipping/preserving `user_files/` and `meta.json`** so the user's SQLite database and Anki configuration are never touched!
4. **UI Trigger (`run_update_check_ui(parent=None)`):**
   - Called when the user clicks **`Tools -> Smart Search -> Verifica aggiornamenti`**.
   - If `update_available == False`:
     - Shows `QMessageBox.information`: `"Smart Search è già aggiornato all'ultima versione (v{current_version})."`
   - If `update_available == True`:
     - Shows `QMessageBox.question`: `"È disponibile una nuova versione di Smart Search (v{latest_version}, attuale: v{current_version}).\n\nVuoi scaricare e installare l'aggiornamento ora?"`
     - If user confirms (`Yes`), calls `download_and_install_update`, then shows `QMessageBox.information`: `"Aggiornamento a v{latest_version} installato con successo.\nRiavvia Anki per applicare le modifiche."`
5. **Unit Tests:**
   - Add unit tests in `tests/test_updater.py` testing version comparison (`0.2.0 > 0.1.1`), mocked GitHub API response parsing, and safe ZIP extraction that preserves `user_files/smart_search.db` and `meta.json`.
