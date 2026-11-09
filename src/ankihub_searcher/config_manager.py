"""Configuration manager for Smart Search.

Works seamlessly inside Anki (via mw.addonManager) and in standalone test environments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": 5,
    "selected_decks": ["AnKing Step Deck", "Mastering Radiology"],
    "search_query": '"deck:AnKing Step Deck"',
    "search_query_history": [
        '"deck:AnKing Step Deck"',
        '"deck:AnKing Step Deck" OR "deck:Mastering Radiology"',
    ],
    "search_input_mode": "keyword",
    "keyword_min_similarity": 0.55,
    "keyword_translate": True,
    "chat_provider": "lm_studio",
    "chat_base_url": "http://127.0.0.1:1234/v1",
    "chat_api_key": "",
    "chat_model": "",
    "chat_max_tokens": 16384,
    "chat_use_json_schema": False,
    "embed_provider": "lm_studio",
    "embed_base_url": "http://127.0.0.1:1234/v1",
    "embed_api_key": "",
    "embed_model": "",
    "embed_query_instruction": "",
    "embed_language_strategy": "translate_to_en",
    "sensitivity": 0.5,
    "relative_score_window": 0.05,
    "max_notes_per_concept": 10,
    "max_cards_per_unit": 150,
    "hybrid_dense_weight": 1.0,
    "similar_scope": "all",
    "similar_custom_query": "",
    "similar_min_score": 0.65,
    "similar_max_notes": 20,
    "menu_location": "tools",
    "fts_use_italian_keywords": False,
    "enable_llm_verification": False,
    "include_active_cards_in_filtered_deck": False,
    "batch_deck_naming_style": "dash",
    "rollback_keep_reviewed_default": True,
    "indexed_fields_by_model": {
        "AnKingOverhaul (AnKing Step Deck / AnKingMed)": ["Text", "Extra"],
        "Mastering Radiology (AnkiCollab) (Mastering Radiology / drradiology)-a1ef2": ["Text", "Extra"],
        "Mastering Radiology (Mastering Radiology / drradiology)": ["Text", "Extra"],
    },
    "default_indexed_field_names": [
        "Text",
        "Extra",
        "Front",
        "Back",
        "Back Extra",
        "Header",
        "Footer",
        "Remarks",
        "Answer",
    ],
    "index_tags": True,
}


@dataclass
class ConfigManager:
    config_version: int = int(DEFAULT_CONFIG["config_version"])
    selected_decks: List[str] = field(default_factory=lambda: list(DEFAULT_CONFIG["selected_decks"]))
    search_query: str = DEFAULT_CONFIG["search_query"]
    search_query_history: List[str] = field(
        default_factory=lambda: list(DEFAULT_CONFIG["search_query_history"])
    )
    search_input_mode: str = str(DEFAULT_CONFIG["search_input_mode"])
    keyword_min_similarity: float = float(DEFAULT_CONFIG["keyword_min_similarity"])
    keyword_translate: bool = bool(DEFAULT_CONFIG["keyword_translate"])
    chat_provider: str = DEFAULT_CONFIG["chat_provider"]
    chat_base_url: str = DEFAULT_CONFIG["chat_base_url"]
    chat_api_key: str = DEFAULT_CONFIG["chat_api_key"]
    chat_model: str = DEFAULT_CONFIG["chat_model"]
    chat_max_tokens: int = int(DEFAULT_CONFIG["chat_max_tokens"])
    chat_use_json_schema: bool = bool(DEFAULT_CONFIG["chat_use_json_schema"])

    embed_provider: str = DEFAULT_CONFIG["embed_provider"]
    embed_base_url: str = DEFAULT_CONFIG["embed_base_url"]
    embed_api_key: str = DEFAULT_CONFIG["embed_api_key"]
    embed_model: str = DEFAULT_CONFIG["embed_model"]
    embed_query_instruction: str = str(DEFAULT_CONFIG.get("embed_query_instruction", ""))
    embed_language_strategy: str = DEFAULT_CONFIG["embed_language_strategy"]

    sensitivity: float = float(DEFAULT_CONFIG["sensitivity"])
    relative_score_window: float = float(DEFAULT_CONFIG["relative_score_window"])
    max_notes_per_concept: int = int(DEFAULT_CONFIG["max_notes_per_concept"])
    max_cards_per_unit: int = int(DEFAULT_CONFIG["max_cards_per_unit"])
    top_k_per_concept: Optional[int] = None
    hybrid_dense_weight: float = float(DEFAULT_CONFIG["hybrid_dense_weight"])
    similar_scope: str = str(DEFAULT_CONFIG["similar_scope"])
    similar_custom_query: str = str(DEFAULT_CONFIG["similar_custom_query"])
    similar_min_score: float = float(DEFAULT_CONFIG["similar_min_score"])
    similar_max_notes: int = int(DEFAULT_CONFIG["similar_max_notes"])
    menu_location: str = str(DEFAULT_CONFIG.get("menu_location", "tools"))
    fts_use_italian_keywords: bool = bool(DEFAULT_CONFIG.get("fts_use_italian_keywords", False))
    enable_llm_verification: bool = bool(DEFAULT_CONFIG["enable_llm_verification"])
    include_active_cards_in_filtered_deck: bool = bool(DEFAULT_CONFIG["include_active_cards_in_filtered_deck"])
    batch_deck_naming_style: str = DEFAULT_CONFIG["batch_deck_naming_style"]
    rollback_keep_reviewed_default: bool = bool(DEFAULT_CONFIG["rollback_keep_reviewed_default"])

    indexed_fields_by_model: Dict[str, List[str]] = field(
        default_factory=lambda: dict(DEFAULT_CONFIG["indexed_fields_by_model"])
    )
    default_indexed_field_names: List[str] = field(
        default_factory=lambda: list(DEFAULT_CONFIG["default_indexed_field_names"])
    )
    index_tags: bool = bool(DEFAULT_CONFIG["index_tags"])

    def get_active_model_id(self) -> str:
        """Returns the unique identifier for the currently configured embedding model."""
        model = self.embed_model.strip() if self.embed_model else "default"
        provider = self.embed_provider.strip() if self.embed_provider else "custom"
        return f"{provider}::{model}"

    def get_min_similarity_threshold(self) -> float:
        """Calculates minimum cosine similarity cutoff from sensitivity.

        sensitivity: 0.0 (Strict / Precision) -> threshold = 0.70
        sensitivity: 0.5 (Default)            -> threshold = 0.60
        sensitivity: 1.0 (Broad / Recall)     -> threshold = 0.50
        formula: 0.70 - clamped * 0.20
        """
        clamped = max(0.0, min(1.0, float(self.sensitivity)))
        return round(0.70 - clamped * 0.20, 4)

    def get_keyword_min_similarity_threshold(self) -> float:
        """Returns the minimum similarity threshold for keyword mode."""
        return float(getattr(self, "keyword_min_similarity", 0.55))

    def get_effective_top_k(self) -> int:
        """Returns the number of top candidate notes to keep per concept."""
        if self.top_k_per_concept is not None:
            return max(1, int(self.top_k_per_concept))
        return max(1, int(self.max_notes_per_concept))

    def get_indexed_fields_for_model(self, model_name: str, available_fields: List[str]) -> List[str]:
        """Resolves which fields should be indexed for a given note model name."""
        if model_name in self.indexed_fields_by_model:
            configured = self.indexed_fields_by_model[model_name]
            matched = [f for f in configured if f in available_fields]
            if matched:
                return matched

        matched_default = [f for f in self.default_indexed_field_names if f in available_fields]
        if matched_default:
            return matched_default

        # Fallback to all fields if none match
        return list(available_fields)

    def to_dict(self) -> Dict[str, Any]:
        """Returns configuration as a plain dictionary."""
        d = asdict(self)
        if "top_k_per_concept" in d and d["top_k_per_concept"] is None:
            del d["top_k_per_concept"]
        return d

    def get_effective_search_query(self) -> str:
        """Returns the active Browser search query, falling back to selected_decks if empty."""
        q = (self.search_query or "").strip()
        if q:
            return q
        if self.selected_decks:
            return " OR ".join(f'"deck:{d}"' for d in self.selected_decks if d.strip())
        return '"deck:AnKing Step Deck"'

    def add_query_to_history(self, query: str, max_items: int = 20) -> None:
        """Sets the active search_query and prepends it to search_query_history without duplicates."""
        clean = (query or "").strip()
        if not clean:
            return
        self.search_query = clean
        new_hist = [clean] + [q for q in (self.search_query_history or []) if q.strip() and q.strip() != clean]
        self.search_query_history = new_hist[:max_items]

    @classmethod
    def migrate_dict(cls, data: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Migrates a raw config dict if config_version is missing or < 5.

        If config_version is missing or < 4, overwrites with v4 defaults:
        - sensitivity: 0.5
        - relative_score_window: 0.05
        - max_notes_per_concept: 10
        - max_cards_per_unit: 150
        - top_k_per_concept: removed if present
        and sets config_version: 4.

        If config_version < 5:
        - sets search_input_mode: "keyword"
        - sets keyword_min_similarity: 0.52
        and sets config_version: 5.

        All other user keys remain untouched.
        Returns (migrated_dict, was_migrated).
        """
        was_migrated = False
        version = data.get("config_version")
        if version is None or int(version) < 4:
            data["sensitivity"] = 0.5
            data["relative_score_window"] = 0.05
            data["max_notes_per_concept"] = 10
            data["max_cards_per_unit"] = 150
            data["hybrid_dense_weight"] = 1.0
            data.pop("top_k_per_concept", None)
            data["config_version"] = 4
            version = 4
            was_migrated = True

        if int(version) < 5:
            if "search_input_mode" not in data:
                data["search_input_mode"] = "keyword"
            if "keyword_min_similarity" not in data:
                data["keyword_min_similarity"] = 0.55
            if "keyword_translate" not in data:
                data["keyword_translate"] = True
            data["config_version"] = 5
            was_migrated = True

        if data.get("keyword_min_similarity") == 0.52:
            data["keyword_min_similarity"] = 0.55
            was_migrated = True

        return data, was_migrated

    def update(self, data: Dict[str, Any]) -> None:
        """Updates instance attributes from dictionary keys."""
        migrated, _ = self.migrate_dict(dict(data))
        has_explicit_query = "search_query" in migrated and bool(str(migrated.get("search_query", "")).strip())
        for key, val in migrated.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if "top_k_per_concept" not in migrated:
            self.top_k_per_concept = None
        if not has_explicit_query and "selected_decks" in migrated and self.selected_decks:
            migrated_q = " OR ".join(f'"deck:{d}"' for d in self.selected_decks if str(d).strip())
            if migrated_q:
                self.search_query = migrated_q
                if migrated_q not in self.search_query_history:
                    self.search_query_history = [migrated_q] + list(self.search_query_history)

    @classmethod
    def load(cls, addon_name: str = "ankihub_searcher", config_path: Optional[str] = None) -> ConfigManager:
        """Loads configuration from Anki addonManager or fallback JSON file."""
        # 1. Try reading from Anki mw if available
        try:
            from aqt import mw  # type: ignore

            if mw and hasattr(mw, "addonManager") and mw.addonManager:
                resolved_name = addon_name
                try:
                    mod_addon = mw.addonManager.addonFromModule(__name__)
                    if mod_addon:
                        resolved_name = mod_addon
                except Exception:
                    pass
                raw_cfg = mw.addonManager.getConfig(resolved_name)
                if isinstance(raw_cfg, dict):
                    # Migrate raw config BEFORE merging with DEFAULT_CONFIG
                    migrated_raw, was_migrated = cls.migrate_dict(dict(raw_cfg))
                    merged = dict(DEFAULT_CONFIG)
                    merged.update(migrated_raw)
                    instance = cls()
                    instance.update(merged)
                    if was_migrated:
                        instance.save(addon_name=resolved_name)
                    return instance
        except Exception:
            pass

        # 2. Try loading from custom config_path or local config.json
        search_paths = []
        if config_path:
            search_paths.append(config_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(base_dir, "config.json"))

        was_file_migrated = False
        target_migrated_path = None
        migrated_file_cfg: Optional[Dict[str, Any]] = None

        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_cfg = json.load(f)
                    if isinstance(file_cfg, dict):
                        migrated_raw, was_migrated = cls.migrate_dict(dict(file_cfg))
                        migrated_file_cfg = migrated_raw
                        if was_migrated:
                            was_file_migrated = True
                            target_migrated_path = path
                        break
                except Exception:
                    pass

        merged = dict(DEFAULT_CONFIG)
        if migrated_file_cfg is not None:
            merged.update(migrated_file_cfg)

        instance = cls()
        instance.update(merged)
        if was_file_migrated and target_migrated_path and config_path and target_migrated_path == config_path:
            instance.save(config_path=target_migrated_path)
        return instance

    def save(self, addon_name: str = "ankihub_searcher", config_path: Optional[str] = None) -> None:
        """Saves current configuration to Anki addonManager or fallback JSON file."""
        data = self.to_dict()

        # 1. Try saving to Anki mw
        try:
            from aqt import mw  # type: ignore

            if mw and hasattr(mw, "addonManager") and mw.addonManager:
                resolved_name = addon_name
                try:
                    mod_addon = mw.addonManager.addonFromModule(__name__)
                    if mod_addon:
                        resolved_name = mod_addon
                except Exception:
                    pass
                mw.addonManager.writeConfig(resolved_name, data)
                return
        except Exception:
            pass

        # 2. Fallback to saving to file
        target_path = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Silently fail if read-only, but log if needed
            print(f"[SmartSearch] Warning: failed to save config file {target_path}: {e}")
