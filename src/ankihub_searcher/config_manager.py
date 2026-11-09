"""Configuration manager for Smart Search.

Works seamlessly inside Anki (via mw.addonManager) and in standalone test environments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "selected_decks": ["AnKing Step Deck", "Mastering Radiology"],
    "search_query": '"deck:AnKing Step Deck"',
    "search_query_history": [
        '"deck:AnKing Step Deck"',
        '"deck:AnKing Step Deck" OR "deck:Mastering Radiology"',
    ],
    "chat_provider": "lm_studio",
    "chat_base_url": "http://127.0.0.1:1234/v1",
    "chat_api_key": "",
    "chat_model": "",
    "chat_max_tokens": 4096,
    "embed_provider": "lm_studio",
    "embed_base_url": "http://127.0.0.1:1234/v1",
    "embed_api_key": "",
    "embed_model": "",
    "embed_language_strategy": "translate_to_en",
    "sensitivity": 0.55,
    "top_k_per_concept": 8,
    "max_cards_per_unit": 80,
    "hybrid_dense_weight": 0.78,
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
    selected_decks: List[str] = field(default_factory=lambda: list(DEFAULT_CONFIG["selected_decks"]))
    search_query: str = DEFAULT_CONFIG["search_query"]
    search_query_history: List[str] = field(
        default_factory=lambda: list(DEFAULT_CONFIG["search_query_history"])
    )
    chat_provider: str = DEFAULT_CONFIG["chat_provider"]
    chat_base_url: str = DEFAULT_CONFIG["chat_base_url"]
    chat_api_key: str = DEFAULT_CONFIG["chat_api_key"]
    chat_model: str = DEFAULT_CONFIG["chat_model"]
    chat_max_tokens: int = int(DEFAULT_CONFIG["chat_max_tokens"])

    embed_provider: str = DEFAULT_CONFIG["embed_provider"]
    embed_base_url: str = DEFAULT_CONFIG["embed_base_url"]
    embed_api_key: str = DEFAULT_CONFIG["embed_api_key"]
    embed_model: str = DEFAULT_CONFIG["embed_model"]
    embed_language_strategy: str = DEFAULT_CONFIG["embed_language_strategy"]

    sensitivity: float = float(DEFAULT_CONFIG["sensitivity"])
    top_k_per_concept: int = int(DEFAULT_CONFIG["top_k_per_concept"])
    max_cards_per_unit: int = int(DEFAULT_CONFIG["max_cards_per_unit"])
    hybrid_dense_weight: float = float(DEFAULT_CONFIG["hybrid_dense_weight"])
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

        sensitivity: 0.0 (Strict / Precision) -> threshold = 0.68
        sensitivity: 1.0 (Broad / Recall)     -> threshold = 0.38
        default 0.55                          -> threshold ~ 0.515
        """
        clamped = max(0.0, min(1.0, float(self.sensitivity)))
        return round(0.68 - clamped * 0.30, 4)

    def get_effective_top_k(self) -> int:
        """Returns the number of top candidate notes to keep per concept."""
        return max(1, int(self.top_k_per_concept))

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
        return asdict(self)

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

    def update(self, data: Dict[str, Any]) -> None:
        """Updates instance attributes from dictionary keys."""
        has_explicit_query = "search_query" in data and bool(str(data.get("search_query", "")).strip())
        for key, val in data.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if not has_explicit_query and "selected_decks" in data and self.selected_decks:
            migrated_q = " OR ".join(f'"deck:{d}"' for d in self.selected_decks if str(d).strip())
            if migrated_q:
                self.search_query = migrated_q
                if migrated_q not in self.search_query_history:
                    self.search_query_history = [migrated_q] + list(self.search_query_history)

    @classmethod
    def load(cls, addon_name: str = "ankihub_searcher", config_path: Optional[str] = None) -> ConfigManager:
        """Loads configuration from Anki addonManager or fallback JSON file."""
        merged: Dict[str, Any] = dict(DEFAULT_CONFIG)

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
                anki_cfg = mw.addonManager.getConfig(resolved_name)
                if isinstance(anki_cfg, dict):
                    merged.update(anki_cfg)
                    instance = cls()
                    instance.update(merged)
                    return instance
        except Exception:
            pass

        # 2. Try loading from custom config_path or local config.json
        search_paths = []
        if config_path:
            search_paths.append(config_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(base_dir, "config.json"))

        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_cfg = json.load(f)
                    if isinstance(file_cfg, dict):
                        merged.update(file_cfg)
                        break
                except Exception:
                    pass

        instance = cls()
        instance.update(merged)
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
