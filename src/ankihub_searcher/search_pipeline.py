"""Multi-stage hybrid search pipeline for Smart Search.

Stage 1: LLM Concept Decomposition & Cross-Lingual Medical Query Expansion
Stage 2: Hybrid Retrieval (Dense Embeddings + SQLite FTS5 BM25) with RRF/linear fusion
Stage 3: Optional LLM Verification & Reranking
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

from .ai_client import AIClientError, OpenAICompatibleClient
from .config_manager import ConfigManager
from .storage import SearchDatabase, normalize_l2
from .text_utils import (
    LectureUnit,
    build_structural_representation,
    compute_structural_similarity,
    extract_primary_field_raw,
    normalize_inline_latex,
)


class Stage1ExtractionError(AIClientError):
    """Raised when Stage 1 concept extraction fails after retries."""

    pass


@dataclass
class ExtractedConcept:
    """An atomic medical concept extracted by LLM from a lecture unit."""

    topic_it: str
    claim_en: str
    keywords_en: List[str]
    keywords_it: List[str] = field(default_factory=list)
    quote_it: str = ""
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_it": self.topic_it,
            "claim_en": self.claim_en,
            "keywords_en": list(self.keywords_en),
            "keywords_it": list(self.keywords_it),
            "quote_it": self.quote_it,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedConcept":
        return cls(
            topic_it=str(data.get("topic_it", "")),
            claim_en=str(data.get("claim_en", "")),
            keywords_en=[str(x) for x in data.get("keywords_en", [])],
            keywords_it=[str(x) for x in data.get("keywords_it", [])],
            quote_it=str(data.get("quote_it", "")),
            source=str(data.get("source", "llm")),
        )


@dataclass
class ScoredNote:
    """A candidate note identified by the search pipeline with its relevance score."""

    nid: int
    score: float
    matched_concepts: List[str]
    clean_text_preview: str = ""


@dataclass
class ConceptCoverage:
    """Tracks how well the Anki collection covers an extracted lecture concept/claim."""

    topic_it: str
    claim_en: str
    keywords_en: List[str] = field(default_factory=list)
    keywords_it: List[str] = field(default_factory=list)
    matched_notes_count: int = 0
    matched_nids: List[int] = field(default_factory=list)
    best_score: float = 0.0
    best_nid: Optional[int] = None
    best_note_preview: str = ""
    status: str = "uncovered"  # "covered" | "partial" | "uncovered"
    quote_it: str = ""
    shared_with: str = ""
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_it": self.topic_it,
            "claim_en": self.claim_en,
            "keywords_en": list(self.keywords_en),
            "keywords_it": list(self.keywords_it),
            "matched_notes_count": int(self.matched_notes_count),
            "matched_nids": list(self.matched_nids),
            "best_score": float(self.best_score),
            "best_nid": self.best_nid,
            "best_note_preview": self.best_note_preview,
            "status": self.status,
            "quote_it": self.quote_it,
            "shared_with": self.shared_with,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptCoverage":
        return cls(
            topic_it=str(data.get("topic_it", "")),
            claim_en=str(data.get("claim_en", "")),
            keywords_en=[str(x) for x in data.get("keywords_en", [])],
            keywords_it=[str(x) for x in data.get("keywords_it", [])],
            matched_notes_count=int(data.get("matched_notes_count", 0)),
            matched_nids=[int(x) for x in data.get("matched_nids", [])],
            best_score=float(data.get("best_score", 0.0)),
            best_nid=int(data["best_nid"]) if data.get("best_nid") is not None else None,
            best_note_preview=str(data.get("best_note_preview", "")),
            status=str(data.get("status", "uncovered")),
            quote_it=str(data.get("quote_it", "")),
            shared_with=str(data.get("shared_with", "")),
            source=str(data.get("source", "llm")),
        )


SYSTEM_PROMPT_STAGE1 = """You are an elite medical professor, USMLE Step 1/Step 2 expert, and AnKing deck specialist.
Your task is to decompose the following Italian lecture text into atomic medical concepts and translate them into high-yield USMLE English claims and keywords.

CRITICAL INSTRUCTIONS:
1. Extract an atomic concept for EVERY distinct entity mentioned in the text:
   - Every disease, syndrome, or pathology.
   - Every molecular/cellular mechanism, pathway, or receptor.
   - Every molecule, protein, cytokine, antibody, or biomarker.
   - Every drug, pharmacological class, or therapeutic agent.
   - Every clinical sign, symptom, or diagnostic finding.
   - Every numerical criterion, cutoff, stage, or diagnostic threshold.
2. STRICT RULES:
   - NEVER merge distinct entities together into a single concept (e.g., do not combine two different drugs or two separate cytokines).
   - Forbid generic concepts without a specific entity (e.g., avoid vague concepts like "general therapy" or "pathology overview").
   - Ignore non-medical digressions entirely (costs, hospital administration, course organization, historical trivia, personal anecdotes).
3. Resolve Italian medical terminology, historical acronyms, and eponyms into standard USMLE English equivalents:
   - e.g., "NALP3" -> "NLRP3 inflammasome"
   - e.g., "pirofosfato di calcio" -> "calcium pyrophosphate dihydrate (CPPD)", "pseudogout"
   - e.g., "acido urico" -> "uric acid", "monosodium urate", "gout"
   - e.g., "fattore di Hageman" -> "Factor XII", "Hageman factor"
   - e.g., "segno di Cullen" -> "Cullen sign", "acute pancreatitis"
4. For each concept, provide:
   - "quote_it": The exact Italian sentence or phrase from the lecture text from which this concept originates.
   - "topic_it": Short Italian topic summary naming the specific entity.
   - "claim_en": 1-2 concise, fact-dense English sentences summarizing the mechanism, pathology, clinical signs, or radiology findings as tested on USMLE/AnKing cards.
   - "keywords_en": 4-10 precise English keywords, synonyms, eponyms, and acronyms.
   - "keywords_it": 3-6 original Italian terms from the lecture.

5. Do NOT overthink or write long reasoning monologues. Be fast, direct, and output ONLY the JSON object immediately.

Respond strictly with a JSON object in this format:
{
  "concepts": [
    {
      "quote_it": "...",
      "topic_it": "...",
      "claim_en": "...",
      "keywords_en": ["...", "..."],
      "keywords_it": ["...", "..."]
    }
  ]
}
"""

SYSTEM_PROMPT_STAGE3 = """You are an expert medical educator reviewing flashcards for a medical student's lecture.
The student is studying the following lecture topic. Determine which candidate flashcards are genuinely relevant and essential to this lecture.
For each candidate card, you are provided with its text and the excerpt/quote from the lecture that retrieved it.

Respond strictly with a JSON object containing the list of relevant note IDs:
{
  "relevant_nids": [12345, 67890]
}
"""


STAGE1_JSON_SCHEMA: Dict[str, Any] = {
    "name": "stage1_concepts",
    "schema": {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote_it": {"type": "string"},
                        "topic_it": {"type": "string"},
                        "claim_en": {"type": "string"},
                        "keywords_en": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "keywords_it": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["quote_it", "topic_it", "claim_en", "keywords_en", "keywords_it"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["concepts"],
        "additionalProperties": False,
    },
}


SYSTEM_PROMPT_KEYWORD_TRANSLATE = """You are an expert medical translation assistant specializing in USMLE and AnKing medical flashcard decks.
Your task is to translate Italian medical keywords into their standard English medical terminology as used in USMLE / AnKing.

Guidelines:
1. For each Italian medical keyword provided, return its standard English medical term.
2. The keywords are grouped by section/unit title to provide context and disambiguate acronyms (for example, in an inflammatory context "PCR" means "C-reactive protein").
3. Well-known universal medical acronyms can remain acronyms, but if an acronym is ambiguous, expand it (e.g., "IDL" -> "IDL (intermediate-density lipoprotein)").
4. Provide only a single short translated term per keyword. Do NOT write full sentences or explanations.
5. Return strictly a JSON object with a "translations" array containing objects with "it" and "en" keys:
{
  "translations": [
    {"it": "ApoB48", "en": "ApoB-48"},
    {"it": "Chilomicroni", "en": "Chylomicrons"},
    {"it": "Recettore LDL", "en": "LDL receptor"}
  ]
}
"""

KEYWORD_TRANSLATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "keyword_translations",
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "it": {"type": "string"},
                        "en": {"type": "string"},
                    },
                    "required": ["it", "en"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}


def extract_first_sentence(text: str) -> str:
    """Extracts the first sentence from a text block, stripping trailing punctuation."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    first = match[0].strip() if match else cleaned
    return first.rstrip(".!?").strip()


def concepts_from_keywords(
    keywords: Sequence[str],
    translations: Optional[Dict[str, str]] = None,
) -> List[ExtractedConcept]:
    """Converts a sequence of keyword strings into ExtractedConcept instances."""
    concepts: List[ExtractedConcept] = []
    trans_lookup: Dict[str, str] = {}
    if translations:
        for k, v in translations.items():
            k_clean = str(k).strip()
            v_clean = str(v).strip()
            if k_clean and v_clean:
                trans_lookup[k_clean] = v_clean
                trans_lookup[k_clean.lower()] = v_clean

    for kw in keywords:
        cleaned = kw.strip()
        if not cleaned:
            continue
        trans = trans_lookup.get(cleaned) or trans_lookup.get(cleaned.lower()) or cleaned
        concepts.append(
            ExtractedConcept(
                topic_it=cleaned,
                claim_en=trans,
                keywords_en=[trans],
                keywords_it=[cleaned],
                quote_it=cleaned,
                source="keyword",
            )
        )
    return concepts


def select_candidates_for_concept(
    dense_scores: Dict[int, float],
    fts_scores: Dict[int, float],
    alpha: float,
    min_threshold: float,
    min_dense_floor: float,
    rel_window: float,
    max_notes_concept: int,
    suspended_nids: Optional[Set[int]] = None,
    include_active_cards: bool = False,
) -> Tuple[List[Tuple[int, float]], Optional[int], float]:
    """Selects top candidate notes for a single concept using score fusion and relative window cutoff.

    Returns: (top_concept_candidates, best_nid, best_score)
    """
    concept_candidates: List[Tuple[int, float]] = []
    all_candidate_nids = set(dense_scores.keys())

    for nid in all_candidate_nids:
        d_score = dense_scores.get(nid, 0.0)
        if d_score < min_dense_floor:
            continue

        if alpha >= 1.0:
            combined = d_score
        else:
            b_score = fts_scores.get(nid, 0.0)
            combined = alpha * d_score + (1.0 - alpha) * b_score

        if combined >= min_threshold:
            concept_candidates.append((nid, combined))

    concept_candidates.sort(key=lambda x: x[1], reverse=True)

    best_nid: Optional[int] = None
    best_score: float = 0.0
    if concept_candidates:
        best_nid, best_score = concept_candidates[0][0], round(concept_candidates[0][1], 4)
        rel_cutoff = max(min_threshold, best_score - rel_window)
        surviving_candidates = [c for c in concept_candidates if c[1] >= rel_cutoff]
    else:
        surviving_candidates = []
        if dense_scores:
            top_d_nid = max(dense_scores.keys(), key=lambda k: dense_scores[k])
            top_d_score = dense_scores[top_d_nid]
            best_nid = top_d_nid
            if alpha >= 1.0:
                best_score = round(top_d_score, 4)
            else:
                best_score = round(alpha * top_d_score + (1.0 - alpha) * fts_scores.get(top_d_nid, 0.0), 4)

    if suspended_nids is not None and not include_active_cards:
        top_concept_candidates: List[Tuple[int, float]] = []
        susp_kept = 0
        active_kept = 0
        for nid, score in surviving_candidates:
            if nid in suspended_nids:
                if susp_kept < max_notes_concept:
                    top_concept_candidates.append((nid, score))
                    susp_kept += 1
            else:
                if active_kept < max_notes_concept:
                    top_concept_candidates.append((nid, score))
                    active_kept += 1
            if susp_kept >= max_notes_concept and active_kept >= max_notes_concept:
                break
    else:
        top_concept_candidates = surviving_candidates[:max_notes_concept]

    return top_concept_candidates, best_nid, best_score


class SmartSearchPipeline:
    """Executes the multi-stage hybrid search pipeline."""

    def __init__(
        self,
        config: ConfigManager,
        storage: SearchDatabase,
        ai_client: OpenAICompatibleClient,
    ):
        self.config = config
        self.storage = storage
        self.ai_client = ai_client
        self.last_concept_coverage: List[ConceptCoverage] = []
        self.last_unit_error: str = ""

    # ---------------- Stage 1: Concept Extraction ----------------

    def extract_concepts(self, unit: LectureUnit) -> List[ExtractedConcept]:
        """Uses the Chat LLM to decompose an Italian lecture unit into atomic English concepts.

        Tries up to 2 times on parsing/LLM failure. If both attempts fail, raises
        Stage1ExtractionError without falling back to ASCII words.
        """
        clean_title = normalize_inline_latex(unit.unit_title)
        clean_body = normalize_inline_latex(unit.body_text)

        user_prompt = f"### Lecture Unit Title: {clean_title}\n"
        if unit.breadcrumb:
            clean_breadcrumb = normalize_inline_latex(unit.breadcrumb)
            user_prompt += f"### Context Hierarchy: {clean_breadcrumb}\n"
        user_prompt += f"### Lecture Text:\n{clean_body}"

        # /no_think only for Qwen models
        if "qwen" in (self.config.chat_model or "").lower():
            user_prompt += "\n\n/no_think"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STAGE1},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_concepts(res_obj: Any) -> List[ExtractedConcept]:
            raw_concepts = []
            if isinstance(res_obj, dict) and "concepts" in res_obj:
                raw_concepts = res_obj["concepts"]
            elif isinstance(res_obj, list):
                raw_concepts = res_obj

            parsed_list: List[ExtractedConcept] = []
            for item in raw_concepts:
                if not isinstance(item, dict):
                    continue
                topic_it = str(item.get("topic_it", "")).strip()
                claim_en = str(item.get("claim_en", "")).strip()
                kws_en = [str(k).strip() for k in item.get("keywords_en", []) if str(k).strip()]
                kws_it = [str(k).strip() for k in item.get("keywords_it", []) if str(k).strip()]
                quote_it = str(item.get("quote_it", "")).strip()

                if claim_en or kws_en or topic_it:
                    parsed_list.append(
                        ExtractedConcept(
                            topic_it=topic_it or unit.unit_title,
                            claim_en=claim_en or topic_it,
                            keywords_en=kws_en or [unit.unit_title],
                            keywords_it=kws_it,
                            quote_it=quote_it,
                        )
                    )
            return parsed_list

        last_err: Optional[Exception] = None
        max_tokens = getattr(self.config, "chat_max_tokens", 16384)
        use_schema = bool(getattr(self.config, "chat_use_json_schema", False))

        for attempt in range(2):
            try:
                schema_to_use = STAGE1_JSON_SCHEMA if use_schema else None
                res = self.ai_client.chat_completion_json(
                    messages=messages,
                    model=self.config.chat_model,
                    base_url=self.config.chat_base_url,
                    api_key=self.config.chat_api_key,
                    temperature=0.1,
                    json_schema=schema_to_use,
                    max_tokens=max_tokens,
                )

                concepts = _parse_concepts(res)
                if concepts:
                    return concepts

                # If parsed response contains 0 concepts and response_format was active,
                # retry once without response_format before counting the attempt as failed.
                if schema_to_use is not None:
                    print(
                        f"[SmartSearch] Notice: 0 concepts with response_format active in attempt {attempt + 1}, retrying once without response_format..."
                    )
                    res_no_schema = self.ai_client.chat_completion_json(
                        messages=messages,
                        model=self.config.chat_model,
                        base_url=self.config.chat_base_url,
                        api_key=self.config.chat_api_key,
                        temperature=0.1,
                        json_schema=None,
                        max_tokens=max_tokens,
                    )
                    concepts_no_schema = _parse_concepts(res_no_schema)
                    if concepts_no_schema:
                        return concepts_no_schema

                raise Stage1ExtractionError("LLM response did not contain any valid medical concepts.")

            except Exception as e:
                last_err = e
                print(f"[SmartSearch] Notice: Stage 1 extraction attempt {attempt + 1} failed: {e}")

        # Both attempts failed: do NOT fall back to arbitrary ASCII words
        raise Stage1ExtractionError(
            f"Stage 1 extraction failed after 2 attempts: {last_err}"
        )

    # ---------------- Keyword Translation ----------------

    def translate_keywords(self, units: Sequence[LectureUnit]) -> Dict[str, str]:
        """Translates Italian medical keywords into USMLE/AnKing English terms in a single LLM call.

        Keywords are grouped by unit title for contextual disambiguation.
        Returns a dict mapping original Italian keyword to translated English term.
        """
        prompt_lines: List[str] = []
        has_keywords = False
        for unit in units:
            kws = [k.strip() for k in getattr(unit, "keywords", []) if k.strip()]
            if not kws:
                continue
            has_keywords = True
            title = unit.unit_title.strip() if unit.unit_title else "Keywords"
            prompt_lines.append(f"### Unit: {title}")
            for kw in kws:
                prompt_lines.append(f"- {kw}")

        if not has_keywords:
            return {}

        user_prompt = (
            "Translate the following Italian keywords to standard English medical terms used in USMLE/AnKing:\n\n"
            + "\n".join(prompt_lines)
        )

        if "qwen" in (self.config.chat_model or "").lower():
            user_prompt += "\n\n/no_think"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_KEYWORD_TRANSLATE},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_translations(res_obj: Any) -> Dict[str, str]:
            trans_dict: Dict[str, str] = {}
            if isinstance(res_obj, dict):
                raw_list = res_obj.get("translations")
                if isinstance(raw_list, list):
                    for item in raw_list:
                        if isinstance(item, dict):
                            it_v = str(item.get("it", "")).strip()
                            en_v = str(item.get("en", "")).strip()
                            if it_v and en_v:
                                trans_dict[it_v] = en_v
                elif isinstance(raw_list, dict):
                    for it_k, en_v in raw_list.items():
                        if str(it_k).strip() and str(en_v).strip():
                            trans_dict[str(it_k).strip()] = str(en_v).strip()
                elif "it" in res_obj and "en" in res_obj:
                    it_v = str(res_obj.get("it", "")).strip()
                    en_v = str(res_obj.get("en", "")).strip()
                    if it_v and en_v:
                        trans_dict[it_v] = en_v
                else:
                    for k, v in res_obj.items():
                        if isinstance(v, str) and k.strip() and v.strip():
                            trans_dict[k.strip()] = v.strip()
            elif isinstance(res_obj, list):
                for item in res_obj:
                    if isinstance(item, dict):
                        it_v = str(item.get("it", "")).strip()
                        en_v = str(item.get("en", "")).strip()
                        if it_v and en_v:
                            trans_dict[it_v] = en_v
            return trans_dict

        all_keywords: List[str] = [
            k.strip() for unit in units for k in getattr(unit, "keywords", []) if k.strip()
        ]

        def _align_to_keywords(res_obj: Any, trans_dict: Dict[str, str]) -> Dict[str, str]:
            """Maps translations back to the original keywords.

            Exact (case-insensitive) matches win; when the model rewrote the Italian side
            but returned one translation per keyword in order, fall back to position.
            """
            lower_map = {k.lower(): v for k, v in trans_dict.items()}
            ordered: List[str] = []
            raw_list = res_obj.get("translations") if isinstance(res_obj, dict) else res_obj
            if isinstance(raw_list, list):
                for item in raw_list:
                    if isinstance(item, dict):
                        ordered.append(str(item.get("en", "")).strip())
            aligned: Dict[str, str] = {}
            for i, kw in enumerate(all_keywords):
                en = trans_dict.get(kw) or lower_map.get(kw.lower())
                if not en and len(ordered) == len(all_keywords):
                    en = ordered[i]
                if en:
                    aligned[kw] = en
            return aligned

        last_err: Optional[Exception] = None
        max_tokens = getattr(self.config, "chat_max_tokens", 16384)
        use_schema = bool(getattr(self.config, "chat_use_json_schema", False))

        for attempt in range(2):
            try:
                schema_to_use = KEYWORD_TRANSLATION_JSON_SCHEMA if use_schema else None
                res = self.ai_client.chat_completion_json(
                    messages=messages,
                    model=self.config.chat_model,
                    base_url=self.config.chat_base_url,
                    api_key=self.config.chat_api_key,
                    temperature=0.1,
                    json_schema=schema_to_use,
                    max_tokens=max_tokens,
                )

                translations = _align_to_keywords(res, _parse_translations(res))
                if translations:
                    return translations

                if schema_to_use is not None:
                    print(
                        f"[SmartSearch] Notice: 0 translations with response_format active in attempt {attempt + 1}, retrying once without response_format..."
                    )
                    res_no_schema = self.ai_client.chat_completion_json(
                        messages=messages,
                        model=self.config.chat_model,
                        base_url=self.config.chat_base_url,
                        api_key=self.config.chat_api_key,
                        temperature=0.1,
                        json_schema=None,
                        max_tokens=max_tokens,
                    )
                    translations_no_schema = _align_to_keywords(res_no_schema, _parse_translations(res_no_schema))
                    if translations_no_schema:
                        return translations_no_schema

                raise RuntimeError("LLM response did not contain any valid keyword translations.")

            except Exception as e:
                last_err = e
                print(f"[SmartSearch] Notice: Keyword translation attempt {attempt + 1} failed: {e}")

        raise RuntimeError(f"Keyword translation failed after 2 attempts: {last_err}")

    # ---------------- Stage 2: Hybrid Retrieval ----------------

    def retrieve_for_concepts(
        self,
        concepts: List[ExtractedConcept],
        candidate_nids: Optional[Union[List[int], Set[int]]] = None,
        active_model_id: str = "",
        suspended_nids: Optional[Set[int]] = None,
        *,
        min_threshold: Optional[float] = None,
        relative_score_window: Optional[float] = None,
        max_notes_per_concept: Optional[int] = None,
        max_cards_per_unit: Optional[int] = None,
        hybrid_dense_weight: Optional[float] = None,
        include_active_cards: Optional[bool] = None,
        concept_vectors: Optional[List[List[float]]] = None,
    ) -> List[ScoredNote]:
        """Performs dense vector search + BM25 lexical search per concept and fuses results."""
        if not concepts:
            return []

        # Allow flexible parameter swapping if candidate_nids was passed as active_model_id string
        if isinstance(candidate_nids, str) and not active_model_id:
            active_model_id = candidate_nids
            candidate_nids = None

        if not active_model_id:
            active_model_id = self.config.get_active_model_id()

        # 1. Build query texts for embedding based on language strategy
        query_texts: List[str] = []
        for c in concepts:
            if getattr(c, "source", "llm") == "keyword":
                text = c.claim_en or c.topic_it
            elif self.config.embed_language_strategy == "multilingual_direct":
                kws = ", ".join(k.strip() for k in c.keywords_en if k.strip())
                text = f"{c.topic_it} | {c.claim_en} | {kws}"
            else:  # translate_to_en (default)
                kws = ", ".join(k.strip() for k in c.keywords_en if k.strip())
                first_sentence = extract_first_sentence(c.claim_en)
                if first_sentence and kws:
                    text = f"{first_sentence} | {kws}"
                else:
                    text = first_sentence or kws or c.claim_en
            query_texts.append(text)

        # Resolve optional query instruction (e.g. for qwen3-embedding)
        instruction = getattr(self.config, "embed_query_instruction", "")
        if not instruction:
            embed_model_name = (self.config.embed_model or "").lower()
            if "qwen3-embedding" in embed_model_name:
                instruction = "Instruct: Given a medical lecture concept, retrieve flashcards that test it\nQuery: "

        if instruction:
            query_texts = [f"{instruction}{t}" for t in query_texts]

        # Batch embed all concept queries at once (unless already provided)
        if concept_vectors is None:
            concept_vectors = self.ai_client.get_embeddings(
                texts=query_texts,
                model=self.config.embed_model,
                base_url=self.config.embed_base_url,
                api_key=self.config.embed_api_key,
            )

        eff_threshold = min_threshold if min_threshold is not None else self.config.get_min_similarity_threshold()
        # Semantic floor: prevents purely lexical hits with 0 semantic relation
        min_dense_floor = max(0.20, eff_threshold - 0.18)
        alpha = hybrid_dense_weight if hybrid_dense_weight is not None else self.config.hybrid_dense_weight
        rel_window = relative_score_window if relative_score_window is not None else getattr(self.config, "relative_score_window", 0.05)
        max_notes_concept = max_notes_per_concept if max_notes_per_concept is not None else getattr(self.config, "max_notes_per_concept", 10)
        max_cards_unit = max_cards_per_unit if max_cards_per_unit is not None else getattr(self.config, "max_cards_per_unit", 150)
        eff_include_active = include_active_cards if include_active_cards is not None else self.config.include_active_cards_in_filtered_deck

        # Track results per note: nid -> {"scores": [float], "concepts": [str]}
        notes_agg: Dict[int, Dict[str, Any]] = {}
        raw_coverage_list: List[ConceptCoverage] = []

        # Pre-load candidate vectors ONCE for all concepts in this unit
        loaded_nids, loaded_vecs, _ = self.storage.load_embeddings_for_notes(
            nids=candidate_nids,
            model_id=active_model_id,
        )

        for idx, concept in enumerate(concepts):
            q_vec = concept_vectors[idx] if idx < len(concept_vectors) else []
            if not q_vec:
                raw_coverage_list.append(
                    ConceptCoverage(
                        topic_it=concept.topic_it,
                        claim_en=concept.claim_en,
                        keywords_en=list(concept.keywords_en),
                        keywords_it=list(concept.keywords_it),
                        quote_it=concept.quote_it,
                        source=getattr(concept, "source", "llm"),
                    )
                )
                continue

            # A. Dense Cosine Similarity against pre-loaded matrix/vectors
            dense_hits = self.storage.compute_similarities_from_loaded(
                nid_list=loaded_nids,
                vecs=loaded_vecs,
                query_vec=q_vec,
            )
            dense_scores: Dict[int, float] = {nid: score for nid, score in dense_hits}

            # B. Lexical FTS5 BM25 (skipped completely when hybrid_dense_weight is 1.0)
            fts_scores: Dict[int, float] = {}
            if alpha < 1.0:
                if getattr(self.config, "fts_use_italian_keywords", False):
                    all_kws = concept.keywords_en + concept.keywords_it
                else:
                    all_kws = list(concept.keywords_en) if concept.keywords_en else list(concept.keywords_it)
                fts_hits = self.storage.search_fts(
                    keywords=all_kws,
                    candidate_nids=candidate_nids,
                    limit=150,
                )
                fts_scores = {nid: score for nid, score in fts_hits}

            # C. Score Fusion & Candidate Selection
            top_concept_candidates, best_nid, best_score = select_candidates_for_concept(
                dense_scores=dense_scores,
                fts_scores=fts_scores,
                alpha=alpha,
                min_threshold=eff_threshold,
                min_dense_floor=min_dense_floor,
                rel_window=rel_window,
                max_notes_concept=max_notes_concept,
                suspended_nids=suspended_nids,
                include_active_cards=eff_include_active,
            )

            matched_c_nids = [nid for nid, _ in top_concept_candidates]
            matched_cnt = len(matched_c_nids)
            if matched_cnt == 0:
                c_status = "uncovered"
            elif matched_cnt == 1 or best_score < (eff_threshold + 0.06):
                c_status = "partial"
            else:
                c_status = "covered"

            raw_coverage_list.append(
                ConceptCoverage(
                    topic_it=concept.topic_it,
                    claim_en=concept.claim_en,
                    keywords_en=list(concept.keywords_en),
                    keywords_it=list(concept.keywords_it),
                    matched_notes_count=matched_cnt,
                    matched_nids=matched_c_nids,
                    best_score=best_score,
                    best_nid=best_nid,
                    status=c_status,
                    quote_it=concept.quote_it,
                    source=getattr(concept, "source", "llm"),
                )
            )

            c_label = concept.topic_it or concept.claim_en[:40]
            for nid, score in top_concept_candidates:
                if nid not in notes_agg:
                    notes_agg[nid] = {"scores": [], "concepts": []}
                notes_agg[nid]["scores"].append(score)
                if c_label not in notes_agg[nid]["concepts"]:
                    notes_agg[nid]["concepts"].append(c_label)

        # Populate best_note_preview for coverage items
        best_nids_to_load = list({c.best_nid for c in raw_coverage_list if c.best_nid is not None})
        best_content_map = self.storage.get_notes_content(best_nids_to_load) if best_nids_to_load else {}
        for cov in raw_coverage_list:
            if cov.best_nid is not None and cov.best_nid in best_content_map:
                txt = best_content_map[cov.best_nid]["clean_text"]
                cov.best_note_preview = (txt[:220] + "...") if len(txt) > 220 else txt

        self.last_concept_coverage = raw_coverage_list

        if not notes_agg:
            return []

        # 3. Cross-Concept Aggregation & Multi-Hit Bonus
        scored_notes: List[ScoredNote] = []
        target_nids = list(notes_agg.keys())
        content_map = self.storage.get_notes_content(target_nids)

        for nid, data in notes_agg.items():
            max_s = max(data["scores"])
            concept_count = len(data["concepts"])
            # Small bonus for notes matching multiple distinct concepts in the lecture
            bonus = (concept_count - 1) * 0.02
            final_score = round(min(1.0, max_s + bonus), 4)

            text_preview = ""
            if nid in content_map:
                c_text = content_map[nid]["clean_text"]
                text_preview = (c_text[:200] + "...") if len(c_text) > 200 else c_text

            scored_notes.append(
                ScoredNote(
                    nid=nid,
                    score=final_score,
                    matched_concepts=data["concepts"],
                    clean_text_preview=text_preview,
                )
            )

        scored_notes.sort(key=lambda x: x.score, reverse=True)
        if suspended_nids is not None and not eff_include_active:
            # Ensure suspended notes get up to max_cards_unit slots without being displaced by active notes
            susp_notes = [sn for sn in scored_notes if sn.nid in suspended_nids][: max_cards_unit]
            act_notes = [sn for sn in scored_notes if sn.nid not in suspended_nids][: max_cards_unit]
            combined_final = susp_notes + act_notes
            combined_final.sort(key=lambda x: x.score, reverse=True)
            return combined_final
        return scored_notes[: max_cards_unit]

    # ---------------- Stage 3: Optional LLM Verification ----------------

    def verify_candidates(
        self,
        unit: LectureUnit,
        candidates: List[ScoredNote],
        batch_size: int = 25,
        concepts: Optional[List[ExtractedConcept]] = None,
    ) -> List[ScoredNote]:
        """Passes candidate notes to the Chat LLM to confirm relevance and filter out false positives."""
        if not self.config.enable_llm_verification or not candidates:
            return candidates

        quote_lookup: Dict[str, str] = {}
        if concepts:
            for conc in concepts:
                if conc.quote_it:
                    if conc.topic_it:
                        quote_lookup[conc.topic_it] = conc.quote_it
                    if conc.claim_en:
                        quote_lookup[conc.claim_en[:40]] = conc.quote_it
                        quote_lookup[conc.claim_en] = conc.quote_it
        if self.last_concept_coverage:
            for cov in self.last_concept_coverage:
                if cov.quote_it:
                    if cov.topic_it:
                        quote_lookup[cov.topic_it] = cov.quote_it
                    if cov.claim_en:
                        quote_lookup[cov.claim_en[:40]] = cov.quote_it
                        quote_lookup[cov.claim_en] = cov.quote_it

        def _get_candidate_quotes(c: ScoredNote) -> str:
            quotes = []
            for mc in c.matched_concepts:
                q = quote_lookup.get(mc)
                if q and q not in quotes:
                    quotes.append(q)
            if self.last_concept_coverage:
                for cov in self.last_concept_coverage:
                    if c.nid in cov.matched_nids and cov.quote_it and cov.quote_it not in quotes:
                        quotes.append(cov.quote_it)
            return " | ".join(quotes)

        verified_nids: Set[int] = set()

        for i in range(0, len(candidates), batch_size):
            chunk = candidates[i : i + batch_size]
            notes_payload = []
            for c in chunk:
                notes_payload.append(
                    {
                        "nid": c.nid,
                        "text": c.clean_text_preview,
                        "matched_concepts": c.matched_concepts,
                        "quote": _get_candidate_quotes(c),
                    }
                )

            user_prompt = f"### Lecture Topic: {unit.unit_title}\n"
            if unit.breadcrumb:
                user_prompt += f"### Hierarchy: {unit.breadcrumb}\n"
            user_prompt += f"### Lecture Excerpt:\n{unit.body_text}\n\n"
            user_prompt += "### Candidate Flashcards:\n"
            for n in notes_payload:
                user_prompt += f"- ID {n['nid']}: {n['text']}\n"
                if n.get("quote"):
                    user_prompt += f"  Matched quote: \"{n['quote']}\"\n"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_STAGE3},
                {"role": "user", "content": user_prompt},
            ]

            try:
                res = self.ai_client.chat_completion_json(
                    messages=messages,
                    model=self.config.chat_model,
                    base_url=self.config.chat_base_url,
                    api_key=self.config.chat_api_key,
                    max_tokens=getattr(self.config, "chat_max_tokens", 16384),
                )
                if isinstance(res, dict) and "relevant_nids" in res:
                    for nid in res["relevant_nids"]:
                        try:
                            verified_nids.add(int(nid))
                        except (ValueError, TypeError):
                            pass
                else:
                    print(f"[SmartSearch] Notice: Stage 3 verification invalid response schema ({res}), keeping batch.")
                    for c in chunk:
                        verified_nids.add(c.nid)
            except Exception as e:
                # If verification call fails, retain chunk rather than losing results
                print(f"[SmartSearch] Notice: Stage 3 verification call error ({e}), keeping batch.")
                for c in chunk:
                    verified_nids.add(c.nid)

        return [c for c in candidates if c.nid in verified_nids]

    # ---------------- End-to-End Search Pipeline ----------------

    def search_unit(
        self,
        unit: LectureUnit,
        candidate_nids: Optional[List[int]],
        active_model_id: str,
        suspended_nids: Optional[Set[int]] = None,
    ) -> List[ScoredNote]:
        """Runs Stage 1, Stage 2, and optional Stage 3 search for a lecture unit."""
        self.last_unit_error = ""
        self.last_concept_coverage = []
        try:
            concepts = self.extract_concepts(unit)
        except Exception as e:
            self.last_unit_error = str(e)
            return []

        if not concepts:
            return []

        candidates = self.retrieve_for_concepts(
            concepts,
            candidate_nids,
            active_model_id,
            suspended_nids=suspended_nids,
        )
        if self.config.enable_llm_verification:
            candidates = self.verify_candidates(unit, candidates, concepts=concepts)
            surviving_nids = {c.nid for c in candidates}
            min_threshold = self.config.get_min_similarity_threshold()
            for cov in self.last_concept_coverage:
                cov.matched_nids = [n for n in cov.matched_nids if n in surviving_nids]
                cov.matched_notes_count = len(cov.matched_nids)
                if cov.matched_notes_count == 0:
                    cov.status = "uncovered"
                elif cov.matched_notes_count == 1 or cov.best_score < (min_threshold + 0.06):
                    cov.status = "partial"
                else:
                    cov.status = "covered"
        return candidates


_STRUCT_REP_CACHE: Dict[int, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
_MAX_STRUCT_CACHE_SIZE = 80000


def compute_tag_similarity_scores(
    selected_nids: Sequence[int],
    notes_tags: Dict[int, Sequence[str]],
    max_tag_df: int = 250,
    candidate_nids: Optional[Set[int]] = None,
) -> Tuple[Dict[int, float], bool]:
    """Computes deck-agnostic hierarchical IDF-weighted tag similarity scores in [0.0, 1.0].

    Automatically filters out overly broad/boilerplate tags whose document frequency
    exceeds dynamic eff_max_df without hardcoding any deck-specific tag names.

    Returns:
        (scores_by_nid, query_has_valid_tags)
    """
    sel_set = {int(n) for n in selected_nids}
    if not sel_set or not notes_tags:
        return {}, False

    # Consider only notes that actually carry at least one tag for DF statistics
    tagged_notes: Dict[int, List[str]] = {}
    for nid, raw_tags in notes_tags.items():
        cleaned = [str(t).strip() for t in raw_tags if str(t).strip()]
        if cleaned:
            tagged_notes[int(nid)] = cleaned

    n_tagged = len(tagged_notes)
    if n_tagged == 0:
        return {}, False

    if n_tagged <= 3:
        eff_max_df = n_tagged
    elif n_tagged < 200:
        eff_max_df = min(max_tag_df, max(3, int(0.50 * n_tagged)))
    else:
        eff_max_df = min(max_tag_df, max(40, int(0.15 * n_tagged)))

    exact_df: Dict[str, int] = collections.Counter()
    prefix_df: Dict[str, int] = collections.Counter()
    for ts in tagged_notes.values():
        unique_lower = {t.lower() for t in ts}
        for tl in unique_lower:
            exact_df[tl] += 1
        prefixes = {tl.rsplit("::", 1)[0] for tl in unique_lower if "::" in tl}
        for p in prefixes:
            prefix_df[p] += 1

    def _extract_features(ts: Sequence[str]) -> Dict[Tuple[str, str], float]:
        feats: Dict[Tuple[str, str], float] = {}
        for t in ts:
            tl = t.lower()
            df = exact_df.get(tl, 0)
            if 1 <= df <= eff_max_df:
                idf = math.log(1.0 + n_tagged / df)
                feats[("L", tl)] = idf * idf
            if "::" in tl:
                p = tl.rsplit("::", 1)[0]
                pdf = prefix_df.get(p, 0)
                if 1 <= pdf <= eff_max_df:
                    pidf = math.log(1.0 + n_tagged / pdf)
                    k = ("P", p)
                    w = 0.35 * pidf * pidf
                    if w > feats.get(k, 0.0):
                        feats[k] = w
        return feats

    # Aggregate query features across selected notes
    q_feat_weights: Dict[Tuple[str, str], float] = {}
    q_feat_counts: Dict[Tuple[str, str], int] = collections.Counter()
    sel_with_tags = [nid for nid in sel_set if nid in tagged_notes]
    if not sel_with_tags:
        return {}, False

    for nid in sel_with_tags:
        feats = _extract_features(tagged_notes[nid])
        for k, w in feats.items():
            q_feat_counts[k] += 1
            if w > q_feat_weights.get(k, 0.0):
                q_feat_weights[k] = w

    if not q_feat_weights:
        return {}, False

    n_sel = len(sel_with_tags)
    q_feats: Dict[Tuple[str, str], float] = {
        k: w * (0.5 + 0.5 * (q_feat_counts[k] / n_sel))
        for k, w in q_feat_weights.items()
    }
    q_sum = sum(q_feats.values())
    if q_sum <= 1e-12:
        return {}, False

    q_leaf_keys = {val for kind, val in q_feats if kind == "L"}
    q_parent_keys = {val for kind, val in q_feats if kind == "P"}

    scores: Dict[int, float] = {}
    for nid, ts in tagged_notes.items():
        if nid in sel_set:
            continue
        if candidate_nids is not None and nid not in candidate_nids:
            continue

        # Fast rejection before computing feature dict
        has_overlap = False
        for t in ts:
            tl = t.lower()
            if tl in q_leaf_keys:
                has_overlap = True
                break
            if "::" in tl and tl.rsplit("::", 1)[0] in q_parent_keys:
                has_overlap = True
                break
        if not has_overlap:
            continue

        c_feats = _extract_features(ts)
        if not c_feats:
            continue

        inter = 0.0
        for k, qw in q_feats.items():
            cw = c_feats.get(k)
            if cw is not None:
                inter += qw if qw <= cw else cw

        if inter <= 0.0:
            continue

        c_sum = sum(c_feats.values())
        cov = inter / q_sum
        dice = (2.0 * inter) / (q_sum + c_sum)
        raw = 0.65 * cov + 0.35 * dice
        # Calibrate concave power curve (0.75) to align tag similarity with cosine scale
        scores[nid] = round(min(1.0, max(0.0, raw ** 0.75)), 4)

    return scores, True


def _get_cached_struct_rep(nid: int, primary_raw: str) -> Dict[str, Any]:
    """Returns cached structural representation for nid if primary_raw signature matches."""
    sig = (len(primary_raw), hash(primary_raw))
    cached = _STRUCT_REP_CACHE.get(nid)
    if cached is not None and cached[0] == sig:
        return cached[1]
    rep = build_structural_representation(primary_raw)
    if len(_STRUCT_REP_CACHE) >= _MAX_STRUCT_CACHE_SIZE:
        _STRUCT_REP_CACHE.clear()
    _STRUCT_REP_CACHE[nid] = (sig, rep)
    return rep


def compute_structural_similarity_scores(
    selected_nids: Sequence[int],
    notes_primary_raw: Dict[int, str],
    candidate_nids: Optional[Set[int]] = None,
    shortlist_limit: int = 800,
) -> Tuple[Dict[int, float], bool]:
    """Computes syntactic/structural template similarity scores in [0.0, 1.0].

    Uses a fast anchor-word pre-filter on collections > 1000 notes and computes exact
    LCS + POS-skeleton + bigram template similarity on shortlisted candidates.

    Returns:
        (scores_by_nid, query_has_valid_structure)
    """
    sel_set = {int(n) for n in selected_nids}
    if not sel_set or not notes_primary_raw:
        return {}, False

    sel_reps: List[Dict[str, Any]] = []
    for nid in sel_set:
        raw = notes_primary_raw.get(nid, "")
        if not raw:
            continue
        rep = _get_cached_struct_rep(nid, raw)
        if len(rep.get("lexical_tokens", [])) >= 2:
            sel_reps.append(rep)

    if not sel_reps:
        return {}, False

    # Collect distinctive query anchor words for fast pre-filtering on large collections
    q_word_weights: Dict[str, int] = {}
    q_min_len = min(len(r["lexical_tokens"]) for r in sel_reps)
    for r in sel_reps:
        for w in r["lexical_tokens"]:
            if len(w) >= 4:
                q_word_weights[w] = max(q_word_weights.get(w, 0), 2)
            elif len(w) == 3:
                q_word_weights[w] = max(q_word_weights.get(w, 0), 1)

    pool_nids = [
        nid
        for nid in notes_primary_raw
        if nid not in sel_set and (candidate_nids is None or nid in candidate_nids)
    ]
    if not pool_nids:
        return {}, True

    if len(pool_nids) > 1000 and q_word_weights:
        distinct_anchors = list(q_word_weights.items())
        min_weighted_hits = 3 if sum(w for _, w in distinct_anchors) >= 6 else 1
        shortlist: List[Tuple[int, int]] = []
        for nid in pool_nids:
            raw = notes_primary_raw[nid]
            if not raw:
                continue
            low = raw[:600].lower()
            w_hits = 0
            for w, weight in distinct_anchors:
                if w in low:
                    w_hits += weight
            if w_hits >= min_weighted_hits:
                shortlist.append((w_hits, nid))
        shortlist.sort(key=lambda x: x[0], reverse=True)
        target_nids = [nid for _, nid in shortlist[:shortlist_limit]]
    else:
        target_nids = pool_nids

    scores: Dict[int, float] = {}
    for nid in target_nids:
        raw = notes_primary_raw.get(nid, "")
        if not raw:
            continue
        c_rep = _get_cached_struct_rep(nid, raw)
        if len(c_rep.get("lexical_tokens", [])) < 2:
            continue
        # If multiple notes selected, use best structural match across selected templates
        best_s = max(compute_structural_similarity(q_rep, c_rep) for q_rep in sel_reps)
        if best_s > 0.0:
            scores[nid] = best_s

    return scores, True


def _load_notes_raw_meta(
    storage: SearchDatabase,
    selected_nids: Sequence[int],
    candidate_nids: Optional[Union[Set[int], Sequence[int]]] = None,
    col: Any = None,
    notes_raw_meta: Optional[Dict[int, Tuple[str, List[str]]]] = None,
) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """Loads primary field raw text and tag lists from notes_raw_meta, Anki col, or storage."""
    primary_texts: Dict[int, str] = {}
    notes_tags: Dict[int, List[str]] = {}

    if notes_raw_meta is not None:
        for nid, (txt, tags) in notes_raw_meta.items():
            nid_i = int(nid)
            if txt:
                primary_texts[nid_i] = txt
            notes_tags[nid_i] = list(tags)
        return primary_texts, notes_tags

    # Try reading directly from Anki collection DB (fast bulk SQL read)
    db_obj = getattr(col, "db", None) if col is not None else None
    if db_obj is not None and hasattr(db_obj, "execute"):
        try:
            cand_set = {int(n) for n in candidate_nids} if candidate_nids is not None else None
            sel_set = {int(n) for n in selected_nids}
            rows = []
            if cand_set is not None:
                needed = list(sel_set | cand_set)
                if len(needed) <= 900:
                    if needed:
                        ph = ",".join(str(x) for x in needed)
                        rows = list(db_obj.execute(f"SELECT id, flds, tags FROM notes WHERE id IN ({ph})"))
                else:
                    target = set(needed)
                    rows = [r for r in db_obj.execute("SELECT id, flds, tags FROM notes") if int(r[0]) in target]
            else:
                rows = list(db_obj.execute("SELECT id, flds, tags FROM notes"))

            for nid_val, flds_str, tags_str in rows:
                nid_i = int(nid_val)
                prim = extract_primary_field_raw(flds_str or "")
                if prim:
                    primary_texts[nid_i] = prim
                notes_tags[nid_i] = [t for t in (tags_str or "").strip().split() if t]
        except Exception:
            pass

    if primary_texts or any(notes_tags.values()):
        return primary_texts, notes_tags

    # Fallback to SmartSearch SQLite storage if available (e.g. standalone without col)
    try:
        fetch_nids: Optional[List[int]] = None
        if candidate_nids is not None:
            fetch_nids = list({int(n) for n in selected_nids} | {int(n) for n in candidate_nids})
        else:
            # Load only selected_nids first; if storage has notes_content, load all
            sel_map = storage.get_notes_content([int(n) for n in selected_nids])
            if sel_map:
                with storage._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT nid, clean_text, clean_tags FROM notes_content")
                    for r in cur.fetchall():
                        nid_i = int(r["nid"])
                        p = extract_primary_field_raw("", fallback_clean_text=r["clean_text"] or "")
                        if p:
                            primary_texts[nid_i] = p
                        t_raw = (r["clean_tags"] or "").strip()
                        if t_raw:
                            notes_tags[nid_i] = [t.strip() for t in t_raw.split(",") if t.strip()]
                return primary_texts, notes_tags
        if fetch_nids:
            c_map = storage.get_notes_content(fetch_nids)
            for nid_i, info in c_map.items():
                p = extract_primary_field_raw("", fallback_clean_text=info.get("clean_text", ""))
                if p:
                    primary_texts[int(nid_i)] = p
                t_raw = (info.get("clean_tags") or "").strip()
                if t_raw:
                    notes_tags[int(nid_i)] = [t.strip() for t in t_raw.split(",") if t.strip()]
    except Exception:
        pass

    return primary_texts, notes_tags


def find_similar_notes(
    storage: SearchDatabase,
    selected_nids: Sequence[int],
    active_model_id: str,
    candidate_nids: Optional[Union[Set[int], Sequence[int]]] = None,
    min_score: float = 0.70,
    max_notes: int = 20,
    *,
    mode: str = "combined",
    col: Any = None,
    notes_raw_meta: Optional[Dict[int, Tuple[str, List[str]]]] = None,
    use_embeddings: bool = True,
    use_tags: bool = True,
    use_structure: bool = True,
    weight_embedding: float = 0.55,
    weight_tags: float = 0.25,
    weight_structure: float = 0.20,
    tag_max_df: int = 250,
    details_out: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[List[Tuple[int, float]], List[int]]:
    """Finds notes similar to the selected notes using semantic embeddings, tags, and/or sentence structure.

    Modes:
    - "combined": fuses all enabled & available signals (Embedding, Tags, Structure) via synergistic fusion.
                  Automatically falls back to Tags/Structure if selected notes lack embeddings.
    - "embedding": pure vector cosine similarity (requires selected notes to be indexed).
    - "tags": hierarchical IDF-weighted tag similarity (no embeddings required).
    - "structure": syntactic/template sentence similarity (no embeddings required).

    Returns:
        (matched_notes, missing_nids)
        where matched_notes is a list of (nid, similarity_score) ordered descending.
    """
    if not selected_nids:
        return [], []

    if candidate_nids is not None and len(candidate_nids) == 0:
        return [], []

    eff_mode = (mode or "combined").strip().lower()
    if eff_mode not in {"combined", "embedding", "tags", "structure"}:
        eff_mode = "combined"

    want_emb = (eff_mode == "embedding") or (eff_mode == "combined" and use_embeddings)
    want_tags = (eff_mode == "tags") or (eff_mode == "combined" and use_tags)
    want_struct = (eff_mode == "structure") or (eff_mode == "combined" and use_structure)

    sel_set = {int(n) for n in selected_nids}
    cand_set: Optional[Set[int]] = {int(n) for n in candidate_nids} if candidate_nids is not None else None

    # 1. Semantic Embedding Similarity
    emb_scores: Dict[int, float] = {}
    emb_active = False
    missing_emb: List[int] = []

    if want_emb:
        sel_nids_list, sel_vecs, _dim = storage.load_embeddings_for_notes(
            list(selected_nids), active_model_id
        )
        indexed_set = set(sel_nids_list)
        missing_emb = [nid for nid in selected_nids if nid not in indexed_set]

        if not missing_emb and sel_nids_list:
            if np is not None and isinstance(sel_vecs, getattr(np, "ndarray", ())):
                mean_arr = np.mean(sel_vecs, axis=0)
                norm_mean = normalize_l2(mean_arr.tolist())
            else:
                if len(sel_vecs) == 1:
                    norm_mean = normalize_l2(sel_vecs[0])
                else:
                    dim_len = len(sel_vecs[0])
                    sum_vec = [0.0] * dim_len
                    for v in sel_vecs:
                        for d in range(dim_len):
                            sum_vec[d] += v[d]
                    norm_mean = normalize_l2(sum_vec)

            cand_list: Optional[Sequence[int]] = list(cand_set) if cand_set is not None else None
            cand_nids_list, cand_vecs, _ = storage.load_embeddings_for_notes(
                cand_list, active_model_id
            )
            if cand_nids_list:
                similarities = storage.compute_similarities_from_loaded(
                    cand_nids_list, cand_vecs, norm_mean
                )
                emb_scores = {
                    int(nid): round(float(score), 4)
                    for nid, score in similarities
                    if int(nid) not in sel_set and float(score) > 0.0
                }
            emb_active = True

    if eff_mode == "embedding":
        if missing_emb:
            return [], missing_emb
        matched = [
            (nid, sc)
            for nid, sc in sorted(emb_scores.items(), key=lambda x: x[1], reverse=True)
            if sc >= min_score
        ][:max_notes]
        if details_out is not None:
            details_out.clear()
            for nid, sc in matched:
                details_out[nid] = {"E": sc}
        return matched, []

    # 2. Load raw primary text & tags when needed for Tags / Structure
    primary_texts: Dict[int, str] = {}
    notes_tags: Dict[int, List[str]] = {}
    if want_tags or want_struct:
        primary_texts, notes_tags = _load_notes_raw_meta(
            storage=storage,
            selected_nids=selected_nids,
            candidate_nids=cand_set,
            col=col,
            notes_raw_meta=notes_raw_meta,
        )

    # 3. Tag Similarity
    tag_scores: Dict[int, float] = {}
    tag_active = False
    if want_tags and notes_tags:
        tag_scores, tag_active = compute_tag_similarity_scores(
            selected_nids=selected_nids,
            notes_tags=notes_tags,
            max_tag_df=tag_max_df,
            candidate_nids=cand_set,
        )

    # 4. Structural / Syntactic Similarity
    struct_scores: Dict[int, float] = {}
    struct_active = False
    if want_struct and primary_texts:
        struct_scores, struct_active = compute_structural_similarity_scores(
            selected_nids=selected_nids,
            notes_primary_raw=primary_texts,
            candidate_nids=cand_set,
        )

    # If combined mode had missing embeddings and neither tags nor structure is available,
    # surface missing_emb so the user is prompted to index (preserving v0.5.x behavior).
    if not emb_active and not tag_active and not struct_active:
        if missing_emb:
            return [], missing_emb
        return [], []

    # Standalone single-criterion modes
    if eff_mode == "tags":
        matched = [
            (nid, sc)
            for nid, sc in sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
            if sc >= min_score
        ][:max_notes]
        if details_out is not None:
            details_out.clear()
            for nid, sc in matched:
                details_out[nid] = {"T": sc}
        return matched, []

    if eff_mode == "structure":
        matched = [
            (nid, sc)
            for nid, sc in sorted(struct_scores.items(), key=lambda x: x[1], reverse=True)
            if sc >= min_score
        ][:max_notes]
        if details_out is not None:
            details_out.clear()
            for nid, sc in matched:
                details_out[nid] = {"S": sc}
        return matched, []

    # 5. Combined Synergistic Fusion across active criteria
    active_weights: Dict[str, float] = {}
    if emb_active:
        active_weights["E"] = max(0.01, float(weight_embedding))
    if tag_active:
        active_weights["T"] = max(0.01, float(weight_tags))
    if struct_active:
        active_weights["S"] = max(0.01, float(weight_structure))

    if not active_weights:
        return [], missing_emb if missing_emb else []

    w_total = sum(active_weights.values())
    all_cands: Set[int] = set()
    if emb_active:
        all_cands.update(emb_scores.keys())
    if tag_active:
        all_cands.update(tag_scores.keys())
    if struct_active:
        all_cands.update(struct_scores.keys())

    scored: List[Tuple[int, float, Dict[str, float]]] = []
    n_active = len(active_weights)

    for nid in all_cands:
        if nid in sel_set:
            continue
        comp: Dict[str, float] = {}
        if emb_active:
            comp["E"] = emb_scores.get(nid, 0.0)
        if tag_active:
            comp["T"] = tag_scores.get(nid, 0.0)
        if struct_active:
            comp["S"] = struct_scores.get(nid, 0.0)

        if n_active == 1:
            final_score = next(iter(comp.values()))
        else:
            s_weighted = sum(active_weights[k] * comp[k] for k in active_weights) / w_total
            s_peak = max(comp.values())
            base = max(s_weighted, 0.85 * s_peak + 0.15 * s_weighted)
            strong_hits = sum(1 for v in comp.values() if v >= 0.45)
            bonus = 0.04 * max(0, strong_hits - 1)
            final_score = round(min(1.0, base + bonus), 4)

        if final_score >= min_score:
            scored.append((nid, final_score, comp))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:max_notes]

    if details_out is not None:
        details_out.clear()
        for nid, _, comp in top_scored:
            details_out[nid] = comp

    return [(nid, sc) for nid, sc, _ in top_scored], []

