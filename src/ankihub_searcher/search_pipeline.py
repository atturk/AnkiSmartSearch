"""Multi-stage hybrid search pipeline for Smart Search.

Stage 1: LLM Concept Decomposition & Cross-Lingual Medical Query Expansion
Stage 2: Hybrid Retrieval (Dense Embeddings + SQLite FTS5 BM25) with RRF/linear fusion
Stage 3: Optional LLM Verification & Reranking
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .ai_client import AIClientError, OpenAICompatibleClient
from .config_manager import ConfigManager
from .storage import SearchDatabase
from .text_utils import LectureUnit


@dataclass
class ExtractedConcept:
    """An atomic medical concept extracted by LLM from a lecture unit."""

    topic_it: str
    claim_en: str
    keywords_en: List[str]
    keywords_it: List[str] = field(default_factory=list)


@dataclass
class ScoredNote:
    """A candidate note identified by the search pipeline with its relevance score."""

    nid: int
    score: float
    matched_concepts: List[str]
    clean_text_preview: str = ""


SYSTEM_PROMPT_STAGE1 = """You are an elite medical professor, USMLE Step 1/Step 2 expert, and AnKing deck specialist.
Your task is to decompose the following Italian lecture notes into atomic medical concepts and translate them into high-yield USMLE English claims and keywords.

CRITICAL INSTRUCTIONS:
1. Deconstruct the lecture text into 3 to 12 atomic medical concepts.
2. Resolve Italian medical terminology, historical acronyms, and eponyms into their standard USMLE English equivalents:
   - e.g., "NALP3" -> "NLRP3 inflammasome"
   - e.g., "pirofosfato di calcio" -> "calcium pyrophosphate dihydrate (CPPD)", "pseudogout"
   - e.g., "acido urico" -> "uric acid", "monosodium urate", "gout"
   - e.g., "fattore di Hageman" -> "Factor XII", "Hageman factor"
   - e.g., "segno di Cullen" -> "Cullen sign", "acute pancreatitis"
3. For each concept, provide:
   - "topic_it": Short Italian topic summary.
   - "claim_en": 1-2 concise, fact-dense English sentences summarizing the mechanism, pathology, clinical signs, or radiology findings as tested on USMLE/AnKing cards.
   - "keywords_en": 4-10 precise English keywords, synonyms, eponyms, and acronyms.
   - "keywords_it": 3-6 original Italian terms from the lecture.

5. Do NOT overthink or write long reasoning monologues. Be fast, direct, and output ONLY the JSON object immediately.

Respond strictly with a JSON object in this format:
{
  "concepts": [
    {
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

Respond strictly with a JSON object containing the list of relevant note IDs:
{
  "relevant_nids": [12345, 67890]
}
"""


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

    # ---------------- Stage 1: Concept Extraction ----------------

    def extract_concepts(self, unit: LectureUnit) -> List[ExtractedConcept]:
        """Uses the Chat LLM to decompose an Italian lecture unit into atomic English concepts.

        Provides a robust fallback if the LLM is unreachable or fails.
        """
        user_prompt = f"### Lecture Unit Title: {unit.unit_title}\n"
        if unit.breadcrumb:
            user_prompt += f"### Context Hierarchy: {unit.breadcrumb}\n"
        user_prompt += f"### Lecture Text:\n{unit.body_text}\n\n/no_think"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STAGE1},
            {"role": "user", "content": user_prompt},
        ]

        try:
            res = self.ai_client.chat_completion_json(
                messages=messages,
                model=self.config.chat_model,
                base_url=self.config.chat_base_url,
                api_key=self.config.chat_api_key,
                temperature=0.1,
            )

            raw_concepts = []
            if isinstance(res, dict) and "concepts" in res:
                raw_concepts = res["concepts"]
            elif isinstance(res, list):
                raw_concepts = res

            concepts: List[ExtractedConcept] = []
            for item in raw_concepts:
                if not isinstance(item, dict):
                    continue
                topic_it = str(item.get("topic_it", "")).strip()
                claim_en = str(item.get("claim_en", "")).strip()
                kws_en = [str(k).strip() for k in item.get("keywords_en", []) if str(k).strip()]
                kws_it = [str(k).strip() for k in item.get("keywords_it", []) if str(k).strip()]

                if claim_en or kws_en or topic_it:
                    concepts.append(
                        ExtractedConcept(
                            topic_it=topic_it or unit.unit_title,
                            claim_en=claim_en or topic_it,
                            keywords_en=kws_en or [unit.unit_title],
                            keywords_it=kws_it,
                        )
                    )

            if concepts:
                return concepts

        except Exception as e:
            # Fallback will activate below
            print(f"[SmartSearch] Notice: Stage 1 LLM extraction fallback used ({e})")

        # Fallback concept when LLM extraction fails or is offline
        words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", unit.body_text)
        sample_kws = words[:12] if words else [unit.unit_title]
        return [
            ExtractedConcept(
                topic_it=unit.unit_title,
                claim_en=unit.unit_title,
                keywords_en=sample_kws,
                keywords_it=sample_kws[:5],
            )
        ]

    # ---------------- Stage 2: Hybrid Retrieval ----------------

    def retrieve_for_concepts(
        self,
        concepts: List[ExtractedConcept],
        candidate_nids: Optional[List[int]],
        active_model_id: str,
        suspended_nids: Optional[Set[int]] = None,
    ) -> List[ScoredNote]:
        """Performs dense vector search + BM25 lexical search per concept and fuses results."""
        if not concepts:
            return []

        # 1. Build query texts for embedding based on language strategy
        query_texts: List[str] = []
        for c in concepts:
            if self.config.embed_language_strategy == "multilingual_direct":
                text = f"{c.topic_it} | {c.claim_en} | {', '.join(c.keywords_en)}"
            else:  # translate_to_en (default)
                text = f"{c.claim_en} | {', '.join(c.keywords_en)}"
            query_texts.append(text)

        # Batch embed all concept queries at once
        concept_vectors = self.ai_client.get_embeddings(
            texts=query_texts,
            model=self.config.embed_model,
            base_url=self.config.embed_base_url,
            api_key=self.config.embed_api_key,
        )

        min_threshold = self.config.get_min_similarity_threshold()
        # Semantic floor: prevents purely lexical hits with 0 semantic relation
        min_dense_floor = max(0.20, min_threshold - 0.18)
        alpha = self.config.hybrid_dense_weight
        effective_top_k = self.config.get_effective_top_k()

        # Track results per note: nid -> {"scores": [float], "concepts": [str]}
        notes_agg: Dict[int, Dict[str, Any]] = {}

        # Pre-load candidate vectors ONCE for all concepts in this unit
        loaded_nids, loaded_vecs, _ = self.storage.load_embeddings_for_notes(
            nids=candidate_nids,
            model_id=active_model_id,
        )

        for idx, concept in enumerate(concepts):
            q_vec = concept_vectors[idx] if idx < len(concept_vectors) else []
            if not q_vec:
                continue

            # A. Dense Cosine Similarity against pre-loaded matrix/vectors
            dense_hits = self.storage.compute_similarities_from_loaded(
                nid_list=loaded_nids,
                vecs=loaded_vecs,
                query_vec=q_vec,
            )
            dense_scores: Dict[int, float] = {nid: score for nid, score in dense_hits}

            # B. Lexical FTS5 BM25
            all_kws = concept.keywords_en + concept.keywords_it
            fts_hits = self.storage.search_fts(
                keywords=all_kws,
                candidate_nids=candidate_nids,
                limit=150,
            )
            fts_scores: Dict[int, float] = {nid: score for nid, score in fts_hits}

            # C. Score Fusion for this concept
            concept_candidates: List[Tuple[int, float]] = []
            all_candidate_nids = set(dense_scores.keys())

            for nid in all_candidate_nids:
                d_score = dense_scores.get(nid, 0.0)
                if d_score < min_dense_floor:
                    continue

                b_score = fts_scores.get(nid, 0.0)
                # Combined linear fusion
                combined = alpha * d_score + (1.0 - alpha) * b_score

                if combined >= min_threshold:
                    concept_candidates.append((nid, combined))

            # Keep top-K for this concept.
            # When include_active_cards_in_filtered_deck is False and suspended_nids is provided,
            # already-active notes do NOT steal top-K slots from suspended notes!
            concept_candidates.sort(key=lambda x: x[1], reverse=True)
            if suspended_nids is not None and not self.config.include_active_cards_in_filtered_deck:
                top_concept_candidates: List[Tuple[int, float]] = []
                susp_kept = 0
                active_kept = 0
                for nid, score in concept_candidates:
                    if nid in suspended_nids:
                        if susp_kept < effective_top_k:
                            top_concept_candidates.append((nid, score))
                            susp_kept += 1
                    else:
                        if active_kept < effective_top_k:
                            top_concept_candidates.append((nid, score))
                            active_kept += 1
                    if susp_kept >= effective_top_k and active_kept >= effective_top_k:
                        break
            else:
                top_concept_candidates = concept_candidates[:effective_top_k]

            c_label = concept.topic_it or concept.claim_en[:40]
            for nid, score in top_concept_candidates:
                if nid not in notes_agg:
                    notes_agg[nid] = {"scores": [], "concepts": []}
                notes_agg[nid]["scores"].append(score)
                if c_label not in notes_agg[nid]["concepts"]:
                    notes_agg[nid]["concepts"].append(c_label)

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
        if suspended_nids is not None and not self.config.include_active_cards_in_filtered_deck:
            # Ensure suspended notes get up to max_cards_per_unit slots without being displaced by active notes
            susp_notes = [sn for sn in scored_notes if sn.nid in suspended_nids][: self.config.max_cards_per_unit]
            act_notes = [sn for sn in scored_notes if sn.nid not in suspended_nids][: self.config.max_cards_per_unit]
            combined_final = susp_notes + act_notes
            combined_final.sort(key=lambda x: x.score, reverse=True)
            return combined_final
        return scored_notes[: self.config.max_cards_per_unit]

    # ---------------- Stage 3: Optional LLM Verification ----------------

    def verify_candidates(
        self,
        unit: LectureUnit,
        candidates: List[ScoredNote],
        batch_size: int = 25,
    ) -> List[ScoredNote]:
        """Passes candidate notes to the Chat LLM to confirm relevance and filter out false positives."""
        if not self.config.enable_llm_verification or not candidates:
            return candidates

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
                    }
                )

            user_prompt = f"### Lecture Topic: {unit.unit_title}\n"
            if unit.breadcrumb:
                user_prompt += f"### Hierarchy: {unit.breadcrumb}\n"
            user_prompt += f"### Lecture Excerpt:\n{unit.body_text[:1200]}\n\n"
            user_prompt += "### Candidate Flashcards:\n"
            for n in notes_payload:
                user_prompt += f"- ID {n['nid']}: {n['text']}\n"

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
                )
                if isinstance(res, dict) and "relevant_nids" in res:
                    for nid in res["relevant_nids"]:
                        verified_nids.add(int(nid))
            except Exception as e:
                # If verification call fails, retain chunk rather than losing results
                print(f"[SmartSearch] Notice: Stage 3 verification call error ({e}), keeping batch.")
                for c in chunk:
                    verified_nids.add(c.nid)

        if not verified_nids:
            return candidates

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
        concepts = self.extract_concepts(unit)
        candidates = self.retrieve_for_concepts(
            concepts,
            candidate_nids,
            active_model_id,
            suspended_nids=suspended_nids,
        )
        if self.config.enable_llm_verification:
            candidates = self.verify_candidates(unit, candidates)
        return candidates
