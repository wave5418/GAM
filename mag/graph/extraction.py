"""LLM direct triple extraction for MAG graph construction.

MAG builds its graph from triples emitted directly by the LLM:
sentence_id + text → (head, relation, tail, source_sentence_id).
The source sentence id is the authority for retrieval-time evidence.
"""

from __future__ import annotations

import json
import logging
import os
from json import JSONDecodeError
from typing import Dict, List, Optional, Set, Tuple

from mag.schema import ExtractedFact, Triple

logger = logging.getLogger(__name__)
_DEFAULT_DIRECT_TRIPLE_MAX_TOKENS = 8192

_DIRECT_TRIPLE_PROMPT = """Extract source-bound atomic facts and knowledge triples from conversation sentences.

Each input item has:
- sentence_id: stable id that MUST be copied exactly into each extracted triple
- date: optional episode date
- text: sentence text, often with speaker prefix
- is_new: true for current source sentences; false for recent context sentences

Task:
- First extract atomic facts. Each fact must preserve the original sentence's facts and meaning. Do not paraphrase beyond reference resolution, summarize, generalize, or add new facts.
- Use recent context sentences only to resolve references inside current source sentences: pronouns, possessives, deixis, ellipsis, and references such as I/me/my/you/your/he/she/they/we/it/this/that/there/then/the former/the latter.
- After reference resolution, each fact should be understandable without reading any other sentence.
- Then extract triples from the extracted facts, not directly from the raw sentence text.
- Include facts/triples about events, attributes, preferences, ownership, locations, dates, recommendations, list items, and counts.
- Prefer complete facts and triples over generic "related" links.

Rules:
1. Use only facts supported by the provided sentences.
2. Extract facts ONLY for input items where is_new is true. Context items are reference-resolution support only.
3. Every fact MUST include a fact_id and a source_sentence_id copied exactly from an is_new input item.
4. If a fact uses context to resolve a reference, also include source_sentence_ids with every supporting sentence_id, including context ids and the source_sentence_id.
5. Every triple MUST include source_fact_id and source_sentence_id. source_fact_id MUST refer to one of the facts you output.
6. If a fact requires multiple adjacent sentences only to resolve references, choose the source_sentence_id that contains the answer-bearing evidence.
7. Keep facts atomic: one event, attribute, preference, ownership, location, date, recommendation, list item, or count per fact.
8. Keep triple heads and tails concise entity/value strings, not whole paragraphs.
9. Do not output facts or triples for pleasantries or unsupported assumptions.
10. Do not use unresolved pronouns or vague references as fact subjects or graph nodes. Avoid heads or tails like "I", "me", "you", "we", "they", "it", "this", "that", "my family", or "her project" unless the phrase has been resolved to the concrete referent.
11. Do not mechanically replace first-person pronouns with the speaker. Resolve only the referenced expressions needed for the original source sentence to stand alone; if the referent is ambiguous, skip that fact/triple.
12. It is valid to output multiple triples between the same two nodes when they have different relations, facts, or source sentences.

Input:
{items}

Return JSON:
{{"facts": [
  {{"fact_id": "f1", "source_sentence_id": "s1", "source_sentence_ids": ["ctx1", "s1"], "fact": "Alice likes piano.", "confidence": 0.9}}
],
"triples": [
  {{"head": "Alice", "relation": "likes", "tail": "piano", "source_fact_id": "f1", "source_sentence_id": "s1", "source_sentence_ids": ["ctx1", "s1"], "confidence": 0.9}}
]}}"""


class DirectTripleExtractor:
    """Extract graph triples directly from sentence ids and texts."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.last_extracted_facts: List[ExtractedFact] = []

    def _max_tokens(self) -> int:
        raw = os.getenv("MAG_DIRECT_TRIPLE_MAX_TOKENS", "")
        try:
            value = int(raw) if raw else _DEFAULT_DIRECT_TRIPLE_MAX_TOKENS
        except ValueError:
            value = _DEFAULT_DIRECT_TRIPLE_MAX_TOKENS
        return max(1024, value)

    def extract_triples_direct(
        self,
        sentence_items: List[Tuple[str, str]],
        timestamps: Optional[Dict[str, str]] = None,
        context_items: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Triple]:
        """Extract triples for current source items, using context only for reference resolution."""
        if not self.llm_client or not sentence_items:
            self.last_extracted_facts = []
            return []

        source_sid_set = {sid for sid, _ in sentence_items}
        context_items = context_items or []
        items = [
            {
                "sentence_id": sid,
                "date": (timestamps or {}).get(sid, ""),
                "text": text,
                "is_new": False,
            }
            for sid, text in context_items
            if sid and text and text.strip() and sid not in source_sid_set
        ] + [
            {
                "sentence_id": sid,
                "date": (timestamps or {}).get(sid, ""),
                "text": text,
                "is_new": True,
            }
            for sid, text in sentence_items
            if sid and text and text.strip()
        ]
        if not items:
            self.last_extracted_facts = []
            return []

        facts, triples = self._extract_items(items, timestamps or {}, source_sid_set)
        self.last_extracted_facts = facts
        return triples

    def _extract_items(
        self,
        items: List[Dict[str, str]],
        timestamps: Dict[str, str],
        source_sid_set: Optional[Set[str]] = None,
    ) -> Tuple[List[ExtractedFact], List[Triple]]:
        if not items:
            return [], []
        source_sid_set = source_sid_set or {
            str(item.get("sentence_id", "")).strip()
            for item in items
            if item.get("is_new", True)
        }

        prompt = _DIRECT_TRIPLE_PROMPT.format(
            items=json.dumps(items, ensure_ascii=False),
        )
        try:
            response = self.llm_client.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract concise knowledge graph triples. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens(),
            )
            data = json.loads(response)
        except JSONDecodeError as e:
            if len(items) > 1:
                midpoint = len(items) // 2
                logger.warning(
                    "Direct triple extraction returned truncated/invalid JSON for %d items; retrying as %d + %d: %s",
                    len(items),
                    midpoint,
                    len(items) - midpoint,
                    str(e)[:200],
                )
                left_source_ids = {
                    str(item.get("sentence_id", "")).strip()
                    for item in items[:midpoint]
                    if str(item.get("sentence_id", "")).strip() in source_sid_set
                }
                right_source_ids = {
                    str(item.get("sentence_id", "")).strip()
                    for item in items[midpoint:]
                    if str(item.get("sentence_id", "")).strip() in source_sid_set
                }
                left_facts, left_triples = self._extract_items(items[:midpoint], timestamps, left_source_ids)
                right_facts, right_triples = self._extract_items(items[midpoint:], timestamps, right_source_ids)
                return left_facts + right_facts, left_triples + right_triples
            logger.warning("Direct triple extraction failed for single item: %s", str(e)[:200])
            return [], []
        except Exception as e:
            logger.warning("Direct triple extraction failed: %s", str(e)[:200])
            return [], []

        if not isinstance(data, dict):
            logger.warning("Direct triple extraction returned non-object JSON: %s", type(data).__name__)
            return [], []

        sid_set = {str(item.get("sentence_id", "")).strip() for item in items}
        facts_by_id: Dict[str, ExtractedFact] = {}
        for item in data.get("facts", []):
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("fact_id", "")).strip()
            fact_text = str(item.get("fact", "")).strip()
            sid = str(item.get("source_sentence_id", "")).strip()
            if not fact_id or not fact_text or sid not in source_sid_set:
                continue
            source_ids = self._valid_source_ids(item.get("source_sentence_ids", sid), sid, sid_set)
            try:
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))
            facts_by_id[fact_id] = ExtractedFact(
                fact_id=fact_id,
                fact=fact_text,
                source_sentence_id=sid,
                confidence=confidence,
                source_sentence_ids=source_ids,
            )
        facts = list(facts_by_id.values())

        triples: List[Triple] = []
        for item in data.get("triples", []):
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if not head or not relation or not tail:
                continue

            source_fact_id = str(item.get("source_fact_id", "")).strip()
            primary_sid = str(item.get("source_sentence_id", "")).strip()
            source_ids = self._valid_source_ids(
                item.get("source_sentence_ids", primary_sid),
                primary_sid,
                sid_set,
            )
            if not primary_sid and source_ids:
                primary_sid = source_ids[0]
            if primary_sid not in source_sid_set:
                continue

            try:
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))

            source_fact = facts_by_id.get(source_fact_id)
            if source_fact_id and (
                source_fact is None or source_fact.source_sentence_id != primary_sid
            ):
                continue
            triples.append(
                Triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    confidence=confidence,
                    source_sentence_id=primary_sid,
                    source_sentence_ids=source_ids,
                    source_fact_id=source_fact_id,
                    source_fact=source_fact.fact if source_fact else "",
                )
            )
        return facts, triples

    @staticmethod
    def _valid_source_ids(raw_ids, primary_sid: str, sid_set: Set[str]) -> List[str]:
        if isinstance(raw_ids, str):
            source_ids = [raw_ids]
        elif isinstance(raw_ids, list):
            source_ids = raw_ids
        else:
            source_ids = [primary_sid]
        result = []
        for source_id in source_ids:
            sid = str(source_id).strip()
            if sid and sid in sid_set and sid not in result:
                result.append(sid)
        if primary_sid and primary_sid in sid_set and primary_sid not in result:
            result.insert(0, primary_sid)
        return result
