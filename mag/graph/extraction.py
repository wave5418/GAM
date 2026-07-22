"""LLM direct triple extraction for MAG graph construction.

MAG builds its graph from triples emitted directly by the LLM:
sentence_id + text → (head, relation, tail, source_sentence_id).
The source sentence id is the authority for retrieval-time evidence.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from mag.schema import Triple

logger = logging.getLogger(__name__)

_DIRECT_TRIPLE_PROMPT = """Extract explicit knowledge triples from the conversation sentences.

Each input item has:
- sentence_id: stable id that MUST be copied exactly into each extracted triple
- date: optional episode date
- text: sentence text, often with speaker prefix

Task:
- Directly extract triples as (head, relation, tail, source_sentence_id)
- Resolve I/me/my/you/your/he/she/they/it to concrete entities when the local context makes it clear
- Include facts about events, attributes, preferences, ownership, locations, dates, recommendations, list items, and counts
- Prefer complete triples over generic "related" links

Rules:
1. Use only facts supported by the provided sentences.
2. Every triple MUST include a source_sentence_id copied exactly from one input item.
3. If a fact requires multiple adjacent sentences, choose the sentence_id that contains the answer-bearing evidence.
4. Keep heads and tails concise entity/value strings, not whole paragraphs.
5. Do not output triples for pleasantries or unsupported assumptions.
6. It is valid to output multiple triples between the same two nodes when they have different relations or source sentences.

Input:
{items}

Return JSON:
{{"triples": [
  {{"head": "Alice", "relation": "likes", "tail": "piano", "source_sentence_id": "s1", "confidence": 0.9}}
]}}"""


class DirectTripleExtractor:
    """Extract graph triples directly from sentence ids and texts."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_triples_direct(
        self,
        sentence_items: List[Tuple[str, str]],
        timestamps: Optional[Dict[str, str]] = None,
    ) -> List[Triple]:
        """Extract triples with LLM-assigned, validated source sentence ids."""
        if not self.llm_client or not sentence_items:
            return []

        sid_set = {sid for sid, _ in sentence_items}
        items = [
            {
                "sentence_id": sid,
                "date": (timestamps or {}).get(sid, ""),
                "text": text,
            }
            for sid, text in sentence_items
            if sid and text and text.strip()
        ]
        if not items:
            return []

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
            )
            data = json.loads(response)
        except Exception as e:
            logger.warning("Direct triple extraction failed: %s", str(e)[:200])
            return []

        triples: List[Triple] = []
        for item in data.get("triples", []):
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if not head or not relation or not tail:
                continue

            source_ids = item.get("source_sentence_ids", item.get("source_sentence_id", ""))
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            if not isinstance(source_ids, list):
                continue

            try:
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))

            for sid in source_ids:
                sid_text = str(sid).strip()
                if sid_text not in sid_set:
                    continue
                triples.append(
                    Triple(
                        head=head,
                        relation=relation,
                        tail=tail,
                        confidence=confidence,
                        source_sentence_id=sid_text,
                    )
                )
        return triples
