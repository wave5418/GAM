"""Source-bound evidence unit construction for MAG ingestion.

The builder converts segmented conversation sentences into independently
understandable sentence units before vector indexing and graph extraction.
It is intentionally conservative: the LLM may only concatenate adjacent
sentences and resolve explicit references.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SENTENCES_PER_CALL = 24

_EVIDENCE_UNIT_PROMPT = """Build source-bound evidence sentence units from conversation sentences.

Each input sentence has:
- sentence_id: stable id that MUST be copied into source_sentence_ids
- speaker: speaker role/name
- date: optional episode date
- text: original sentence text

Goal:
- Each output unit must be a single independently understandable sentence or sentence sequence.
- The stored/indexed unit will replace the source sentences. Therefore every input sentence must be covered exactly once.

Allowed operations:
1. Concatenate adjacent input sentences when needed for complete semantics.
2. Resolve explicit references using local context: pronouns, possessives, ellipsis, this/that/these/those, here/there, then, the former/latter, "the project", "the event", "my friend", or similar definite references.

Hard rules:
1. Do NOT summarize, paraphrase, compress, reorder, infer new facts, or change sentence-internal wording.
2. Except for replacing a reference expression with its concrete referent, preserve the original words.
3. Only merge adjacent sentences; never merge non-adjacent sentences.
4. If a reference is ambiguous, do not guess. Keep the original expression and add it to unresolved_references.
5. Do not mechanically replace first-person or second-person pronouns with the speaker. Resolve only when the local referent is clear.
6. Do not merge across unrelated topics only because the same entity appears.
7. Use source_sentence_ids copied exactly from inputs, in original order.
8. Every input sentence_id must appear in exactly one output unit.

Input:
{items}

Return JSON:
{{"units": [
  {{
    "source_sentence_ids": ["s0", "s1"],
    "text": "Resolved/concatenated unit text.",
    "resolved_references": [{{"expression": "it", "referent": "the pottery class"}}],
    "unresolved_references": [],
    "merge_reason": "adjacent sentences form one complete event",
    "confidence": 0.9
  }}
]}}"""


@dataclass
class EvidenceUnit:
    """A stored/indexed MAG sentence unit derived from one or more raw sentences."""

    text: str
    speaker: str
    timestamp: datetime
    source_sentence_ids: List[str]
    source_texts: List[str]
    source_speakers: List[str]
    source_timestamps: List[str]
    resolved_references: List[Dict[str, str]] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    merge_reason: str = ""
    confidence: float = 1.0


class EvidenceUnitBuilder:
    """Build independently understandable sentence units with one LLM call per batch."""

    def __init__(self, llm_client=None, max_sentences_per_call: int = _DEFAULT_MAX_SENTENCES_PER_CALL):
        self.llm_client = llm_client
        self.max_sentences_per_call = max(1, int(max_sentences_per_call or _DEFAULT_MAX_SENTENCES_PER_CALL))

    def build(self, sentences: Sequence[Tuple[str, str, datetime]]) -> List[EvidenceUnit]:
        """Return evidence units; fall back to one unit per input sentence on failure."""
        if not sentences:
            return []
        if not self.llm_client:
            return self.raw_units(sentences)

        units: List[EvidenceUnit] = []
        for start in range(0, len(sentences), self.max_sentences_per_call):
            chunk = sentences[start : start + self.max_sentences_per_call]
            try:
                units.extend(self._build_chunk(chunk, start))
            except Exception as exc:
                logger.warning("Evidence unit construction failed; using raw sentences: %s", str(exc)[:200])
                units.extend(self._fallback_units(chunk, start))
        return units

    @staticmethod
    def raw_units(sentences: Sequence[Tuple[str, str, datetime]]) -> List[EvidenceUnit]:
        """Build one evidence unit per input sentence without LLM transformation."""
        return EvidenceUnitBuilder._fallback_units(sentences, 0)

    def _build_chunk(
        self,
        sentences: Sequence[Tuple[str, str, datetime]],
        offset: int,
    ) -> List[EvidenceUnit]:
        raw_ids = [f"s{offset + idx}" for idx in range(len(sentences))]
        items = [
            {
                "sentence_id": raw_ids[idx],
                "speaker": speaker,
                "date": timestamp.isoformat()[:10] if hasattr(timestamp, "isoformat") else "",
                "text": text,
            }
            for idx, (text, speaker, timestamp) in enumerate(sentences)
        ]
        prompt = _EVIDENCE_UNIT_PROMPT.format(items=json.dumps(items, ensure_ascii=False))
        response = self.llm_client.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": "Construct source-bound evidence sentence units. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response)
        if not isinstance(data, dict):
            return self._fallback_units(sentences, offset)

        source_by_id = {sid: idx for idx, sid in enumerate(raw_ids)}
        used: set[str] = set()
        units: List[EvidenceUnit] = []
        for item in data.get("units", []):
            unit = self._parse_unit(item, sentences, raw_ids, source_by_id, used)
            if unit is not None:
                units.append(unit)
                used.update(unit.source_sentence_ids)

        for idx, sid in enumerate(raw_ids):
            if sid not in used:
                units.extend(self._fallback_units([sentences[idx]], offset + idx))

        return sorted(units, key=lambda unit: source_by_id.get(unit.source_sentence_ids[0], 10**9))

    def _parse_unit(
        self,
        item: Any,
        sentences: Sequence[Tuple[str, str, datetime]],
        raw_ids: List[str],
        source_by_id: Dict[str, int],
        used: set[str],
    ) -> EvidenceUnit | None:
        if not isinstance(item, dict):
            return None
        source_ids = item.get("source_sentence_ids", [])
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        if not isinstance(source_ids, list):
            return None
        source_ids = [str(source_id).strip() for source_id in source_ids if str(source_id).strip()]
        if not source_ids or any(source_id not in source_by_id for source_id in source_ids):
            return None
        if any(source_id in used for source_id in source_ids):
            return None

        indices = [source_by_id[source_id] for source_id in source_ids]
        if indices != sorted(indices):
            return None
        if indices != list(range(indices[0], indices[-1] + 1)):
            return None

        text = str(item.get("text", "")).strip()
        if not text:
            return None
        source_text = " ".join(sentences[idx][0] for idx in indices)
        if not self._looks_source_bound(text, source_text):
            return None

        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))

        source_speakers = [str(sentences[idx][1]) for idx in indices]
        speaker = source_speakers[0] if len(set(source_speakers)) == 1 else "mixed"
        timestamp = sentences[indices[0]][2]
        resolved_references = item.get("resolved_references", [])
        if not isinstance(resolved_references, list):
            resolved_references = []
        unresolved = item.get("unresolved_references", [])
        if isinstance(unresolved, str):
            unresolved = [unresolved]
        if not isinstance(unresolved, list):
            unresolved = []

        return EvidenceUnit(
            text=text,
            speaker=speaker,
            timestamp=timestamp,
            source_sentence_ids=source_ids,
            source_texts=[sentences[idx][0] for idx in indices],
            source_speakers=source_speakers,
            source_timestamps=[
                sentences[idx][2].isoformat() if hasattr(sentences[idx][2], "isoformat") else ""
                for idx in indices
            ],
            resolved_references=[
                ref for ref in resolved_references if isinstance(ref, dict)
            ],
            unresolved_references=[str(ref) for ref in unresolved if str(ref).strip()],
            merge_reason=str(item.get("merge_reason", "")).strip(),
            confidence=confidence,
        )

    @staticmethod
    def _looks_source_bound(text: str, source_text: str) -> bool:
        """Reject obvious summaries or hallucinated rewrites without blocking reference edits."""
        if len(text) > max(len(source_text) * 2.0 + 120, 240):
            return False
        source_tokens = EvidenceUnitBuilder._content_tokens(source_text)
        if len(source_tokens) < 4:
            return True
        output_tokens = set(EvidenceUnitBuilder._content_tokens(text))
        overlap = sum(1 for token in source_tokens if token in output_tokens)
        return overlap / max(1, len(source_tokens)) >= 0.45

    @staticmethod
    def _content_tokens(text: str) -> List[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "for",
            "from",
            "i",
            "in",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "she",
            "that",
            "the",
            "they",
            "this",
            "to",
            "was",
            "we",
            "you",
        }
        return [
            token
            for token in re.findall(r"[A-Za-z0-9']+", text.lower())
            if token not in stopwords and len(token) > 1
        ]

    @staticmethod
    def _fallback_units(
        sentences: Sequence[Tuple[str, str, datetime]],
        offset: int,
    ) -> List[EvidenceUnit]:
        return [
            EvidenceUnit(
                text=text,
                speaker=str(speaker),
                timestamp=timestamp,
                source_sentence_ids=[f"s{offset + idx}"],
                source_texts=[text],
                source_speakers=[str(speaker)],
                source_timestamps=[timestamp.isoformat() if hasattr(timestamp, "isoformat") else ""],
            )
            for idx, (text, speaker, timestamp) in enumerate(sentences)
        ]
