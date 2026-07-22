"""
实体/关系抽取工具。

当前 MAG 建图主路径是 LLM 直接抽取 triples:
sentence_id + text → (head, relation, tail, source_sentence_id)。
旧的实体抽取和判别式 pairwise relation detector 仅保留为兼容工具。
"""

from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from mag.schema import Triple

logger = logging.getLogger(__name__)

# spaCy 依存标签 → 关系角色映射
_PREDICATE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl"}

# ====================================================================
# LLM 判别式 Prompt — 给定句子+实体对，判断是否有关联
# ====================================================================

_DISCRIMINATIVE_PROMPT = """Given a sentence and a list of entity pairs from that sentence, determine which pairs have a semantic relationship.

For each pair that HAS a relationship:
- Write a concise relation label (e.g. "works_at", "located_in", "knows", "uses", "created", "owns", "is_a", "part_of")
- Set "has_relation": true

For each pair with NO relationship:
- Set "has_relation": false

Rules:
1. Only use entities exactly as they appear in the sentence.
2. A relationship exists ONLY if the sentence explicitly states a connection between the two entities.
3. If entities are merely mentioned in the same sentence without a direct connection, mark as false.
4. The relation label must be a short verb or prepositional phrase describing the connection.

Sentence: {sentence}

Entity pairs to evaluate:
{entity_pairs}

Return JSON: {{"relations": [{{"head": "Google", "tail": "Mountain View", "relation": "located_in", "has_relation": true, "confidence": 0.9}}, ...]}}"""

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


class EntityExtractor:
    """实体抽取 — 仅抽实体名称和类型，不抽关系"""

    def __init__(self, llm_client=None, use_llm: bool = False, use_mem0: bool = False):
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.use_mem0 = use_mem0
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using regex fallback")
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def extract(self, sentence: str) -> List[Tuple[str, str]]:
        """返回 [(entity_name, entity_type), ...]"""
        if not sentence or not sentence.strip():
            return []

        if self.use_llm and self.llm_client:
            return self._extract_by_llm(sentence)
        elif self.use_mem0:
            return self._extract_by_mem0(sentence)
        else:
            return self._extract_by_spacy(sentence)

    def extract_batch(self, sentences: List[str]) -> List[List[Tuple[str, str]]]:
        return [self.extract(s) for s in sentences]

    def _extract_by_spacy(self, sentence: str) -> List[Tuple[str, str]]:
        nlp = self.nlp
        if nlp is None:
            return self._extract_by_regex(sentence)

        doc = nlp(sentence)
        seen = set()
        results: Dict[str, str] = {}

        # 1. NER
        for ent in doc.ents:
            name = ent.text.strip().lower()
            if len(name) > 1 and name not in seen:
                seen.add(name); results[name] = ent.label_

        # 2. 正则 fallback: 大写单词 spacy NER 漏掉的（如人名 Caroline）
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for m in re.findall(pattern, sentence):
            m_lower = m.lower()
            if m_lower not in seen and len(m) > 1:
                seen.add(m_lower); results[m_lower] = "PERSON"

        return [(name, label) for name, label in results.items()]

    @staticmethod
    def _extract_by_mem0(sentence: str) -> List[Tuple[str, str]]:
        """使用 mem0 的多级规则实体抽取（PROPER+QUOTED+COMPOUND+NOUN）"""
        try:
            from mem0.utils.entity_extraction import extract_entities as _mem0_extract
            return [(name, etype) for etype, name in _mem0_extract(sentence)]
        except Exception:
            return EntityExtractor._extract_by_regex(sentence)

    @staticmethod
    def _extract_by_regex(sentence: str) -> List[Tuple[str, str]]:
        """Regex 兜底 — 识别首字母大写的多词短语"""
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, sentence)
        seen = set()
        result = []
        for m in matches:
            m_lower = m.lower()
            if m_lower not in seen and len(m) > 1:
                seen.add(m_lower)
                result.append((m, "NP"))
        return result

    def _extract_by_llm(self, sentence: str) -> List[Tuple[str, str]]:
        try:
            resp = self.llm_client.generate_response(
                messages=[{
                    "role": "system",
                    "content": "Extract named entities (people, orgs, locations, products, dates, etc.) from the sentence."
                }, {
                    "role": "user",
                    "content": (
                        f'Sentence: "{sentence}"\n\n'
                        'Return JSON: {{"entities": [{{"name": "Google", "type": "ORG"}}, ...]}}'
                    ),
                }],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp)
            return [(e["name"], e.get("type", "")) for e in data.get("entities", [])]
        except Exception as e:
            logger.warning("LLM entity extraction failed: %s", str(e))
            return self._extract_by_spacy(sentence)


class DiscriminativeRelationDetector:
    """Relation detector facade.

    `extract_triples_direct()` is the MAG graph-construction path. The older
    pairwise discriminative methods remain for backward compatibility.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_triples_direct(
        self,
        sentence_items: List[Tuple[str, str]],
        timestamps: Optional[Dict[str, str]] = None,
    ) -> List[Triple]:
        """Extract graph triples directly from sentences with LLM-assigned source ids.

        This is the graph-construction path for MAG. It does not depend on
        spaCy/entity pre-extraction or pairwise discriminative relation checks.
        """
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

    # ------------------------------------------------------------------
    # 规则指代消解 — 零 API，基于说话人标签替换第一/二人称
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_coreferences_rule(
        sentence_items: List[Tuple[str, str]],  # [(sentence_id, text), ...]
    ) -> Dict[str, str]:
        """
        基于说话人标签的轻量指代消解：
        - I/me/my → 替换为说话人
        - you/your → 替换为对话另一方

        speaker 信息应当已经在前缀中（如 'Caroline: ...'），
        如果 text 不含 speaker 前缀则不会正确工作。
        """
        import re
        result = {}

        # 追踪最近见过的两个说话人
        last_speaker = None
        prev_speaker = None

        for sid, text in sentence_items:
            # 提取说话人前缀
            speaker = None
            if ': ' in text:
                parts = text.split(': ', 1)
                speaker = parts[0].strip()
                content = parts[1]
            else:
                content = text

            if speaker:
                prev_speaker = last_speaker
                last_speaker = speaker

            # 仅替换第一人称 I/me/my → speaker (100% 准确)
            # 第二人称 you/your → 另一方 (对话场景准确)
            # 不碰 they/we/them/it — 需要上下文理解，规则做不到
            if last_speaker:
                modified = content
                modified = re.sub(r'\bI\b', last_speaker, modified)
                modified = re.sub(r'\bmy\b', last_speaker + "'s", modified)
                modified = re.sub(r'\bme\b', last_speaker, modified)
                modified = re.sub(r'\bMy\b', last_speaker + "'s", modified)
                modified = re.sub(r'\bMe\b', last_speaker, modified)
                modified = re.sub(r"\bI've\b", last_speaker + " has", modified)
                modified = re.sub(r"\bI'm\b", last_speaker + " is", modified)
                if prev_speaker:
                    modified = re.sub(r'\byou\b', prev_speaker, modified)
                    modified = re.sub(r'\byour\b', prev_speaker + "'s", modified)
                    modified = re.sub(r'\bYou\b', prev_speaker, modified)
                    modified = re.sub(r'\bYour\b', prev_speaker + "'s", modified)

                if modified != content:
                    full_text = f"{last_speaker}: {modified}" if speaker else modified
                    result[sid] = full_text

        return result

    # ------------------------------------------------------------------
    # LLM 指代消解 — 独立的 LLM 调用（慢但更准）
    # ------------------------------------------------------------------

    def resolve_coreferences(
        self,
        sentence_items: List[Tuple[str, str]],  # [(sentence_id, text), ...]
    ) -> Dict[str, str]:
        """
        用 LLM 将对话题中的代词消解为具体实体名。
        独立于关系检测，可单独 ablation。

        Returns: {sentence_id: resolved_text, ...} — 仅包含被修改的句子
        """
        if not self.llm_client or len(sentence_items) < 2:
            return {}

        import json
        items_json = json.dumps(
            {sid[:8]: text for sid, text in sentence_items},
            ensure_ascii=False,
        )
        prompt = (
            "Rewrite each sentence to replace ALL pronouns with the specific entity/person "
            "they refer to, based on conversation context. This includes: "
            "I, my, me, mine (→ speaker's name), you, your (→ listener's name), "
            "it, they, he, she, that, this, we, them, him, her.\n\n"
            "Rules:\n"
            "1. 'I' 'my' 'me': replace with the SPEAKER's name (e.g. Caroline, Jon).\n"
            "2. 'you' 'your': replace with who the speaker is talking to.\n"
            "3. 'we': replace with the group implied by context (e.g. 'Caroline and Melanie').\n"
            "4. 'that' as a determiner ('that book', 'that day'): do NOT replace.\n"
            "5. Keep all other words exactly as they are.\n"
            "6. Return ONLY sentences where you actually changed pronouns.\n\n"
            f"Sentences:\n{items_json}\n\n"
            'Return JSON: {"resolved": {"sentence_id_prefix": "rewritten text", ...}}'
        )
        try:
            response = self.llm_client.generate_response(
                messages=[{"role": "system", "content": "Resolve pronouns to entities based on context. Return JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
            raw = data.get("resolved", {})
        except Exception as e:
            logger.warning("Coref resolution failed: %s", str(e)[:100])
            return {}

        sid_map = {sid[:8]: sid for sid, _ in sentence_items}
        result = {}
        for short_id, text in raw.items():
            if short_id in sid_map and isinstance(text, str) and text.strip():
                result[sid_map[short_id]] = text.strip()
        return result

    def detect(
        self,
        sentence: str,
        entities: List[Tuple[str, str]],  # [(name, type), ...]
        sentence_id: str = "",
    ) -> List[Triple]:
        """
        输入: 一个句子 + 已抽取的实体列表
        输出: LLM 确认存在关系的 Triple 列表 (每条带 source_sentence_id)
        """
        if not entities or len(entities) < 2:
            return []
        if not self.llm_client:
            return self._detect_by_heuristic(sentence, entities, sentence_id)

        return self._detect_by_llm(sentence, entities, sentence_id)

    def detect_batch(
        self,
        sentences: List[str],
        entity_lists: List[List[Tuple[str, str]]],
        sentence_ids: List[str] = None,
    ) -> List[List[Triple]]:
        if sentence_ids is None:
            sentence_ids = [""] * len(sentences)

        results = []
        for sent, ents, sid in zip(sentences, entity_lists, sentence_ids):
            try:
                results.append(self.detect(sent, ents, sid))
            except Exception as e:
                logger.debug("Relation detection failed for '%s': %s", sent[:50], str(e))
                results.append([])
        return results

    # ------------------------------------------------------------------
    # LLM 联合分割+消解+关系判别 — 一次调用，三个产出
    # ------------------------------------------------------------------

    def segment_and_relate(
        self,
        sentence_items: List[Tuple[str, str]],
        carry_over: List[Tuple[str, str]] = None,
        timestamps: Dict[str, str] = None,  # sid → ISO date
    ) -> Dict[str, Any]:
        """一次 LLM 调用完成: 分句 + 消解 + 关系判别"""
        if not self.llm_client:
            return {"segments": [], "triples": [], "carry_over": []}

        import json

        # Step 1: 拼接上下文
        all_items = (carry_over or []) + list(sentence_items)
        sid_to_text = {sid: text for sid, text in all_items}
        ts = timestamps or {}

        # Step 2: 补全 speaker 前缀 — 碎片句继承前一 句的 speaker
        last_speaker = ""
        display_items = []
        for i, (sid, text) in enumerate(sentence_items):
            if ": " in text:
                last_speaker = text.split(": ", 1)[0].strip()
                display_items.append({"i": i, "text": text})
            elif last_speaker:
                display_items.append({"i": i, "text": f"{last_speaker}: {text}"})
            else:
                display_items.append({"i": i, "text": text})
        items_json = json.dumps(display_items, ensure_ascii=False)
        prompt = (
            "Resolve ALL pronouns and merge adjacent sentences into self-contained facts.\n\n"
            "RULES:\n"
            "- Replace I/me/my → speaker name (e.g. 'Caroline', 'Jon')\n"
            "- Replace you/your → listener name from context\n"
            "- Replace we/they/it → the specific group/thing referenced\n"
            "- Merge consecutive sentences that form a continuous utterance into ONE sentence\n"
            "  (including both short reactions AND longer semantically-continuous sentences)\n"
            "- Use connectors: ', and ', '; ', '. Then ', ', so ', ', but '\n"
            "- CRITICAL: Do NOT rewrite or rephrase. ONLY replace pronouns and "
            "join sentences with connectors. Every original word must remain in output.\n"
            "- Output FEWER sentences than input when merging is appropriate\n\n"
            f"Sentences:\n{items_json}\n\n"
            'Return JSON:\n'
            '{{"facts": ["resolved sentence 1", "resolved sentence 2", ...]}}'
        )
        try:
            response = self.llm_client.generate_response(
                messages=[{"role": "system", "content": "Resolve pronouns and merge sentences. Keep original wording. Return JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
        except Exception as e:
            logger.warning("segment_and_relate failed: %s", str(e)[:200])
            return {"segments": [], "facts": [], "triples": [], "carry_over": [], "merged_away": []}

        # Step 3: 直接使用 LLM 输出的 resolved facts
        facts = [f.strip() for f in data.get("facts", []) if isinstance(f, str) and f.strip()]
        merged_away = {sid for sid, _ in sentence_items}
        carry = [(facts[-1], None)] if facts else []

        return {"segments": [], "facts": facts, "triples": [],
                "carry_over": carry, "merged_away": list(merged_away)}

    # ------------------------------------------------------------------
    # 新方法: 基于完整对话上下文抽取关系
    # ------------------------------------------------------------------

    def extract_from_context(
        self,
        sentence_items: List[Tuple[str, str]],  # [(sentence_id, text), ...]
        pre_extracted: Dict[str, List[Tuple[str, str]]] = None,  # sid → [(name,type)]
    ) -> List[Triple]:
        """
        对于连续的几轮对话：先抽实体 → 取实体出现的上下文句子 → LLM 判别式判断关联。
        如果提供 pre_extracted，跳过实体抽取步骤，直接使用已存实体。
        不做 SPO，不要求边属性，只判 YES/NO。
        """
        if not self.llm_client or len(sentence_items) < 2:
            return []

        # Step 1: 抽所有实体（复用 S2 已存的实体，避免重复抽取）
        sid_to_text = {sid: text for sid, text in sentence_items}
        entity_sentences: Dict[str, Set[str]] = {}  # entity_lower -> {sid, ...}
        for sid, text in sentence_items:
            entities = (pre_extracted or {}).get(sid)
            if not entities:
                entities = self._extract_entities(text)
            for e, _ in entities:
                e_lower = e.strip().lower()
                if e_lower not in entity_sentences:
                    entity_sentences[e_lower] = set()
                entity_sentences[e_lower].add(sid)

        entity_names = list(entity_sentences.keys())
        if len(entity_names) < 2:
            return []

        # Step 2: 找共现实体对
        pairs = set()
        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                e1, e2 = entity_names[i], entity_names[j]
                if entity_sentences[e1] & entity_sentences[e2]:
                    pairs.add((e1, e2))
        if not pairs:
            return []

        # Step 3: 取实体出现的上下文句子
        pair_contexts = []
        for e1, e2 in pairs:
            sids = entity_sentences[e1] | entity_sentences[e2]
            texts = [sid_to_text.get(sid, "") for sid in sids]
            pair_contexts.append({
                "entity1": e1, "entity2": e2,
                "context": " | ".join(texts[:3]),
                "sids": list(sids),
            })

        # Step 4: LLM 自由建边 — 给实体列表+上下文，不做共现预筛
        entities_json = json.dumps(list(entity_names)[:80], ensure_ascii=False)
        context_json = json.dumps(
            [text for _, text in sentence_items], ensure_ascii=False)
        prompt = (
            "Given a conversation context and a list of entities, identify entity pairs "
            "that have a semantic relationship based on the conversation.\n\n"
            "Rules:\n"
            "1. Propose relations ONLY if the conversation explicitly states a connection.\n"
            "2. Relation types: works_at, located_in, knows, uses, created, owns, is_a, part_of\n"
            "3. Use the entity names EXACTLY as listed.\n\n"
            f"Entities: {entities_json}\n\n"
            f"Conversation:\n{context_json}\n\n"
            'Return JSON: {"related": [{"head": "caroline", "relation": "works_at", "tail": "google", "confidence": 0.9}, ...]}'
        )
        try:
            response = self.llm_client.generate_response(
                messages=[{"role": "system", "content": "Determine entity relationships."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
            related = data.get("related", [])
            logger.warning("LLM relation: entities=%d related=%d keys=%s first=%s",
                           len(entity_names), len(related), list(data.keys())[:5],
                           str(related[:3])[:200] if related else 'EMPTY')
        except Exception as e:
            logger.warning("LLM relation detection failed: %s", str(e))
            return []

        # Step 5: 连边 — 自由建边模式，不要求共现
        triples = []
        found_any = False
        for item in related:
            if not isinstance(item, dict):
                continue
            e1 = item.get("head", item.get("entity1", "")).strip().lower()
            e2 = item.get("tail", item.get("entity2", "")).strip().lower()
            conf = float(item.get("confidence", 0.5))
            # 找任一包含 e1 的句子作为 source
            src_sid = ""
            for sid, text in sentence_items:
                if e1 in text.lower():
                    src_sid = sid; break
            if src_sid:
                triples.append(Triple(
                    head=e1, relation="related", tail=e2,
                    confidence=conf, source_sentence_id=src_sid,
                ))
        logger.warning("LLM relation: %d pairs → %d triples created", len(related), len(triples))
        return triples

    @staticmethod
    def _extract_entities(text: str) -> List[Tuple[str, str]]:
        """抽实体: mem0 多级规则 (PROPER+QUOTED+COMPOUND+NOUN)"""
        try:
            from mem0.utils.entity_extraction import extract_entities as _mem0_extract
            return [(name, etype) for etype, name in _mem0_extract(text)]
        except Exception:
            import re
            return [(m.strip().lower(), "NP") for m in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)]


    def _detect_by_llm(
        self,
        sentence: str,
        entities: List[Tuple[str, str]],
        sentence_id: str,
    ) -> List[Triple]:
        """
        LLM 基于完整对话上下文抽取关系（非逐对判断）。

        输入: 一批句子及其 ID
              [(sid1, "Alice works at Google"), (sid2, "Bob uses Rust"), ...]
        LLM 看到: 完整对话列表，理解上下文后抽取所有 SPO 三元组
        输出: 每条 Triple 携带 source_sentence_id
              Alice-Mountain View: NO
              Google-Mountain View: YES, "located_in", 0.85
        输出:
              [Triple("Alice","works_at","Google",0.95,sid),
               Triple("Google","located_in","Mountain View",0.85,sid)]
        """
        # 生成所有实体对 (不重复)
        entity_names = [e[0] for e in entities]
        pairs = list(combinations(entity_names, 2))

        if not pairs:
            return []

        # 格式化 pairs 供 LLM 判断
        pairs_str = "\n".join(
            f"  [{i}] ({a}, {b})" for i, (a, b) in enumerate(pairs)
        )

        prompt = _DISCRIMINATIVE_PROMPT.format(
            sentence=sentence, entity_pairs=pairs_str,
        )

        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "You detect semantic relationships between entity pairs. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
            relations = data.get("relations", [])
        except Exception as e:
            logger.warning("LLM relation detection failed: %s", str(e))
            return []

        # 只保留 has_relation=true 的
        triples = []
        for item in relations:
            if not isinstance(item, dict):
                continue
            if not item.get("has_relation", False):
                continue

            head = item.get("head", "").strip()
            relation = item.get("relation", "").strip()
            tail = item.get("tail", "").strip()
            confidence = float(item.get("confidence", 0.5))

            if head and relation and tail:
                triples.append(Triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    confidence=confidence,
                    source_sentence_id=sentence_id,
                ))

        return triples

    # ------------------------------------------------------------------
    # 启发式兜底 (无 LLM 时)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_by_heuristic(
        sentence: str,
        entities: List[Tuple[str, str]],
        sentence_id: str,
    ) -> List[Triple]:
        """
        无 LLM 时的简单启发式 — 基于依存解析 (如果 spaCy 可用)
        否则返回空列表。
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            return []

        doc = nlp(sentence)
        entity_names = [e[0] for e in entities]
        entity_set = set(name.lower() for name in entity_names)

        triples = []
        for token in doc:
            if token.dep_ not in _PREDICATE_DEPS:
                continue

            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "iobj", "attr")]

            for subj in subjects:
                subj_text = subj.text.strip()
                if subj_text.lower() not in entity_set:
                    continue

                relation = token.lemma_ or token.text

                for obj in objects:
                    obj_text = obj.text.strip()
                    if obj_text.lower() not in entity_set:
                        continue

                    distance = abs(token.i - subj.i) + abs(token.i - obj.i)
                    confidence = max(0.2, 0.8 - 0.05 * distance)

                    triples.append(Triple(
                        head=subj_text,
                        relation=relation,
                        tail=obj_text,
                        confidence=confidence,
                        source_sentence_id=sentence_id,
                    ))

        return triples
