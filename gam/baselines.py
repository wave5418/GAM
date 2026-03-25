from __future__ import annotations

from typing import Tuple

from memory.answer_formatter import AnswerFormatter
from memory.trg_memory import QueryContext


class RetrievalOnlyBaseline:
    """Simple retrieval baseline with a narrow, framework-owned implementation."""

    def __init__(self, builder, mode: str):
        self.builder = builder
        self.mode = mode
        self.answer_formatter = AnswerFormatter()

    def _vector_nodes(self, question: str, top_k: int):
        context = self.builder.trg.query(question, max_results=top_k)
        return context.anchor_nodes if context else []

    def _keyword_nodes(self, question: str, top_k: int):
        words = [w.strip('.,!?;:"\'-').lower() for w in question.split()]
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "what", "when", "where", "who", "how", "did", "does", "do"}
        matched_ids = []
        for word in words:
            if not word or word in stop_words:
                continue
            matched_ids.extend(list(self.builder.node_index.get(word, [])))

        nodes = []
        seen = set()
        for node_id in matched_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.builder.trg.graph_db.nodes.get(node_id)
            if node is not None:
                nodes.append(node)
            if len(nodes) >= top_k:
                break
        return nodes

    def _scan_nodes(self, question: str, top_k: int):
        words = [w.strip('.,!?;:"\'-').lower() for w in question.split() if len(w) > 2]
        scored = []
        for node in self.builder.trg.graph_db.nodes.values():
            if not hasattr(node, "attributes"):
                continue
            content = getattr(node, "content_narrative", "") or getattr(node, "summary", "")
            content_lower = content.lower()
            score = sum(1 for word in words if word in content_lower)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def query(self, question: str, top_k: int = 15) -> Tuple[QueryContext, str]:
        if self.mode == "vector_only":
            nodes = self._vector_nodes(question, top_k)
        elif self.mode == "keyword_only":
            nodes = self._keyword_nodes(question, top_k)
        elif self.mode == "scan_only":
            nodes = self._scan_nodes(question, top_k)
        else:
            raise ValueError(f"Unsupported baseline mode: {self.mode}")

        query_context = QueryContext(
            query_text=question,
            anchor_nodes=nodes,
            traversal_paths=[],
            narrative_context=f"Baseline mode: {self.mode}"
        )
        query_context.metadata = {
            "query_type": self.mode,
            "baseline_mode": self.mode,
            "top_k_returned": len(nodes),
        }
        answer_context = self.answer_formatter.format_context_for_qa(nodes, question, session_nodes=[])
        return query_context, answer_context
