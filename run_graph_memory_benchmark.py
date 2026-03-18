#!/usr/bin/env python3
"""
Unified benchmark framework for Graph Agent Memory experiments on LoCoMo.

Features:
- User-customizable LLM `base_url`, `api_key`, `model_name`
- User-customizable embedding endpoint/model
- Side-by-side comparison across graph memory methods and earlier retrieval baselines
- Structured JSON output for reproducible experiments
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_dataset import load_locomo_dataset
from memory.answer_formatter import AnswerFormatter
from memory.best_of_n_selector import BestOfNSelector
from memory.evaluator import Evaluator
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.test_harness import TestHarness
from memory.trg_memory import QueryContext


@dataclass
class EndpointConfig:
    api_key: Optional[str]
    base_url: Optional[str]
    model_name: str


@dataclass
class EmbeddingConfig:
    backend: str
    model_name: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]


def normalize_name(value: str) -> str:
    return value.replace("/", "_").replace(".", "_").replace("-", "_")


class SimpleRetrievalEngine:
    """Baseline retrieval-only engine with the same query interface as QueryEngine."""

    def __init__(self, builder: MemoryBuilder, mode: str):
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

        seen = set()
        nodes = []
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


def set_env_if_present(name: str, value: Optional[str]) -> None:
    if value:
        os.environ[name] = value


def create_builder(sample_id: int, args, llm_cfg: EndpointConfig, emb_cfg: EmbeddingConfig) -> MemoryBuilder:
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    sample_cache = cache_root / f"sample_{sample_id}"
    sample_cache.mkdir(parents=True, exist_ok=True)

    return MemoryBuilder(
        cache_dir=str(sample_cache),
        llm_model=llm_cfg.model_name,
        use_episodes=args.use_episodes,
        embedding_model=emb_cfg.backend,
        llm_api_key=llm_cfg.api_key,
        llm_base_url=llm_cfg.base_url,
        embedding_model_name=emb_cfg.model_name,
        embedding_api_key=emb_cfg.api_key,
        embedding_base_url=emb_cfg.base_url,
    )


def create_engine(method: str, builder: MemoryBuilder):
    if method in {"graph_full", "basic_retrieval", "no_causal", "no_temporal", "flat_graph"}:
        ablation_config = {}
        if method != "graph_full":
            ablation_key_map = {
                "basic_retrieval": "basic_retrieval",
                "no_causal": "no_causal",
                "no_temporal": "no_temporal",
                "flat_graph": "flat_graph",
            }
            ablation_config[ablation_key_map[method]] = True

        return QueryEngine(
            builder.trg,
            builder.node_index,
            entity_session_map=getattr(builder, "entity_session_map", None),
            entity_dia_map=getattr(builder, "entity_dia_map", None),
            llm_controller=builder.llm_controller,
            ablation_config=ablation_config,
        )

    if method in {"vector_only", "keyword_only", "scan_only"}:
        return SimpleRetrievalEngine(builder, method)

    raise ValueError(f"Unsupported method: {method}")


def summarize_results(results: List[Dict], evaluator: Evaluator) -> Dict:
    evaluation_results = []
    for result in results:
        evaluation_results.append(
            {
                "is_correct": result.get("correct", False),
                "metrics": result.get("metrics"),
                "llm_judge_score": result.get("llm_judge_score", 0.0),
                "category": result.get("category"),
            }
        )

    aggregate = evaluator.compute_aggregate_stats(evaluation_results)
    by_category = evaluator.compute_category_stats(evaluation_results)
    aggregate["avg_latency_sec"] = (
        sum(r.get("processing_time", 0.0) for r in results) / len(results) if results else 0.0
    )
    aggregate["questions"] = len(results)
    return {
        "overall": aggregate,
        "by_category": by_category,
    }


def run_method(method: str, sample, builder: MemoryBuilder, evaluator: Evaluator, args) -> Dict:
    engine = create_engine(method, builder)
    tester = TestHarness(builder, engine, evaluator=evaluator)
    tester.best_of_n = max(1, args.best_of_n)
    tester.best_of_n_method = args.best_of_n_method
    if tester.best_of_n > 1:
        tester.best_of_n_selector = BestOfNSelector(
            n_attempts=tester.best_of_n,
            selection_method=tester.best_of_n_method
        )

    sample_for_test = copy.copy(sample)
    allowed_categories = set(args.categories)
    sample_for_test.qa = [qa for qa in sample.qa if qa.category in allowed_categories]

    if args.parallel:
        results = tester.test_questions_parallel(sample_for_test, max_questions=args.max_questions, n_workers=args.n_workers)
    else:
        results = tester.test_questions(sample_for_test, max_questions=args.max_questions)

    return {
        "method": method,
        "stats": summarize_results(results, evaluator),
        "results": results,
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Unified benchmark for Graph Agent Memory on LoCoMo")
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", type=str, default="graph_full,basic_retrieval,no_causal,no_temporal,flat_graph,vector_only,keyword_only")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cache-dir", default="./benchmark_cache")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--use-episodes", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--best-of-n", type=int, default=1)
    parser.add_argument("--best-of-n-method", choices=["llm_judge", "voting", "f1"], default="llm_judge")

    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model-name", default=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"))

    parser.add_argument("--embedding-backend", choices=["minilm", "openai"], default=os.getenv("DEFAULT_EMBEDDING_BACKEND", "minilm"))
    parser.add_argument("--embedding-model-name", default=os.getenv("DEFAULT_EMBEDDING_MODEL"))
    parser.add_argument("--embedding-api-key", default=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"))

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            normalized_key = key.replace("-", "_")
            if hasattr(args, normalized_key):
                setattr(args, normalized_key, value)

    llm_cfg = EndpointConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
    )
    emb_cfg = EmbeddingConfig(
        backend=args.embedding_backend,
        model_name=args.embedding_model_name,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )

    set_env_if_present("OPENAI_API_KEY", llm_cfg.api_key)
    set_env_if_present("OPENAI_BASE_URL", llm_cfg.base_url)

    samples = load_locomo_dataset(args.dataset)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "samples": args.sample,
        "methods": methods,
        "llm": asdict(llm_cfg),
        "embedding": asdict(emb_cfg),
        "max_questions": args.max_questions,
        "categories": args.categories,
        "runs": [],
    }

    for sample_id in args.sample:
        sample = samples[sample_id]
        builder = create_builder(sample_id, args, llm_cfg, emb_cfg)
        cache_file = Path(builder.cache_dir) / "graph.json"

        if cache_file.exists() and not args.rebuild:
            builder.load()
        else:
            builder.build_memory(sample)
            builder.save()

        evaluator = Evaluator(
            llm_controller=builder.llm_controller,
            use_llm_judge=builder.llm_controller is not None,
        )

        sample_runs = []
        for method in methods:
            sample_runs.append(run_method(method, sample, builder, evaluator, args))

        report["runs"].append(
            {
                "sample_id": sample_id,
                "memory_stats": builder.trg.get_statistics(),
                "method_runs": sample_runs,
            }
        )

    model_tag = normalize_name(llm_cfg.model_name)
    output_file = output_dir / f"graph_memory_benchmark_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Benchmark finished. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
