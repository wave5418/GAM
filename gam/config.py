from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMSettings:
    api_key: Optional[str]
    base_url: Optional[str]
    model_name: str


@dataclass
class EmbeddingSettings:
    backend: str
    model_name: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]


@dataclass
class BenchmarkConfig:
    dataset_format: str
    dataset: str
    sample: List[int]
    methods: List[str]
    max_questions: int
    categories: List[int]
    cache_dir: str
    output_dir: str
    rebuild: bool
    use_episodes: bool
    parallel: bool
    n_workers: int
    best_of_n: int
    best_of_n_method: str
    llm: LLMSettings
    embedding: EmbeddingSettings

    @classmethod
    def from_sources(cls, args) -> "BenchmarkConfig":
        config_data = {}
        if getattr(args, "config", None):
            with open(args.config, "r", encoding="utf-8") as f:
                config_data = json.load(f)

        def read(name: str, default: Any) -> Any:
            value = getattr(args, name, None)
            if value is not None:
                return value
            return config_data.get(name, default)

        llm = LLMSettings(
            api_key=read("api_key", os.getenv("OPENAI_API_KEY")),
            base_url=read("base_url", os.getenv("OPENAI_BASE_URL")),
            model_name=read("model_name", os.getenv("DEFAULT_MODEL", "gpt-4o-mini")),
        )
        embedding = EmbeddingSettings(
            backend=read("embedding_backend", os.getenv("DEFAULT_EMBEDDING_BACKEND", "minilm")),
            model_name=read("embedding_model_name", os.getenv("DEFAULT_EMBEDDING_MODEL")),
            api_key=read("embedding_api_key", os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")),
            base_url=read("embedding_base_url", os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
        )

        methods = read(
            "methods",
            ["graph_full", "basic_retrieval", "no_causal", "no_temporal", "flat_graph", "vector_only", "keyword_only"],
        )
        if isinstance(methods, str):
            methods = [item.strip() for item in methods.split(",") if item.strip()]

        return cls(
            dataset_format=read("dataset_format", "locomo"),
            dataset=read("dataset", "data/locomo10.json"),
            sample=read("sample", [0]),
            methods=methods,
            max_questions=int(read("max_questions", 20)),
            categories=read("categories", [1, 2, 3, 4]),
            cache_dir=read("cache_dir", "./benchmark_cache"),
            output_dir=read("output_dir", "./benchmark_results"),
            rebuild=bool(read("rebuild", False)),
            use_episodes=bool(read("use_episodes", False)),
            parallel=bool(read("parallel", False)),
            n_workers=int(read("n_workers", 3)),
            best_of_n=int(read("best_of_n", 1)),
            best_of_n_method=read("best_of_n_method", "llm_judge"),
            llm=llm,
            embedding=embedding,
        )

    def export(self) -> Dict[str, Any]:
        return asdict(self)

    def output_path(self) -> Path:
        safe_model = self.llm.model_name.replace("/", "_").replace(".", "_").replace("-", "_")
        return Path(self.output_dir) / f"graph_memory_benchmark_{safe_model}.json"
