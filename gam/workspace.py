from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path

from load_dataset import load_locomo_dataset
from memory.best_of_n_selector import BestOfNSelector
from memory.evaluator import Evaluator
from memory.memory_builder import MemoryBuilder
from memory.test_harness import TestHarness

from .methods import MethodRegistry
from .reporting import summarize_results


class BenchmarkWorkspace:
    """Independent orchestration layer for benchmark runs."""

    def __init__(self, config):
        self.config = config
        self._seed_environment()

    def _seed_environment(self) -> None:
        if self.config.llm.api_key:
            os.environ["OPENAI_API_KEY"] = self.config.llm.api_key
        if self.config.llm.base_url:
            os.environ["OPENAI_BASE_URL"] = self.config.llm.base_url

    def load_samples(self):
        return load_locomo_dataset(self.config.dataset)

    def create_builder(self, sample_id: int) -> MemoryBuilder:
        cache_root = Path(self.config.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        sample_cache = cache_root / f"sample_{sample_id}"
        sample_cache.mkdir(parents=True, exist_ok=True)

        return MemoryBuilder(
            cache_dir=str(sample_cache),
            llm_model=self.config.llm.model_name,
            use_episodes=self.config.use_episodes,
            embedding_model=self.config.embedding.backend,
            llm_api_key=self.config.llm.api_key,
            llm_base_url=self.config.llm.base_url,
            embedding_model_name=self.config.embedding.model_name,
            embedding_api_key=self.config.embedding.api_key,
            embedding_base_url=self.config.embedding.base_url,
        )

    def ensure_memory(self, builder: MemoryBuilder, sample) -> None:
        cache_file = Path(builder.cache_dir) / "graph.json"
        if cache_file.exists() and not self.config.rebuild:
            builder.load()
        else:
            builder.build_memory(sample)
            builder.save()

    def run_method(self, method: str, sample, builder: MemoryBuilder, evaluator: Evaluator):
        engine = MethodRegistry.create(method, builder)
        tester = TestHarness(builder, engine, evaluator=evaluator)
        tester.best_of_n = max(1, self.config.best_of_n)
        tester.best_of_n_method = self.config.best_of_n_method
        if tester.best_of_n > 1:
            tester.best_of_n_selector = BestOfNSelector(
                n_attempts=tester.best_of_n,
                selection_method=tester.best_of_n_method,
            )

        scoped_sample = copy.copy(sample)
        scoped_sample.qa = [qa for qa in sample.qa if qa.category in set(self.config.categories)]

        if self.config.parallel:
            results = tester.test_questions_parallel(
                scoped_sample,
                max_questions=self.config.max_questions,
                n_workers=self.config.n_workers,
            )
        else:
            results = tester.test_questions(
                scoped_sample,
                max_questions=self.config.max_questions,
            )

        return {
            "method": method,
            "stats": summarize_results(results, evaluator),
            "results": results,
        }

    def build_report(self) -> dict:
        samples = self.load_samples()
        report = {
            "timestamp": datetime.now().isoformat(),
            "framework": "gam",
            "dataset": self.config.dataset,
            "samples": self.config.sample,
            "methods": self.config.methods,
            "config": self.config.export(),
            "runs": [],
        }

        for sample_id in self.config.sample:
            sample = samples[sample_id]
            builder = self.create_builder(sample_id)
            self.ensure_memory(builder, sample)

            evaluator = Evaluator(
                llm_controller=builder.llm_controller,
                use_llm_judge=builder.llm_controller is not None,
            )

            report["runs"].append(
                {
                    "sample_id": sample_id,
                    "memory_stats": builder.trg.get_statistics(),
                    "method_runs": [
                        self.run_method(method, sample, builder, evaluator)
                        for method in self.config.methods
                    ],
                }
            )

        return report
