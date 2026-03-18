from __future__ import annotations

from typing import Dict, List


def summarize_results(results: List[Dict], evaluator) -> Dict:
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
        sum(item.get("processing_time", 0.0) for item in results) / len(results) if results else 0.0
    )
    aggregate["questions"] = len(results)
    return {
        "overall": aggregate,
        "by_category": by_category,
    }
