#!/usr/bin/env python3
"""Summarize LOCOMO answer failures by likely root cause.

This is intentionally heuristic: it uses the saved benchmark JSON only, so it
can run after any LOCOMO experiment without re-querying models or stores.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "her", "his",
    "their", "did", "when", "what", "where", "who", "had", "has", "have",
    "was", "were", "will", "would", "could", "about", "before", "after",
    "week", "month", "year", "years", "date", "time",
}

TEMPORAL_WORDS = {
    "date", "time", "year", "month", "day", "temporal", "week", "before",
    "after", "yesterday", "tomorrow", "friday", "saturday", "sunday",
    "monday", "tuesday", "wednesday", "thursday",
}


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def content_tokens(text: Any) -> list[str]:
    return [
        token
        for token in normalize(text).split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def cutoff_result(evaluation: dict[str, Any], cutoff: str) -> dict[str, Any]:
    cutoffs = evaluation.get("cutoff_results")
    if isinstance(cutoffs, dict) and isinstance(cutoffs.get(cutoff), dict):
        return cutoffs[cutoff]
    for value in evaluation.values():
        if isinstance(value, dict) and "score" in value:
            return value
    return {}


def classify_failure(evaluation: dict[str, Any], result: dict[str, Any], cutoff_k: int) -> tuple[str, list[str]]:
    gt_tokens = content_tokens(evaluation.get("ground_truth_answer", ""))
    retrieved = evaluation.get("retrieval", {}).get("search_results", [])
    retrieved_text = normalize("\n".join(r.get("memory", "") for r in retrieved[:cutoff_k]))
    token_hits = sum(1 for token in gt_tokens if token in retrieved_text)
    hit_ratio = token_hits / max(1, len(gt_tokens))

    reason = normalize(result.get("reason", ""))
    generated = normalize(result.get("generated_answer", ""))
    tags: list[str] = []
    if any(word in reason or word in generated for word in TEMPORAL_WORDS):
        tags.append("temporal_or_date_wrong")
    if any(phrase in reason for phrase in ("not include", "does not include", "missing", "no correct item")):
        tags.append("missing_required_item")
    if any(phrase in reason for phrase in ("not explicitly", "not enough", "do not specify", "cannot determine", "no information")):
        tags.append("abstained_or_insufficient_context")
    if any(phrase in reason for phrase in ("different", "incorrect", "not consistent", "not match", "contradict")):
        tags.append("wrong_or_conflicting_fact")

    if hit_ratio >= 0.8:
        root = "evidence_present_answer_failed"
    elif hit_ratio >= 0.3:
        root = "partial_evidence_or_distractor"
    else:
        root = "evidence_missing_or_low_recall"
    return root, tags


def build_report(path: Path, cutoff: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    evaluations = data.get("evaluations", [])
    cutoff_k = int(cutoff.split("_", 1)[1]) if cutoff.startswith("top_") else 10

    totals = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    tags = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for evaluation in evaluations:
        result = cutoff_result(evaluation, cutoff)
        score = float(result.get("score") or 0)
        category = evaluation.get("category_name") or str(evaluation.get("category", "unknown"))
        if score >= 0.5:
            totals["correct"] += 1
            by_category[category]["correct"] += 1
            continue

        root, root_tags = classify_failure(evaluation, result, cutoff_k)
        totals[root] += 1
        totals["wrong"] += 1
        by_category[category][root] += 1
        by_category[category]["wrong"] += 1
        tags.update(root_tags)

        if len(examples[root]) < 8:
            retrieved = evaluation.get("retrieval", {}).get("search_results", [])
            examples[root].append({
                "question_id": evaluation.get("question_id"),
                "category": category,
                "question": evaluation.get("question"),
                "ground_truth_answer": evaluation.get("ground_truth_answer"),
                "generated_answer": result.get("generated_answer"),
                "reason": result.get("reason"),
                "top_retrieval": [
                    {
                        "score": r.get("score"),
                        "memory": (r.get("memory") or "")[:240],
                    }
                    for r in retrieved[:3]
                ],
            })

    total = len(evaluations)
    return {
        "source": str(path),
        "cutoff": cutoff,
        "total": total,
        "accuracy": totals["correct"] / total if total else 0.0,
        "root_causes": dict(totals),
        "by_category": {category: dict(counter) for category, counter in sorted(by_category.items())},
        "reason_tags": dict(tags),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--cutoff", default="top_10")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.results_json, args.cutoff)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
