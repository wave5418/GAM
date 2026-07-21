"""Compare two LOCOMO result files and label retrieval/judgment regressions.

The script is intentionally heuristic and offline-only: it does not call an LLM
or mutate benchmark outputs. It focuses on making bad-case diffs inspectable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _judgment(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("judgment", "UNKNOWN"))


def _answer(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("generated_answer", ""))


def _reason(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("reason", ""))


def _tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _result_text(result: dict[str, Any]) -> str:
    return str(result.get("memory") or result.get("text") or result.get("content") or "")


def _top_results(item: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results = item.get("retrieval", {}).get("search_results", [])
    return [result for result in results[:limit] if isinstance(result, dict)]


def _top_texts(item: dict[str, Any], limit: int) -> list[str]:
    return [_result_text(result) for result in _top_results(item, limit)]


def _route_name(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    route_scores = metadata.get("route_scores") if isinstance(metadata.get("route_scores"), dict) else {}
    route = metadata.get("route") or result.get("source") or route_scores.get("route") or "unknown"
    return str(route)


def _route_composition(item: dict[str, Any], limit: int) -> Counter[str]:
    return Counter(_route_name(result) for result in _top_results(item, limit))


def _graph_ratio(item: dict[str, Any], limit: int) -> float:
    results = _top_results(item, limit)
    if not results:
        return 0.0
    graph_hits = sum(1 for result in results if "graph_bfs" in _route_name(result))
    return graph_hits / len(results)


def _score_stats(item: dict[str, Any], limit: int) -> dict[str, float]:
    scores = [
        float(result["score"])
        for result in _top_results(item, limit)
        if isinstance(result.get("score"), (int, float))
    ]
    if not scores:
        return {"avg": 0.0, "max": 0.0}
    return {"avg": sum(scores) / len(scores), "max": max(scores)}


def _top_overlap(left: dict[str, Any], right: dict[str, Any], limit: int) -> float:
    left_texts = {re.sub(r"\s+", " ", text.strip().lower()) for text in _top_texts(left, limit)}
    right_texts = {re.sub(r"\s+", " ", text.strip().lower()) for text in _top_texts(right, limit)}
    if not left_texts and not right_texts:
        return 1.0
    return len(left_texts & right_texts) / max(1, len(left_texts | right_texts))


def _gt_support(item: dict[str, Any], limit: int) -> float:
    gt_tokens = _tokens(item.get("ground_truth_answer", ""))
    if not gt_tokens:
        return 0.0
    top_tokens = _tokens("\n".join(_top_texts(item, limit)))
    return len(gt_tokens & top_tokens) / max(1, len(gt_tokens))


def _timing(item: dict[str, Any]) -> dict[str, float]:
    query_debug = item.get("retrieval", {}).get("query_debug")
    if not isinstance(query_debug, dict):
        return {}
    routes = query_debug.get("routes")
    if not isinstance(routes, dict):
        return {}
    timing = routes.get("timing_ms")
    if not isinstance(timing, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in timing.items()
        if isinstance(value, (int, float))
    }


def _timing_bottleneck(timing: dict[str, float]) -> str:
    staged = {key: value for key, value in timing.items() if key != "total"}
    if not staged:
        return ""
    return max(staged.items(), key=lambda item: item[1])[0]


def _transition(old_item: dict[str, Any], new_item: dict[str, Any], cutoff: str) -> str:
    old_judgment = _judgment(old_item, cutoff)
    new_judgment = _judgment(new_item, cutoff)
    if old_judgment == "CORRECT" and new_judgment == "CORRECT":
        return "both_correct"
    if old_judgment != "CORRECT" and new_judgment != "CORRECT":
        return "both_wrong"
    if old_judgment == "CORRECT":
        return "lost_correct_to_wrong"
    return "gained_wrong_to_correct"


def _looks_abstained(answer: str) -> bool:
    normalized = answer.lower()
    return any(
        marker in normalized
        for marker in (
            "not specify",
            "not specified",
            "not mentioned",
            "no information",
            "not available",
            "do not know",
            "don't know",
        )
    )


def _symptoms(old_item: dict[str, Any], new_item: dict[str, Any], cutoff: str, limit: int) -> list[str]:
    symptoms: list[str] = []
    overlap = _top_overlap(old_item, new_item, limit)
    old_support = _gt_support(old_item, limit)
    new_support = _gt_support(new_item, limit)
    old_graph = _graph_ratio(old_item, limit)
    new_graph = _graph_ratio(new_item, limit)
    category = str(new_item.get("category_name") or new_item.get("category") or "").lower()

    if overlap < 0.35:
        symptoms.append("topk_changed")
    if new_support + 0.15 < old_support:
        symptoms.append("gt_support_dropped")
    if new_graph > old_graph + 0.2:
        symptoms.append("graph_ratio_increased")
    if "temporal" in category:
        symptoms.append("temporal_question")
    if _looks_abstained(_answer(new_item, cutoff)):
        symptoms.append("new_answer_abstained")
    if _looks_abstained(_answer(old_item, cutoff)):
        symptoms.append("old_answer_abstained")

    new_timing = _timing(new_item)
    if new_timing.get("total", 0.0) > 3000:
        symptoms.append("slow_search")
    return symptoms


def _primary_issue(symptoms: list[str], transition: str) -> str:
    symptom_set = set(symptoms)
    if "slow_search" in symptom_set:
        return "latency"
    if transition == "lost_correct_to_wrong":
        if "gt_support_dropped" in symptom_set or "topk_changed" in symptom_set:
            if "graph_ratio_increased" in symptom_set:
                return "bfs_noise_or_graph_reroute"
            if "temporal_question" in symptom_set:
                return "temporal_retrieval_shift"
            return "retrieval_shift"
        if "new_answer_abstained" in symptom_set:
            return "generation_abstention"
    if transition == "both_wrong":
        if "gt_support_dropped" in symptom_set:
            return "missing_or_buried_evidence"
        if "graph_ratio_increased" in symptom_set:
            return "bfs_noise"
    if transition == "gained_wrong_to_correct":
        return "fixed"
    if transition == "both_correct":
        return "stable"
    return "judge_or_generation_variance"


def _item_key(item: dict[str, Any]) -> str:
    return str(item.get("question_id", ""))


def _sort_key(question_id: str) -> tuple[int, int, str]:
    match = re.search(r"conv(\d+)_q(\d+)", question_id)
    if not match:
        return (math.inf, math.inf, question_id)
    return (int(match.group(1)), int(match.group(2)), question_id)


def _load_items(path: Path, conversations: set[int] | None) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    items = data.get("evaluations", [])
    output = {}
    for item in items:
        if conversations is not None and item.get("conversation_idx") not in conversations:
            continue
        qid = _item_key(item)
        if qid:
            output[qid] = item
    return output


def _diagnose(
    old_items: dict[str, dict[str, Any]],
    new_items: dict[str, dict[str, Any]],
    cutoff: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(set(old_items) & set(new_items), key=_sort_key)
    for qid in keys:
        old_item = old_items[qid]
        new_item = new_items[qid]
        transition = _transition(old_item, new_item, cutoff)
        symptoms = _symptoms(old_item, new_item, cutoff, top_k)
        old_timing = _timing(old_item)
        new_timing = _timing(new_item)
        row = {
            "question_id": qid,
            "conversation_idx": new_item.get("conversation_idx"),
            "category": new_item.get("category_name") or new_item.get("category"),
            "transition": transition,
            "primary_issue": _primary_issue(symptoms, transition),
            "symptoms": symptoms,
            "question": new_item.get("question"),
            "ground_truth_answer": new_item.get("ground_truth_answer"),
            "old_judgment": _judgment(old_item, cutoff),
            "new_judgment": _judgment(new_item, cutoff),
            "old_answer": _answer(old_item, cutoff),
            "new_answer": _answer(new_item, cutoff),
            "old_reason": _reason(old_item, cutoff),
            "new_reason": _reason(new_item, cutoff),
            "topk_text_overlap": round(_top_overlap(old_item, new_item, top_k), 4),
            "old_gt_support": round(_gt_support(old_item, top_k), 4),
            "new_gt_support": round(_gt_support(new_item, top_k), 4),
            "old_graph_ratio": round(_graph_ratio(old_item, top_k), 4),
            "new_graph_ratio": round(_graph_ratio(new_item, top_k), 4),
            "old_route_composition": dict(_route_composition(old_item, top_k)),
            "new_route_composition": dict(_route_composition(new_item, top_k)),
            "old_score_avg": round(_score_stats(old_item, top_k)["avg"], 4),
            "new_score_avg": round(_score_stats(new_item, top_k)["avg"], 4),
            "new_search_total_ms": new_timing.get("total"),
            "new_search_bottleneck": _timing_bottleneck(new_timing),
            "old_search_total_ms": old_timing.get("total"),
            "old_search_bottleneck": _timing_bottleneck(old_timing),
        }
        rows.append(row)

    summary = {
        "aligned_questions": len(keys),
        "old_only": len(set(old_items) - set(new_items)),
        "new_only": len(set(new_items) - set(old_items)),
        "transitions": dict(Counter(row["transition"] for row in rows)),
        "primary_issues": dict(Counter(row["primary_issue"] for row in rows)),
        "by_conversation": {},
        "by_category": {},
    }
    for field in ("conversation_idx", "category"):
        bucket = defaultdict(Counter)
        for row in rows:
            bucket[str(row[field])][row["transition"]] += 1
        target = "by_conversation" if field == "conversation_idx" else "by_category"
        summary[target] = {key: dict(value) for key, value in sorted(bucket.items())}
    return rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "conversation_idx",
        "category",
        "transition",
        "primary_issue",
        "symptoms",
        "topk_text_overlap",
        "old_gt_support",
        "new_gt_support",
        "old_graph_ratio",
        "new_graph_ratio",
        "old_score_avg",
        "new_score_avg",
        "new_search_total_ms",
        "new_search_bottleneck",
        "old_search_total_ms",
        "old_search_bottleneck",
        "question",
        "ground_truth_answer",
        "old_answer",
        "new_answer",
        "old_reason",
        "new_reason",
        "old_route_composition",
        "new_route_composition",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list))
                else row.get(key)
                for key in fieldnames
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path, help="Baseline LOCOMO result JSON")
    parser.add_argument("--new", required=True, type=Path, help="New LOCOMO result JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="bad_case_diff")
    parser.add_argument("--cutoff", default="top_10")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--conversations", default="", help="Comma-separated conversation ids")
    args = parser.parse_args()

    conversations = None
    if args.conversations.strip():
        conversations = {int(part) for part in args.conversations.split(",") if part.strip()}

    old_items = _load_items(args.old, conversations)
    new_items = _load_items(args.new, conversations)
    rows, summary = _diagnose(old_items, new_items, args.cutoff, args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.name}_summary.json"
    jsonl_path = args.output_dir / f"{args.name}_items.jsonl"
    csv_path = args.output_dir / f"{args.name}_items.csv"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)

    print(f"Summary: {summary_path}")
    print(f"Items JSONL: {jsonl_path}")
    print(f"Items CSV: {csv_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
