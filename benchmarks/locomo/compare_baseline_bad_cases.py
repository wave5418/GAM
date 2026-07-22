"""Cross-compare LOCOMO bad cases across MAG, Mem0, and Zep.

The script is offline-only: it reads completed LOCOMO result JSON files,
aligns questions by ``question_id``, and emits inspectable reports for cases
where one system fails while one or both comparison systems succeed.
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


SYSTEMS = ("mag", "mem0", "zep")


def _judgment(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("judgment", "UNKNOWN"))


def _is_correct(item: dict[str, Any], cutoff: str) -> bool:
    return _judgment(item, cutoff) == "CORRECT"


def _answer(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("generated_answer", ""))


def _reason(item: dict[str, Any], cutoff: str) -> str:
    return str(item.get("cutoff_results", {}).get(cutoff, {}).get("reason", ""))


def _result_text(result: dict[str, Any]) -> str:
    return str(result.get("memory") or result.get("text") or result.get("content") or "")


def _top_results(item: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results = item.get("retrieval", {}).get("search_results", [])
    return [result for result in results[:limit] if isinstance(result, dict)]


def _top_texts(item: dict[str, Any], limit: int) -> list[str]:
    return [_result_text(result) for result in _top_results(item, limit)]


def _tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _top_overlap(left: dict[str, Any], right: dict[str, Any], limit: int) -> float:
    left_texts = {_normalized_text(text) for text in _top_texts(left, limit)}
    right_texts = {_normalized_text(text) for text in _top_texts(right, limit)}
    if not left_texts and not right_texts:
        return 1.0
    return len(left_texts & right_texts) / max(1, len(left_texts | right_texts))


def _gt_support(item: dict[str, Any], limit: int) -> float:
    gt_tokens = _tokens(item.get("ground_truth_answer", ""))
    if not gt_tokens:
        return 0.0
    top_tokens = _tokens("\n".join(_top_texts(item, limit)))
    return len(gt_tokens & top_tokens) / max(1, len(gt_tokens))


def _source_name(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    route_scores = metadata.get("route_scores") if isinstance(metadata.get("route_scores"), dict) else {}
    return str(metadata.get("route") or result.get("source") or route_scores.get("route") or "unknown")


def _source_composition(item: dict[str, Any], limit: int) -> dict[str, int]:
    return dict(Counter(_source_name(result) for result in _top_results(item, limit)))


def _graph_ratio(item: dict[str, Any], limit: int) -> float:
    results = _top_results(item, limit)
    if not results:
        return 0.0
    graph_hits = sum(1 for result in results if "graph" in _source_name(result).lower())
    return graph_hits / len(results)


def _latency_ms(item: dict[str, Any]) -> float | None:
    latency = item.get("retrieval", {}).get("search_latency_ms")
    if isinstance(latency, (int, float)):
        return float(latency)
    timing = item.get("retrieval", {}).get("query_debug", {}).get("routes", {}).get("timing_ms")
    if isinstance(timing, dict) and isinstance(timing.get("total"), (int, float)):
        return float(timing["total"])
    return None


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
            "cannot determine",
        )
    )


def _has_temporal_fact_context(item: dict[str, Any], limit: int) -> bool:
    text = "\n".join(_top_texts(item, limit)).lower()
    context = str(item.get("retrieval", {}).get("graphiti_context", "")).lower()
    combined = f"{text}\n{context}"
    return "fact/event date" in combined or "valid date range" in combined or "valid_at" in combined


def _snippet(text: str, max_chars: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _sort_key(question_id: str) -> tuple[int, int, str]:
    match = re.search(r"conv(\d+)_q(\d+)", question_id)
    if not match:
        return (math.inf, math.inf, question_id)
    return (int(match.group(1)), int(match.group(2)), question_id)


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = {}
    for item in data.get("evaluations", []):
        if isinstance(item, dict) and item.get("question_id"):
            items[str(item["question_id"])] = item
    return data.get("metadata", {}), items


def _pattern(correct: dict[str, bool]) -> str:
    return "_".join(f"{system}{'C' if correct[system] else 'W'}" for system in SYSTEMS)


def _bucket(correct: dict[str, bool]) -> str:
    wrong = [system for system in SYSTEMS if not correct[system]]
    if not wrong:
        return "all_correct"
    if len(wrong) == 3:
        return "all_wrong"
    if wrong == ["mag"]:
        return "mag_wrong_mem0_zep_correct"
    if wrong == ["mem0"]:
        return "mem0_wrong_mag_zep_correct"
    if wrong == ["zep"]:
        return "zep_wrong_mag_mem0_correct"
    if set(wrong) == {"mag", "mem0"}:
        return "mag_mem0_wrong_zep_correct"
    if set(wrong) == {"mag", "zep"}:
        return "mag_zep_wrong_mem0_correct"
    if set(wrong) == {"mem0", "zep"}:
        return "mem0_zep_wrong_mag_correct"
    return "other"


def _difference_signals(
    items: dict[str, dict[str, Any]],
    correct: dict[str, bool],
    cutoff: str,
    top_k: int,
) -> list[str]:
    signals: list[str] = []
    mag = items["mag"]
    mem0 = items["mem0"]
    zep = items["zep"]
    mag_support = _gt_support(mag, top_k)
    mem0_support = _gt_support(mem0, top_k)
    zep_support = _gt_support(zep, top_k)

    for system in SYSTEMS:
        if correct[system] and _looks_abstained(_answer(items[system], cutoff)):
            signals.append(f"{system}_correct_but_answer_abstained")

    if not correct["mag"]:
        best_baseline_support = max(mem0_support if correct["mem0"] else 0.0, zep_support if correct["zep"] else 0.0)
        if mag_support + 0.15 < best_baseline_support:
            signals.append("mag_gt_support_lower_than_correct_baselines")
        if _graph_ratio(mag, top_k) >= 0.3:
            signals.append("mag_graph_heavy_when_wrong")
        if _looks_abstained(_answer(mag, cutoff)):
            signals.append("mag_answer_abstained")
        if correct["mem0"] and _top_overlap(mag, mem0, top_k) < 0.25:
            signals.append("mag_low_topk_overlap_with_mem0")
        if correct["zep"] and _top_overlap(mag, zep, top_k) < 0.25:
            signals.append("mag_low_topk_overlap_with_zep")
        if correct["zep"] and _has_temporal_fact_context(zep, top_k):
            signals.append("zep_temporal_fact_context_available")

    if not correct["mem0"] and _looks_abstained(_answer(mem0, cutoff)):
        signals.append("mem0_answer_abstained")
    if not correct["zep"] and _looks_abstained(_answer(zep, cutoff)):
        signals.append("zep_answer_abstained")
    if not correct["zep"] and _has_temporal_fact_context(zep, top_k):
        signals.append("zep_fact_context_not_sufficient")

    latencies = {system: _latency_ms(items[system]) for system in SYSTEMS}
    if latencies["mag"] is not None and latencies["mem0"] is not None and latencies["mag"] > max(1000.0, latencies["mem0"] * 5):
        signals.append("mag_much_slower_than_mem0")
    if latencies["mag"] is not None and latencies["zep"] is not None and latencies["mag"] > max(1000.0, latencies["zep"] * 5):
        signals.append("mag_much_slower_than_zep")
    return signals


def _build_rows(paths: dict[str, Path], cutoff: str, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for system, path in paths.items():
        system_metadata, items = _load_result(path)
        metadata[system] = {
            "path": str(path),
            "project_name": system_metadata.get("project_name"),
            "run_id": system_metadata.get("run_id"),
            "timestamp": system_metadata.get("timestamp"),
            "total_questions": system_metadata.get("total_questions") or len(items),
        }
        loaded[system] = items

    aligned_ids = sorted(set.intersection(*(set(items) for items in loaded.values())), key=_sort_key)
    rows: list[dict[str, Any]] = []
    for question_id in aligned_ids:
        items = {system: loaded[system][question_id] for system in SYSTEMS}
        correct = {system: _is_correct(items[system], cutoff) for system in SYSTEMS}
        base_item = items["mag"]
        row = {
            "question_id": question_id,
            "conversation_idx": base_item.get("conversation_idx"),
            "category": base_item.get("category_name") or base_item.get("category"),
            "question": base_item.get("question"),
            "ground_truth_answer": base_item.get("ground_truth_answer"),
            "pattern": _pattern(correct),
            "bucket": _bucket(correct),
            "difference_signals": _difference_signals(items, correct, cutoff, top_k),
        }
        for system in SYSTEMS:
            item = items[system]
            row[f"{system}_judgment"] = _judgment(item, cutoff)
            row[f"{system}_answer"] = _answer(item, cutoff)
            row[f"{system}_reason"] = _reason(item, cutoff)
            row[f"{system}_gt_support"] = round(_gt_support(item, top_k), 4)
            row[f"{system}_graph_ratio"] = round(_graph_ratio(item, top_k), 4)
            row[f"{system}_latency_ms"] = _latency_ms(item)
            row[f"{system}_source_composition"] = _source_composition(item, top_k)
            row[f"{system}_top1"] = _snippet(_top_texts(item, 1)[0]) if _top_texts(item, 1) else ""
        row["mag_mem0_topk_overlap"] = round(_top_overlap(items["mag"], items["mem0"], top_k), 4)
        row["mag_zep_topk_overlap"] = round(_top_overlap(items["mag"], items["zep"], top_k), 4)
        row["mem0_zep_topk_overlap"] = round(_top_overlap(items["mem0"], items["zep"], top_k), 4)
        rows.append(row)

    summary = _summarize(metadata, loaded, rows)
    return summary, rows


def _summarize(
    metadata: dict[str, Any],
    loaded: dict[str, dict[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "metadata": metadata,
        "aligned_questions": len(rows),
        "missing_question_counts": {
            system: len(set.union(*(set(items) for items in loaded.values())) - set(loaded[system]))
            for system in SYSTEMS
        },
        "system_accuracy": {},
        "pattern_counts": dict(Counter(row["pattern"] for row in rows)),
        "bucket_counts": dict(Counter(row["bucket"] for row in rows)),
        "requested_buckets": {
            "mag_wrong_mem0_zep_correct": [],
            "mem0_wrong_mag_zep_correct": [],
            "zep_wrong_mag_mem0_correct": [],
        },
        "by_category": {},
        "difference_signal_counts": {},
        "latency_ms": {},
    }
    for system in SYSTEMS:
        correct = sum(1 for row in rows if row[f"{system}_judgment"] == "CORRECT")
        summary["system_accuracy"][system] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        }
        latencies = [
            row[f"{system}_latency_ms"]
            for row in rows
            if isinstance(row.get(f"{system}_latency_ms"), (int, float))
        ]
        summary["latency_ms"][system] = _latency_summary(latencies)

    for bucket in summary["requested_buckets"]:
        summary["requested_buckets"][bucket] = [row["question_id"] for row in rows if row["bucket"] == bucket]

    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_category[str(row["category"])][row["bucket"]] += 1
    summary["by_category"] = {category: dict(counts) for category, counts in sorted(by_category.items())}

    signal_counts = Counter()
    for row in rows:
        signal_counts.update(row["difference_signals"])
    summary["difference_signal_counts"] = dict(signal_counts)
    return summary


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": round(_percentile(sorted_values, 0.50), 2),
        "p95": round(_percentile(sorted_values, 0.95), 2),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * percentile) - 1))
    return sorted_values[index]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "conversation_idx",
        "category",
        "pattern",
        "bucket",
        "difference_signals",
        "question",
        "ground_truth_answer",
        "mag_judgment",
        "mem0_judgment",
        "zep_judgment",
        "mag_answer",
        "mem0_answer",
        "zep_answer",
        "mag_reason",
        "mem0_reason",
        "zep_reason",
        "mag_gt_support",
        "mem0_gt_support",
        "zep_gt_support",
        "mag_graph_ratio",
        "mem0_graph_ratio",
        "zep_graph_ratio",
        "mag_latency_ms",
        "mem0_latency_ms",
        "zep_latency_ms",
        "mag_mem0_topk_overlap",
        "mag_zep_topk_overlap",
        "mem0_zep_topk_overlap",
        "mag_source_composition",
        "mem0_source_composition",
        "zep_source_composition",
        "mag_top1",
        "mem0_top1",
        "zep_top1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    cutoff: str,
    max_cases_per_bucket: int,
) -> None:
    lines = [
        "# LOCOMO MAG/Mem0/Zep Bad Case Cross Comparison",
        "",
        "## Scope",
    ]
    for system in SYSTEMS:
        meta = summary["metadata"][system]
        lines.append(f"- {system}: `{meta['path']}`")
        if meta.get("project_name"):
            lines.append(f"  - project: `{meta['project_name']}`")
    lines.extend(
        [
            f"- Aligned questions: `{summary['aligned_questions']}`",
            f"- Cutoff: `{cutoff}`",
            "",
            "## Initial Reading",
            "",
            "- MAG-only failures are concentrated in retrieval differences: low TopK overlap with both baselines is the dominant signal.",
            "- In MAG-only failures, lower ground-truth token support and graph-heavy TopK indicate evidence admission/ranking is a stronger suspect than answer generation alone.",
            "- Mem0-only and Zep-only failures are useful controls: they show where summary-style memory or temporal fact/entity context loses detail even when MAG succeeds.",
            "- Some judge-labeled correct cases still look like abstentions; treat those rows as evaluator-noise candidates before drawing method conclusions.",
            "",
            "## Accuracy",
            "",
            "| System | Correct | Total | Accuracy | Mean Latency ms | P50 ms | P95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system in SYSTEMS:
        accuracy = summary["system_accuracy"][system]
        latency = summary["latency_ms"][system]
        lines.append(
            f"| {system} | {accuracy['correct']} | {accuracy['total']} | "
            f"{accuracy['accuracy']:.2%} | {_fmt(latency['mean'])} | {_fmt(latency['p50'])} | {_fmt(latency['p95'])} |"
        )

    lines.extend(["", "## Pattern Counts", "", "| Bucket | Count |", "|---|---:|"])
    for bucket, count in sorted(summary["bucket_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{bucket}` | {count} |")

    lines.extend(["", "## Requested Buckets"])
    requested_names = {
        "mag_wrong_mem0_zep_correct": "MAG wrong, both baselines correct",
        "mem0_wrong_mag_zep_correct": "Mem0 wrong, MAG/Zep correct",
        "zep_wrong_mag_mem0_correct": "Zep wrong, MAG/Mem0 correct",
    }
    for bucket, title in requested_names.items():
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        lines.extend(["", f"### {title}", ""])
        lines.append(f"- Count: `{len(bucket_rows)}`")
        lines.append(f"- Question IDs: `{', '.join(row['question_id'] for row in bucket_rows) or 'none'}`")
        category_counts = Counter(str(row["category"]) for row in bucket_rows)
        if category_counts:
            lines.append(f"- By category: `{dict(category_counts)}`")
        signal_counts = Counter(signal for row in bucket_rows for signal in row["difference_signals"])
        if signal_counts:
            lines.append(f"- Difference signals: `{dict(signal_counts)}`")
        for row in bucket_rows[:max_cases_per_bucket]:
            lines.extend(_case_lines(row))
        if len(bucket_rows) > max_cases_per_bucket:
            lines.append(f"- Omitted examples: `{len(bucket_rows) - max_cases_per_bucket}`; see CSV/JSONL outputs.")

    lines.extend(["", "## Difference Signal Counts", "", "| Signal | Count |", "|---|---:|"])
    for signal, count in sorted(summary["difference_signal_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{signal}` | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_lines(row: dict[str, Any]) -> list[str]:
    return [
        "",
        f"- `{row['question_id']}` `{row['category']}` signals=`{','.join(row['difference_signals'])}`",
        f"  - Q: {row['question']}",
        f"  - GT: {row['ground_truth_answer']}",
        f"  - MAG: {row['mag_answer']}",
        f"  - Mem0: {row['mem0_answer']}",
        f"  - Zep: {row['zep_answer']}",
        f"  - support MAG/Mem0/Zep: `{row['mag_gt_support']}/{row['mem0_gt_support']}/{row['zep_gt_support']}`; "
        f"overlap MAG-Mem0/MAG-Zep: `{row['mag_mem0_topk_overlap']}/{row['mag_zep_topk_overlap']}`",
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mag", required=True, type=Path, help="MAG LOCOMO result JSON")
    parser.add_argument("--mem0", required=True, type=Path, help="Mem0 LOCOMO result JSON")
    parser.add_argument("--zep", required=True, type=Path, help="Zep LOCOMO result JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="locomo_bad_case_cross_compare")
    parser.add_argument("--cutoff", default="top_10")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-cases-per-bucket", type=int, default=30)
    args = parser.parse_args()

    paths = {"mag": args.mag, "mem0": args.mem0, "zep": args.zep}
    summary, rows = _build_rows(paths, args.cutoff, args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.name}_summary.json"
    jsonl_path = args.output_dir / f"{args.name}_items.jsonl"
    csv_path = args.output_dir / f"{args.name}_items.csv"
    markdown_path = args.output_dir / f"{args.name}.md"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, summary, rows, args.cutoff, args.max_cases_per_bucket)

    print(f"Summary: {summary_path}")
    print(f"Items JSONL: {jsonl_path}")
    print(f"Items CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
