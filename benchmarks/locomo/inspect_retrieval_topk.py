"""Inspect LOCOMO retrieval evidence coverage for saved predict-only outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _result_text(result: dict[str, Any]) -> str:
    parts = [str(result.get("memory") or result.get("text") or result.get("content") or "")]
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    for key in ("supporting_context", "supporting_graph_context"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item.get("memory", item)) for item in value)
    return "\n".join(parts)


def _route(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    route_scores = metadata.get("route_scores") if isinstance(metadata.get("route_scores"), dict) else {}
    return str(metadata.get("route") or result.get("source") or route_scores.get("route") or "unknown")


def _load_evidence_lookup(dataset_path: Path) -> dict[tuple[int, str], list[str]]:
    data = json.loads(dataset_path.read_text())
    lookup = {}
    for conv_idx, conv in enumerate(data):
        conversation = conv["conversation"]
        session_dates = {}
        for key in conversation:
            if key.startswith("session_") and key.endswith("_date_time"):
                session_num = key.replace("session_", "").replace("_date_time", "")
                session_dates[session_num] = conversation[key]
        for key, turns in conversation.items():
            if not key.startswith("session_") or key.endswith("date_time") or not isinstance(turns, list):
                continue
            for turn in turns:
                dia_id = turn.get("dia_id", "")
                if not dia_id:
                    continue
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                query = turn.get("query") or turn.get("img_query")
                blip = turn.get("blip_caption") or turn.get("blip")
                variants = [text] if text else []
                if query:
                    variants.append(f"Sharing image - query: {query}")
                if blip:
                    variants.append(f"The image shows: {blip}")
                merged_text = " ".join(variants)
                dia_match = re.match(r"D(\d+):", dia_id)
                date_suffix = ""
                if dia_match:
                    session_date = session_dates.get(dia_match.group(1), "")
                    if session_date:
                        date_suffix = f", said on {session_date}"
                lookup[(conv_idx, dia_id)] = [
                    f'[{dia_id}{date_suffix}] {speaker}: "{merged_text}"',
                    *variants,
                ]
    return lookup


def _rank_evidence(results: list[dict[str, Any]], evidence_variants: list[str]) -> dict[str, Any]:
    best_rank = None
    best_recall = 0.0
    best_result: dict[str, Any] | None = None
    best_variant = ""
    for idx, result in enumerate(results, start=1):
        result_tokens = _tokens(_result_text(result))
        for variant in evidence_variants:
            evidence_tokens = _tokens(variant)
            if not evidence_tokens:
                continue
            recall = len(evidence_tokens & result_tokens) / len(evidence_tokens)
            if recall > best_recall:
                best_recall = recall
                best_rank = idx
                best_result = result
                best_variant = variant
    hit = best_recall >= 0.55
    if not hit:
        best_rank = None
    matched = best_result or {}
    return {
        "rank": best_rank,
        "recall": round(best_recall, 4),
        "source": _route(matched) if matched else "",
        "score": matched.get("score") if matched else None,
        "memory": str(matched.get("memory", ""))[:300] if matched else "",
        "matched_variant": best_variant[:200],
    }


def _graph_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    edges = data.get("edges", [])
    sentence_ids = set()
    multi_edges = Counter()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        multi_edges[(edge.get("u"), edge.get("v"))] += 1
        edge_data = edge.get("data") if isinstance(edge.get("data"), dict) else {}
        for sid in edge_data.get("source_sentence_ids", []) or []:
            sentence_ids.add(sid)
    return {
        "entities": len(data.get("entities", {})),
        "edges": len(edges),
        "edge_pairs_with_multiplicity": sum(1 for count in multi_edges.values() if count > 1),
        "source_sentence_ids": len(sentence_ids),
    }


def inspect(result_dir: Path, dataset_path: Path, graph_path: Path | None, output_dir: Path) -> dict[str, Any]:
    evidence_lookup = _load_evidence_lookup(dataset_path)
    rows: list[dict[str, Any]] = []
    summary = {
        "questions": 0,
        "all_evidence_in_top10": 0,
        "all_evidence_in_top20": 0,
        "all_evidence_in_top50": 0,
        "all_evidence_in_top200": 0,
        "graph": _graph_stats(graph_path),
    }

    for path in sorted(result_dir.glob("conv*_q*.json")):
        item = json.loads(path.read_text())
        results = item.get("retrieval", {}).get("search_results", [])
        if not isinstance(results, list):
            results = []
        summary["questions"] += 1

        evidence_ranks = []
        for ref in item.get("evidence", []) or []:
            variants = evidence_lookup.get((item["conversation_idx"], ref), [ref])
            match = _rank_evidence(results, variants)
            evidence_ranks.append({"ref": ref, **match})

        ranked = [entry["rank"] for entry in evidence_ranks if entry["rank"] is not None]
        for cutoff in (10, 20, 50, 200):
            if evidence_ranks and len(ranked) == len(evidence_ranks) and max(ranked) <= cutoff:
                summary[f"all_evidence_in_top{cutoff}"] += 1

        top10 = results[:10]
        route_counts = Counter(_route(result) for result in top10)
        query_debug = item.get("retrieval", {}).get("query_debug", {})
        timing = query_debug.get("routes", {}).get("timing_ms", {}) if isinstance(query_debug, dict) else {}
        rows.append(
            {
                "question_id": item.get("question_id"),
                "category": item.get("category_name") or item.get("category"),
                "question": item.get("question"),
                "ground_truth_answer": item.get("ground_truth_answer"),
                "evidence_ranks": evidence_ranks,
                "top10_graph_hits": sum(count for route, count in route_counts.items() if "graph_bfs" in route),
                "top10_routes": dict(route_counts),
                "search_latency_ms": item.get("retrieval", {}).get("search_latency_ms"),
                "timing_ms": timing,
                "top5": [
                    {
                        "rank": idx,
                        "source": result.get("source", ""),
                        "score": result.get("score", 0),
                        "memory": str(result.get("memory", ""))[:300],
                    }
                    for idx, result in enumerate(results[:5], start=1)
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "topk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / "topk_items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with (output_dir / "topk_items.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "category",
                "question",
                "ground_truth_answer",
                "evidence_ranks",
                "top10_graph_hits",
                "search_latency_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return {"summary": summary, "items": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--dataset-path", default=Path("datasets/locomo/locomo10.json"), type=Path)
    parser.add_argument("--graph-path", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.result_dir, args.dataset_path, args.graph_path, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
