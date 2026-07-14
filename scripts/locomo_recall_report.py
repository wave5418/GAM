#!/usr/bin/env python3
"""Generate LOCOMO retrieval recall report at multiple top-k cutoffs.

Definition used here:
- Evidence recalled if retrieved context semantically matches the same event
  described by the evidence turn (paraphrase allowed).

Outputs:
1) locomo_recall_summary.json
2) locomo_recall_per_question.jsonl          (machine-friendly)
3) locomo_recall_per_question_readable.json  (human-readable)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "with", "you", "your",
}


class SemanticScorer:
    """Optional sentence-transformer scorer with cache and lexical fallback."""

    def __init__(self) -> None:
        self.model = None
        self.cache: dict[str, Any] = {}
        self.enabled = False
        self.model_name = ""
        try:
            from sentence_transformers import SentenceTransformer

            # If model is not available locally and cannot be downloaded,
            # we gracefully fall back to lexical heuristics.
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.enabled = True
        except Exception:
            self.model = None
            self.enabled = False
            self.model_name = ""

    def semantic_similarity(self, a: str, b: str) -> float | None:
        if not self.enabled or self.model is None:
            return None
        import numpy as np

        def emb(t: str):
            if t in self.cache:
                return self.cache[t]
            v = self.model.encode([t], normalize_embeddings=True)[0]
            self.cache[t] = v
            return v

        va = emb(a)
        vb = emb(b)
        return float(np.dot(va, vb))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_list(text: str) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return [t for t in norm.split() if len(t) > 1 and t not in STOPWORDS]


def token_set(text: str) -> set[str]:
    return set(token_list(text))


def strip_evidence_meta(evidence_text: str) -> str:
    # [D1:3, said on ...] Caroline: I went ...
    s = evidence_text
    s = re.sub(r"^\[[^\]]*\]\s*", "", s)
    s = re.sub(r"^[A-Za-z][A-Za-z\s\-']{0,40}:\s*", "", s)
    return s.strip()


def lexical_recall_score(evidence_content: str, memory_text: str) -> tuple[float, int]:
    ev_toks = token_set(evidence_content)
    mem_toks = token_set(memory_text)
    if not ev_toks or not mem_toks:
        return 0.0, 0
    inter = len(ev_toks & mem_toks)
    return inter / max(1, len(ev_toks)), inter


def is_semantic_event_match(
    evidence_text: str,
    memory_text: str,
    scorer: SemanticScorer,
) -> tuple[bool, dict[str, float | int | None]]:
    ev_content = strip_evidence_meta(evidence_text)
    lex_recall, inter = lexical_recall_score(ev_content, memory_text)
    sem = scorer.semantic_similarity(ev_content, memory_text)

    # Event-level semantic hit (paraphrase allowed):
    # 1) Strong semantic similarity, with at least light lexical anchor.
    # 2) Moderate semantic similarity + moderate lexical recall.
    # 3) Pure lexical high-recall fallback (for near-copy summaries).
    matched = False
    if sem is not None:
        if sem >= 0.58 and inter >= 1:
            matched = True
        elif sem >= 0.52 and lex_recall >= 0.35 and inter >= 2:
            matched = True
    if not matched:
        if lex_recall >= 0.60 and inter >= 3:
            matched = True
        elif lex_recall >= 0.50 and inter >= 4:
            matched = True

    return matched, {
        "semantic_similarity": round(sem, 4) if sem is not None else None,
        "lexical_recall": round(lex_recall, 4),
        "token_overlap": inter,
    }


def best_match_rank_for_evidence(
    evidence_text: str,
    retrieved: list[dict[str, Any]],
    scorer: SemanticScorer,
) -> tuple[int | None, dict[str, Any]]:
    """Find first rank that passes a calibrated semantic-event criterion.

    More grounded criterion:
    - Absolute semantic relevance (cosine) should be high enough.
    - And/or it should stand out from same-question candidate distribution
      (mean/std and top1-top2 margin), reducing accidental topical matches.
    """
    candidates: list[dict[str, Any]] = []
    for r in retrieved:
        matched, diag = is_semantic_event_match(evidence_text, r["memory"], scorer)
        sem = diag.get("semantic_similarity")
        sem = float(sem) if sem is not None else 0.0
        candidates.append(
            {
                "rank": r["rank"],
                "memory": r["memory"],
                "sem": sem,
                "lex": float(diag["lexical_recall"]),
                "overlap": int(diag["token_overlap"]),
                "base_match": bool(matched),
            }
        )

    if not candidates:
        return None, {
            "method": "calibrated_semantic_event_match",
            "best_rank": None,
            "best_semantic_similarity": 0.0,
            "mu": 0.0,
            "sigma": 0.0,
            "top1_top2_margin": 0.0,
            "rule_passed": None,
        }

    sems = sorted((c["sem"] for c in candidates), reverse=True)
    mu = sum(sems) / len(sems)
    var = sum((x - mu) ** 2 for x in sems) / len(sems)
    sigma = var ** 0.5
    top1 = sems[0]
    top2 = sems[1] if len(sems) > 1 else 0.0
    margin = top1 - top2
    zlike = (top1 - mu) / max(1e-6, sigma)

    # Rules are checked in rank order; first candidate passing any rule is hit.
    # R1: absolute semantic + basic lexical anchor.
    # R2: semantic stands out from in-query candidates (distribution-aware).
    # R3: near-copy lexical fallback.
    for c in candidates:
        r1 = c["sem"] >= 0.60 and c["overlap"] >= 1
        r2 = c["sem"] >= 0.54 and (c["sem"] >= mu + 0.9 * sigma or (margin >= 0.06 and zlike >= 0.8))
        r3 = c["lex"] >= 0.60 and c["overlap"] >= 3
        if r1 or r2 or r3:
            rule = "R1" if r1 else ("R2" if r2 else "R3")
            return c["rank"], {
                "method": "calibrated_semantic_event_match",
                "best_rank": c["rank"],
                "best_semantic_similarity": round(c["sem"], 4),
                "best_lexical_recall": round(c["lex"], 4),
                "best_token_overlap": c["overlap"],
                "mu": round(mu, 4),
                "sigma": round(sigma, 4),
                "top1_top2_margin": round(margin, 4),
                "zlike": round(zlike, 4),
                "rule_passed": rule,
            }

    best = max(candidates, key=lambda x: x["sem"])
    return None, {
        "method": "calibrated_semantic_event_match",
        "best_rank": None,
        "best_semantic_similarity": round(best["sem"], 4),
        "best_lexical_recall": round(best["lex"], 4),
        "best_token_overlap": best["overlap"],
        "mu": round(mu, 4),
        "sigma": round(sigma, 4),
        "top1_top2_margin": round(margin, 4),
        "zlike": round(zlike, 4),
        "rule_passed": None,
    }


def load_evidence_lookup(dataset_path: Path) -> dict[tuple[int, str], str]:
    with dataset_path.open() as f:
        data = json.load(f)

    lookup: dict[tuple[int, str], str] = {}
    for conv_idx, conv in enumerate(data):
        conversation = conv["conversation"]
        session_dates = {}
        for key in conversation:
            if key.endswith("_date_time") and key.startswith("session_"):
                session_num = key.replace("session_", "").replace("_date_time", "")
                session_dates[session_num] = conversation[key]
        for key in conversation:
            if key.startswith("session_") and not key.endswith("date_time"):
                if not isinstance(conversation[key], list):
                    continue
                for turn in conversation[key]:
                    dia_id = turn.get("dia_id", "")
                    if not dia_id:
                        continue
                    speaker = turn.get("speaker", "")
                    text = turn.get("text", "")
                    dia_match = re.match(r"D(\d+):", dia_id)
                    date_suffix = ""
                    if dia_match:
                        snum = dia_match.group(1)
                        sdate = session_dates.get(snum, "")
                        if sdate:
                            date_suffix = f", said on {sdate}"
                    lookup[(conv_idx, dia_id)] = f"[{dia_id}{date_suffix}] {speaker}: {text}"
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCOMO retrieval recall analysis")
    parser.add_argument("--dataset-path", required=True, help="Path to locomo10.json")
    parser.add_argument("--results-path", required=True, help="Path to unified result JSON")
    parser.add_argument("--cutoffs", default="1,3,5,10,20", help="Comma-separated k values")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results_path = Path(args.results_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = sorted({int(x) for x in args.cutoffs.split(",") if x.strip()})

    with results_path.open() as f:
        unified = json.load(f)
    evaluations = unified["evaluations"]
    evidence_lookup = load_evidence_lookup(dataset_path)

    scorer = SemanticScorer()

    total_questions = len(evaluations)
    agg_question_hits = {k: 0 for k in cutoffs}
    agg_evidence_total = 0
    agg_evidence_hits = {k: 0 for k in cutoffs}

    by_cat_q_total = defaultdict(int)
    by_cat_q_hits = {k: defaultdict(int) for k in cutoffs}
    by_cat_e_total = defaultdict(int)
    by_cat_e_hits = {k: defaultdict(int) for k in cutoffs}

    detail_jsonl_path = out_dir / "locomo_recall_per_question.jsonl"
    readable_path = out_dir / "locomo_recall_per_question_readable.json"
    readable_rows = []

    with detail_jsonl_path.open("w", encoding="utf-8") as fout:
        for e in evaluations:
            conv_idx = e["conversation_idx"]
            category_name = e.get("category_name", "unknown")
            ev_refs = e.get("evidence", []) or []
            search_results = e.get("retrieval", {}).get("search_results", []) or []

            retrieved = []
            for idx, item in enumerate(search_results, start=1):
                retrieved.append(
                    {
                        "rank": idx,
                        "memory": item.get("memory", ""),
                        "score": item.get("score"),
                    }
                )

            evidence_items = []
            for ref in ev_refs:
                ev_text = evidence_lookup.get((conv_idx, ref), "")
                best_rank, best_diag = best_match_rank_for_evidence(ev_text, retrieved, scorer)
                best_memory = ""
                if best_rank is not None:
                    for r in retrieved:
                        if r["rank"] == best_rank:
                            best_memory = r["memory"]
                            break
                if not best_memory and best_diag:
                    best_memory = ""

                evidence_items.append(
                    {
                        "evidence_ref": ref,
                        "evidence_text": ev_text,
                        "best_match_rank": best_rank,
                        "best_match_memory": best_memory,
                        "best_match_diagnostics": best_diag,
                    }
                )

            per_k = {}
            q_hit_cache = {}
            for k in cutoffs:
                matched_evs = sum(
                    1
                    for item in evidence_items
                    if item["best_match_rank"] is not None and item["best_match_rank"] <= k
                )
                q_hit = matched_evs > 0
                q_hit_cache[k] = q_hit
                ev_total = len(evidence_items)
                per_k[f"top_{k}"] = {
                    "question_hit": q_hit,
                    "matched_evidence": matched_evs,
                    "total_evidence": ev_total,
                    "evidence_recall": (matched_evs / ev_total) if ev_total else None,
                }

            by_cat_q_total[category_name] += 1
            by_cat_e_total[category_name] += len(evidence_items)
            agg_evidence_total += len(evidence_items)

            for k in cutoffs:
                if q_hit_cache[k]:
                    agg_question_hits[k] += 1
                    by_cat_q_hits[k][category_name] += 1
                matched_evs = per_k[f"top_{k}"]["matched_evidence"]
                agg_evidence_hits[k] += matched_evs
                by_cat_e_hits[k][category_name] += matched_evs

            row = {
                "question_id": e.get("question_id"),
                "conversation_idx": conv_idx,
                "category_name": category_name,
                "question": e.get("question"),
                "ground_truth_answer": e.get("ground_truth_answer"),
                "evidence": evidence_items,
                "retrieved_contexts_top20": retrieved[:20],
                "per_k": per_k,
            }

            # machine-friendly line-delimited JSON
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            readable_rows.append(row)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_results": str(results_path),
        "matching_mode": "semantic_event_match_with_paraphrase",
        "semantic_model_enabled": scorer.enabled,
        "semantic_model_name": scorer.model_name,
        "total_questions": total_questions,
        "total_evidence_refs": agg_evidence_total,
        "cutoffs": cutoffs,
        "overall_recall": {},
        "by_category": {},
    }

    cats = sorted(by_cat_q_total.keys())
    for k in cutoffs:
        q_recall = agg_question_hits[k] / total_questions if total_questions else 0.0
        e_recall = agg_evidence_hits[k] / agg_evidence_total if agg_evidence_total else 0.0
        summary["overall_recall"][f"top_{k}"] = {
            "question_hit_count": agg_question_hits[k],
            "question_recall": q_recall,
            "evidence_hit_count": agg_evidence_hits[k],
            "evidence_recall": e_recall,
        }

    for cat in cats:
        summary["by_category"][cat] = {
            "question_total": by_cat_q_total[cat],
            "evidence_total": by_cat_e_total[cat],
        }
        for k in cutoffs:
            q_total = by_cat_q_total[cat]
            e_total = by_cat_e_total[cat]
            q_hit = by_cat_q_hits[k][cat]
            e_hit = by_cat_e_hits[k][cat]
            summary["by_category"][cat][f"top_{k}"] = {
                "question_hit_count": q_hit,
                "question_recall": (q_hit / q_total) if q_total else 0.0,
                "evidence_hit_count": e_hit,
                "evidence_recall": (e_hit / e_total) if e_total else 0.0,
            }

    summary_path = out_dir / "locomo_recall_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with readable_path.open("w", encoding="utf-8") as f:
        json.dump(readable_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote details(JSONL): {detail_jsonl_path}")
    print(f"Wrote details(readable): {readable_path}")
    print(f"Semantic model enabled: {scorer.enabled} ({scorer.model_name or 'fallback'})")

    for k in cutoffs:
        s = summary["overall_recall"][f"top_{k}"]
        print(
            f"top_{k}: question_recall={s['question_recall']:.4f} "
            f"({s['question_hit_count']}/{total_questions}), "
            f"evidence_recall={s['evidence_recall']:.4f} "
            f"({s['evidence_hit_count']}/{agg_evidence_total})"
        )


if __name__ == "__main__":
    main()
