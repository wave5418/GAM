#!/usr/bin/env python3
"""Plot similarity distribution grouped by answer correctness.

For each question and each cutoff label (e.g. top_10/top_20/top_50):
- take retrieved memories within effective-k (min(k, retrieved_count))
- compute max semantic similarity between any evidence text and any retrieved memory
- split by cutoff judgment correctness (score >= 0.5 vs < 0.5)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer


def strip_meta(evidence_text: str) -> str:
    s = re.sub(r"^\[[^\]]*\]\s*", "", evidence_text)
    s = re.sub(r"^[A-Za-z][A-Za-z\s\-']{0,40}:\s*", "", s)
    return s.strip()


def parse_cutoff(label: str) -> int:
    m = re.match(r"top_(\d+)", label)
    return int(m.group(1)) if m else 0


def load_evidence_lookup(dataset_path: Path) -> dict[tuple[int, str], str]:
    with dataset_path.open() as f:
        data = json.load(f)
    lookup = {}
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


def summarize(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=float) if vals else np.array([0.0])
    return {
        "count": int(len(vals)),
        "mean": float(arr.mean()) if len(vals) else 0.0,
        "median": float(np.median(arr)) if len(vals) else 0.0,
        "p25": float(np.percentile(arr, 25)) if len(vals) else 0.0,
        "p75": float(np.percentile(arr, 75)) if len(vals) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--labels", default="top_10,top_20,top_50")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.results_path).open() as f:
        unified = json.load(f)
    evaluations = unified["evaluations"]
    ev_lookup = load_evidence_lookup(Path(args.dataset_path))

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb_cache: dict[str, np.ndarray] = {}

    def emb(text: str) -> np.ndarray:
        if text in emb_cache:
            return emb_cache[text]
        v = model.encode([text], normalize_embeddings=True)[0]
        emb_cache[text] = v
        return v

    def sim(a: str, b: str) -> float:
        return float(np.dot(emb(a), emb(b)))

    splits = {lbl: {"correct": [], "wrong": []} for lbl in labels}
    effective_k_counts = {lbl: [] for lbl in labels}

    for e in evaluations:
        conv_idx = e["conversation_idx"]
        ev_refs = e.get("evidence", []) or []
        ev_texts = [strip_meta(ev_lookup.get((conv_idx, ref), "")) for ref in ev_refs]
        ev_texts = [t for t in ev_texts if t]
        retrieved = e.get("retrieval", {}).get("search_results", []) or []
        memories = [r.get("memory", "") for r in retrieved if r.get("memory")]
        if not ev_texts or not memories:
            continue

        for lbl in labels:
            k = parse_cutoff(lbl)
            eff_k = min(k, len(memories))
            effective_k_counts[lbl].append(eff_k)
            top_mems = memories[:eff_k]
            best = 0.0
            for ev_t in ev_texts:
                for mem in top_mems:
                    s = sim(ev_t, mem)
                    if s > best:
                        best = s

            score = e.get("cutoff_results", {}).get(lbl, {}).get("score", 0.0)
            if score == 1:
                splits[lbl]["correct"].append(best)
            else:
                splits[lbl]["wrong"].append(best)

    # Plot
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    bins = np.linspace(0.0, 1.0, 31)
    stats = {}
    for i, lbl in enumerate(labels):
        ax = axes[0][i]
        cvals = splits[lbl]["correct"]
        wvals = splits[lbl]["wrong"]
        ax.hist(wvals, bins=bins, alpha=0.6, density=True, color="#d95f02", label=f"Wrong (n={len(wvals)})")
        ax.hist(cvals, bins=bins, alpha=0.6, density=True, color="#1b9e77", label=f"Correct (n={len(cvals)})")
        ax.set_title(lbl)
        ax.set_xlabel("Max semantic similarity (evidence vs retrieved)")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=8)
        stats[lbl] = {
            "correct": summarize(cvals),
            "wrong": summarize(wvals),
            "effective_k": summarize(effective_k_counts[lbl]),
        }

    fig.suptitle("Answer Correct vs Wrong: Evidence-Memory Semantic Similarity", fontsize=14)
    fig.tight_layout()

    png = out_dir / "answer_correctness_similarity_distribution.png"
    stats_path = out_dir / "answer_correctness_similarity_stats.json"
    fig.savefig(png, dpi=180)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote plot: {png}")
    print(f"Wrote stats: {stats_path}")
    for lbl in labels:
        s = stats[lbl]
        print(
            f"{lbl}: correct_mean={s['correct']['mean']:.4f} (n={s['correct']['count']}), "
            f"wrong_mean={s['wrong']['mean']:.4f} (n={s['wrong']['count']}), "
            f"effective_k_mean={s['effective_k']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()
