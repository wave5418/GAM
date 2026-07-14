#!/usr/bin/env python3
"""Plot similarity distributions for recall success vs failure.

Input:
- locomo_recall_per_question_readable.json from semantic-model recall run.

Output:
- similarity_distribution_success_failure.png
- similarity_distribution_stats.json
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


def compute_similarity_splits(rows: list[dict], cutoffs: list[int]) -> dict[int, dict[str, list[float]]]:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cache: dict[str, np.ndarray] = {}

    def emb(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        v = model.encode([text], normalize_embeddings=True)[0]
        cache[text] = v
        return v

    def sim(a: str, b: str) -> float:
        return float(np.dot(emb(a), emb(b)))

    out: dict[int, dict[str, list[float]]] = {k: {"success": [], "failure": []} for k in cutoffs}

    for row in rows:
        ev_texts = [strip_meta(e.get("evidence_text", "")) for e in row.get("evidence", [])]
        ev_texts = [t for t in ev_texts if t]
        mems = [m.get("memory", "") for m in row.get("retrieved_contexts_top20", []) if m.get("memory")]
        if not ev_texts or not mems:
            for k in cutoffs:
                ok = bool(row.get("per_k", {}).get(f"top_{k}", {}).get("question_hit", False))
                (out[k]["success"] if ok else out[k]["failure"]).append(0.0)
            continue

        for k in cutoffs:
            top_mems = mems[:k]
            best = 0.0
            for ev in ev_texts:
                for mem in top_mems:
                    s = sim(ev, mem)
                    if s > best:
                        best = s
            ok = bool(row.get("per_k", {}).get(f"top_{k}", {}).get("question_hit", False))
            (out[k]["success"] if ok else out[k]["failure"]).append(best)

    return out


def summarize(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=float) if vals else np.array([0.0])
    return {
        "count": int(len(vals)),
        "mean": float(arr.mean()) if len(vals) else 0.0,
        "median": float(np.median(arr)) if len(vals) else 0.0,
        "p25": float(np.percentile(arr, 25)) if len(vals) else 0.0,
        "p75": float(np.percentile(arr, 75)) if len(vals) else 0.0,
    }


def plot_distributions(splits: dict[int, dict[str, list[float]]], out_png: Path) -> None:
    cutoffs = sorted(splits.keys())
    n = len(cutoffs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), squeeze=False)

    bins = np.linspace(0.0, 1.0, 31)
    for idx, k in enumerate(cutoffs):
        ax = axes[idx // cols][idx % cols]
        suc = splits[k]["success"]
        fail = splits[k]["failure"]
        ax.hist(fail, bins=bins, alpha=0.6, label=f"Failure (n={len(fail)})", color="#d95f02", density=True)
        ax.hist(suc, bins=bins, alpha=0.6, label=f"Success (n={len(suc)})", color="#1b9e77", density=True)
        ax.set_title(f"Top-{k}")
        ax.set_xlabel("Max semantic similarity per question")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=8)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("LOCOMO Recall Success vs Failure Similarity Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to locomo_recall_per_question_readable.json")
    parser.add_argument("--cutoffs", default="1,3,5,10,20", help="Comma-separated cutoffs")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = sorted({int(x) for x in args.cutoffs.split(",") if x.strip()})

    with in_path.open() as f:
        rows = json.load(f)

    splits = compute_similarity_splits(rows, cutoffs)

    stats = {}
    for k in cutoffs:
        stats[f"top_{k}"] = {
            "success": summarize(splits[k]["success"]),
            "failure": summarize(splits[k]["failure"]),
        }

    png_path = out_dir / "similarity_distribution_success_failure.png"
    stats_path = out_dir / "similarity_distribution_stats.json"
    plot_distributions(splits, png_path)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote plot: {png_path}")
    print(f"Wrote stats: {stats_path}")
    for k in cutoffs:
        s = stats[f"top_{k}"]
        print(
            f"top_{k}: success_mean={s['success']['mean']:.4f} (n={s['success']['count']}), "
            f"failure_mean={s['failure']['mean']:.4f} (n={s['failure']['count']})"
        )


if __name__ == "__main__":
    main()
