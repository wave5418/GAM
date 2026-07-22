# Direct-Triple MAG Bad-Case Rebuild — 2026-07-22

## Scope

Goal: rebuild a small representative subset of MAG-only wrong LOCOMO bad cases with the new direct-triple graph builder, then check whether the missing evidence enters retrieved TopK.

Source comparison file:

- MAG: `/home/lhw/MAG/x/memory-benchmarks/results/locomo/locomo_results_20260716_175424.json`
- Mem0: `/home/lhw/MAG/results/locomo/locomo_results_mem0.json`
- Zep: `/home/lhw/MAG/results/locomo/locomo_results_20260625_zep.json`
- Cross-compare: `/tmp/locomo_baseline_cross_compare/locomo_mag_mem0_zep_20260716_items.jsonl`

Representative question IDs selected from `mag_wrong_mem0_zep_correct`:

- `conv0_q51` — multi-hop, "What has Melanie painted?"
- `conv0_q55` — multi-hop, "What subject have Caroline and Melanie both painted?"
- `conv0_q111` — single-hop, "What creative project do Mel and her kids do together besides pottery?"
- `conv2_q82`, `conv2_q147`, `conv2_q148` were initially queued as single-hop cases, but the run was stopped after the completed `conv0` slice because full-conversation direct-triple ingestion was too slow for a quick probe.

## Run

Command:

```bash
MAG_GRAPH_PERSIST_DIR=/home/lhw/MAG/results/graphs \
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
PYTHONUNBUFFERED=1 \
python -m benchmarks.locomo.run \
  --backend mag \
  --project-name direct_triples_badcases_20260722 \
  --conversations 0,2 \
  --question-ids conv0_q51,conv0_q55,conv0_q111,conv2_q82,conv2_q147,conv2_q148 \
  --predict-only \
  --score-debug \
  --top-k 200 \
  --top-k-cutoffs 10,20,50,200 \
  --max-workers 1 \
  --output-dir /home/lhw/MAG/results/locomo/direct_triple_badcases
```

Important environment note: localhost Qdrant access timed out until proxy variables were cleared and `NO_PROXY=localhost,127.0.0.1` was set.

Outputs:

- Per-question retrieval: `/home/lhw/MAG/results/locomo/direct_triple_badcases/predicted_direct_triples_badcases_20260722/`
- Persisted graph: `/home/lhw/MAG/results/graphs/mag_direct_triples_badcases_20260722_graph.json`
- TopK inspection: `/home/lhw/MAG/results/locomo/direct_triple_badcases/analysis_20260722/`

Graph caveat: the graph contains completed `conv0` plus partial `conv2` ingestion up to `session_5_c14`, because the run was gracefully interrupted after the target `conv0` questions completed. The saved `conv0` retrieval outputs are complete and reusable.

## TopK Evidence Coverage

Inspection command:

```bash
python -m benchmarks.locomo.inspect_retrieval_topk \
  --result-dir /home/lhw/MAG/results/locomo/direct_triple_badcases/predicted_direct_triples_badcases_20260722 \
  --dataset-path datasets/locomo/locomo10.json \
  --graph-path /home/lhw/MAG/results/graphs/mag_direct_triples_badcases_20260722_graph.json \
  --output-dir /home/lhw/MAG/results/locomo/direct_triple_badcases/analysis_20260722
```

Summary:

- Questions completed: 3
- All evidence in Top10: 1/3
- All evidence in Top20: 1/3
- All evidence in Top50: 2/3
- All evidence in Top200: 3/3
- Graph size at interruption: 942 entities, 964 edges, 39 multi-edge node pairs, 760 edge source sentence IDs

Per-question observations:

- `conv0_q111`: evidence `D8:5` ranked 1, source `vector+bm25_validator`; direct triples/BFS were not required for this single-hop case.
- `conv0_q51`: answer facts are present but scattered: `sunrise` appears around rank 17, `sunset` around rank 20/29, and `horse` around rank 36/42/44. All answer components are inside Top50, but one official evidence turn `D1:12` ranks 146 because the turn text is conversational and the useful fact is in image metadata.
- `conv0_q55`: both supporting image-query evidence variants rank around 30, so the required `sunset` evidence is in Top50 but not Top10. The needed bridge "both painted the same subject" is not promoted strongly enough despite graph/BFS participation.

## Interpretation

Direct-triple extraction improves evidence availability versus the old failure mode: in this small completed slice, missing facts are no longer absent from Top200, and the answer-bearing terms for the multi-hop/list-style painting questions appear by Top50. The remaining failure is ranking and aggregation, not pure graph coverage.

The main technical mismatch with the desired graph-as-primary design is:

- BFS/direct-triple retrieval can surface evidence, but the final ranking still allows generic lexical matches to dominate.
- Multi-hop/list questions need answer-level aggregation across several evidence sentences; current retrieval returns scattered facts without a strong mechanism that groups `Melanie -> painted -> {horse, sunset, sunrise}` or intersects `Caroline` and `Melanie` on a shared object.
- Context-window expansion and diversity selection dominate latency in this run, and `graph_search_paths` is also high for q55.

Next precision work should therefore target generic graph-side aggregation/ranking, not dataset-specific query hacks.
