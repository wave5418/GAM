# Direct-Triple MAG Conv0 Bad-Case Top10 Check — 2026-07-22

## Scope

This check reruns every `conv0` Top10 wrong case from the original MAG result:

- Original MAG result: `/home/lhw/MAG/x/memory-benchmarks/results/locomo/locomo_results_20260716_175424.json`
- New project: `direct_triples_badcases_20260722`
- Reused ingest checkpoint: `run_id=93b4e8dc`
- Completed ingest scope: full `conv0`, 419/419 chunks
- Persisted graph: `/home/lhw/MAG/results/graphs/mag_direct_triples_badcases_20260722_graph.json`

Question IDs:

```text
conv0_q111,conv0_q112,conv0_q121,conv0_q22,conv0_q29,conv0_q41,conv0_q44,conv0_q49,
conv0_q51,conv0_q55,conv0_q63,conv0_q64,conv0_q67,conv0_q71,conv0_q76,conv0_q90
```

## Commands

Predict-only retrieval completion:

```bash
MAG_GRAPH_PERSIST_DIR=/home/lhw/MAG/results/graphs \
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
PYTHONUNBUFFERED=1 \
python -m benchmarks.locomo.run \
  --backend mag \
  --project-name direct_triples_badcases_20260722 \
  --run-id 93b4e8dc \
  --resume \
  --conversations 0 \
  --question-ids conv0_q111,conv0_q112,conv0_q121,conv0_q22,conv0_q29,conv0_q41,conv0_q44,conv0_q49,conv0_q51,conv0_q55,conv0_q63,conv0_q64,conv0_q67,conv0_q71,conv0_q76,conv0_q90 \
  --predict-only \
  --score-debug \
  --top-k 200 \
  --top-k-cutoffs 10 \
  --max-workers 1 \
  --output-dir /home/lhw/MAG/results/locomo/direct_triple_badcases
```

Top10 answer and judge:

```bash
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
PYTHONUNBUFFERED=1 \
python -m benchmarks.locomo.run \
  --project-name direct_triples_badcases_20260722 \
  --run-id 93b4e8dc \
  --evaluate-only \
  --rejudge \
  --answerer-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --provider openai \
  --conversations 0 \
  --question-ids conv0_q111,conv0_q112,conv0_q121,conv0_q22,conv0_q29,conv0_q41,conv0_q44,conv0_q49,conv0_q51,conv0_q55,conv0_q63,conv0_q64,conv0_q67,conv0_q71,conv0_q76,conv0_q90 \
  --top-k 200 \
  --top-k-cutoffs 10 \
  --max-workers 3 \
  --output-dir /home/lhw/MAG/results/locomo/direct_triple_badcases
```

Note: the default `qwen-turbo` run failed with 403 model-access errors, so the actual evaluation uses the original benchmark's `gpt-4o-mini` answerer and judge.

## Results

Unified result:

- `/home/lhw/MAG/results/locomo/direct_triple_badcases/locomo_results_20260722_200255.json`

Official LLM judge:

- Overall Top10: 6/16, 37.5%
- Single-hop: 1/4
- Multi-hop: 2/4
- Temporal: 3/6
- Open-domain: 0/2

Evidence coverage from saved retrieval:

- All official evidence in Top10: 3/16
- All official evidence in Top20: 6/16
- All official evidence in Top50: 9/16
- All official evidence in Top200: 14/16
- Search latency: average 27.49s, median 26.52s, min 23.54s, max 35.35s

## Per-Question Outcomes

| qid | category | judge | generated Top10 answer | evidence rank signal |
| --- | --- | --- | --- | --- |
| `conv0_q111` | single-hop | CORRECT | painting | evidence rank 1 |
| `conv0_q112` | single-hop | WRONG | abstains | evidence rank 15 |
| `conv0_q121` | single-hop | WRONG | abstains | evidence rank 18 |
| `conv0_q22` | open-domain | WRONG | says no | evidence rank 17 |
| `conv0_q29` | temporal | WRONG | August 24, 2023 | evidence rank 27 |
| `conv0_q41` | temporal | WRONG | October 17, 2023 | evidence not matched |
| `conv0_q44` | temporal | WRONG | abstains | evidence not matched |
| `conv0_q49` | temporal | CORRECT | abstains | evidence rank 125 |
| `conv0_q51` | multi-hop | CORRECT | nature-inspired pieces, landscapes, still life | evidence ranks 36, 20, 146 |
| `conv0_q55` | multi-hop | CORRECT | Nature | evidence ranks 30, 30 |
| `conv0_q63` | temporal | CORRECT | September 2023 | evidence rank 2 |
| `conv0_q64` | open-domain | WRONG | abstains | evidence rank 32 |
| `conv0_q67` | temporal | CORRECT | weekend of 9-10 September 2023 | evidence rank 3 |
| `conv0_q71` | multi-hop | WRONG | Charlotte's Web | evidence ranks 121, 80 |
| `conv0_q76` | multi-hop | WRONG | abstains | evidence rank 196 |
| `conv0_q90` | single-hop | WRONG | abstains | evidence rank 115 |

## Reading

The official 6/16 should be treated as optimistic because the judge accepts at least three questionable answers:

- `conv0_q49`: generated answer abstains, but judge marks CORRECT.
- `conv0_q51`: generated answer omits explicit `horse`, `sunset`, and `sunrise`, but judge marks CORRECT.
- `conv0_q55`: generated answer says `Nature`, while the gold answer is `Sunsets`, but judge marks CORRECT.

A stricter manual read gives 3/16 clear Top10 wins: `conv0_q111`, `conv0_q63`, and `conv0_q67`.

The direct-triple graph improves broad evidence availability: most previously wrong conv0 cases now have evidence somewhere in Top50/Top200. It does not yet solve the core Top10 problem. The failures are mainly ranking and aggregation:

- Evidence often appears just outside Top10 (`q112`, `q121`, `q22`, `q29`, `q55`, `q64`).
- Multi-hop/list evidence is scattered and not converted into answer-level grouped facts (`q51`, `q55`, `q71`, `q76`).
- Some temporal/open-domain cases retrieve related but wrong events, causing confident wrong answers (`q29`, `q41`).
- Latency is still dominated by retrieval-side processing rather than answer/judge, with ~23–35s per search on this slice.
