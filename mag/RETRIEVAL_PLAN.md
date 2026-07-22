# MAG Retrieval Fix Plan

This document fixes the retrieval direction after the BFS regression analysis.
The intended design is graph-first retrieval: graph traversal should be a
primary recall path, while vector/BM25/rerank validate and order graph evidence.

## Fixed Now

- MAG search no longer runs the base vector search before `_mag_search()`.
- The legacy inline BFS inside `_search_vector_store()` is disabled by default.
- BFS candidate materialization validates `user_id`, `agent_id`, and `run_id`
  before a sentence can enter the result pool.
- Context-window expansion validates the same scope before adding or boosting a
  neighboring sentence.
- LinearRAG fetches now apply the same scope validation.
- Entity links and relation edges now store session scope, and BFS/LinearRAG
  graph traversal can filter by the active `user_id`/`agent_id`/`run_id` scope.
- Added graph-scope regression tests covering entity links, scoped neighbors,
  and BFS candidate isolation.
- `_mag_search()` now builds a scoped graph candidate pool first (100-300
  candidates), uses vector/BM25 as validator scores, records per-candidate
  `route_scores`, and only adds vector-only memories as fallback when the graph
  pool is too small.
- Graph construction now uses LLM direct triple extraction: the LLM reads a
  batch of sentence ids and texts, then emits `(head, relation, tail,
  source_sentence_id)` records. Graph nodes and edges are derived from those
  triples rather than spaCy entity extraction plus pairwise relation judging.

## Remaining Design Work

1. Scope the graph itself.
   - Backfill or rebuild old persisted graph files that lack per-sentence scope.
   - Keep final payload validation as a defense-in-depth guard.

2. Replace raw path score with evidence score.
   - Combine path confidence, query-sentence similarity, query-triple similarity,
     entity coverage, temporal compatibility, and source-session diversity.
   - Prevent relation-only or hub-entity paths from dominating without sentence
     evidence.
   - Preserve and score multiple edges between the same two nodes, because
     different relations and source sentences can provide different evidence.

3. Change context-window behavior.
   - Attach previous/next sentences to the owning candidate as supporting text.
   - Do not add context sentences as independent high-score candidates unless
     they pass the same graph/vector validator.

4. Add diversity controls.
   - Limit repeated results from the same entity, session, and graph path.
   - Prevent a local graph cluster from filling the full top-k.

5. Add regression checks.
   - Assert no retrieved result crosses `user_id` boundaries.
   - Track top-k route composition, evidence recall, and answer accuracy by
     category.
   - Compare `graph_only`, `vector_only`, `graph+validator`, and
     `graph+validator+context` runs on the same saved predictions.
