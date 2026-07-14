#!/usr/bin/env python3
"""
Build LOCOMO predict outputs using Graphiti retrieval.

This script writes files compatible with:
  python -m benchmarks.locomo.run --evaluate-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from openai import AsyncOpenAI
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_DATASET_DIR = "datasets/locomo"
CHUNK_SIZE = 20
CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "open-domain",
    4: "temporal",
}


class _SimpleLogger:
    def info(self, msg: str, *args) -> None:
        if args:
            msg = msg % args
        print(msg)


class _ChunkedOpenAIEmbedder(OpenAIEmbedder):
    """DashScope-compatible embedder wrapper with bounded batch size."""

    def __init__(self, *args, batch_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if len(input_data_list) <= self.batch_size:
            return await super().create_batch(input_data_list)
        out: list[list[float]] = []
        for i in range(0, len(input_data_list), self.batch_size):
            out.extend(await super().create_batch(input_data_list[i : i + self.batch_size]))
        return out


def parse_locomo_date(date_str: str) -> datetime | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        paired.append((key, conversation.get(date_key, ""), conversation[key]))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str) -> list[list[dict]]:
    messages = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"
        else:
            photo_tag = ""
        if photo_tag:
            text = f"{text} {photo_tag}" if text else photo_tag
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})

    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i : i + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
    return chunks


def download_dataset(dataset_dir: str, logger: _SimpleLogger) -> str:
    path = Path(dataset_dir) / "locomo10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.info("Dataset already exists: %s", path)
        return str(path)
    logger.info("Downloading LOCOMO-10 dataset...")
    data = requests.get(DATASET_URL, timeout=60).json()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Downloaded: %s (%d conversations)", path, len(data))
    return str(path)


def load_dataset(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LOCOMO predict files with Graphiti")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default="results/locomo")
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks per conversation for smoke tests")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--driver", default="kuzu", choices=["kuzu", "neo4j", "falkor"])
    parser.add_argument("--kuzu-db", default=None, help="Path to Kuzu DB file (driver=kuzu)")
    parser.add_argument("--falkor-host", default="localhost")
    parser.add_argument("--falkor-port", type=int, default=6379)
    parser.add_argument("--falkor-db", default="graphiti")
    parser.add_argument("--llm-model", default="qwen-turbo")
    parser.add_argument("--llm-client", default="generic", choices=["generic", "openai"])
    parser.add_argument("--llm-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedder-model", default="text-embedding-v3")
    parser.add_argument("--embedder-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--http-timeout-sec", type=int, default=30, help="HTTP timeout for LLM/embedder API calls")
    parser.add_argument("--op-timeout-sec", type=int, default=120, help="Timeout for each add/search op")
    parser.add_argument("--embed-batch-size", type=int, default=10)
    parser.add_argument("--rebuild-indexes", action="store_true", help="Rebuild graph indexes/constraints before ingest")
    parser.add_argument("--source-description", default="locomo")
    return parser.parse_args()


def edge_to_result(edge) -> dict:
    return {
        "memory": getattr(edge, "fact", "") or "",
        "score": 0.0,
        "id": getattr(edge, "uuid", "") or "",
        "created_at": str(getattr(edge, "created_at", "") or ""),
    }


async def async_main() -> None:
    args = parse_args()
    run_id = args.run_id or uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir) / f"predicted_{args.project_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset_path or download_dataset(DEFAULT_DATASET_DIR, logger=_SimpleLogger())
    dataset = load_dataset(dataset_path)

    conv_indices = [int(c) for c in args.conversations.split(",")]
    categories = [int(c) for c in args.categories.split(",")]

    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    api_key = os.getenv("OPENAI_API_KEY")
    llm_http_client = AsyncOpenAI(api_key=api_key, base_url=args.llm_base_url, timeout=args.http_timeout_sec)
    embed_http_client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.embedder_base_url,
        timeout=args.http_timeout_sec,
    )
    llm_cfg = LLMConfig(
        api_key=api_key,
        model=args.llm_model,
        base_url=args.llm_base_url,
        small_model=args.llm_model,
    )
    if args.llm_client == "generic":
        llm = OpenAIGenericClient(llm_cfg, client=llm_http_client)
    else:
        llm = OpenAIClient(llm_cfg, client=llm_http_client)
    embedder = _ChunkedOpenAIEmbedder(
        OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=args.embedder_base_url,
            embedding_model=args.embedder_model,
        ),
        client=embed_http_client,
        batch_size=args.embed_batch_size,
    )
    if args.driver == "kuzu":
        kuzu_db = args.kuzu_db or f"/home/lhw/MAG/x/memory-benchmarks/datasets/graphiti_kuzu_{run_id}.db"
        print(f"[graphiti] driver=kuzu db={kuzu_db}")
        g = Graphiti(graph_driver=KuzuDriver(db=kuzu_db), llm_client=llm, embedder=embedder)
    elif args.driver == "falkor":
        print(f"[graphiti] driver=falkor host={args.falkor_host} port={args.falkor_port} db={args.falkor_db}")
        g = Graphiti(
            graph_driver=FalkorDriver(
                host=args.falkor_host,
                port=args.falkor_port,
                database=args.falkor_db,
            ),
            llm_client=llm,
            embedder=embedder,
        )
    else:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise RuntimeError("Missing NEO4J_PASSWORD env var")
        g = Graphiti(uri=neo4j_uri, user=neo4j_user, password=neo4j_password, llm_client=llm, embedder=embedder)

    try:
        print(f"[graphiti] build indices (rebuild={args.rebuild_indexes})")
        await g.build_indices_and_constraints(delete_existing=args.rebuild_indexes)
        for conv_idx in conv_indices:
            print(f"[graphiti] start conversation {conv_idx}")
            entry = dataset[conv_idx]
            conversation = entry["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            group_id = f"loc_{conv_idx}_{run_id}"

            # Ingest
            chunk_budget = args.max_chunks
            for session_key, session_date, turns in get_sorted_sessions(conversation):
                chunks = session_to_chunks(turns, speaker_a, speaker_b)
                session_dt = parse_locomo_date(session_date) or datetime.now(timezone.utc)
                for i, messages in enumerate(chunks):
                    if chunk_budget is not None and chunk_budget <= 0:
                        break
                    print(f"[graphiti] ingest {session_key} chunk={i} messages={len(messages)}")
                    episode_body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    kwargs = {}
                    if args.driver in ("neo4j", "falkor"):
                        kwargs["group_id"] = group_id
                    try:
                        await asyncio.wait_for(g.add_episode(
                            name=f"{session_key}_chunk_{i}",
                            episode_body=episode_body,
                            source_description=args.source_description,
                            reference_time=session_dt.replace(tzinfo=timezone.utc),
                            **kwargs,
                        ), timeout=args.op_timeout_sec)
                    except Exception as e:
                        print(f"[graphiti] WARNING: ingest failed {session_key} chunk={i}: {e}")
                        continue
                    if chunk_budget is not None:
                        chunk_budget -= 1
                if chunk_budget is not None and chunk_budget <= 0:
                    print(f"[graphiti] chunk limit reached for conv {conv_idx}")
                    break

            # Retrieve
            questions = entry.get("qa", entry.get("qa_pairs", []))
            scoped = [(qi, qa) for qi, qa in enumerate(questions) if qa.get("category") in categories]
            if args.max_questions is not None:
                scoped = scoped[: args.max_questions]

            for qi, qa in scoped:
                question_id = f"conv{conv_idx}_q{qi}"
                question = qa["question"]
                answer = str(qa["answer"])
                category = qa["category"]

                start = time.monotonic()
                print(f"[graphiti] search {question_id}")
                try:
                    if args.driver == "neo4j":
                        edges = await asyncio.wait_for(
                            g.search(question, group_ids=[group_id], num_results=args.top_k),
                            timeout=args.op_timeout_sec,
                        )
                    else:
                        edges = await asyncio.wait_for(
                            g.search(question, num_results=args.top_k),
                            timeout=args.op_timeout_sec,
                        )
                except Exception as e:
                    print(f"[graphiti] WARNING: search failed {question_id}: {e}")
                    edges = []
                latency_ms = (time.monotonic() - start) * 1000
                print(f"[graphiti] search_done {question_id} edges={len(edges)} latency_ms={latency_ms:.1f}")

                formatted = [edge_to_result(e) for e in edges]
                result = {
                    "question_id": question_id,
                    "conversation_idx": conv_idx,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, "unknown"),
                    "question": question,
                    "ground_truth_answer": answer,
                    "evidence": qa.get("evidence", []),
                    "user_id": group_id,
                    "reference_date": qa.get("date"),
                    "retrieval": {
                        "search_query": question,
                        "search_results": formatted,
                        "search_latency_ms": round(latency_ms, 1),
                        "total_results": len(formatted),
                    },
                }

                (out_dir / f"{question_id}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
                print(f"[graphiti] wrote {out_dir / f'{question_id}.json'}")

        print(f"Done. Predict outputs: {out_dir}")
        print(f"Next: python -m benchmarks.locomo.run --project-name {args.project_name} --evaluate-only")
    except Exception as exc:
        print(f"[graphiti] ERROR: {exc}")
        traceback.print_exc()
        raise
    finally:
        print("[graphiti] closing")
        await g.close()


if __name__ == "__main__":
    asyncio.run(async_main())
