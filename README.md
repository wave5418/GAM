# GAM: Graph Agent Memory Benchmark Framework

`GAM` is a benchmark-oriented framework for testing graph-based agent memory systems on long-horizon dialogue tasks.

This repository now combines two layers:

- `gam/`: a more standalone experiment framework for configuration, method registration, execution, and reporting
- `memory/`: the underlying graph-memory implementation used by the benchmark adapters

The goal is to make experiments easier to run, compare, and evolve without tying the whole workflow to one large evaluation script.

## What It Supports

- User-defined LLM `base_url`, `api_key`, and `model_name`
- User-defined embedding backend and embedding model
- Side-by-side comparison across graph-memory variants and retrieval baselines
- Structured JSON benchmark outputs
- Compatibility entrypoint for older usage patterns

## Methods Available

The benchmark layer currently supports the following methods:

- `graph_full`
- `basic_retrieval`
- `no_causal`
- `no_temporal`
- `flat_graph`
- `vector_only`
- `keyword_only`
- `scan_only`

## Project Layout

```text
GAM/
├── gam/                           # Standalone benchmark package
│   ├── cli.py                    # CLI entry
│   ├── config.py                 # Experiment configuration
│   ├── methods.py                # Method registry
│   ├── baselines.py              # Retrieval-only baselines
│   ├── workspace.py              # Run orchestration
│   └── reporting.py              # Metrics/report helpers
├── gam_cli.py                    # Preferred CLI entrypoint
├── memory/                       # Graph memory implementation
├── data/                         # Dataset files
├── examples/                     # Sample inputs
└── GRAPH_AGENT_MEMORY_FRAMEWORK.md
```

## Installation

### Recommended

```bash
git clone https://github.com/wave5418/GAM.git
cd GAM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Minimal smoke-test dependencies

If you only want to verify the CLI and OpenAI-compatible integration first:

```bash
pip install networkx openai python-dotenv faiss-cpu
```

For full local MiniLM-based experiments, install the full `requirements.txt`.

## Quick Start

### 1. Run the new standalone CLI

```bash
python gam_cli.py \
  --dataset data/locomo10.json \
  --sample 0 \
  --methods graph_full,basic_retrieval,no_causal,no_temporal,flat_graph,vector_only,keyword_only \
  --model-name gpt-4o-mini \
  --embedding-backend minilm \
  --max-questions 10 \
  --parallel
```

### 2. Use an OpenAI-compatible endpoint

```bash
python gam_cli.py \
  --dataset data/locomo10.json \
  --sample 0 \
  --methods graph_full,vector_only,keyword_only \
  --model-name Qwen/Qwen2.5-72B-Instruct \
  --base-url https://api.siliconflow.cn/v1 \
  --api-key sk-your-key \
  --embedding-backend openai \
  --embedding-model-name text-embedding-3-small \
  --embedding-base-url https://api.siliconflow.cn/v1 \
  --embedding-api-key sk-your-key
```

### 3. Run ALFWorld-format data

```bash
python gam_cli.py \
  --dataset-format alfworld \
  --dataset examples/alfworld_sample.json \
  --sample 0 \
  --methods graph_full,vector_only,keyword_only,scan_only \
  --max-questions 1 \
  --categories 4
```

ALFWorld adapter currently accepts:

- JSON list of episodes
- JSON object with `episodes`
- JSONL (one episode per line)

Each episode is converted into one internal sample (single session with Agent/Environment turns).

### 4. Show all CLI options

```bash
python gam_cli.py --help
```

## Datasets

### LoCoMo

- Path: `data/locomo10.json`
- Included in the repository
- Best current fit for the standalone `gam/` benchmark flow

### LongMemEval

- Sample file included in `examples/longmemeval_sample.json`
- Full dataset needs to be downloaded separately if you want to run the older LongMemEval scripts

### ALFWorld

- Use `--dataset-format alfworld` to switch parser mode
- Works with common trajectory fields like `trajectory` / `steps` / `history`
- If no QA is provided in source data, the loader auto-generates a minimal factual QA so evaluation can run

## Output

Benchmark results are written to:

```text
benchmark_results/graph_memory_benchmark_<model>.json
```

Each report includes:

- experiment config
- selected samples
- memory statistics
- per-method results
- aggregate metrics
- category breakdown

## Configuration

You can pass settings directly by CLI flags or use a JSON config file.

Example:

```bash
python gam_cli.py --config configs/graph_memory_benchmark.example.json
```

Configurable fields include:

- dataset path
- sample ids
- compared methods
- model endpoint settings
- embedding settings
- cache and output directories
- parallel workers
- best-of-n strategy

## Current Design Notes

- The `gam/` package is intentionally more independent from the original script layout.
- The benchmark layer is now framework-owned, with clearer separation between config, method selection, orchestration, and reporting.
- The underlying `memory/` implementation is still reused through adapters rather than being fully rewritten from scratch.

## Related Docs

- `GRAPH_AGENT_MEMORY_FRAMEWORK.md`: practical benchmark usage notes
- `MAMGA_API_CONFIG.md`: API and endpoint configuration notes

## License

MIT License. See `LICENSE`.
