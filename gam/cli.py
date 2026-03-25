from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

from .config import BenchmarkConfig
from .workspace import BenchmarkWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original benchmark CLI for Graph Agent Memory")
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample", type=int, nargs="+", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="+", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--use-episodes", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--best-of-n", type=int, default=None)
    parser.add_argument("--best-of-n-method", choices=["llm_judge", "voting", "f1"], default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--embedding-backend", choices=["minilm", "openai"], default=None)
    parser.add_argument("--embedding-model-name", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    config = BenchmarkConfig.from_sources(args)
    workspace = BenchmarkWorkspace(config)
    report = workspace.build_report()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = config.output_path()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"GAM benchmark finished. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
