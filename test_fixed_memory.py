#!/usr/bin/env python3
"""
Fixed TRG Memory System - Refactored with modular architecture

This script now uses the following modules:
- memory.memory_builder: Memory construction and indexing
- memory.query_engine: Query execution and retrieval
- memory.test_harness: Testing and evaluation

The core logic has been moved to proper modules under memory/ folder.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_dataset import load_locomo_dataset
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.test_harness import TestHarness
from memory.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def score_only_mode(args):
    """
    Re-score existing results without rebuilding memory or querying.

    This mode loads existing result files and re-evaluates them with
    potentially different scoring models or methods.
    """
    import os
    from utils.memory_layer import LLMController
    from tqdm import tqdm

    print("\n" + "="*70)
    print("  SCORE-ONLY MODE: Re-evaluating existing results")
    print("="*70)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Cannot perform LLM-based evaluation.")
        return 1

    llm_controller = LLMController(
        backend='openai',
        model=args.model,
        api_key=api_key
    )

    evaluator = Evaluator(llm_controller=llm_controller, use_llm_judge=True)
    print(f"Evaluator initialized with model: {args.model}\n")

    all_sample_results = []

    for sample_id in args.sample:
        print(f"\n{'='*70}")
        print(f"Re-scoring Sample {sample_id}")
        print(f"{'='*70}")

        if args.input_results:
            input_file = args.input_results
        else:
            model_name_normalized = args.model.replace(".", "_").replace("-", "_")
            model_specific_file = f"results_{model_name_normalized}/fixed_results_sample{sample_id}.json"
            default_file = f"results/fixed_results_sample{sample_id}.json"

            if Path(model_specific_file).exists():
                input_file = model_specific_file
            elif Path(default_file).exists():
                input_file = default_file
            else:
                input_file = default_file

        if not Path(input_file).exists():
            print(f"Error: Results file not found: {input_file}")
            print(f"Please run the full test first or specify --input-results")
            continue

        with open(input_file, 'r') as f:
            data = json.load(f)

        existing_results = data.get('results', [])
        print(f"Loaded {len(existing_results)} results from {input_file}")

        updated_results = []
        print("Re-evaluating answers...")

        for result in tqdm(existing_results, desc="Re-scoring"):
            question = result['question']
            expected = result['expected']
            predicted = result['predicted']
            category = result.get('category')

            eval_result = evaluator.evaluate_answer(
                question,
                str(expected) if expected else "",
                predicted,
                question_category=category
            )

            result['metrics'] = eval_result.get('metrics')
            result['correct'] = eval_result.get('is_correct', False)
            result['llm_judge_score'] = eval_result.get('llm_judge_score', 0.0)

            updated_results.append(result)

        total = len(updated_results)
        correct = sum(1 for r in updated_results if r['correct'])
        not_found = sum(1 for r in updated_results if r['predicted'] == "Information not found")

        f1_scores = [r['metrics']['f1'] for r in updated_results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in updated_results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r['llm_judge_score'] for r in updated_results if r.get('llm_judge_score', 0) >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        results_no_cat5 = [r for r in updated_results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r['llm_judge_score'] for r in results_no_cat5 if r.get('llm_judge_score', 0) >= 0) / total_no_cat5 * 100
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in updated_results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if r.get('llm_judge_score', 0) >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Re-scored Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/rescored_results_sample{sample_id}.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'original_file': input_file,
                'results': updated_results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown
                }
            }, f, indent=2, default=str)

        print(f"\nRe-scored results saved to {output_file}")

        wrong_answers = [r for r in test_results if r.get('llm_judge_score', 0) < 0.5]
        if wrong_answers:
            wrong_output_file = f"{results_dir}/fixed_results_sample{sample_id}_wrong.json"

            category_names = {
                1: "Multi-hop",
                2: "Temporal",
                3: "Open-domain",
                4: "Single-hop",
                5: "Adversarial"
            }

            formatted_wrong = []
            for wa in wrong_answers:
                formatted_wrong.append({
                    'q_id': wa['question_id'],
                    'category': f"{wa['category']} ({category_names.get(wa['category'], 'Unknown')})",
                    'question': wa['question'],
                    'expected': wa['expected'],
                    'predicted': wa['predicted'],
                    'f1': wa.get('metrics', {}).get('f1', 0),
                    'llm_score': wa.get('llm_judge_score', 0),
                    'top_nodes': wa.get('search_details', {}).get('top_nodes', []),
                    'answer_context': wa.get('answer_context', '')
                })

            wrong_summary = {
                'sample_id': sample_id,
                'total_questions': len(test_results),
                'total_wrong': len(wrong_answers),
                'wrong_percentage': len(wrong_answers) / len(test_results) * 100,
                'wrong_questions': formatted_wrong
            }

            with open(wrong_output_file, 'w') as f:
                json.dump(wrong_summary, f, indent=2, default=str)

            print(f"Wrong answers ({len(wrong_answers)}) saved to {wrong_output_file}")

        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'category_breakdown': category_breakdown
        })

    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RE-SCORED RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/rescored_results_aggregate_samples_{'_'.join(map(str, args.sample))}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'samples': args.sample,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5
                }
            }, f, indent=2)
        print(f"\nAggregate re-scored results saved to {aggregate_output}")

    return 0

def main():
    """Main test function - supports multiple samples"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, nargs='+', default=[0],
                       help="Sample IDs to test (can specify multiple, e.g., --sample 0 1 2)")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--cache-dir", default="./locomo_trg_fixed")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild memory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to use (e.g., gpt-4o-mini, gpt-4.1-mini, gpt-3.5-turbo, gpt-4o)")
    parser.add_argument("--embedding-model", type=str, default="minilm",
                       choices=["minilm", "openai"],
                       help="Embedding model to use: 'minilm' (all-MiniLM-L6-v2, 384-dim) or 'openai' (text-embedding-3-small, 1536-dim)")
    parser.add_argument("--use-episodes", action="store_true",
                       help="Use episode-based segmentation instead of turn-based (groups related turns)")
    parser.add_argument("--score-only", action="store_true",
                       help="Only re-score existing results without rebuilding memory or querying")
    parser.add_argument("--input-results", type=str, default=None,
                       help="Input results file to re-score (used with --score-only)")
    parser.add_argument("--skip-category-5", action="store_true",
                       help="Skip category 5 (Adversarial) questions during testing")
    parser.add_argument("--category-to-test", type=str, default="1,2,3,4",
                       help="Comma-separated list of categories to test (e.g., '1,2,3,4' or '1,3'). Default: '1,2,3,4'")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel testing (parallel is enabled by default for 3x speedup)")
    parser.add_argument("--n-workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    parser.add_argument("--best-of-n", type=int, default=3,
                       help="Run each question N times and select best answer (default: 3 = best-of-3)")
    parser.add_argument("--best-of-n-method", type=str, default="llm_judge",
                       choices=["llm_judge", "voting", "f1"],
                       help="Method for selecting best answer: 'llm_judge', 'voting', or 'f1' (default: llm_judge)")
    parser.add_argument("--ablation", type=str, default=None,
                       choices=["basic_retrieval", "no_causal", "no_temporal", "flat_graph"],
                       help="Run ablation study with specific configuration")
    parser.add_argument("--debug-search", action="store_true",
                       help="Enable full retrieval trace logging for every question and write a separate debug JSON")
    args = parser.parse_args()

    # Parallel is enabled by default
    args.parallel = not args.no_parallel

    print("="*70)
    print("  Fixed TRG Memory System Test")
    print(f"  Model: {args.model}")
    if args.score_only:
        print(f"  Mode: SCORE-ONLY (Re-evaluation)")
    else:
        print(f"  Mode: {'Episode-based' if args.use_episodes else 'Turn-based'}")
    print(f"  Samples: {args.sample}")

    # Parse categories to test
    categories_to_test = [int(c.strip()) for c in args.category_to_test.split(',')]

    # Handle legacy skip_category_5 flag
    if args.skip_category_5 and 5 in categories_to_test:
        categories_to_test.remove(5)

    print(f"  Categories to test: {sorted(categories_to_test)}")
    category_names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
    print("  Category types: " + ', '.join([f"{c}:{category_names.get(c, 'Unknown')}" for c in sorted(categories_to_test)]))

    # Show parallel mode status
    # if args.parallel:
    #     print(f"  Parallel mode: ✓ ENABLED ({args.n_workers} workers, ~{args.n_workers}x speedup)")
    # else:
    #     print(f"  Parallel mode: ✗ DISABLED (use default for 3x speedup)")

    print("="*70)

    # Handle score-only mode
    if args.score_only:
        return score_only_mode(args)

    # Load dataset
    samples = load_locomo_dataset(args.dataset)

    # Validate sample IDs
    for sample_id in args.sample:
        if sample_id < 0 or sample_id >= len(samples):
            print(f"Error: Sample ID {sample_id} is out of range (0-{len(samples)-1})")
            return 1

    # Process each sample
    all_sample_results = []

    for sample_idx, sample_id in enumerate(args.sample, 1):
        sample = samples[sample_id]

        print(f"\n{'='*70}")
        print(f"Processing Sample {sample_id} ({sample_idx}/{len(args.sample)})")
        print(f"{'='*70}")

        # Auto-generate cache directory with sample number, embedding model, and LLM model
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Normalize model name for folder (replace dots and hyphens with underscores)
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")

        if args.cache_dir == "./locomo_trg_fixed":
            # User didn't specify custom cache dir, use auto-naming with model name
            if args.use_episodes:
                cache_dir = f"./locomo_trg_episodes_{model_name_normalized}/sample{sample_id}{embedding_suffix}"
            else:
                cache_dir = f"./locomo_trg_cache_{model_name_normalized}/sample{sample_id}{embedding_suffix}"
            print(f"Auto cache directory: {cache_dir}")
            print(f"Embedding model: {args.embedding_model}")
        else:
            # User specified custom cache dir with sample ID in subdirectory
            cache_dir = f"{args.cache_dir}/sample{sample_id}{embedding_suffix}"
            print(f"Custom cache directory: {cache_dir}")
            print(f"Embedding model: {args.embedding_model}")

        # Initialize memory builder
        builder = MemoryBuilder(
            cache_dir=cache_dir,
            llm_model=args.model,
            use_episodes=args.use_episodes,
            embedding_model=args.embedding_model
        )

        # Build or load memory
        cache_file = Path(cache_dir) / "graph.json"
        if cache_file.exists() and not args.rebuild:
            logger.info("Loading cached memory...")
            builder.load()
        else:
            logger.info("Building memory...")
            stats = builder.build_memory(sample)
            builder.save()
            # print(f"Memory built: {stats}")

        # Get memory stats
        mem_stats = builder.trg.get_statistics()
        print(f"\nMemory Statistics:")
        print(f"  Total nodes: {mem_stats['total_nodes']}")
        print(f"  Total links: {mem_stats['links_created']}")
        if mem_stats['total_nodes'] > 0:
            print(f"  Links per node: {mem_stats['links_created']/mem_stats['total_nodes']:.1f}")
        print(f"  Node types: {mem_stats['node_types']}")
        print(f"  Link types: {mem_stats['link_types']}")

        # Initialize query engine with entity-session mapping
        # Prepare ablation configuration
        ablation_config = {}
        if args.ablation:
            ablation_config[args.ablation] = True
            print(f"  ⚠️ ABLATION MODE: {args.ablation}")

        query_engine = QueryEngine(
            builder.trg,
            builder.node_index,
            entity_session_map=builder.entity_session_map if hasattr(builder, 'entity_session_map') else None,
            entity_dia_map=builder.entity_dia_map if hasattr(builder, 'entity_dia_map') else None,
            ablation_config=ablation_config
        )

        # Initialize evaluator (separate from memory and query)
        evaluator = Evaluator(
            llm_controller=builder.llm_controller,
            use_llm_judge=True
        )

        # Initialize test harness with evaluator
        tester = TestHarness(builder, query_engine, evaluator=evaluator)
        tester.debug_search = args.debug_search

        # Setup best-of-N if requested
        if args.best_of_n > 1:
            from memory.best_of_n_selector import BestOfNSelector
            # print(f"\n✓ Using Best-of-{args.best_of_n} selection (method: {args.best_of_n_method})")
            tester.best_of_n = args.best_of_n
            tester.best_of_n_method = args.best_of_n_method
            tester.best_of_n_selector = BestOfNSelector(
                n_attempts=args.best_of_n,
                selection_method=args.best_of_n_method
            )

        # Filter questions based on categories to test
        original_count = len(sample.qa)
        sample.qa = [qa for qa in sample.qa if qa.category in categories_to_test]
        filtered_count = len(sample.qa)

        if filtered_count < original_count:
            print(f"\nFiltered questions: {original_count} → {filtered_count}")
            print(f"Testing categories: {sorted(categories_to_test)}\n")
        else:
            print(f"\nTesting all {filtered_count} questions\n")

        # Test questions (parallel is now default)
        if args.parallel:
            results = tester.test_questions_parallel(sample, args.max_questions, n_workers=args.n_workers)
        else:
            results = tester.test_questions(sample, args.max_questions)

        # Calculate results for this sample
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        not_found = sum(1 for r in results if r['predicted'] == "Information not found")

        # Calculate average scores
        f1_scores = [r['metrics']['f1'] for r in results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r['llm_judge_score'] for r in results if r.get('llm_judge_score', 0) >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        # Calculate scores WITHOUT Category 5
        results_no_cat5 = [r for r in results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r['llm_judge_score'] for r in results_no_cat5 if r.get('llm_judge_score', 0) >= 0) / total_no_cat5 * 100
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        # Calculate category-wise breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if r.get('llm_judge_score', 0) >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        # Print results for this sample
        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        # Print category breakdown
        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        # Save per-sample results with model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Create model-specific results directory
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/fixed_results_sample{sample_id}{embedding_suffix}.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'results': results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown,
                    'memory_stats': mem_stats
                }
            }, f, indent=2, default=str)

        print(f"Results saved to {output_file}")

        if args.debug_search:
            debug_output_file = f"{results_dir}/fixed_results_sample{sample_id}{embedding_suffix}_debug.json"
            debug_payload = {
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'debug_search': True,
                'results': [
                    {
                        'question_id': r.get('question_id'),
                        'question': r.get('question'),
                        'category': r.get('category'),
                        'expected': r.get('expected'),
                        'predicted': r.get('predicted'),
                        'llm_judge_score': r.get('llm_judge_score'),
                        'processing_time': r.get('processing_time'),
                        'search_details': r.get('search_details', {}),
                        'answer_context': r.get('answer_context')
                    }
                    for r in results
                ]
            }
            with open(debug_output_file, 'w') as f:
                json.dump(debug_payload, f, indent=2, default=str)
            print(f"Debug search trace saved to {debug_output_file}")

        # Store for aggregation
        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'category_breakdown': category_breakdown
        })

    # Print summary if multiple samples
    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        # Calculate averages
        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        # Aggregate category breakdown across all samples
        print(f"\n{'-'*70}")
        print(f"AGGREGATE RESULTS BY CATEGORY (Average across {len(args.sample)} samples):")

        # Collect all categories across all samples
        all_categories = set()
        for r in all_sample_results:
            all_categories.update(r['category_breakdown'].keys())

        # Calculate average stats per category
        aggregate_category_stats = {}
        for cat_key in sorted(all_categories, key=lambda x: int(x.split('_')[1])):
            cat_num = int(cat_key.split('_')[1])
            total_samples_with_cat = 0
            sum_total = sum_correct = sum_acc = sum_f1 = sum_bleu = sum_llm = 0

            for r in all_sample_results:
                if cat_key in r['category_breakdown']:
                    cat_data = r['category_breakdown'][cat_key]
                    total_samples_with_cat += 1
                    sum_total += cat_data['total']
                    sum_correct += cat_data['correct']
                    sum_acc += cat_data['accuracy']
                    sum_f1 += cat_data['avg_f1']
                    sum_bleu += cat_data['avg_bleu1']
                    sum_llm += cat_data['avg_llm']

            if total_samples_with_cat > 0:
                aggregate_category_stats[cat_num] = {
                    'avg_total': sum_total / total_samples_with_cat,
                    'avg_correct': sum_correct / total_samples_with_cat,
                    'avg_accuracy': sum_acc / total_samples_with_cat,
                    'avg_f1': sum_f1 / total_samples_with_cat,
                    'avg_bleu1': sum_bleu / total_samples_with_cat,
                    'avg_llm': sum_llm / total_samples_with_cat,
                    'samples_count': total_samples_with_cat
                }

        print(f"  {'Cat':<5} {'Avg Total':<10} {'Avg Corr':<10} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7} {'#Samples':<9}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
        for cat in sorted(aggregate_category_stats.keys()):
            stats = aggregate_category_stats[cat]
            print(f"  {cat:<5} {stats['avg_total']:<10.1f} {stats['avg_correct']:<10.1f} "
                  f"{stats['avg_accuracy']:<7.1f} {stats['avg_f1']:<7.1f} {stats['avg_bleu1']:<7.1f} "
                  f"{stats['avg_llm']:<7.1f} {stats['samples_count']:<9}")

        # Save aggregate results in model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Use same model-specific results directory
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/fixed_results_aggregate_samples_{'_'.join(map(str, args.sample))}{embedding_suffix}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'samples': args.sample,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5,
                    'category_breakdown': aggregate_category_stats
                }
            }, f, indent=2)
        print(f"\nAggregate results saved to {aggregate_output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
