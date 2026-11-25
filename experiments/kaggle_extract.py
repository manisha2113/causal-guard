from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

# Ensure repository root is on sys.path so script can be run directly or from Kaggle cells
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ai_engine.cli import run_batch
from ai_engine.scanner import CausalScanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python experiments/kaggle_extract.py",
        description="Run batch causal attribution extraction on Kaggle",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode (none, 4bit, 8bit)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing task dataset CSVs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save extracted feature CSVs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["lie", "jailbreak", "toxicity", "truthfulqa"],
        help="Tasks to process (default: lie jailbreak toxicity truthfulqa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process per task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_short_name = args.model.rstrip("/").split("/")[-1].lower()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 60)
    print("CausalGuard Kaggle Batch Extraction")
    print("=" * 60)
    print(f"Model:           {args.model}")
    print(f"Short Name:      {model_short_name}")
    print(f"Quantize:        {args.quantize}")
    print(f"Compute Device:  {device_name}")
    print(f"Data Directory:  {data_dir}")
    print(f"Out Directory:   {out_dir}")
    print(f"Tasks:           {args.tasks}")
    if args.limit is not None:
        print(f"Row Limit:       {args.limit}")
    print("=" * 60)

    # Initialize scanner once to avoid reloading model across tasks
    print("\nLoading model and initializing CausalScanner...")
    scanner = CausalScanner(
        model_name=args.model,
        quantize=args.quantize,
    )

    for task in args.tasks:
        input_csv = data_dir / f"{task}.csv"
        if not input_csv.exists():
            print(f"\n[Warning] Input CSV not found for task '{task}': {input_csv}. Skipping.")
            continue

        output_csv = out_dir / f"features_{model_short_name}_{task}.csv"
        print(f"\nProcessing task '{task}':")
        print(f"  Input:  {input_csv}")
        print(f"  Output: {output_csv}")

        run_batch(
            input_path=input_csv,
            output_path=output_csv,
            model_name=args.model,
            quantize=args.quantize,
            limit=args.limit,
            scanner=scanner,
        )

    # Summary table of rows done per task
    summary: list[tuple[str, int, str]] = []
    for task in args.tasks:
        output_csv = out_dir / f"features_{model_short_name}_{task}.csv"
        if output_csv.exists():
            with open(output_csv, "r", encoding="utf-8") as f:
                lines = sum(1 for line in f if line.strip())
                row_count = max(0, lines - 1)  # Exclude header
            status = "Completed"
        else:
            row_count = 0
            status = "Not Found"
        summary.append((task, row_count, status))

    print("\n" + "=" * 50)
    print("Batch Extraction Summary")
    print("=" * 50)
    print(f"{'Task':<15} | {'Rows Done':>10} | {'Status':<12}")
    print("-" * 50)
    for task, count, status in summary:
        print(f"{task:<15} | {count:>10} | {status:<12}")
    print("=" * 50)


if __name__ == "__main__":
    main()
