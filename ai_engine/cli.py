from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ai_engine.scanner import CausalScanner
    from ai_engine.causal_map import CausalMap
except ImportError:
    from scanner import CausalScanner
    from causal_map import CausalMap


def run_scan(
    prompt: str,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    out: str | None = None,
    use_chat_template: bool = True,
    quantize: str = "none",
    scanner: CausalScanner | None = None,
) -> CausalMap:
    """Execute scan on a single prompt and print diagnostic information."""
    if scanner is None:
        scanner = CausalScanner(
            model_name=model_name,
            use_chat_template=use_chat_template,
            quantize=quantize,
        )

    t0 = time.perf_counter()
    causal_map = scanner.scan(prompt)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)

    print("\nToken Causal Effects:")
    print(f"{'Idx':>4} | {'Token':<22} | {'Causal Effect':>14}")
    print("-" * 46)
    for idx, (tok, eff) in enumerate(zip(causal_map.tokens, causal_map.token_effects)):
        print(f"{idx:>4} | {tok:<22} | {eff:>14.6f}")

    print("\nLayer Causal Effects:")
    print(f"{'Layer':>6} | {'Causal Effect':>15}")
    print("-" * 25)
    for layer_idx, eff in enumerate(causal_map.layer_effects):
        print(f"{layer_idx:>6} | {eff:>15.6f}")

    if causal_map.features is not None:
        print("\nToken Statistics (5 moments):")
        stat_names = ["mean", "std", "range", "skewness", "kurtosis"]
        for name, val in zip(stat_names, causal_map.features["token_features"]):
            print(f"  {name:>8}: {val:.6f}")

    print(f"\nTime taken: {elapsed:.4f}s")

    out_file = out if out is not None else "result.json"
    causal_map.save_json(out_file)
    print(f"Saved CausalMap JSON to {out_file}")

    return causal_map


def run_batch(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_chat_template: bool = True,
    limit: int | None = None,
    quantize: str = "none",
    scanner: CausalScanner | None = None,
) -> None:
    """Run batch scanning across a CSV file with resume and error recovery support."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    errors_file = Path(str(output_path) + ".errors.csv")
    tokens_file = Path(str(output_path) + ".tokens.jsonl")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing IDs for resume functionality
    processed_ids: set[str] = set()
    header_written = False
    existing_has_quantize = True
    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and "id" in header:
                header_written = True
                id_idx = header.index("id")
                if "quantize" not in header:
                    print(
                        f"[Warning] Existing output file '{output_file}' does not contain a 'quantize' column."
                    )
                    existing_has_quantize = False
                for row in reader:
                    if row and len(row) > id_idx:
                        processed_ids.add(row[id_idx])

    errors_header_written = errors_file.exists() and errors_file.stat().st_size > 0

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter out already processed rows
    pending_rows = [r for r in all_rows if str(r["id"]) not in processed_ids]
    if limit is not None:
        pending_rows = pending_rows[:limit]

    if not pending_rows:
        print("No new rows to process (all rows completed or limit reached).")
        return

    if scanner is None:
        scanner = CausalScanner(
            model_name=model_name,
            use_chat_template=use_chat_template,
            quantize=quantize,
        )

    pbar = tqdm(pending_rows, desc="Batch scanning")
    for row in pbar:
        row_id = str(row["id"])
        prompt = row["prompt"]
        label = row.get("label", "")
        task = row.get("task", "")

        t0 = time.perf_counter()
        try:
            causal_map = scanner.scan(prompt)
            elapsed = time.perf_counter() - t0
            pbar.set_postfix(id=row_id, time=f"{elapsed:.2f}s")

            assert causal_map.features is not None
            feature_names = causal_map.features["feature_names"]
            feature_vector = causal_map.features["vector"]

            # Output CSV columns: id, task, label, model_name, quantize, feature_names..., n_tokens, first_token_text, seconds
            if existing_has_quantize:
                full_header = (
                    ["id", "task", "label", "model_name", "quantize"]
                    + feature_names
                    + ["n_tokens", "first_token_text", "seconds"]
                )
                row_values = (
                    [row_id, task, label, causal_map.model_name, scanner.quantize]
                    + feature_vector
                    + [len(causal_map.tokens), causal_map.first_token_text, f"{elapsed:.4f}"]
                )
            else:
                full_header = (
                    ["id", "task", "label", "model_name"]
                    + feature_names
                    + ["n_tokens", "first_token_text", "seconds"]
                )
                row_values = (
                    [row_id, task, label, causal_map.model_name]
                    + feature_vector
                    + [len(causal_map.tokens), causal_map.first_token_text, f"{elapsed:.4f}"]
                )

            with open(output_file, "a", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                if not header_written:
                    writer.writerow(full_header)
                    header_written = True
                writer.writerow(row_values)
                f_out.flush()

            # Append full token effects to JSONL
            token_record = {
                "id": row_id,
                "tokens": causal_map.tokens,
                "token_effects": causal_map.token_effects,
                "layer_effects": causal_map.layer_effects,
            }
            with open(tokens_file, "a", encoding="utf-8") as f_tok:
                f_tok.write(json.dumps(token_record, ensure_ascii=False) + "\n")
                f_tok.flush()

            processed_ids.add(row_id)

        except Exception as e:
            print(f"\n[Error] ID {row_id}: {e}")
            with open(errors_file, "a", newline="", encoding="utf-8") as f_err:
                err_writer = csv.writer(f_err)
                if not errors_header_written:
                    err_writer.writerow(["id", "error"])
                    errors_header_written = True
                err_writer.writerow([row_id, str(e)])
                f_err.flush()
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ai_engine.cli",
        description="CausalGuard CLI for LLM attribution scanning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan subcommand
    scan_p = subparsers.add_parser("scan", help="Scan a single prompt")
    scan_p.add_argument("--prompt", required=True, type=str, help="Prompt text to scan")
    scan_p.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or local path",
    )
    scan_p.add_argument(
        "--quantize",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Model quantization (none, 8bit, 4bit)",
    )
    scan_p.add_argument("--out", type=str, default="result.json", help="Path to save CausalMap JSON")
    scan_p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not apply chat template",
    )

    # batch subcommand
    batch_p = subparsers.add_parser("batch", help="Run batch scanning over a CSV file")
    batch_p.add_argument("--input", required=True, type=str, help="Input CSV path")
    batch_p.add_argument("--output", required=True, type=str, help="Output CSV path")
    batch_p.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or local path",
    )
    batch_p.add_argument(
        "--quantize",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Model quantization (none, 8bit, 4bit)",
    )
    batch_p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not apply chat template",
    )
    batch_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process",
    )

    args = parser.parse_args()

    if args.command == "scan":
        run_scan(
            prompt=args.prompt,
            model_name=args.model,
            out=args.out,
            use_chat_template=not args.no_chat_template,
            quantize=args.quantize,
        )
    elif args.command == "batch":
        run_batch(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            use_chat_template=not args.no_chat_template,
            limit=args.limit,
            quantize=args.quantize,
        )


if __name__ == "__main__":
    main()
