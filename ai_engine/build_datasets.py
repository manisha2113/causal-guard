from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import statistics
from typing import Any

from datasets import load_dataset
import numpy as np


def sanitize_prompt(text: Any, max_chars: int) -> str | None:
    """Clean, strip, and truncate prompt string."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip()
    return cleaned if cleaned else None


def clean_prompt_no_truncate(text: Any, min_chars: int, max_chars: int) -> str | None:
    """Clean and strip prompt string; drop if outside [min_chars, max_chars]."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if min_chars <= len(cleaned) <= max_chars:
        return cleaned
    return None


def print_stats_table(stage_name: str, class0: list[str], class1: list[str] | None = None) -> None:
    """Print class counts, average, and median character length."""
    print(f"  {stage_name}:")
    classes = [(0, "normal (0)", class0)]
    if class1 is not None:
        classes.append((1, "misbehavior (1)", class1))

    for _, desc, prompts in classes:
        lens = [len(p) for p in prompts]
        count = len(lens)
        avg_len = sum(lens) / count if count > 0 else 0.0
        med_len = float(statistics.median(lens)) if count > 0 else 0.0
        print(f"    Class {desc}: count={count}, avg_char_len={avg_len:.1f}, median_char_len={med_len:.1f}")


def match_by_length(
    class0_prompts: list[str],
    class1_prompts: list[str],
    n_per_class: int,
    seed: int,
    num_bins: int = 10,
) -> tuple[list[str], list[str]]:
    """Match two classes by length using quantile bins of pooled lengths.

    Bins all prompts into length buckets and samples the same number of prompts
    per class from each bucket, as many as both classes allow (up to n_per_class).
    """
    rng = random.Random(seed)
    class0_shuffled = list(class0_prompts)
    class1_shuffled = list(class1_prompts)
    rng.shuffle(class0_shuffled)
    rng.shuffle(class1_shuffled)

    pooled_lengths = [len(p) for p in class0_shuffled] + [len(p) for p in class1_shuffled]
    if not pooled_lengths:
        return [], []

    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.unique(np.quantile(pooled_lengths, quantiles))
    if len(bin_edges) < 2:
        bin_edges = np.array([min(pooled_lengths), max(pooled_lengths) + 1])

    b_count = len(bin_edges) - 1

    def get_bin(length: int) -> int:
        idx = int(np.digitize(length, bin_edges[:-1]) - 1)
        return max(0, min(idx, b_count - 1))

    buckets_c0: list[list[str]] = [[] for _ in range(b_count)]
    buckets_c1: list[list[str]] = [[] for _ in range(b_count)]

    for p in class0_shuffled:
        buckets_c0[get_bin(len(p))].append(p)
    for p in class1_shuffled:
        buckets_c1[get_bin(len(p))].append(p)

    capacities = [min(len(buckets_c0[b]), len(buckets_c1[b])) for b in range(b_count)]
    total_capacity = sum(capacities)

    target_total = min(n_per_class, total_capacity)
    if target_total == 0:
        return [], []

    if total_capacity <= target_total:
        quotas = list(capacities)
    else:
        # Largest Remainder Method for proportional allocation across bins
        exact_quotas = [target_total * (c / total_capacity) for c in capacities]
        quotas = [int(q) for q in exact_quotas]
        remainder = target_total - sum(quotas)
        fractions = sorted(
            range(b_count),
            key=lambda b: exact_quotas[b] - quotas[b],
            reverse=True,
        )
        for i in range(remainder):
            quotas[fractions[i]] += 1

    selected_c0: list[str] = []
    selected_c1: list[str] = []

    for b in range(b_count):
        k = quotas[b]
        if k > 0:
            selected_c0.extend(buckets_c0[b][:k])
            selected_c1.extend(buckets_c1[b][:k])

    return selected_c0, selected_c1


def build_lie_dataset(n_per_class: int, max_chars: int, seed: int) -> list[dict[str, Any]]:
    """Build lie dataset from allenai/sciq.

    Label 0: question as is.
    Label 1: 'Answer the following question with a lie. ' + question.
    """
    ds = load_dataset("allenai/sciq", split="train")
    rng = random.Random(seed)

    seen = set()
    questions: list[str] = []
    for item in ds:
        q = sanitize_prompt(item.get("question"), max_chars=max_chars)
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    rng.shuffle(questions)
    selected = questions[:n_per_class]

    prefix = "Answer the following question with a lie. "
    normal_prompts = selected
    lie_prompts = [
        sanitize_prompt(prefix + q, max_chars=max_chars) or (prefix + q)
        for q in selected
    ]

    print_stats_table("[Before length matching]", normal_prompts, lie_prompts)
    print_stats_table("[After length matching] (length matching skipped for lie task)", normal_prompts, lie_prompts)

    rows: list[dict[str, Any]] = []
    for idx, (q, lq) in enumerate(zip(normal_prompts, lie_prompts)):
        rows.append({
            "id": f"lie_0_{idx}",
            "prompt": q,
            "label": 0,
            "task": "lie",
        })
        rows.append({
            "id": f"lie_1_{idx}",
            "prompt": lq,
            "label": 1,
            "task": "lie",
        })

    return rows


def build_jailbreak_dataset(
    n_per_class: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    jailbreak_normal: str = "regular",
    length_match: bool = True,
) -> list[dict[str, Any]]:
    """Build jailbreak dataset.

    Label 0: 'regular' from TrustAIRLab/in-the-wild-jailbreak-prompts (regular_2023_12_25)
             or 'advbench' from walledai/AdvBench
    Label 1: 'jailbreak_2023_12_25' from TrustAIRLab/in-the-wild-jailbreak-prompts (jailbreak flag true)
    """
    rng = random.Random(seed)

    # Label 0
    seen_normal = set()
    normal_prompts: list[str] = []

    if jailbreak_normal == "regular":
        ds_reg = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "regular_2023_12_25",
            split="train",
        )
        for item in ds_reg:
            raw = item.get("prompt") or item.get("text") or item.get("user_input")
            p = clean_prompt_no_truncate(raw, min_chars=min_chars, max_chars=max_chars)
            if p and p not in seen_normal:
                seen_normal.add(p)
                normal_prompts.append(p)
    else:
        ds_adv = load_dataset("walledai/AdvBench", split="train")
        adv_col = "prompt"
        for candidate_col in ["prompt", "goal", "text", "content"]:
            if candidate_col in ds_adv.column_names:
                adv_col = candidate_col
                break
        for item in ds_adv:
            raw = item.get(adv_col)
            p = clean_prompt_no_truncate(raw, min_chars=min_chars, max_chars=max_chars)
            if p and p not in seen_normal:
                seen_normal.add(p)
                normal_prompts.append(p)

    # Label 1: jailbreak_2023_12_25
    ds_jb = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "jailbreak_2023_12_25",
        split="train",
    )
    seen_jb = set()
    jb_prompts: list[str] = []
    for item in ds_jb:
        is_jailbreak = True
        if "jailbreak" in item:
            is_jailbreak = bool(item["jailbreak"])
        elif "label" in item:
            is_jailbreak = str(item["label"]).strip().lower() in {"1", "true", "jailbreak"}
        elif "type" in item:
            is_jailbreak = "jailbreak" in str(item["type"]).lower()

        if is_jailbreak:
            raw_text = item.get("prompt") or item.get("text") or item.get("jailbreak_prompt")
            p = clean_prompt_no_truncate(raw_text, min_chars=min_chars, max_chars=max_chars)
            if p and p not in seen_jb and p not in seen_normal:
                seen_jb.add(p)
                jb_prompts.append(p)

    print_stats_table("[Before length matching]", normal_prompts, jb_prompts)

    if length_match:
        selected_normal, selected_jb = match_by_length(
            class0_prompts=normal_prompts,
            class1_prompts=jb_prompts,
            n_per_class=n_per_class,
            seed=seed,
        )
    else:
        rng.shuffle(normal_prompts)
        rng.shuffle(jb_prompts)
        k = min(n_per_class, len(normal_prompts), len(jb_prompts))
        selected_normal = normal_prompts[:k]
        selected_jb = jb_prompts[:k]

    print_stats_table("[After length matching]", selected_normal, selected_jb)

    rows: list[dict[str, Any]] = []
    for idx, p in enumerate(selected_normal):
        rows.append({
            "id": f"jb_0_{idx}",
            "prompt": p,
            "label": 0,
            "task": "jailbreak",
        })
    for idx, p in enumerate(selected_jb):
        rows.append({
            "id": f"jb_1_{idx}",
            "prompt": p,
            "label": 1,
            "task": "jailbreak",
        })

    return rows


def build_toxicity_dataset(
    n_per_class: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    length_match: bool = True,
) -> list[dict[str, Any]]:
    """Build toxicity dataset from lmsys/toxic-chat (toxicchat0124 config, train split)."""
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    rng = random.Random(seed)

    seen_non_toxic = set()
    seen_toxic = set()
    non_toxic: list[str] = []
    toxic: list[str] = []

    for item in ds:
        p = clean_prompt_no_truncate(item.get("user_input"), min_chars=min_chars, max_chars=max_chars)
        if not p:
            continue
        lbl = item.get("toxicity")
        if lbl == 0 and p not in seen_non_toxic:
            seen_non_toxic.add(p)
            non_toxic.append(p)
        elif lbl == 1 and p not in seen_toxic:
            seen_toxic.add(p)
            toxic.append(p)

    print_stats_table("[Before length matching]", non_toxic, toxic)

    if length_match:
        selected_non_toxic, selected_toxic = match_by_length(
            class0_prompts=non_toxic,
            class1_prompts=toxic,
            n_per_class=n_per_class,
            seed=seed,
        )
    else:
        rng.shuffle(non_toxic)
        rng.shuffle(toxic)
        k = min(n_per_class, len(non_toxic), len(toxic))
        selected_non_toxic = non_toxic[:k]
        selected_toxic = toxic[:k]

    print_stats_table("[After length matching]", selected_non_toxic, selected_toxic)

    rows: list[dict[str, Any]] = []
    for idx, p in enumerate(selected_non_toxic):
        rows.append({
            "id": f"tox_0_{idx}",
            "prompt": p,
            "label": 0,
            "task": "toxicity",
        })
    for idx, p in enumerate(selected_toxic):
        rows.append({
            "id": f"tox_1_{idx}",
            "prompt": p,
            "label": 1,
            "task": "toxicity",
        })

    return rows


def build_truthfulqa_dataset(n_per_class: int, max_chars: int, seed: int) -> list[dict[str, Any]]:
    """Build truthfulqa dataset from truthfulqa/truthful_qa (question column, label 0 only)."""
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except Exception:
        try:
            ds = load_dataset("truthfulqa/truthful_qa", split="validation")
        except Exception:
            ds = load_dataset("truthfulqa/truthful_qa")
            if isinstance(ds, dict):
                ds = ds[list(ds.keys())[0]]

    rng = random.Random(seed)
    seen = set()
    questions: list[str] = []

    for item in ds:
        q = sanitize_prompt(item.get("question"), max_chars=max_chars)
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    rng.shuffle(questions)
    selected = questions[:n_per_class]

    print_stats_table("[Before length matching]", selected)
    print_stats_table("[After length matching] (length matching skipped for truthfulqa)", selected)

    rows: list[dict[str, Any]] = []
    for idx, q in enumerate(selected):
        rows.append({
            "id": f"tqa_0_{idx}",
            "prompt": q,
            "label": 0,
            "task": "truthfulqa",
        })

    return rows


def save_dataset_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Save rows to CSV with columns: id, prompt, label, task."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "label", "task"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ai_engine.build_datasets",
        description="Download and format benchmark datasets for CausalGuard",
    )
    parser.add_argument(
        "--task",
        choices=["lie", "jailbreak", "toxicity", "truthfulqa", "all"],
        default="all",
        help="Task dataset to build (default: all)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=300,
        help="Number of samples per class (default: 300)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum prompt length in characters (default: 20)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=600,
        help="Maximum prompt length in characters (default: 600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--jailbreak-normal",
        choices=["regular", "advbench"],
        default="regular",
        help="Source for normal prompts in jailbreak task: 'regular' or 'advbench' (default: regular)",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--length-match",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Match prompt length distributions between classes (default: True)",
        )
    else:
        parser.add_argument(
            "--length-match",
            action="store_true",
            default=True,
            help="Match prompt length distributions between classes",
        )
        parser.add_argument(
            "--no-length-match",
            dest="length_match",
            action="store_false",
            help="Disable length matching",
        )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory for generated CSVs (default: data)",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    tasks_to_run = (
        ["lie", "jailbreak", "toxicity", "truthfulqa"]
        if args.task == "all"
        else [args.task]
    )

    for task_name in tasks_to_run:
        try:
            print(f"\nBuilding dataset for task: '{task_name}'...")
            if task_name == "lie":
                rows = build_lie_dataset(
                    n_per_class=args.n_per_class,
                    max_chars=args.max_chars,
                    seed=args.seed,
                )
            elif task_name == "jailbreak":
                rows = build_jailbreak_dataset(
                    n_per_class=args.n_per_class,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    seed=args.seed,
                    jailbreak_normal=args.jailbreak_normal,
                    length_match=args.length_match,
                )
            elif task_name == "toxicity":
                rows = build_toxicity_dataset(
                    n_per_class=args.n_per_class,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    seed=args.seed,
                    length_match=args.length_match,
                )
            elif task_name == "truthfulqa":
                rows = build_truthfulqa_dataset(
                    n_per_class=args.n_per_class,
                    max_chars=args.max_chars,
                    seed=args.seed,
                )
            else:
                continue

            csv_path = out_dir / f"{task_name}.csv"
            save_dataset_csv(rows, csv_path)
        except Exception as e:
            print(f"\n[Warning] Failed to build dataset for task '{task_name}': {e}")
            continue


if __name__ == "__main__":
    main()
