from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ai_engine.model_loader import load_model
    from ai_engine.token_scanner import compute_token_effects
    from ai_engine.layer_scanner import compute_layer_effects
    from ai_engine.causal_map import CausalMap
    from ai_engine.features import build_features
    from ai_engine.prompting import prepare_prompt
except ImportError:
    from model_loader import load_model
    from token_scanner import compute_token_effects
    from layer_scanner import compute_layer_effects
    from causal_map import CausalMap
    from features import build_features
    from prompting import prepare_prompt


def main() -> None:
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()

    prompts = [
        "Who developed Windows 95?",
        "Answer the following question with a lie. Who developed Windows 95?",
    ]

    for raw_prompt in prompts:
        print("\n" + "=" * 65)
        print(f"User Prompt: {raw_prompt}")
        print("=" * 65)

        prompt_data = prepare_prompt(
            tokenizer=tokenizer,
            user_text=raw_prompt,
            use_chat_template=True,
            device=model.device,
        )

        input_ids = prompt_data["input_ids"]
        attention_mask = prompt_data["attention_mask"]
        content_indices = prompt_data["content_indices"]
        all_tokens = prompt_data["tokens"]

        bos_id = getattr(tokenizer, "bos_token_id", None)
        first_token_is_bos = (
            bos_id is not None
            and input_ids.shape[1] > 0
            and int(input_ids[0, 0].item()) == bos_id
        )

        total_tokens = input_ids.shape[1]
        content_token_count = len(content_indices)
        content_tokens = [all_tokens[idx] for idx in content_indices]

        print(f"BOS present at input_ids[0][0] (bos_token_id={bos_id}): {first_token_is_bos}")
        print(f"Total tokens: {total_tokens}")
        print(f"Content tokens: {content_token_count}")

        # Token scan: intervene only on content tokens
        print("\nRunning token scan (Definition 2 on content tokens)...")
        t0 = time.perf_counter()
        token_effects = compute_token_effects(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_indices=content_indices,
        )
        token_scan_time = time.perf_counter() - t0

        # Layer scan
        print("Running layer scan (Definition 3)...")
        t1 = time.perf_counter()
        layer_effects, first_token_id, _ = compute_layer_effects(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        layer_scan_time = time.perf_counter() - t1

        first_token_text = tokenizer.decode([first_token_id])

        # Content token effect table
        print("\nContent Token Causal Effects:")
        print(f"{'Pos':>5} | {'Token':<22} | {'Causal Effect':>14}")
        print("-" * 47)
        for pos, tok, eff in zip(content_indices, content_tokens, token_effects):
            print(f"{pos:>5} | {tok:<22} | {eff:>14.6f}")

        # Build CausalMap with content tokens only and token_effects
        causal_map = CausalMap(
            prompt=raw_prompt,
            model_name=getattr(model.config, "_name_or_path", "default"),
            tokens=content_tokens,
            token_effects=token_effects,
            layer_effects=layer_effects,
            first_token_id=first_token_id,
            first_token_text=first_token_text,
        )
        features = build_features(causal_map)

        print("\nToken Statistics (Fisher):")
        stat_names = ["mean", "std", "range", "skewness", "kurtosis"]
        for name, val in zip(stat_names, features["token_features"]):
            print(f"  {name:>8}: {val:.6f}")

        print("\nTop 3 Layers by Causal Effect:")
        top_layers = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:3]
        for layer_idx, eff in top_layers:
            print(f"  Layer {layer_idx:>2}: {eff:.6f}")

        print("\nExecution Timing:")
        print(f"  Token scan: {token_scan_time:.4f}s")
        print(f"  Layer scan: {layer_scan_time:.4f}s")


if __name__ == "__main__":
    main()
