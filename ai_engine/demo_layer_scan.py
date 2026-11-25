from __future__ import annotations

import sys
from pathlib import Path
import torch

# Ensure parent directory is in sys.path when script is executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ai_engine.model_loader import load_model
    from ai_engine.layer_scanner import compute_layer_effects
except ImportError:
    from model_loader import load_model
    from layer_scanner import compute_layer_effects


def main() -> None:
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()

    raw_prompt = "Who developed Windows 95?"
    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = raw_prompt
    else:
        text = raw_prompt

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    num_tokens = input_ids.shape[1]
    print(f"\nPrompt: {raw_prompt}")
    print(f"Number of tokens: {num_tokens}")

    # First run of compute_layer_effects
    print("\nRunning layer scan (first pass)...")
    layer_effects, first_token_id, first_token_logit = compute_layer_effects(
        model, input_ids=input_ids, attention_mask=attention_mask
    )
    first_token_text = tokenizer.decode([first_token_id])

    print(f"First predicted token ID: {first_token_id}")
    print(f"First predicted token text: {repr(first_token_text)}")
    print(f"First predicted token logit: {first_token_logit:.4f}")

    print("\nLayer Causal Effects (Definition 3):")
    print(f"{'Layer':>6} | {'Causal Effect':>15}")
    print("-" * 25)
    for idx, effect in enumerate(layer_effects):
        print(f"{idx:>6} | {effect:>15.6f}")

    # Second run to verify hook removal
    print("\nRunning layer scan (second pass)...")
    layer_effects_2, first_token_id_2, first_token_logit_2 = compute_layer_effects(
        model, input_ids=input_ids, attention_mask=attention_mask
    )

    with torch.no_grad():
        plain_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        plain_logits = plain_outputs.logits[0, -1]
        plain_first_token_logit = float(plain_logits[first_token_id_2].item())
        are_identical = torch.allclose(
            plain_logits,
            model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1],
        ) and (first_token_logit_2 == plain_first_token_logit)

    print("\nHook removal check:")
    print(f"  Second run logit: {first_token_logit_2:.6f}")
    print(f"  Plain pass logit: {plain_first_token_logit:.6f}")
    print(f"  Are normal logits identical to plain forward pass? {are_identical}")


if __name__ == "__main__":
    main()
