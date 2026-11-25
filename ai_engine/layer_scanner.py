from __future__ import annotations

from typing import Any
import torch

try:
    from ai_engine.model_loader import get_decoder_layers
except ImportError:
    from model_loader import get_decoder_layers


def _forward_with_optional_logits_to_keep(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> Any:
    """Run model forward pass, requesting logits_to_keep=1 if supported."""
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    except TypeError:
        return model(input_ids=input_ids, attention_mask=attention_mask)


@torch.no_grad()
def compute_layer_effects(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[float], int, float]:
    """Compute layer causal effects following Definition 3 of the LLMScan paper.

    Definition 3 (Layer Causal Effect):
        CE_l = |logit_0 - logit_0^(-l)|
    measures the change in the first generated token's logit when decoder layer l
    is skipped (bypassed via shortcut from layer l-1 to l+1).

    Args:
        model: Hugging Face causal language model.
        input_ids: Input token tensor of shape (batch_size, seq_len).
        attention_mask: Optional attention mask tensor of shape (batch_size, seq_len).

    Returns:
        tuple[list[float], int, float]:
            - layer_effects: List of causal effect values for each layer.
            - first_token_id: Predicted next-token ID from the unperturbed run.
            - first_token_logit: Logit value for first_token_id from the unperturbed run.
    """
    model.eval()

    # Normal run: unperturbed model forward pass
    normal_outputs = _forward_with_optional_logits_to_keep(
        model, input_ids=input_ids, attention_mask=attention_mask
    )
    normal_logits = normal_outputs.logits[0, -1]
    first_token_id = int(torch.argmax(normal_logits).item())
    first_token_logit = float(normal_logits[first_token_id].item())

    layers = get_decoder_layers(model)
    layer_effects: list[float] = []

    for layer in layers:
        def bypass_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            # Shortcut: pass input hidden states directly to bypass layer l
            input_hidden = args[0] if len(args) > 0 else kwargs["hidden_states"]
            if isinstance(output, tuple):
                return (input_hidden,) + tuple(output[1:])
            return input_hidden

        handle = layer.register_forward_hook(bypass_hook, with_kwargs=True)
        try:
            skipped_outputs = _forward_with_optional_logits_to_keep(
                model, input_ids=input_ids, attention_mask=attention_mask
            )
            skipped_logits = skipped_outputs.logits[0, -1]
            skipped_logit = float(skipped_logits[first_token_id].item())
            effect = abs(first_token_logit - skipped_logit)
            layer_effects.append(effect)
        finally:
            handle.remove()

    return layer_effects, first_token_id, first_token_logit
