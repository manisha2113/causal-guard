from __future__ import annotations

from typing import Any
import torch

try:
    from ai_engine.model_loader import get_decoder_layers
except ImportError:
    from model_loader import get_decoder_layers


def get_intervention_token_id(tokenizer: Any) -> int:
    """Retrieve the token ID for the intervention token '-' without special tokens.

    Args:
        tokenizer: Hugging Face tokenizer.

    Returns:
        int: The token ID for '-'.

    Raises:
        ValueError: If encoding '-' produces no tokens.
    """
    ids = tokenizer.encode("-", add_special_tokens=False)
    if not ids:
        raise ValueError("Failed to encode intervention token '-' with tokenizer.")
    return int(ids[-1])


@torch.no_grad()
def compute_token_effects(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    aggregation: str = "mean",
    token_indices: list[int] | None = None,
    chunk_size: int = 8,
) -> list[float]:
    """Compute token causal effects following Definition 2 of the LLMScan paper.

    Definition 2 (Token Causal Effect):
        CE_xi = || AS - AS'_xi ||
    measures the Frobenius (L2) distance between attention scores of the original
    prompt (AS) and the attention scores (AS'_xi) when token xi is replaced by '-'.

    Args:
        model: Hugging Face causal language model loaded with eager attention.
        tokenizer: Hugging Face tokenizer.
        input_ids: Input token tensor of shape (1, seq_len) or (seq_len,).
        attention_mask: Optional attention mask tensor.
        layers: Indices of decoder layers to evaluate. Defaults to [0, (L-1)//2, L-1].
        heads: Indices of attention heads to evaluate. Defaults to [0, (H-1)//2, H-1].
        aggregation: Method to aggregate layer-head norms into a scalar ('mean', 'sum', 'max').
        token_indices: Token positions to intervene on. Defaults to all positions.
        chunk_size: Batch size of intervened prompts evaluated per forward pass.

    Returns:
        list[float]: Token causal effects for each position in token_indices.

    Raises:
        ValueError: If attention implementation is not 'eager' or invalid aggregation is given.
    """
    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl != "eager":
        raise ValueError(
            f"Model attention implementation must be 'eager' to extract attention weights, "
            f"but found '{attn_impl}'. Load model with attn_implementation='eager'."
        )

    model.eval()

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    seq_len = input_ids.shape[1]

    # Resolve default layers: first, middle, last [0, (L-1)//2, L-1]
    if layers is None:
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is None:
            try:
                num_layers = len(get_decoder_layers(model))
            except Exception:
                num_layers = len(model.model.layers)
        layers = [0, (num_layers - 1) // 2, num_layers - 1]

    # Resolve default heads: first, middle, last [0, (H-1)//2, H-1]
    if heads is None:
        num_heads = getattr(model.config, "num_attention_heads", None)
        if num_heads is None:
            raise ValueError("Unable to determine num_attention_heads from model.config.")
        heads = [0, (num_heads - 1) // 2, num_heads - 1]

    sub_token_id = get_intervention_token_id(tokenizer)

    if token_indices is None:
        token_indices = list(range(seq_len))

    # Normal run: compute baseline attention scores
    normal_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    # Extract selected layers and heads: shape (1, len(layers), len(heads), T, T)
    normal_attn = torch.stack(
        [normal_out.attentions[l][0, heads, :, :] for l in layers], dim=0
    ).unsqueeze(0)
    del normal_out

    token_effects: list[float] = []

    # Run interventions in batches of chunk_size
    for start_idx in range(0, len(token_indices), chunk_size):
        chunk_positions = token_indices[start_idx : start_idx + chunk_size]
        b_size = len(chunk_positions)

        # Create copies of input_ids with position i replaced by '-'
        batch_input_ids = input_ids.repeat(b_size, 1)
        for b_idx, pos in enumerate(chunk_positions):
            batch_input_ids[b_idx, pos] = sub_token_id

        batch_mask = torch.ones_like(batch_input_ids)

        chunk_out = model(
            input_ids=batch_input_ids,
            attention_mask=batch_mask,
            output_attentions=True,
        )

        # Shape: (B, len(layers), len(heads), T, T)
        intervened_attn = torch.stack(
            [chunk_out.attentions[l][:, heads, :, :] for l in layers], dim=1
        )
        del chunk_out

        # Compute Frobenius norm || AS - AS'_xi || for each (layer, head) pair
        diff = normal_attn - intervened_attn
        norms = torch.linalg.matrix_norm(diff, ord="fro")  # shape (B, len(layers), len(heads))
        del intervened_attn, diff

        flattened_norms = norms.view(b_size, -1)
        if aggregation == "mean":
            agg = flattened_norms.mean(dim=-1)
        elif aggregation == "sum":
            agg = flattened_norms.sum(dim=-1)
        elif aggregation == "max":
            agg = flattened_norms.amax(dim=-1)
        else:
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'. Choose from 'mean', 'sum', or 'max'."
            )
        del norms, flattened_norms

        for val in agg:
            token_effects.append(float(val.item()))
        del agg

    del normal_attn
    return token_effects
