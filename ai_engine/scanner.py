from __future__ import annotations

from typing import Any
import torch

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


def extract_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool = True,
    device: Any = None,
) -> dict[str, Any]:
    """Helper to tokenize prompt and isolate user-content token positions.

    Excludes fixed template formatting tokens and prepended BOS tokens.
    """
    data = prepare_prompt(
        tokenizer=tokenizer,
        user_text=prompt,
        use_chat_template=use_chat_template,
        device=device,
    )
    # If no chat template or raw text used and content_indices is empty, fallback to non-BOS tokens
    if len(data["content_indices"]) == 0:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        input_ids = data["input_ids"]
        start_idx = (
            1
            if (
                bos_id is not None
                and input_ids.shape[1] > 0
                and int(input_ids[0, 0].item()) == bos_id
            )
            else 0
        )
        data["content_indices"] = list(range(start_idx, input_ids.shape[1]))
    return data


class CausalScanner:
    """Orchestrates causal token and layer scanning on language models."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        use_chat_template: bool = True,
        aggregation: str = "mean",
        layers: list[int] | None = None,
        heads: list[int] | None = None,
        chunk_size: int = 8,
        quantize: str = "none",
    ) -> None:
        """Initialize and load the model once."""
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self.aggregation = aggregation
        self.layers = layers
        self.heads = heads
        self.chunk_size = chunk_size
        self.quantize = quantize

        self.model, self.tokenizer = load_model(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            quantize=self.quantize,
        )

    def scan(self, prompt: str) -> CausalMap:
        """Perform causal attribution scan on the given prompt.

        Args:
            prompt: User prompt string.

        Returns:
            CausalMap: Complete causal map with token and layer attributions and features.

        Raises:
            ValueError: If the prompt contains zero content tokens.
        """
        prompt_data = extract_prompt_tokens(
            tokenizer=self.tokenizer,
            prompt=prompt,
            use_chat_template=self.use_chat_template,
            device=self.model.device,
        )

        input_ids = prompt_data["input_ids"]
        attention_mask = prompt_data["attention_mask"]
        content_indices = prompt_data["content_indices"]
        all_tokens = prompt_data["tokens"]

        if len(content_indices) == 0:
            raise ValueError(f"Prompt '{prompt}' contains zero content tokens.")

        # Compute token causal effects on content positions only
        token_effects = compute_token_effects(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            layers=self.layers,
            heads=self.heads,
            aggregation=self.aggregation,
            token_indices=content_indices,
            chunk_size=self.chunk_size,
        )

        # Compute layer causal effects
        layer_effects, first_token_id, _ = compute_layer_effects(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        content_tokens = [all_tokens[idx] for idx in content_indices]
        first_token_text = self.tokenizer.decode([first_token_id])
        model_name = getattr(self.model.config, "_name_or_path", self.model_name)

        causal_map = CausalMap(
            prompt=prompt,
            model_name=model_name,
            tokens=content_tokens,
            token_effects=token_effects,
            layer_effects=layer_effects,
            first_token_id=first_token_id,
            first_token_text=first_token_text,
        )
        build_features(causal_map)
        return causal_map
