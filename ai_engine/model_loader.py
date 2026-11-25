from __future__ import annotations

from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str | None = None,
    dtype: torch.dtype | None = None,
    quantize: str = "none",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and its tokenizer.

    Args:
        model_name: Hugging Face model identifier or local path.
        device: Device to load the model onto ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        dtype: PyTorch dtype to load model weights with. Defaults to float16 for CUDA and float32 for CPU.
        quantize: Quantization mode ('none', '8bit', '4bit'). Defaults to 'none'.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model in eval mode and tokenizer.
    """
    if quantize not in ("none", "8bit", "4bit"):
        raise ValueError(
            f"Invalid quantize option '{quantize}'. Choices are: 'none', '8bit', '4bit'."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize in ("4bit", "8bit"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Model quantization ({quantize}) requires a CUDA GPU, but CUDA is not available."
            )

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Quantization requires the 'bitsandbytes' package. "
                "Please install it (e.g. pip install bitsandbytes)."
            ) from exc

        if quantize == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:  # 8bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": 0},
            attn_implementation="eager",
        )
        model.eval()
        return model, tokenizer

    # quantize == "none"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    return model, tokenizer


def get_decoder_layers(model: Any) -> torch.nn.ModuleList:
    """Retrieve the decoder transformer layers from the model.

    Supports Llama, Mistral, and Qwen2-style architectures (model.model.layers).

    Args:
        model: Loaded Hugging Face causal language model.

    Returns:
        torch.nn.ModuleList: The list/ModuleList of transformer decoder layers.

    Raises:
        ValueError: If decoder layers cannot be located on the model.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers

    raise ValueError(
        f"Could not find decoder layers in model of type '{model.__class__.__name__}'. "
        "Expected `model.model.layers` (supported architectures: Llama, Mistral, Qwen2)."
    )


def get_model_info(model: Any) -> dict[str, Any]:
    """Extract architectural metadata from the model.

    Args:
        model: Loaded Hugging Face causal language model.

    Returns:
        dict: Dictionary containing num_layers, num_heads, hidden_size, and model_name.
    """
    config = getattr(model, "config", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        try:
            num_layers = len(get_decoder_layers(model))
        except ValueError:
            num_layers = None

    num_heads = getattr(config, "num_attention_heads", None) if config else None
    hidden_size = getattr(config, "hidden_size", None) if config else None
    model_name = getattr(config, "_name_or_path", "") if config else ""

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "model_name": model_name,
    }
