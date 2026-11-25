from __future__ import annotations

from typing import Any
import torch


def prepare_prompt(
    tokenizer: Any,
    user_text: str,
    use_chat_template: bool = True,
    device: Any = None,
) -> dict[str, Any]:
    """Prepare and tokenize a prompt, isolating content tokens from template formatting.

    Args:
        tokenizer: Hugging Face tokenizer.
        user_text: The raw user prompt text.
        use_chat_template: Whether to apply the tokenizer's chat template if available.
        device: Device to move the resulting input_ids and attention_mask tensors to.

    Returns:
        dict[str, Any]: Dictionary containing:
            - input_ids: Tensor of shape (1, T)
            - attention_mask: Tensor of shape (1, T)
            - content_indices: List of token positions belonging to user_text itself
            - tokens: List of token strings for ALL positions via convert_ids_to_tokens
            - text: The final formatted prompt string

    Raises:
        ValueError: If the user_text character span cannot be located inside formatted text.
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    if use_chat_template and has_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_text

    try:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError):
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    input_ids: torch.Tensor = encoding["input_ids"]
    attention_mask: torch.Tensor = encoding.get(
        "attention_mask", torch.ones_like(input_ids)
    )

    raw_offsets = encoding.get("offset_mapping")
    if raw_offsets is not None:
        if isinstance(raw_offsets, torch.Tensor):
            offsets: list[tuple[int, int]] = [
                (int(s), int(e)) for s, e in raw_offsets[0].tolist()
            ]
        else:
            offsets = [(int(s), int(e)) for s, e in raw_offsets[0]]
    else:
        # Fallback: estimate offsets if tokenizer does not support return_offsets_mapping
        tokens_temp = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        offsets = []
        cursor = 0
        for tok in tokens_temp:
            clean_tok = tok.replace(" ", " ").replace("Ġ", " ")
            pos = text.find(clean_tok.strip(), cursor)
            if pos != -1 and clean_tok.strip():
                offsets.append((pos, pos + len(clean_tok.strip())))
                cursor = pos + len(clean_tok.strip())
            else:
                offsets.append((cursor, cursor))

    # BOS handling: prepend BOS if model uses it and not already present
    bos_id = getattr(tokenizer, "bos_token_id", None)
    prepended_bos = False
    if (
        bos_id is not None
        and input_ids.shape[1] > 0
        and int(input_ids[0, 0].item()) != bos_id
    ):
        bos_tensor = torch.tensor([[bos_id]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        mask_tensor = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([mask_tensor, attention_mask], dim=1)
        offsets = [(0, 0)] + offsets
        prepended_bos = True

    # Find character span of user_text inside formatted text
    start_char = text.find(user_text)
    if start_char == -1:
        start_char = text.rfind(user_text)
    if start_char == -1:
        raise ValueError(
            f"Could not find character span for user_text inside formatted text:\n{text}"
        )
    end_char = start_char + len(user_text)

    content_indices: list[int] = []
    start_pos = 1 if prepended_bos else 0
    for pos in range(start_pos, len(offsets)):
        s, e = offsets[pos]
        # Overlap check between [s, e) and [start_char, end_char)
        if e > s and s < end_char and e > start_char:
            content_indices.append(pos)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "content_indices": content_indices,
        "tokens": tokens,
        "text": text,
    }
