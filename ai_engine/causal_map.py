from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class CausalMap:
    """Represents causal attribution maps computed for a prompt and model.

    Attributes:
        prompt: The input prompt evaluated.
        model_name: Identifier or path of the model used.
        tokens: Token strings representing the tokenized prompt.
        token_effects: Token-level causal effects following Definition 2 of the LLMScan
            paper (L2 distance of attention scores after replacing a token).
        layer_effects: Layer-level causal effects following Definition 3 of the LLMScan
            paper (logit change when a layer is skipped).
        first_token_id: Token ID of the first generated response token.
        first_token_text: Text representation of the first generated response token.
        features: Optional extracted features or extra metadata associated with the scan.
    """

    prompt: str
    model_name: str
    tokens: list[str]
    token_effects: list[float]
    layer_effects: list[float]
    first_token_id: int
    first_token_text: str = ""
    features: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the CausalMap instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalMap:
        """Create a CausalMap instance from a dictionary."""
        return cls(
            prompt=d["prompt"],
            model_name=d["model_name"],
            tokens=list(d["tokens"]),
            token_effects=[float(x) for x in d["token_effects"]],
            layer_effects=[float(x) for x in d["layer_effects"]],
            first_token_id=int(d["first_token_id"]),
            first_token_text=d.get("first_token_text", ""),
            features=d.get("features"),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the CausalMap instance to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> CausalMap:
        """Load a CausalMap instance from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
