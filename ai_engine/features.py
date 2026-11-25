from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats

try:
    from ai_engine.causal_map import CausalMap
except ImportError:
    from causal_map import CausalMap


def token_statistics(token_effects: list[float] | np.ndarray) -> dict[str, float]:
    """Calculate 5 summary statistics from token causal effect values.

    Returns a dictionary with exactly 5 keys in order:
    'mean', 'std', 'range', 'skewness', 'kurtosis'.

    Uses the Fisher definition of kurtosis (via scipy.stats). Returns 0.0 for
    skewness and kurtosis if fewer than 3 values are provided or if all values
    are identical. Never outputs NaN or inf.

    Args:
        token_effects: List or 1D array of token causal effects.

    Returns:
        dict[str, float]: Dictionary containing mean, std, range, skewness, kurtosis.
    """
    arr = np.asarray(token_effects, dtype=np.float64)

    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = float(max_val - min_val)

    if arr.size < 3 or range_val == 0.0 or np.isclose(max_val, min_val):
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(arr, bias=True))
        kurt_val = float(stats.kurtosis(arr, fisher=True, bias=True))

    return {
        "mean": float(np.nan_to_num(mean_val, nan=0.0, posinf=0.0, neginf=0.0)),
        "std": float(np.nan_to_num(std_val, nan=0.0, posinf=0.0, neginf=0.0)),
        "range": float(np.nan_to_num(range_val, nan=0.0, posinf=0.0, neginf=0.0)),
        "skewness": float(np.nan_to_num(skew_val, nan=0.0, posinf=0.0, neginf=0.0)),
        "kurtosis": float(np.nan_to_num(kurt_val, nan=0.0, posinf=0.0, neginf=0.0)),
    }


def build_features(causal_map: CausalMap) -> dict[str, Any]:
    """Build unified feature representations from a CausalMap.

    Computes the 5 token summary statistics and combines them with layer effects
    into a joint feature vector. Stores the result in causal_map.features and returns it.

    Args:
        causal_map: CausalMap instance containing token and layer effects.

    Returns:
        dict[str, Any]: Dictionary with token_features, layer_features, vector,
            and feature_names.
    """
    stats_dict = token_statistics(causal_map.token_effects)
    token_features: list[float] = [
        stats_dict["mean"],
        stats_dict["std"],
        stats_dict["range"],
        stats_dict["skewness"],
        stats_dict["kurtosis"],
    ]
    layer_features: list[float] = list(causal_map.layer_effects)
    vector: list[float] = token_features + layer_features

    token_feature_names = [
        "tok_mean",
        "tok_std",
        "tok_range",
        "tok_skewness",
        "tok_kurtosis",
    ]
    layer_feature_names = [f"layer_{i}" for i in range(len(layer_features))]
    feature_names = token_feature_names + layer_feature_names

    features: dict[str, Any] = {
        "token_features": token_features,
        "layer_features": layer_features,
        "vector": vector,
        "feature_names": feature_names,
    }

    causal_map.features = features
    return features
