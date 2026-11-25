from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from ai_engine.scanner import CausalScanner
from ai_engine.model_loader import get_decoder_layers
from ai_engine.cli import run_batch

MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def scanner() -> CausalScanner:
    return CausalScanner(model_name=MODEL_NAME, device="cpu")


def test_scanner_output_lengths(scanner: CausalScanner) -> None:
    """len(token_effects) == len(tokens) and len(layer_effects) == number of decoder layers."""
    cm = scanner.scan("Who developed Windows 95?")
    assert len(cm.token_effects) == len(cm.tokens)
    expected_layers = len(get_decoder_layers(scanner.model))
    assert len(cm.layer_effects) == expected_layers


def test_features_no_nan_or_inf(scanner: CausalScanner) -> None:
    """Features contain no NaN or inf."""
    cm = scanner.scan("Sample prompt to check features")
    assert cm.features is not None
    vec = cm.features["vector"]
    assert len(vec) > 0
    assert not any(np.isnan(x) or np.isinf(x) for x in vec)


def test_layer_hooks_removed(scanner: CausalScanner) -> None:
    """Layer-skip hooks are removed after scanning: scanning the same prompt twice gives identical layer_effects."""
    decoder_layers = get_decoder_layers(scanner.model)
    initial_hook_counts = [len(layer._forward_hooks) for layer in decoder_layers]

    prompt = "Windows 95 development"
    cm1 = scanner.scan(prompt)
    cm2 = scanner.scan(prompt)

    assert np.allclose(cm1.layer_effects, cm2.layer_effects, atol=1e-6)

    final_hook_counts = [len(layer._forward_hooks) for layer in decoder_layers]
    assert final_hook_counts == initial_hook_counts


def test_batch_resume(scanner: CausalScanner, tmp_path: Path) -> None:
    """Batch resume works: run batch on a 4-row CSV with --limit 2, then without limit."""
    input_csv = tmp_path / "test_input.csv"
    output_csv = tmp_path / "test_output.csv"

    rows = [
        {"id": "1", "prompt": "Question one?", "label": "0", "task": "test"},
        {"id": "2", "prompt": "Question two?", "label": "0", "task": "test"},
        {"id": "3", "prompt": "Question three?", "label": "1", "task": "test"},
        {"id": "4", "prompt": "Question four?", "label": "1", "task": "test"},
    ]
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "label", "task"])
        writer.writeheader()
        writer.writerows(rows)

    # First run with limit=2
    run_batch(input_path=input_csv, output_path=output_csv, scanner=scanner, limit=2)

    with open(output_csv, "r", encoding="utf-8") as f:
        first_reader = list(csv.DictReader(f))
        first_run_ids = [r["id"] for r in first_reader]
        assert first_run_ids == ["1", "2"]
        first_run_rows = list(first_reader)

    # Second run without limit (resumes and completes remaining rows)
    run_batch(input_path=input_csv, output_path=output_csv, scanner=scanner)

    with open(output_csv, "r", encoding="utf-8") as f:
        final_reader = list(csv.DictReader(f))
        final_ids = [r["id"] for r in final_reader]
        assert final_ids == ["1", "2", "3", "4"]
        assert len(final_ids) == 4
        # Verify first 2 rows were preserved exactly without being recomputed
        for i in range(2):
            assert final_reader[i] == first_run_rows[i]

    # Verify tokens JSONL has all 4 rows
    tokens_jsonl = Path(str(output_csv) + ".tokens.jsonl")
    assert tokens_jsonl.exists()
    with open(tokens_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4
        assert [item["id"] for item in lines] == ["1", "2", "3", "4"]


def test_batch_error_handling(scanner: CausalScanner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt that raises an error is logged to the errors file and the batch continues."""
    input_csv = tmp_path / "test_err_input.csv"
    output_csv = tmp_path / "test_err_output.csv"
    errors_csv = Path(str(output_csv) + ".errors.csv")

    rows = [
        {"id": "good_1", "prompt": "Good prompt 1", "label": "0", "task": "test"},
        {"id": "bad_1", "prompt": "Bad prompt that fails", "label": "0", "task": "test"},
        {"id": "good_2", "prompt": "Good prompt 2", "label": "0", "task": "test"},
    ]
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "label", "task"])
        writer.writeheader()
        writer.writerows(rows)

    original_scan = scanner.scan

    def mock_scan(prompt: str):
        if prompt == "Bad prompt that fails":
            raise ValueError("Forced error for testing")
        return original_scan(prompt)

    monkeypatch.setattr(scanner, "scan", mock_scan)

    run_batch(input_path=input_csv, output_path=output_csv, scanner=scanner)

    # Valid prompts were processed and written
    with open(output_csv, "r", encoding="utf-8") as f:
        output_rows = list(csv.DictReader(f))
        output_ids = [r["id"] for r in output_rows]
        assert output_ids == ["good_1", "good_2"]

    # Error file logged the failure
    assert errors_csv.exists()
    with open(errors_csv, "r", encoding="utf-8") as f:
        err_rows = list(csv.DictReader(f))
        assert len(err_rows) == 1
        assert err_rows[0]["id"] == "bad_1"
        assert "Forced error for testing" in err_rows[0]["error"]
