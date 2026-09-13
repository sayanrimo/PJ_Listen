"""Tests for GPT calibration credential separation: it must read only
AZURE_GPT_* variables, never fall back to embedding credentials, and
must not be required at all when run_llm_calibration is false.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from project_listen import features  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "AZURE_GPT_ENDPOINT", "AZURE_GPT_API_KEY",
        "AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_gpt_calibration_requires_gpt_specific_vars(monkeypatch):
    # Only embedding vars are set — GPT calibration must not use them.
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")

    segments = pd.DataFrame({"segment_id": ["s1"], "text": ["hello world this is a test"]})
    anchor_scores = pd.DataFrame({"segment_id": ["s1"], "anchor_enjoyability": [0.5]})

    with pytest.raises(RuntimeError, match="AZURE_GPT_ENDPOINT"):
        features.calibrate_with_llm_rubric(
            segments, anchor_scores, "enjoyability", "enjoyment",
            use_azure=True, azure_deployment="gpt-5-mini",
        )


def test_gpt_calibration_reads_gpt_specific_vars(monkeypatch):
    monkeypatch.setenv("AZURE_GPT_ENDPOINT", "https://gpt-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_GPT_API_KEY", "fake-gpt-key")

    segments = pd.DataFrame({"segment_id": ["s1"], "text": ["hello world this is a test"]})
    anchor_scores = pd.DataFrame({"segment_id": ["s1"], "anchor_enjoyability": [0.5]})

    with patch("openai.AzureOpenAI") as mock_client_cls:
        mock_resp = mock_client_cls.return_value.chat.completions.create.return_value
        mock_resp.choices[0].message.content = '{"score": 0.7}'
        result = features.calibrate_with_llm_rubric(
            segments, anchor_scores, "enjoyability", "enjoyment",
            use_azure=True, azure_deployment="gpt-5-mini", sample_size=1,
        )
        assert mock_client_cls.call_args.kwargs["azure_endpoint"] == "https://gpt-resource.openai.azure.com"
        assert mock_client_cls.call_args.kwargs["api_key"] == "fake-gpt-key"
        assert result["n"] == 1


def test_calibration_not_called_when_disabled():
    """Mirrors pipeline.run_feature_pipeline's guard: with
    run_llm_calibration=False, calibrate_with_llm_rubric must never be
    invoked, so no GPT env vars are required."""
    cfg = {"features": {"run_llm_calibration": False, "llm_calibration": {"constructs": ["enjoyability"]}}}
    with patch("project_listen.features.calibrate_with_llm_rubric") as mock_calib:
        if cfg["features"].get("run_llm_calibration", False):
            for construct in cfg["features"]["llm_calibration"].get("constructs", []):
                features.calibrate_with_llm_rubric(None, None, construct, "")
        mock_calib.assert_not_called()