"""Tests for the embedding-credential separation, fail-fast validation,
and output-dimension checks added to embed.py.

All network calls are mocked — no real Azure services are contacted.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from project_listen import embed  # noqa: E402


def _cfg(**overrides) -> embed.EmbedConfig:
    base = dict(
        backend="azure",
        model_name="qwen--qwen3-embedding-8b",
        azure_deployment="qwen--qwen3-embedding-8b",
        azure_api_version="2024-06-01",
        azure_endpoint_env="AZURE_EMBEDDING_ENDPOINT",
        azure_api_key_env="AZURE_EMBEDDING_API_KEY",
        azure_api_style="deployment",
        output_dim=4096,
    )
    base.update(overrides)
    return embed.EmbedConfig(**base)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY", "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_GPT_ENDPOINT", "AZURE_GPT_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------
# Reads the correct, embedding-specific env vars
# ---------------------------------------------------------------------

def test_reads_azure_embedding_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with patch("openai.AzureOpenAI") as mock_client:
        backend = embed.AzureOpenAIBackend(_cfg())
        assert mock_client.call_args.kwargs["azure_endpoint"] == "https://my-embed-resource.openai.azure.com"


def test_reads_azure_embedding_api_key(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with patch("openai.AzureOpenAI") as mock_client:
        embed.AzureOpenAIBackend(_cfg())
        assert mock_client.call_args.kwargs["api_key"] == "fake-embed-key"


def test_does_not_fall_back_to_gpt_key(monkeypatch):
    # Only the GPT vars are set — embedding backend must NOT pick these up.
    monkeypatch.setenv("AZURE_GPT_ENDPOINT", "https://gpt-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_GPT_API_KEY", "fake-gpt-key")
    with pytest.raises(embed.ConfigurationError, match="AZURE_EMBEDDING_ENDPOINT"):
        embed.AzureOpenAIBackend(_cfg())


# ---------------------------------------------------------------------
# Fail-fast: missing config fails before any HTTP request
# ---------------------------------------------------------------------

def test_missing_embedding_key_fails_before_request(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    with patch("openai.AzureOpenAI") as mock_client:
        with pytest.raises(embed.ConfigurationError, match="AZURE_EMBEDDING_API_KEY"):
            embed.AzureOpenAIBackend(_cfg())
        mock_client.assert_not_called()


def test_missing_embedding_endpoint_fails_before_request(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with patch("openai.AzureOpenAI") as mock_client:
        with pytest.raises(embed.ConfigurationError, match="AZURE_EMBEDDING_ENDPOINT"):
            embed.AzureOpenAIBackend(_cfg())
        mock_client.assert_not_called()


def test_responses_in_endpoint_fails_before_request(monkeypatch):
    monkeypatch.setenv(
        "AZURE_EMBEDDING_ENDPOINT",
        "https://my-resource.services.ai.azure.com/openai/v1/responses",
    )
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with patch("openai.AzureOpenAI") as mock_client:
        with pytest.raises(embed.ConfigurationError, match="deployment"):
            embed.AzureOpenAIBackend(_cfg())
        mock_client.assert_not_called()


def test_deployment_style_rejects_openai_v1(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-resource.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with pytest.raises(embed.ConfigurationError, match="openai/v1"):
        embed.AzureOpenAIBackend(_cfg(azure_api_style="deployment"))


def test_v1_style_produces_correct_base_url(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-resource.services.ai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with patch("openai.OpenAI") as mock_client:
        embed.AzureOpenAIBackend(_cfg(azure_api_style="v1"))
        assert mock_client.call_args.kwargs["base_url"] == "https://my-resource.services.ai.azure.com/openai/v1/"


def test_v1_style_rejects_responses(monkeypatch):
    monkeypatch.setenv(
        "AZURE_EMBEDDING_ENDPOINT",
        "https://my-resource.services.ai.azure.com/openai/v1/responses",
    )
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")
    with pytest.raises(embed.ConfigurationError, match="responses"):
        embed.AzureOpenAIBackend(_cfg(azure_api_style="v1"))


# ---------------------------------------------------------------------
# Output-dim validation
# ---------------------------------------------------------------------

def test_output_dim_larger_than_native_raises(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")

    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1] * 1536)]

    with patch("openai.AzureOpenAI") as mock_client_cls:
        mock_client_cls.return_value.embeddings.create.return_value = fake_resp
        backend = embed.AzureOpenAIBackend(_cfg(output_dim=4096))
        with pytest.raises(embed.ConfigurationError, match="native dimension=1536"):
            backend.embed_batch(["hello"], instruction=None)


def test_output_dim_within_native_succeeds(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")

    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1] * 4096)]

    with patch("openai.AzureOpenAI") as mock_client_cls:
        mock_client_cls.return_value.embeddings.create.return_value = fake_resp
        backend = embed.AzureOpenAIBackend(_cfg(output_dim=1024))
        vectors = backend.embed_batch(["hello"], instruction=None)
        assert vectors.shape == (1, 4096)  # truncation happens later in embed_texts, not embed_batch


# ---------------------------------------------------------------------
# Configuration errors are never retried; transient errors still are
# ---------------------------------------------------------------------

def test_config_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")

    with patch("openai.AzureOpenAI") as mock_client_cls:
        mock_client_cls.return_value.embeddings.create.side_effect = embed.ConfigurationError("bad config")
        backend = embed.AzureOpenAIBackend(_cfg(max_retries=5))
        with pytest.raises(embed.ConfigurationError):
            backend.embed_batch(["hello"], instruction=None)
        assert mock_client_cls.return_value.embeddings.create.call_count == 1


def test_transient_error_is_retried(monkeypatch):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "fake-embed-key")

    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1] * 4096)]

    with patch("openai.AzureOpenAI") as mock_client_cls, patch("time.sleep"):
        mock_client_cls.return_value.embeddings.create.side_effect = [
            RuntimeError("transient network blip"), fake_resp,
        ]
        backend = embed.AzureOpenAIBackend(_cfg(max_retries=3))
        vectors = backend.embed_batch(["hello"], instruction=None)
        assert vectors.shape == (1, 4096)
        assert mock_client_cls.return_value.embeddings.create.call_count == 2


# ---------------------------------------------------------------------
# API key never appears in logs
# ---------------------------------------------------------------------

def test_api_key_not_in_diagnostics_log(monkeypatch, caplog):
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://my-embed-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "super-secret-key-value")
    with caplog.at_level("INFO"):
        embed.log_embedding_diagnostics(_cfg())
    assert "super-secret-key-value" not in caplog.text