"""Stage 4 — Embed.

Wraps an embedding model (default: Qwen3-Embedding-8B, 4096-dim,
instruction-aware) behind a cached, retrying batch client. Three
backends are supported behind one interface:

  - "api": calls a generic OpenAI-compatible /embeddings endpoint
    (DashScope, a self-hosted TEI/vLLM server, etc). Configure via
    QWEN_API_KEY / QWEN_API_BASE env vars.
  - "azure": calls an Azure OpenAI resource's embeddings deployment
    (e.g. a text-embedding-3-large deployment, or any embedding model
    you've deployed on Azure AI Foundry — including a Qwen3 model if
    your Azure resource serves one). Configure via AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and a deployment
    name (config.embed.azure_deployment).
  - "local": loads a sentence-transformers model (e.g.
    Qwen/Qwen3-Embedding-8B) for teams running it on their own GPU box.
    Same interface, no network calls.

Both implement instruction-aware embedding (a task prefix prepended to
the query-side text — here, every segment is treated as the "document"
side, and anchors as the "query" side, per Qwen3's asymmetric
instruction convention), last-token pooling (handled internally by
sentence-transformers / by the API), L2 normalization, and optional
MRL truncation (Matryoshka: take the first ``k`` dims of the 4096-dim
vector and re-normalize — Qwen3 embeddings are trained to be truncation
-robust).

Caching: every (text, model, instruction, dim) tuple is hashed to a
cache key; cached vectors are stored in a parquet file so re-running
the pipeline after a code change downstream of embed.py costs nothing.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

EMBED_DIM_FULL = 4096

# Qwen3 embedding instruction convention: documents get no prefix,
# queries/anchors get an instruction prefix. We embed *segments* as
# documents and *concept anchors* as queries, so cosine-similarity
# reflects "does this document satisfy this instruction/query".
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a market research construct description, retrieve consumer "
    "verbatim responses that best express that construct\nQuery: {text}"
)


def _hash_key(text: str, model: str, instruction: str | None, dim: int) -> str:
    payload = f"{model}|{instruction or ''}|{dim}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _mrl_truncate(mat: np.ndarray, target_dim: int | None) -> np.ndarray:
    """Matryoshka truncation: slice to the first target_dim dims and
    re-normalize. No-op if target_dim is None or >= current dim."""
    if target_dim is None or target_dim >= mat.shape[1]:
        return mat
    truncated = mat[:, :target_dim]
    return _l2_normalize(truncated)


@dataclass
class EmbedConfig:
    backend: Literal["api", "azure", "local"] = "api"
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    api_base: str = field(default_factory=lambda: os.environ.get("QWEN_API_BASE", ""))
    api_key_env: str = "QWEN_API_KEY"
    output_dim: int = EMBED_DIM_FULL      # MRL truncation target (<=4096)
    batch_size: int = 16
    max_retries: int = 5
    backoff_base_s: float = 1.5
    cache_path: str | Path = "data/cache/embeddings.parquet"
    # Azure-only fields (used when backend == "azure"). model_name is
    # ignored for the Azure API call itself — Azure routes by deployment
    # name, not model name — but is still used as part of the cache key
    # so switching deployments doesn't collide with old cached vectors.
    #
    # IMPORTANT: these are the *embedding-specific* credential names.
    # They must never be defaulted to the generic AZURE_OPENAI_* names
    # shared with a GPT/chat deployment — the embedding backend and the
    # GPT calibration backend (features.calibrate_with_llm_rubric) use
    # entirely separate env vars, and neither silently falls back to
    # the other's credentials.
    azure_deployment: str = ""
    azure_api_version: str = "2024-06-01"
    azure_endpoint_env: str = "AZURE_EMBEDDING_ENDPOINT"
    azure_api_key_env: str = "AZURE_EMBEDDING_API_KEY"
    # "deployment" -> AzureOpenAI client, SDK builds the
    #   /openai/deployments/<name>/embeddings route itself from a bare
    #   resource endpoint.
    # "v1" -> plain OpenAI client pointed at a base_url ending in
    #   exactly "/openai/v1/" (Azure AI Foundry's OpenAI-compatible
    #   route). Never combine the two conventions.
    azure_api_style: Literal["deployment", "v1"] = "deployment"


class ConfigurationError(ValueError):
    """Deterministic embedding-configuration problem (bad/missing env
    var, malformed endpoint, impossible output_dim, ...).

    Raised *before* any HTTP request is attempted, and never retried —
    retrying a bad endpoint or a missing key five times just burns time
    for an error that will never resolve itself.
    """


class EmbeddingBackend:
    """Interface both backends satisfy."""

    def embed_batch(self, texts: Sequence[str], instruction: str | None) -> np.ndarray:
        raise NotImplementedError


class QwenAPIBackend(EmbeddingBackend):
    """OpenAI-compatible /embeddings endpoint backend."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "")
        if not self.api_key:
            logger.warning(
                "embed: env var %s is not set — API calls will fail until it is.",
                cfg.api_key_env,
            )
        try:
            import openai  # lazy import; optional dependency
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for backend='api'. "
                "Install it via requirements.txt or switch to backend='local'."
            ) from exc
        self._client = openai.OpenAI(api_key=self.api_key, base_url=cfg.api_base or None)

    def embed_batch(self, texts: Sequence[str], instruction: str | None) -> np.ndarray:
        payload = [instruction.format(text=t) if instruction else t for t in texts]
        cfg = self.cfg
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                resp = self._client.embeddings.create(model=cfg.model_name, input=payload)
                vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
                return vectors
            except Exception as exc:  # broad: network/API errors of many types
                last_exc = exc
                sleep_s = cfg.backoff_base_s * (2 ** attempt)
                logger.warning(
                    "embed: API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, cfg.max_retries, exc, sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Qwen embedding API failed after {cfg.max_retries} attempts") from last_exc


# Substrings that indicate a full request/route was pasted into what
# should be a bare Azure resource endpoint. Any of these on a
# "deployment"-style endpoint means the AzureOpenAI SDK will double up
# the path when it appends /openai/deployments/<name>/embeddings —
# this is exactly the malformed-URL bug this module guards against.
_DEPLOYMENT_STYLE_FORBIDDEN_SUBSTRINGS = (
    "/responses",
    "/embeddings",
    "/deployments/",
    "?api-version=",
    "/openai/v1",
)


def _validate_deployment_style_endpoint(endpoint: str, env_name: str) -> str:
    for bad in _DEPLOYMENT_STYLE_FORBIDDEN_SUBSTRINGS:
        if bad in endpoint:
            raise ConfigurationError(
                f"{env_name} must contain only the Azure resource endpoint when "
                "azure_api_style='deployment'. Do not include /openai/v1, /responses, "
                "/deployments, /embeddings, or an api-version query string. "
                f"Got: {endpoint!r}"
            )
    return endpoint.rstrip("/")


def _validate_v1_style_endpoint(endpoint: str, env_name: str) -> str:
    if "/responses" in endpoint:
        raise ConfigurationError(
            f"{env_name} must not contain '/responses' when azure_api_style='v1'. "
            f"Got: {endpoint!r}"
        )
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base + "/"
    if "/openai/v1" in base:
        raise ConfigurationError(
            f"{env_name} contains an unexpected path around '/openai/v1'; it must "
            f"resolve to exactly '<resource>/openai/v1/'. Got: {endpoint!r}"
        )
    return base + "/openai/v1/"


def _sanitize_endpoint_for_logging(endpoint: str) -> str:
    """Strip query params and never show a full request path — safe to
    print in diagnostics (still not a secret, but keep logs tidy)."""
    return endpoint.split("?", 1)[0].rstrip("/")


def _validate_embedding_credentials(cfg: "EmbedConfig") -> tuple[str, str]:
    """Fail-fast validation, run once before any HTTP request or retry
    loop. Returns (endpoint, api_key). Raises ConfigurationError on any
    deterministic problem — these are never retried.

    Deliberately reads *only* cfg.azure_endpoint_env / cfg.azure_api_key_env
    (the embedding-specific names threaded through from config.yaml) —
    never a hardcoded AZURE_OPENAI_* name, and never a GPT credential.
    """
    if not cfg.azure_deployment:
        raise ConfigurationError(
            "config.embed.azure_deployment must be set when backend='azure' "
            "(this is the exact Azure deployment name, not the model name)."
        )

    api_key = os.environ.get(cfg.azure_api_key_env, "")
    endpoint = os.environ.get(cfg.azure_endpoint_env, "")

    if not endpoint:
        raise ConfigurationError(
            f"{cfg.azure_endpoint_env} is not set. The embedding backend reads this "
            "specific env var (config.embed.azure_endpoint_env) and will not fall "
            "back to any GPT/chat endpoint variable."
        )
    if not api_key:
        raise ConfigurationError(
            f"{cfg.azure_api_key_env} is not set. The embedding backend reads this "
            "specific env var (config.embed.azure_api_key_env) and will not fall "
            "back to any GPT/chat API key."
        )

    if cfg.azure_api_style == "deployment":
        endpoint = _validate_deployment_style_endpoint(endpoint, cfg.azure_endpoint_env)
    elif cfg.azure_api_style == "v1":
        endpoint = _validate_v1_style_endpoint(endpoint, cfg.azure_endpoint_env)
    else:
        raise ConfigurationError(
            f"Unknown config.embed.azure_api_style={cfg.azure_api_style!r}; "
            "must be 'deployment' or 'v1'."
        )

    return endpoint, api_key


def log_embedding_diagnostics(cfg: "EmbedConfig") -> None:
    """Safe startup diagnostics: which env vars are configured and
    whether they resolved to a value — never the value itself.

    Only meaningful for backend='azure'; a no-op otherwise so predict
    runs on 'api'/'local' backends don't log irrelevant Azure lines.
    """
    if cfg.backend != "azure":
        logger.info("pipeline: embedding backend=%s", cfg.backend)
        return

    endpoint_raw = os.environ.get(cfg.azure_endpoint_env, "")
    key_raw = os.environ.get(cfg.azure_api_key_env, "")
    logger.info("pipeline: embedding backend=azure")
    logger.info("pipeline: embedding endpoint env=%s loaded=%s", cfg.azure_endpoint_env, bool(endpoint_raw))
    if endpoint_raw:
        logger.info(
            "pipeline: embedding endpoint (sanitized)=%s",
            _sanitize_endpoint_for_logging(endpoint_raw),
        )
    logger.info("pipeline: embedding key env=%s loaded=%s", cfg.azure_api_key_env, bool(key_raw))
    logger.info("pipeline: embedding deployment=%s", cfg.azure_deployment)
    logger.info("pipeline: embedding API style=%s", cfg.azure_api_style)
    logger.info("pipeline: embedding API version=%s", cfg.azure_api_version)
    logger.info("pipeline: embedding output_dim=%s", cfg.output_dim)


class AzureOpenAIBackend(EmbeddingBackend):
    """Azure embeddings backend supporting two, mutually-exclusive
    routing conventions (config.embed.azure_api_style):

      - "deployment": the AzureOpenAI SDK client, given a *bare*
        resource endpoint (e.g. https://<resource>.openai.azure.com),
        builds the /openai/deployments/<name>/embeddings route itself.
      - "v1": a plain OpenAI client pointed at a base_url ending in
        exactly "/openai/v1/" (Azure AI Foundry's OpenAI-compatible
        route). The deployment name is still passed as `model=`.

    Mixing these — e.g. handing an "/openai/v1/..." endpoint to the
    AzureOpenAI client — is exactly what produces the malformed
    ".../openai/v1/responses/openai/deployments/.../embeddings" URL
    this backend guards against via fail-fast validation before any
    request is attempted.
    """

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._native_dim: Optional[int] = None
        endpoint, api_key = _validate_embedding_credentials(cfg)

        try:
            import openai  # lazy import; optional dependency
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for backend='azure'. "
                "Install it via requirements.txt or switch to backend='local'."
            ) from exc

        if cfg.azure_api_style == "deployment":
            self._client = openai.AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=cfg.azure_api_version,
            )
        else:  # "v1", already validated/normalized to end in /openai/v1/
            self._client = openai.OpenAI(api_key=api_key, base_url=endpoint)

    def embed_batch(self, texts: Sequence[str], instruction: str | None) -> np.ndarray:
        payload = [instruction.format(text=t) if instruction else t for t in texts]
        cfg = self.cfg
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                resp = self._client.embeddings.create(model=cfg.azure_deployment, input=payload)
                vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
                self._check_native_dim(vectors.shape[1])
                return vectors
            except ConfigurationError:
                raise  # deterministic — never retry
            except Exception as exc:  # broad: network/API errors of many types
                last_exc = exc
                sleep_s = cfg.backoff_base_s * (2 ** attempt)
                logger.warning(
                    "embed: Azure API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, cfg.max_retries, exc, sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Azure OpenAI embedding call failed after {cfg.max_retries} attempts") from last_exc

    def _check_native_dim(self, dim: int) -> None:
        """Record the native dimension on first response; if the
        configured output_dim exceeds it, fail clearly rather than
        padding/fabricating or silently changing the configured value.
        This is a one-time, post-success check — it doesn't interact
        with the retry loop above."""
        if self._native_dim is None:
            self._native_dim = dim
        if self.cfg.output_dim > self._native_dim:
            raise ConfigurationError(
                f"Configured output_dim={self.cfg.output_dim}, but deployment "
                f"'{self.cfg.azure_deployment}' returned native dimension={self._native_dim}. "
                "output_dim cannot exceed the native embedding dimension."
            )


class QwenLocalBackend(EmbeddingBackend):
    """sentence-transformers local backend, for self-hosted GPU inference."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required for backend='local'. "
                "Install it via requirements.txt or switch to backend='api'."
            ) from exc
        logger.info("embed: loading local model %s (this can take a while the first time)", cfg.model_name)
        self._model = SentenceTransformer(cfg.model_name, trust_remote_code=True)

    def embed_batch(self, texts: Sequence[str], instruction: str | None) -> np.ndarray:
        payload = [instruction.format(text=t) if instruction else t for t in texts]
        # sentence-transformers handles last-token pooling internally
        # for Qwen3-Embedding models via its Pooling config.
        vectors = self._model.encode(
            payload, batch_size=self.cfg.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return vectors.astype(np.float32)


def _build_backend(cfg: EmbedConfig) -> EmbeddingBackend:
    if cfg.backend == "api":
        return QwenAPIBackend(cfg)
    if cfg.backend == "azure":
        return AzureOpenAIBackend(cfg)
    if cfg.backend == "local":
        return QwenLocalBackend(cfg)
    raise ValueError(f"Unknown embed backend: {cfg.backend!r}")


class EmbeddingCache:
    """Parquet-backed cache keyed by sha256(text|model|instruction|dim)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._df = pd.read_parquet(self.path)
        else:
            self._df = pd.DataFrame(columns=["key", "vector"])
        self._keys = set(self._df["key"]) if len(self._df) else set()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not len(self._df):
            return {}
        hits = self._df[self._df["key"].isin(keys)]
        return {row.key: np.array(row.vector, dtype=np.float32) for row in hits.itertuples()}

    def put_many(self, keys: list[str], vectors: np.ndarray) -> None:
        new_rows = pd.DataFrame({"key": keys, "vector": [v.tolist() for v in vectors]})
        self._df = pd.concat([self._df, new_rows], ignore_index=True)
        self._df = self._df.drop_duplicates(subset="key", keep="last")
        self._keys |= set(keys)

    def save(self) -> None:
        self._df.to_parquet(self.path, index=False)
        logger.info("embed: cache saved (%d entries) -> %s", len(self._df), self.path)


def embed_texts(
    texts: Sequence[str],
    cfg: EmbedConfig,
    instruction: str | None = None,
    backend: EmbeddingBackend | None = None,
    cache: EmbeddingCache | None = None,
) -> np.ndarray:
    """Embed a list of texts with caching, MRL truncation, and L2 norm.

    Args:
        texts: raw strings to embed (documents OR instruction-formatted
            queries — pass ``instruction=None`` for documents).
        cfg: EmbedConfig controlling model/backend/dim/batching.
        instruction: optional "{text}"-templated instruction string
            applied before hashing/embedding (query side).
        backend: reuse an existing backend instance (avoids reloading a
            local model); built from cfg if omitted.
        cache: reuse an existing cache instance; built from cfg if
            omitted.

    Returns:
        (n_texts, cfg.output_dim) float32 array, L2-normalized.
    """
    if backend is None:
        backend = _build_backend(cfg)
    if cache is None:
        cache = EmbeddingCache(cfg.cache_path)

    # Include the Azure deployment name in the cache key too, so
    # pointing config at a different deployment (potentially a
    # different underlying model) never silently reuses stale vectors.
    cache_model_id = f"{cfg.model_name}::{cfg.azure_deployment}" if cfg.backend == "azure" else cfg.model_name
    keys = [_hash_key(t, cache_model_id, instruction, cfg.output_dim) for t in texts]
    cached = cache.get_many(keys)

    missing_idx = [i for i, k in enumerate(keys) if k not in cached]
    logger.info(
        "embed: %d/%d texts hit cache; embedding %d new text(s)",
        len(texts) - len(missing_idx), len(texts), len(missing_idx),
    )

    if missing_idx:
        new_vectors: list[np.ndarray] = []
        new_keys: list[str] = []
        for start in range(0, len(missing_idx), cfg.batch_size):
            batch_positions = missing_idx[start:start + cfg.batch_size]
            batch_texts = [texts[i] for i in batch_positions]
            raw = backend.embed_batch(batch_texts, instruction)
            raw = _l2_normalize(raw.astype(np.float32))
            raw = _mrl_truncate(raw, cfg.output_dim)
            raw = _l2_normalize(raw)  # re-normalize post-truncation
            new_vectors.append(raw)
            new_keys.extend(keys[i] for i in batch_positions)
        stacked = np.vstack(new_vectors)
        cache.put_many(new_keys, stacked)
        cache.save()
        cached.update({k: v for k, v in zip(new_keys, stacked)})

    dim = cfg.output_dim
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = cached[k]
    return out