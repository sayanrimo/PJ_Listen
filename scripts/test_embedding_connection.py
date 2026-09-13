"""Pre-flight check: confirms the Qwen embedding deployment is reachable
with AZURE_EMBEDDING_ENDPOINT / AZURE_EMBEDDING_API_KEY before running
the full pipeline.

Reuses the same EmbedConfig / _build_backend / embed_texts code path as
the production pipeline — no duplicated connection logic.

Usage (from repo root, venv active):
    python scripts/test_embedding_connection.py

Or without activating the venv:
    .\.venv\Scripts\python.exe scripts/test_embedding_connection.py

Prints success/failure, deployment name, native dimension, and the
first five vector values only. Never prints the API key. Exits
non-zero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from project_listen import embed  # noqa: E402

TEST_TEXT = "Project LISTEN embedding connection test."


def main() -> int:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    embed_cfg = embed.EmbedConfig(
        backend=cfg["embed"]["backend"],
        model_name=cfg["embed"]["model_name"],
        api_base=cfg["embed"].get("api_base", ""),
        api_key_env=cfg["embed"].get("api_key_env", "QWEN_API_KEY"),
        output_dim=cfg["embed"]["output_dim"],
        batch_size=cfg["embed"]["batch_size"],
        max_retries=cfg["embed"]["max_retries"],
        backoff_base_s=cfg["embed"]["backoff_base_s"],
        cache_path=cfg["embed"]["cache_path"],
        azure_deployment=cfg["embed"].get("azure_deployment", ""),
        azure_api_version=cfg["embed"].get("azure_api_version", "2024-06-01"),
        azure_endpoint_env=cfg["embed"].get("azure_endpoint_env", "AZURE_EMBEDDING_ENDPOINT"),
        azure_api_key_env=cfg["embed"].get("azure_api_key_env", "AZURE_EMBEDDING_API_KEY"),
        azure_api_style=cfg["embed"].get("azure_api_style", "deployment"),
    )

    embed.log_embedding_diagnostics(embed_cfg)

    if embed_cfg.backend != "azure":
        print(f"backend={embed_cfg.backend!r} — this script only checks the 'azure' backend.")
        return 1

    try:
        backend = embed._build_backend(embed_cfg)
        vectors = backend.embed_batch([TEST_TEXT], instruction=None)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("SUCCESS: True")
    print(f"deployment: {embed_cfg.azure_deployment}")
    print(f"native dimension: {vectors.shape[1]}")
    print(f"first five values: {vectors[0][:5].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())