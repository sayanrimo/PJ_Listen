"""Pre-flight check: confirms the GPT calibration deployment is
reachable with AZURE_GPT_ENDPOINT / AZURE_GPT_API_KEY.

Independent of the embedding connection test — uses only GPT-specific
credentials, never AZURE_EMBEDDING_*. Not run as part of predict mode.

Usage:
    python scripts/test_gpt_connection.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> int:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    calib_cfg = cfg["features"]["llm_calibration"]
    endpoint_env = calib_cfg.get("azure_endpoint_env", "AZURE_GPT_ENDPOINT")
    key_env = calib_cfg.get("azure_api_key_env", "AZURE_GPT_API_KEY")
    deployment = calib_cfg.get("azure_deployment", "")

    print("This test is independent of the embedding connection — it uses only "
          f"{endpoint_env} / {key_env}, never the AZURE_EMBEDDING_* variables.")

    endpoint = os.environ.get(endpoint_env, "")
    api_key = os.environ.get(key_env, "")
    print(f"gpt endpoint env={endpoint_env} loaded={bool(endpoint)}")
    print(f"gpt key env={key_env} loaded={bool(api_key)}")

    if not endpoint or not api_key:
        print(f"FAILED: {endpoint_env} and {key_env} must both be set.")
        return 1
    if not deployment:
        print("FAILED: features.llm_calibration.azure_deployment is not set.")
        return 1

    try:
        import openai
        client = openai.AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint,
            api_version=calib_cfg.get("azure_api_version", "2024-06-01"),
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0,
        )
        reply = resp.choices[0].message.content
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("SUCCESS: True")
    print(f"deployment: {deployment}")
    print(f"reply: {reply!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())