"""Embedding and anchor-construct quality diagnostics.

Runs three checks, none of which need GPT/LLM calibration:

  1. Embedding sanity — are vectors properly L2-normalized, and does
     the backend distinguish similar vs. unrelated text at all (catches
     a silently-broken backend returning near-identical or garbage
     vectors for everything).
  2. Anchor cohesion — do the paraphrase phrases for each construct
     actually agree with each other in embedding space? Low cohesion
     means the resulting centroid is averaging incompatible directions,
     producing a noisy construct score.
  3. Construct separation — are any two constructs' centroids so close
     together that they're not really measuring different things?
  4. (If ad_level_features.csv already exists) score-variance check —
     does each construct actually vary across ads, or is it flat/near-
     zero variance (meaning it isn't discriminating between ads)?

Reuses the same EmbedConfig / backend / anchor-loading code as the
production pipeline — no duplicated embedding logic.

Usage:
    python scripts/check_embedding_quality.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from project_listen import embed, features, segment  # noqa: E402

# Thresholds — adjust here rather than hunting through the report logic.
COHESION_WARN_BELOW = 0.40      # mean pairwise cosine among one construct's anchor phrases
SEPARATION_WARN_ABOVE = 0.85    # cosine between two different constructs' centroids
SCORE_VARIANCE_WARN_BELOW = 0.02  # std of a construct's ad-level mean across ads

# A few hand-picked probe sentences with a known expected relationship,
# used to sanity-check the backend isn't returning degenerate vectors.
SIMILAR_PAIR = (
    "I really enjoyed watching this ad.",
    "I found this ad delightful and fun.",
)
UNRELATED_PAIR = (
    "I really enjoyed watching this ad.",
    "I was confused about what this ad was trying to say.",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_embed_cfg(cfg: dict) -> embed.EmbedConfig:
    return embed.EmbedConfig(
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


def check_embedding_sanity(embed_cfg, backend, cache) -> list[str]:
    print("\n=== 1. Embedding sanity ===")
    issues = []

    texts = list(SIMILAR_PAIR) + list(UNRELATED_PAIR)
    vectors = embed.embed_texts(texts, embed_cfg, instruction=None, backend=backend, cache=cache)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"L2 norms (should all be ~1.0): {norms.round(4).tolist()}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        issues.append("Embeddings are not properly L2-normalized — check embed.py's normalization step.")

    sim_score = _cosine(vectors[0], vectors[1])
    unrel_score = _cosine(vectors[2], vectors[3])
    print(f"Similar-pair cosine similarity:   {sim_score:.3f}  (expect clearly higher)")
    print(f"Unrelated-pair cosine similarity: {unrel_score:.3f}  (expect clearly lower)")

    if sim_score <= unrel_score:
        issues.append(
            "Similar-meaning text did NOT score higher than unrelated text — the embedding "
            "backend may be returning degenerate/garbage vectors. Do not trust downstream scores "
            "until this is fixed."
        )
    elif sim_score - unrel_score < 0.05:
        issues.append(
            f"Similar vs. unrelated gap is very small ({sim_score - unrel_score:.3f}) — "
            "the embedding space may not be discriminating meaning well."
        )
    else:
        print("PASS: embedding space discriminates similar vs. unrelated text as expected.")

    return issues


def check_anchor_cohesion(anchor_sets, embed_cfg, backend, cache) -> tuple[list[str], dict]:
    print("\n=== 2. Anchor cohesion (per construct) ===")
    issues = []
    centroids: dict[str, np.ndarray] = {}

    for aset in anchor_sets:
        vectors = embed.embed_texts(aset.phrases, embed_cfg, instruction=None, backend=backend, cache=cache)
        pairwise = [
            _cosine(vectors[i], vectors[j])
            for i, j in combinations(range(len(vectors)), 2)
        ]
        mean_cohesion = float(np.mean(pairwise)) if pairwise else float("nan")
        centroid = vectors.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)
        centroids[aset.name] = centroid

        flag = "LOW" if mean_cohesion < COHESION_WARN_BELOW else "ok"
        print(f"  {aset.name:<22} mean pairwise cosine = {mean_cohesion:.3f}  [{flag}]")
        if mean_cohesion < COHESION_WARN_BELOW:
            issues.append(
                f"Anchor cohesion for '{aset.name}' is LOW ({mean_cohesion:.3f} < {COHESION_WARN_BELOW}) "
                "— its paraphrase phrases don't agree with each other in meaning; the centroid may be "
                "averaging incompatible directions. Consider revising the phrases in config.yaml."
            )

    return issues, centroids


def check_construct_separation(centroids: dict[str, np.ndarray]) -> list[str]:
    print("\n=== 3. Construct separation (pairwise, flag > %.2f) ===" % SEPARATION_WARN_ABOVE)
    issues = []
    names = list(centroids.keys())
    close_pairs = []
    for a, b in combinations(names, 2):
        sim = _cosine(centroids[a], centroids[b])
        if sim > SEPARATION_WARN_ABOVE:
            close_pairs.append((a, b, sim))

    if not close_pairs:
        print("PASS: no construct pairs exceed the separation threshold.")
    for a, b, sim in close_pairs:
        print(f"  HIGH OVERLAP: '{a}' vs '{b}' = {sim:.3f}")
        issues.append(
            f"'{a}' and '{b}' construct centroids are highly correlated ({sim:.3f}) — they may not "
            "be measuring meaningfully different things. Review whether both are needed."
        )
    return issues


def check_score_variance(cfg: dict) -> list[str]:
    print("\n=== 4. Ad-level score variance (skipped if no prior run) ===")
    issues = []
    out_dir = Path(cfg["output"]["output_dir"])
    ad_features_path = out_dir / cfg["output"]["ad_features_filename"]
    if not ad_features_path.exists():
        print(f"No {ad_features_path} found yet — run the pipeline first to enable this check.")
        return issues

    df = pd.read_csv(ad_features_path)
    mean_cols = [c for c in df.columns if c.endswith("_overall_mean") and c.startswith("anchor_")]
    for col in mean_cols:
        std = df[col].std()
        flag = "FLAT" if (pd.notna(std) and std < SCORE_VARIANCE_WARN_BELOW) else "ok"
        print(f"  {col:<45} std across ads = {std:.4f}  [{flag}]")
        if pd.notna(std) and std < SCORE_VARIANCE_WARN_BELOW:
            issues.append(
                f"'{col}' has near-zero variance across ads (std={std:.4f}) — this construct isn't "
                "discriminating between your ads and may not be useful for driver analysis."
            )
    return issues


def main() -> int:
    cfg = load_config()
    embed_cfg = build_embed_cfg(cfg)
    embed.log_embedding_diagnostics(embed_cfg)

    backend = embed._build_backend(embed_cfg)
    cache = embed.EmbeddingCache(embed_cfg.cache_path)

    all_issues: list[str] = []

    all_issues += check_embedding_sanity(embed_cfg, backend, cache)

    active_themes = set(segment.get_active_themes(cfg["segment"]["themes"]).keys())
    anchor_sets = features.load_anchor_sets(
        cfg["features"]["anchors"], cfg["features"]["theme_construct_map"], active_themes,
    )
    cohesion_issues, centroids = check_anchor_cohesion(anchor_sets, embed_cfg, backend, cache)
    all_issues += cohesion_issues

    all_issues += check_construct_separation(centroids)
    all_issues += check_score_variance(cfg)

    print("\n" + "=" * 60)
    if not all_issues:
        print("ALL CHECKS PASSED — no embedding or anchor-construct issues found.")
        return 0

    print(f"{len(all_issues)} ISSUE(S) FOUND:\n")
    for i, issue in enumerate(all_issues, 1):
        print(f"{i}. {issue}")
    print("\nThese are warnings, not hard failures — review and decide whether to act on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())