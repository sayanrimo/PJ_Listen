"""Stage 5 — Features.

Three complementary feature families, each independently toggle-able
via config so the pipeline degrades gracefully if UMAP/HDBSCAN or an
OpenAI key aren't available:

  (a) PRIMARY — concept-anchor cosine similarity. Interpretable,
      zero-shot, and directly maps to LINK's own dimension names, so
      it's the backbone feature set and the one driver analysis (Step
      7) will lean on for explainability.
  (b) DISCOVERY — UMAP -> HDBSCAN topic modelling over segment
      embeddings, with c-TF-IDF labels, to surface diagnostics *we
      didn't think to anchor* (e.g. a recurring objection specific to
      one brand). Per-ad topic-share features.
  (c) OPTIONAL — GPT rubric scoring on a small calibration sample,
      purely to sanity-check that anchor cosine similarity agrees with
      an LLM judge before trusting it at scale. Never run at full
      volume (cost + latency), and never used as a production feature
      itself — it's a validation signal, reported as a correlation.

CHANGE (one-question-one-construct + dynamic on/off + 0-1 scoring):

  - Every construct in config.yaml's `features.anchors` now carries an
    `enabled: true/false` flag (was previously "present in the dict =
    active"). Set it to false to drop a construct from scoring without
    deleting its phrase list.

  - `features.theme_construct_map` (config.yaml) says which *one*
    Theme's segment text each construct is scored against — e.g.
    "enjoyability" is only compared to the Emotion-theme answer, never
    to Memory/Meaning/Optimisation too. This replaces the old behavior
    of scoring every construct against every theme and averaging
    across themes, which produced a meaningless cross-theme matrix
    (e.g. an Optimisation answer getting an Enjoyability score) and
    meant "one score per construct" wasn't actually true — you'd get
    up to 4 different theme-level numbers plus an "_overall" average.
    Now each construct yields exactly ONE raw score per
    (Respondent ID, Ad ID), full stop.

  - If a construct's mapped theme is disabled in config.yaml's
    segment.themes (e.g. the study never asked Emotion), that
    construct is automatically skipped with a warning rather than
    silently scoring against text that doesn't exist.

  - Raw cosine similarity lives in [-1, 1]. Every anchor score is now
    always rescaled to [0, 1] via (cosine + 1) / 2 before it reaches
    aggregate.py or any output file — this is not configurable, it's
    the fixed default scale.

  - An OPTIONAL further linear rescale ("calibration") can stretch the
    0-1 score to an arbitrary client-facing range (e.g. 0-10, 0-100).
    This is OFF by default (config.yaml's `scoring.calibration.enabled:
    false`) — turning it on and setting `scoring.calibration.range` is
    the only way to get anything other than 0-1 output.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .embed import EmbedConfig, EmbeddingBackend, EmbeddingCache, embed_texts
from .logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# (a) Concept-anchor cosine similarity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorSet:
    """A named construct with several paraphrase anchor phrases, and
    the single Theme it is scored against."""
    name: str
    phrases: list[str]
    theme: str


def load_anchor_sets(
    anchors_cfg: dict[str, dict],
    theme_construct_map: dict[str, str],
    active_themes: set[str],
) -> list[AnchorSet]:
    """Build AnchorSet objects from config, honoring the enabled flag
    and the theme mapping.

    Args:
        anchors_cfg: `features.anchors` block — mapping
            construct_name -> {"enabled": bool, "phrases": [...]}.
        theme_construct_map: `features.theme_construct_map` — mapping
            construct_name -> the single Theme it should be scored
            against. Every *enabled* construct must have an entry
            here, or this raises.
        active_themes: set of Theme names currently enabled in
            segment.themes (see segment.get_active_themes). A
            construct whose mapped theme isn't in this set is skipped
            with a warning, not an error — a study legitimately may
            not field every question, and that shouldn't crash the run.

    Returns:
        List of AnchorSet for constructs that are enabled AND whose
        mapped theme is active.
    """
    sets: list[AnchorSet] = []
    for name, spec in anchors_cfg.items():
        if not spec.get("enabled", True):
            logger.info("features: construct '%s' is disabled in config — skipped.", name)
            continue

        phrases = spec["phrases"]
        if len(phrases) < 3:
            logger.warning(
                "features: anchor set '%s' has only %d phrase(s); "
                "5-10 is recommended for a stable centroid.", name, len(phrases),
            )

        if name not in theme_construct_map:
            raise ValueError(
                f"Construct '{name}' is enabled but has no entry in "
                "features.theme_construct_map — every enabled construct must "
                "declare exactly one Theme it is scored against."
            )
        theme = theme_construct_map[name]

        if theme not in active_themes:
            logger.warning(
                "features: construct '%s' maps to theme '%s', which is disabled "
                "in segment.themes — this construct will not be scored this run.",
                name, theme,
            )
            continue

        sets.append(AnchorSet(name=name, phrases=phrases, theme=theme))

    logger.info("features: %d construct(s) active this run: %s", len(sets), [a.name for a in sets])
    return sets


def compute_anchor_centroids(
    anchor_sets: list[AnchorSet],
    embed_cfg: EmbedConfig,
    backend: Optional[EmbeddingBackend] = None,
    cache: Optional[EmbeddingCache] = None,
) -> dict[str, np.ndarray]:
    """Embed each anchor phrase and average into one centroid per construct.

    Anchors are embedded as *documents* (no instruction prefix) so they
    live in the same representation space as the segments they'll be
    compared against; cosine similarity between two L2-normalized
    document embeddings is a standard, symmetric semantic-similarity
    signal, which is what we want when anchors are themselves natural-
    language paraphrases rather than a search query.
    """
    centroids: dict[str, np.ndarray] = {}
    for aset in anchor_sets:
        vectors = embed_texts(aset.phrases, embed_cfg, instruction=None, backend=backend, cache=cache)
        centroid = vectors.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)
        centroids[aset.name] = centroid
    logger.info("features: computed %d anchor centroids", len(centroids))
    return centroids


def score_anchor_similarity(
    segments: pd.DataFrame,
    segment_embeddings: np.ndarray,
    anchor_sets: list[AnchorSet],
    centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Cosine similarity of each segment against ONLY the anchor
    centroid(s) whose mapped Theme matches that segment's Theme.

    This is the one-question-one-construct scoring step: a construct
    mapped to the Emotion theme only ever gets compared against
    Emotion-theme segments. Rows for other themes get NaN in that
    construct's column (not zero — zero would falsely imply "measured
    and found absent"; NaN means "not applicable, wasn't scored here").
    Downstream mean/std aggregation (aggregate.py) skips NaN by
    default, so this naturally collapses to exactly one score per
    (Respondent ID, Ad ID) per construct with no further pivoting.

    Args:
        segments: must include 'segment_id' and 'Theme', row-aligned
            with segment_embeddings (same order, same length).
        segment_embeddings: (n_segments, dim) L2-normalized array.
        anchor_sets: output of load_anchor_sets (already filtered to
            enabled constructs with an active mapped theme).
        centroids: construct_name -> (dim,) L2-normalized centroid.

    Returns:
        DataFrame indexed by segment_id, one column
        `anchor_<construct>` per construct, raw cosine values in
        [-1, 1] (rescale to [0, 1] with rescale_cosine_to_unit before
        this leaves the pipeline).
    """
    theme_arr = segments["Theme"].values
    out = pd.DataFrame({"segment_id": segments["segment_id"].values})

    for aset in anchor_sets:
        col = f"anchor_{aset.name}"
        sims = np.full(len(segments), np.nan, dtype=np.float32)
        mask = theme_arr == aset.theme
        if mask.any():
            sims[mask] = segment_embeddings[mask] @ centroids[aset.name]
        else:
            logger.warning(
                "features: no segments found for theme '%s' (construct '%s') — "
                "all scores for this construct will be NaN.", aset.theme, aset.name,
            )
        out[col] = sims

    return out


def rescale_cosine_to_unit(df: pd.DataFrame, anchor_cols: list[str]) -> pd.DataFrame:
    """Linearly rescale cosine similarity [-1, 1] -> [0, 1] in place
    (returns the same df object for convenient chaining). NaNs pass
    through unchanged. This is the fixed default output scale — not
    configurable; see apply_calibration for an optional further
    rescale on top of this."""
    for col in anchor_cols:
        df[col] = (df[col] + 1.0) / 2.0
    return df


def apply_calibration(
    df: pd.DataFrame,
    anchor_cols: list[str],
    calibration_cfg: dict,
) -> pd.DataFrame:
    """Optional further linear rescale of an already-[0,1] score to a
    client-facing range (e.g. 0-10, 0-100). OFF by default — if
    calibration_cfg['enabled'] is falsy, returns df unchanged and
    scores stay on the 0-1 default scale.

    Args:
        df: DataFrame with anchor_<construct> columns already in [0, 1].
        anchor_cols: list of anchor_<construct> column names to rescale.
        calibration_cfg: `scoring.calibration` block from config.yaml,
            e.g. {"enabled": True, "range": {"min": 0, "max": 10}}.
    """
    if not calibration_cfg.get("enabled", False):
        return df

    rng = calibration_cfg.get("range", {})
    lo, hi = rng.get("min"), rng.get("max")
    if lo is None or hi is None:
        raise ValueError(
            "scoring.calibration.enabled is true but scoring.calibration.range "
            "is missing 'min'/'max' in config.yaml."
        )
    if hi <= lo:
        raise ValueError(f"scoring.calibration.range must have max > min; got min={lo}, max={hi}.")

    logger.info("features: calibration ENABLED — rescaling scores from [0, 1] to [%s, %s]", lo, hi)
    for col in anchor_cols:
        df[col] = df[col] * (hi - lo) + lo
    return df


# ---------------------------------------------------------------------------
# (b) Discovery: UMAP -> HDBSCAN -> c-TF-IDF topic shares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryResult:
    topic_assignments: pd.DataFrame   # segment_id, topic_id
    topic_labels: dict[int, str]      # topic_id -> top c-TF-IDF terms
    topic_share_by_ad: pd.DataFrame   # Ad ID x topic_share_<id> columns


def _ctfidf_labels(texts_by_topic: dict[int, list[str]], top_n: int = 6) -> dict[int, str]:
    """Class-based TF-IDF labels: term frequency within a topic's
    concatenated docs, weighted down by how common the term is across
    *all* topics' documents (the standard BERTopic c-TF-IDF idea,
    implemented directly with sklearn so we don't hard-depend on the
    bertopic package)."""
    from sklearn.feature_extraction.text import CountVectorizer

    topic_ids = sorted(texts_by_topic.keys())
    docs = [" ".join(texts_by_topic[t]) for t in topic_ids]
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    counts = vectorizer.fit_transform(docs).toarray().astype(float)
    vocab = np.array(vectorizer.get_feature_names_out())

    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    df_count = (counts > 0).sum(axis=0)
    idf = np.log(1 + len(topic_ids) / np.maximum(df_count, 1))
    ctfidf = tf * idf

    labels: dict[int, str] = {}
    for row_i, topic_id in enumerate(topic_ids):
        top_idx = np.argsort(-ctfidf[row_i])[:top_n]
        labels[topic_id] = ", ".join(vocab[top_idx])
    return labels


def discover_topics(
    segments: pd.DataFrame,
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_cluster_size: int = 8,
    min_dist: float = 0.0,
    umap_dim: int = 5,
    random_state: int = 42,
) -> DiscoveryResult:
    """UMAP dimensionality reduction -> HDBSCAN density clustering ->
    c-TF-IDF topic labels -> per-ad topic-share features.

    Unassigned segments (HDBSCAN label -1, "noise") are kept as their
    own pseudo-topic in assignments but excluded from topic-share
    features and from c-TF-IDF labelling, matching standard BERTopic
    convention.

    Args:
        segments: long segment table (must include 'segment_id',
            'Ad ID', 'text' aligned row-for-row with ``embeddings``).
        embeddings: (n_segments, dim) L2-normalized array, same row
            order as ``segments``.
        n_neighbors, min_dist, umap_dim: UMAP hyperparameters.
        min_cluster_size: HDBSCAN hyperparameter; with ~300 segments
            per theme this keeps clusters from being single-respondent
            noise while still finding minority topics.
        random_state: seed for UMAP reproducibility.
    """
    import umap
    import hdbscan

    logger.info(
        "features: discovery clustering on %d segments (umap_dim=%d, min_cluster_size=%d)",
        len(segments), umap_dim, min_cluster_size,
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=umap_dim,
        metric="cosine", random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    topic_ids = clusterer.fit_predict(reduced)

    assignments = pd.DataFrame({
        "segment_id": segments["segment_id"].values,
        "Ad ID": segments["Ad ID"].values,
        "topic_id": topic_ids,
    })

    texts_by_topic: dict[int, list[str]] = {}
    for tid, text in zip(topic_ids, segments["text"].values):
        if tid == -1:
            continue
        texts_by_topic.setdefault(int(tid), []).append(text)

    labels = _ctfidf_labels(texts_by_topic) if texts_by_topic else {}

    valid = assignments[assignments["topic_id"] != -1]
    if len(valid):
        share = (
            valid.groupby(["Ad ID", "topic_id"]).size().unstack(fill_value=0)
        )
        share = share.div(share.sum(axis=1), axis=0)
        share.columns = [f"topic_share_{c}" for c in share.columns]
        share = share.reset_index()
    else:
        share = pd.DataFrame({"Ad ID": segments["Ad ID"].unique()})

    n_topics = len(labels)
    n_noise = int((topic_ids == -1).sum())
    logger.info(
        "features: discovery found %d topics, %d/%d segments unassigned (noise)",
        n_topics, n_noise, len(segments),
    )
    for tid, label in labels.items():
        logger.info("features: topic %d = [%s]", tid, label)

    return DiscoveryResult(topic_assignments=assignments, topic_labels=labels, topic_share_by_ad=share)


# ---------------------------------------------------------------------------
# (c) Optional GPT rubric calibration (small sample only — NOT the same
#     as scoring.calibration above; this is the LLM-vs-anchor validation
#     check, kept under its original name for backward compatibility).
# ---------------------------------------------------------------------------

_RUBRIC_SYSTEM_PROMPT = (
    "You are scoring consumer ad-response verbatims against a market-research "
    "construct. Given a construct name+definition and a verbatim, output ONLY a "
    "JSON object: {{\"score\": <float 0-1>}}. 0 = construct entirely absent, "
    "1 = construct strongly and clearly present. No other text."
)


def calibrate_with_llm_rubric(
    segments: pd.DataFrame,
    anchor_scores: pd.DataFrame,
    construct: str,
    construct_definition: str,
    sample_size: int = 30,
    model: str = "gpt-4o",
    random_state: int = 42,
    use_azure: bool = False,
    azure_deployment: str = "",
    azure_api_version: str = "2024-06-01",
    azure_endpoint_env: str = "AZURE_GPT_ENDPOINT",
    azure_api_key_env: str = "AZURE_GPT_API_KEY",
) -> dict[str, float]:
    """Score a small random sample with an LLM rubric and report
    correlation against the anchor-cosine score for the same construct.

    This is a *validation* step, not a production feature: it never
    runs at full respondent volume (cost/latency), and its output is a
    diagnostic correlation coefficient, not a column that flows into
    aggregate.py.

    Args:
        segments: long segment table with 'segment_id' and 'text'.
        anchor_scores: output of ``score_anchor_similarity`` (must
            contain 'segment_id' and f'anchor_{construct}').
        construct: name of the construct being validated, must match
            an anchor set name.
        construct_definition: short human definition given to the LLM.
        sample_size: number of segments to sample (kept small by
            design — this is a spot check, not a full re-score).
        model: OpenAI chat model name (ignored if use_azure=True; use
            azure_deployment instead, since Azure routes by deployment).
        random_state: sampling seed.
        use_azure: if True, calls an Azure OpenAI chat deployment
            instead of api.openai.com. Requires the env vars named by
            azure_api_key_env / azure_endpoint_env (GPT-specific,
            separate from the embedding credentials in embed.py — this
            function never reads or falls back to the embedding key).
        azure_deployment: Azure deployment name for the chat model
            (e.g. your "gpt-4o" deployment name — may differ from the
            underlying model name). Required if use_azure=True.
        azure_api_version: Azure OpenAI API version string.
        azure_endpoint_env: env var name holding the GPT resource
            endpoint. Defaults to AZURE_GPT_ENDPOINT.
        azure_api_key_env: env var name holding the GPT API key.
            Defaults to AZURE_GPT_API_KEY.

    Returns:
        {"n": int, "pearson_r": float, "spearman_r": float}
    """
    try:
        import openai
    except ImportError as exc:
        raise ImportError("The 'openai' package is required for LLM calibration.") from exc

    if use_azure:
        # GPT-specific credentials only — deliberately never falls back
        # to the embedding backend's AZURE_EMBEDDING_* variables.
        api_key = os.environ.get(azure_api_key_env, "")
        endpoint = os.environ.get(azure_endpoint_env, "")
        if not api_key or not endpoint:
            raise RuntimeError(
                f"{azure_api_key_env} and {azure_endpoint_env} must both be set "
                "to run LLM calibration with use_azure=True."
            )
        if not azure_deployment:
            raise ValueError("azure_deployment is required when use_azure=True.")
        client = openai.AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=azure_api_version)
        chat_model = azure_deployment
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — cannot run LLM calibration.")
        client = openai.OpenAI(api_key=api_key)
        chat_model = model

    col = f"anchor_{construct}"
    if col not in anchor_scores.columns:
        raise ValueError(f"'{col}' not found in anchor_scores; did you compute anchors for '{construct}'?")

    merged = segments.merge(anchor_scores[["segment_id", col]], on="segment_id")
    merged = merged[merged["text"].str.len() > 0]
    sample = merged.sample(n=min(sample_size, len(merged)), random_state=random_state)

    llm_scores: list[float] = []
    for text in sample["text"]:
        prompt = (
            f"Construct: {construct}\nDefinition: {construct_definition}\n"
            f"Verbatim: \"{text}\""
        )
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": _RUBRIC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            score = float(json.loads(match.group(0))["score"]) if match else float("nan")
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("features: could not parse LLM calibration score from: %r", raw)
            score = float("nan")
        llm_scores.append(score)

    sample = sample.assign(llm_score=llm_scores).dropna(subset=["llm_score"])
    pearson_r = float(sample[col].corr(sample["llm_score"], method="pearson"))
    spearman_r = float(sample[col].corr(sample["llm_score"], method="spearman"))

    logger.info(
        "features: LLM calibration for '%s' (n=%d): pearson=%.3f spearman=%.3f",
        construct, len(sample), pearson_r, spearman_r,
    )
    return {"n": len(sample), "pearson_r": pearson_r, "spearman_r": spearman_r}