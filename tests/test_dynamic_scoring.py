"""Tests for the three new requirements: (1) constructs/themes are
independently toggle-able via config, (2) each construct gets exactly
ONE score per (Respondent, Ad) via its single mapped theme — not one
per theme, not a cross-theme average, and (3) default scoring is
[0, 1], with an off-by-default calibration to rescale further.

Pure unit tests — no embedding API / network calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from project_listen import aggregate, features  # noqa: E402

ANCHORS_CFG = {
    "enjoyability": {"enabled": True, "phrases": ["I liked it", "fun ad", "enjoyable", "pleasant"]},
    "branding": {"enabled": False, "phrases": ["clear brand", "obvious brand", "branded well", "clear logo"]},
    "relevance": {"enabled": True, "phrases": ["relevant to me", "speaks to me", "relatable", "connects with me"]},
}
THEME_MAP = {"enjoyability": "Emotion", "branding": "Memory", "relevance": "Meaning"}


# ---------------------------------------------------------------------
# 1. Toggling
# ---------------------------------------------------------------------

def test_disabled_construct_is_skipped():
    sets = features.load_anchor_sets(ANCHORS_CFG, THEME_MAP, active_themes={"Emotion", "Memory", "Meaning"})
    names = {a.name for a in sets}
    assert names == {"enjoyability", "relevance"}  # branding excluded — enabled: false


def test_construct_skipped_if_its_mapped_theme_is_disabled():
    # enjoyability needs "Emotion", which isn't in active_themes here
    sets = features.load_anchor_sets(ANCHORS_CFG, THEME_MAP, active_themes={"Meaning"})
    names = {a.name for a in sets}
    assert names == {"relevance"}


def test_enabled_construct_missing_from_theme_map_raises():
    bad_cfg = {**ANCHORS_CFG, "nostalgia": {"enabled": True, "phrases": ["old times", "the past", "nostalgic"]}}
    with pytest.raises(ValueError, match="nostalgia"):
        features.load_anchor_sets(bad_cfg, THEME_MAP, active_themes={"Emotion", "Memory", "Meaning"})


# ---------------------------------------------------------------------
# 2. One score per construct per (Respondent, Ad) — theme-scoped scoring
# ---------------------------------------------------------------------

def test_score_anchor_similarity_is_theme_scoped():
    segments = pd.DataFrame({
        "segment_id": ["r1__a1__Emotion", "r1__a1__Meaning"],
        "Theme": ["Emotion", "Meaning"],
    })
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    anchor_sets = [
        features.AnchorSet(name="enjoyability", phrases=["x"], theme="Emotion"),
        features.AnchorSet(name="relevance", phrases=["y"], theme="Meaning"),
    ]
    centroids = {"enjoyability": np.array([1.0, 0.0]), "relevance": np.array([0.0, 1.0])}
    out = features.score_anchor_similarity(segments, embeddings, anchor_sets, centroids)

    emo_row = out[out["segment_id"] == "r1__a1__Emotion"].iloc[0]
    mean_row = out[out["segment_id"] == "r1__a1__Meaning"].iloc[0]

    # enjoyability only scored on the Emotion-theme segment
    assert emo_row["anchor_enjoyability"] == pytest.approx(1.0)
    assert pd.isna(mean_row["anchor_enjoyability"])
    # relevance only scored on the Meaning-theme segment
    assert mean_row["anchor_relevance"] == pytest.approx(1.0)
    assert pd.isna(emo_row["anchor_relevance"])


def test_aggregate_respondent_level_yields_one_score_not_a_pivot():
    """The old behavior produced anchor_x_Emotion, anchor_x_Meaning, ...
    plus anchor_x_overall_mean/std. The new behavior must produce a
    single anchor_x column with one value per (Respondent, Ad)."""
    segments = pd.DataFrame({
        "segment_id": ["r1__a1__Emotion", "r1__a1__Meaning", "r1__a1__Memory", "r1__a1__Optimisation"],
        "Respondent ID": ["r1"] * 4,
        "Ad ID": ["a1"] * 4,
        "Theme": ["Emotion", "Meaning", "Memory", "Optimisation"],
        "Order Shown": [1, 1, 1, 1],
        "Brand": ["Acme"] * 4,
        "Ad / Platform": ["YouTube"] * 4,
        "quarantined": [False, False, False, False],
    })
    anchor_scores = pd.DataFrame({
        "segment_id": ["r1__a1__Emotion", "r1__a1__Meaning", "r1__a1__Memory", "r1__a1__Optimisation"],
        "anchor_enjoyability": [0.8, np.nan, np.nan, np.nan],  # only scored on Emotion
    })

    result = aggregate.aggregate_to_respondent_level(segments, anchor_scores)

    assert list(result["anchor_enjoyability"]) == pytest.approx([0.8])
    # no per-theme columns like anchor_enjoyability_Emotion or _overall_mean
    assert not any(c.startswith("anchor_enjoyability_") for c in result.columns)
    assert len(result) == 1  # one row per (Respondent, Ad)


def test_aggregate_ad_level_averages_only_over_the_mapped_theme():
    segments = pd.DataFrame({
        "segment_id": ["r1__a1__Emotion", "r1__a1__Meaning", "r2__a1__Emotion", "r2__a1__Meaning"],
        "Respondent ID": ["r1", "r1", "r2", "r2"],
        "Ad ID": ["a1"] * 4,
        "Theme": ["Emotion", "Meaning", "Emotion", "Meaning"],
        "Order Shown": [1, 1, 2, 2],
        "quarantined": [False, False, False, False],
    })
    anchor_scores = pd.DataFrame({
        "segment_id": ["r1__a1__Emotion", "r1__a1__Meaning", "r2__a1__Emotion", "r2__a1__Meaning"],
        "anchor_enjoyability": [0.6, np.nan, 0.8, np.nan],
    })
    result = aggregate.aggregate_to_ad_level(segments, anchor_scores)
    row = result[result["Ad ID"] == "a1"].iloc[0]
    assert row["anchor_enjoyability_mean"] == pytest.approx(0.7)  # mean of 0.6, 0.8 — Meaning's NaN skipped
    assert not any(c.startswith("anchor_enjoyability_Emotion") or c.startswith("anchor_enjoyability_Meaning")
                   for c in result.columns)


# ---------------------------------------------------------------------
# 3. 0-1 default scale + off-by-default calibration
# ---------------------------------------------------------------------

def test_rescale_cosine_to_unit_range():
    df = pd.DataFrame({"anchor_x": [-1.0, 0.0, 1.0, np.nan]})
    out = features.rescale_cosine_to_unit(df.copy(), ["anchor_x"])
    vals = out["anchor_x"].tolist()
    assert vals[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert pd.isna(vals[3])


def test_calibration_off_by_default_leaves_0_1_scale():
    df = pd.DataFrame({"anchor_x": [0.0, 0.5, 1.0]})
    out = features.apply_calibration(df.copy(), ["anchor_x"], {"enabled": False, "range": {"min": 0, "max": 10}})
    assert out["anchor_x"].tolist() == [0.0, 0.5, 1.0]


def test_calibration_rescales_when_enabled():
    df = pd.DataFrame({"anchor_x": [0.0, 0.5, 1.0]})
    out = features.apply_calibration(df.copy(), ["anchor_x"], {"enabled": True, "range": {"min": 0, "max": 10}})
    assert out["anchor_x"].tolist() == pytest.approx([0.0, 5.0, 10.0])


def test_calibration_enabled_without_range_raises():
    df = pd.DataFrame({"anchor_x": [0.0, 0.5, 1.0]})
    with pytest.raises(ValueError, match="range"):
        features.apply_calibration(df.copy(), ["anchor_x"], {"enabled": True})