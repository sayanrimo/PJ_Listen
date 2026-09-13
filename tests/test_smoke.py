"""Tiny smoke test: runs ingest -> segment -> quality on the first 15
rows of the real data file and asserts basic shape/columns. Does NOT
call any embedding API (no network / API key needed), so it's safe to
run in CI on every commit.

CHANGE (dynamic themes): theme_config is now read from config.yaml and
passed explicitly into ingest.load_raw / segment.build_segments —
there's no more hardcoded ingest.REQUIRED_COLUMNS / segment.THEMES.

Run with:  pytest tests/test_smoke.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from project_listen import ingest, quality, segment  # noqa: E402

DATA_PATH = Path(__file__).parent.parent / "data" / "dummy_data.xlsx"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@pytest.fixture
def theme_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["segment"]["themes"]


@pytest.fixture
def first_15_rows(theme_config) -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip(f"smoke test data not found at {DATA_PATH}; copy dummy_data.xlsx there to run.")
    result = ingest.load_raw(DATA_PATH, theme_config=theme_config)
    return result.df.head(15)


def test_ingest_schema(first_15_rows, theme_config):
    assert list(first_15_rows.columns) == ingest.required_columns(theme_config)
    assert len(first_15_rows) == 15
    assert first_15_rows["Order Shown"].dtype.kind in "iu"  # int-like


def test_segment_shape(first_15_rows, theme_config):
    seg_result = segment.build_segments(first_15_rows, theme_config)
    n_active = len(segment.get_active_themes(theme_config))
    # 15 respondent rows x n_active themes
    assert seg_result.n_segments == 15 * n_active
    assert set(seg_result.df["Theme"].unique()) == set(seg_result.active_themes)
    assert seg_result.df["segment_id"].is_unique


def test_quality_gate_runs(first_15_rows, theme_config):
    seg_result = segment.build_segments(first_15_rows, theme_config)
    qc = quality.run_quality_gate(seg_result.df, min_tokens=4)
    assert "quarantined" in qc.df.columns
    assert len(qc.df) == len(seg_result.df)
    assert set(qc.report["Theme"]) == set(seg_result.active_themes)


def test_optimisation_concatenation(first_15_rows, theme_config):
    if not theme_config.get("Optimisation", {}).get("enabled", True):
        pytest.skip("Optimisation theme is disabled in config.yaml")
    seg_result = segment.build_segments(first_15_rows, theme_config)
    opt = seg_result.df[seg_result.df["Theme"] == "Optimisation"]
    assert " || " in opt.iloc[0]["text"] or opt.iloc[0]["text"] != ""


def test_disabling_a_theme_drops_its_column_requirement(theme_config):
    """Core requirement: a theme with enabled: false must not require
    its source column(s) to be present in the raw xlsx — this is what
    lets a study that never fielded a question run without error."""
    reduced = {k: dict(v) for k, v in theme_config.items()}
    if "Optimisation" not in reduced:
        pytest.skip("no Optimisation theme in config to disable")
    reduced["Optimisation"] = {**reduced["Optimisation"], "enabled": False}
    cols = ingest.required_columns(reduced)
    assert "Optimisation response - one improvement" not in cols
    assert "Why improvement would help" not in cols


def test_disabling_all_themes_raises(theme_config):
    all_off = {k: {**v, "enabled": False} for k, v in theme_config.items()}
    with pytest.raises(ValueError, match="No themes are enabled"):
        segment.get_active_themes(all_off)