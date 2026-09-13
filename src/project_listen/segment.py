"""Stage 2 — Segment.

Melts wide respondent-level rows into a long "segment" table: one row
per (Respondent ID, Ad ID, Theme, text). This is the atomic unit every
downstream stage (quality, embed, features) operates on.

CHANGE (dynamic themes): the Theme taxonomy is no longer a hardcoded
4-item constant. It's read from config.yaml's `segment.themes` block,
so studies that didn't field every question (e.g. no Optimisation
question this wave) can disable that theme entirely — no rows are
emitted for it, and no anchor construct mapped to it will be scored
(see features.theme_construct_map).

A theme can still map to more than one source column (e.g.
Optimisation = "what to improve" + "why it would help") — these are
concatenated with " || " in the order given in config.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

METADATA_COLUMNS: list[str] = ["Brand", "Ad / Platform", "Order Shown"]
SEGMENT_ID_COLUMNS: list[str] = ["Respondent ID", "Ad ID", "Theme"]


@dataclass(frozen=True)
class SegmentResult:
    df: pd.DataFrame
    n_segments: int
    active_themes: list[str]


def get_active_themes(theme_config: dict[str, dict]) -> dict[str, list[str]]:
    """Returns {theme_name: [source_column, ...]} for enabled themes only.

    Raises ValueError if nothing is enabled — the pipeline has nothing
    to score with zero active themes.
    """
    active = {
        name: cfg["source_columns"]
        for name, cfg in theme_config.items()
        if cfg.get("enabled", True)
    }
    if not active:
        raise ValueError(
            "No themes are enabled in config.yaml (segment.themes) — "
            "at least one theme must be enabled to run the pipeline."
        )
    return active


def _concat_columns(row: pd.Series, columns: list[str]) -> str:
    parts = [row[c] for c in columns if row[c]]
    return " || ".join(parts)


def build_segments(df: pd.DataFrame, theme_config: dict[str, dict]) -> SegmentResult:
    """Melt wide respondent x ad rows into long respondent x ad x theme rows.

    Args:
        df: the validated, normalized frame from ``ingest.load_raw``,
            already restricted to columns required by the active themes.
        theme_config: the `segment.themes` block from config.yaml.

    Returns:
        SegmentResult with one row per (Respondent ID, Ad ID, Theme)
        for every *enabled* theme only.
    """
    active_themes = get_active_themes(theme_config)
    records: list[dict] = []

    for _, row in df.iterrows():
        base = {
            "Respondent ID": row["Respondent ID"],
            "Ad ID": row["Ad ID"],
            "Brand": row["Brand"],
            "Ad / Platform": row["Ad / Platform"],
            "Order Shown": row["Order Shown"],
        }
        for theme, columns in active_themes.items():
            text = _concat_columns(row, columns)
            records.append({**base, "Theme": theme, "text": text})

    long_df = pd.DataFrame.from_records(records)
    long_df = long_df[["Respondent ID", "Ad ID", "Theme", "text"] + METADATA_COLUMNS]

    # Stable segment_id for caching/joining downstream.
    long_df["segment_id"] = (
        long_df["Respondent ID"] + "__" + long_df["Ad ID"] + "__" + long_df["Theme"]
    )

    logger.info(
        "Segment: built %d segments from %d respondent rows (%d active theme(s): %s)",
        len(long_df), len(df), len(active_themes), list(active_themes.keys()),
    )

    return SegmentResult(df=long_df, n_segments=len(long_df), active_themes=list(active_themes.keys()))