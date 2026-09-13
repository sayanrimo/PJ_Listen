"""Stage 1 — Ingest.

Reads the raw respondent-level xlsx, validates it has the columns the
rest of the pipeline depends on (failing loudly if not), normalizes
whitespace in free-text fields, and coerces dtypes.

CHANGE (dynamic themes): which text columns are required is no longer
a hardcoded constant. It's derived from config.yaml's `segment.themes`
block, so a study that never fielded a given question (e.g. no
Optimisation question this wave) simply sets that theme's `enabled` to
false — ingest then does not require its source column(s) to exist.

Design note (unchanged): we validate column names literally rather
than fuzzy-matching them. Silent renaming of a market-research
deliverable is a worse failure mode than a loud crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

# These are never toggle-able — every row needs a respondent/ad key,
# brand/platform metadata, and order-shown, regardless of which theme
# questions were fielded.
ALWAYS_REQUIRED_COLUMNS: list[str] = [
    "Respondent ID",
    "Ad ID",
    "Brand",
    "Ad / Platform",
    "Order Shown",
]


class SchemaValidationError(ValueError):
    """Raised when the input xlsx does not match the required schema."""


@dataclass(frozen=True)
class IngestResult:
    df: pd.DataFrame
    n_respondents: int
    n_ads: int
    n_rows: int


def required_columns(theme_config: dict[str, dict]) -> list[str]:
    """Columns the raw xlsx must contain: always-required metadata
    columns + the source column(s) of every *enabled* theme.

    A theme with `enabled: false` contributes no columns here — if a
    study didn't field that question, its column(s) may be absent from
    the file entirely without failing ingest.
    """
    cols = list(ALWAYS_REQUIRED_COLUMNS)
    for name, cfg in theme_config.items():
        if cfg.get("enabled", True):
            cols += cfg["source_columns"]
    return cols


def text_columns(theme_config: dict[str, dict]) -> list[str]:
    """Source text column(s) of every enabled theme, for normalization."""
    cols: list[str] = []
    for name, cfg in theme_config.items():
        if cfg.get("enabled", True):
            cols += cfg["source_columns"]
    return cols


def _normalize_text(value: object) -> str:
    """Strip/collapse whitespace; return "" for null-like values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    text = " ".join(text.split())
    return text


def validate_columns(df: pd.DataFrame, sheet_name: str, required: list[str]) -> None:
    """Fail loudly if any required column is missing.

    Extra/unexpected columns are tolerated (and logged) — this now
    includes source columns belonging to *disabled* themes, since
    those are legitimately allowed to be present-but-ignored (e.g. a
    study xlsx that still has an Optimisation column even though this
    wave's config turned that theme off).
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"xlsx sheet '{sheet_name}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Column names must match the schema contract exactly, and every "
            "*enabled* theme in config.yaml's segment.themes needs its source "
            "column(s) present. If this question genuinely wasn't fielded, set "
            "that theme's `enabled: false` in config.yaml instead."
        )
    extra = [c for c in df.columns if c not in required]
    if extra:
        logger.info(
            "Ingest: ignoring extra/disabled-theme column(s) not required by "
            "the current config: %s", extra,
        )


def load_raw(
    path: str | Path,
    sheet_name: str = "Sheet1",
    theme_config: dict[str, dict] | None = None,
) -> IngestResult:
    """Load and validate the respondent-level xlsx.

    Args:
        path: path to the input xlsx (or any file matching the schema).
        sheet_name: worksheet to read (default "Sheet1").
        theme_config: the `segment.themes` block from config.yaml —
            required, since it determines which columns are mandatory.

    Returns:
        IngestResult with the cleaned DataFrame and basic counts.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        SchemaValidationError: if a required column is missing.
        ValueError: if theme_config is omitted or has no enabled theme.
    """
    if theme_config is None:
        raise ValueError(
            "load_raw requires theme_config (cfg['segment']['themes']) — "
            "required columns now depend on which themes are enabled."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    required = required_columns(theme_config)
    txt_cols = text_columns(theme_config)

    logger.info("Ingest: reading %s [sheet=%s]", path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    validate_columns(df, sheet_name, required)

    # Keep only the contract columns actually required by this config,
    # in canonical order. Columns from disabled themes are dropped
    # here even if present in the file.
    df = df[required].copy()

    for col in txt_cols:
        df[col] = df[col].map(_normalize_text)

    df["Brand"] = df["Brand"].map(_normalize_text)
    df["Ad / Platform"] = df["Ad / Platform"].map(_normalize_text)
    df["Respondent ID"] = df["Respondent ID"].astype(str).str.strip()
    df["Ad ID"] = df["Ad ID"].astype(str).str.strip()

    # Coerce Order Shown to int; fail loudly on non-numeric junk rather
    # than silently coercing to NaN -> 0, which would look like real data.
    try:
        df["Order Shown"] = pd.to_numeric(df["Order Shown"], errors="raise").astype(int)
    except (ValueError, TypeError) as exc:
        bad = df[pd.to_numeric(df["Order Shown"], errors="coerce").isna()]
        raise SchemaValidationError(
            f"'Order Shown' contains non-numeric values in {len(bad)} row(s), "
            f"e.g. respondent(s) {bad['Respondent ID'].head(5).tolist()}."
        ) from exc

    # Drop fully-blank rows (no respondent/ad key at all) — these are
    # spreadsheet artifacts, not data.
    key_blank = (df["Respondent ID"] == "") | (df["Ad ID"] == "")
    if key_blank.any():
        logger.warning("Ingest: dropping %d row(s) with blank Respondent ID / Ad ID", key_blank.sum())
        df = df[~key_blank].reset_index(drop=True)

    n_respondents = df["Respondent ID"].nunique()
    n_ads = df["Ad ID"].nunique()
    logger.info(
        "Ingest: loaded %d rows | %d respondents | %d ads | active themes: %s",
        len(df), n_respondents, n_ads,
        [name for name, cfg in theme_config.items() if cfg.get("enabled", True)],
    )

    return IngestResult(df=df, n_respondents=n_respondents, n_ads=n_ads, n_rows=len(df))