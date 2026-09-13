"""Stage 3 — Quality gating.

Applies cheap, deterministic rule-based checks to every segment
*before* any embedding spend. Flagged segments are quarantined (kept
in the output with flags set, excluded from scoring) rather than
silently dropped, so the QC report is auditable and nothing
disappears without a trace.

Checks:
  - empty: no text at all.
  - too_short: fewer than ``min_tokens`` whitespace tokens.
  - duplicate_verbatim: exact text (case-insensitive) repeated by the
    same respondent across ads, or copy-pasted wholesale — a common
    straight-lining signal in open-ends.
  - near_identical_cross_respondent: text that is a near-exact match
    (post-normalization) to another respondent's answer for the same
    Ad ID + Theme — a proxy for panel fraud / copy-paste / boilerplate
    survey text, without needing an embedding call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+")


def _token_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _normalize_for_match(text: str) -> str:
    """Aggressive normalization used only for near-duplicate matching
    (not for the text that gets embedded)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


@dataclass(frozen=True)
class QualityResult:
    df: pd.DataFrame          # segments + flag columns + `quarantined` bool
    report: pd.DataFrame      # summary counts per flag, per theme


def run_quality_gate(
    segments: pd.DataFrame,
    min_tokens: int = 4,
    near_duplicate_min_len: int = 20,
) -> QualityResult:
    """Flag and quarantine low-quality segments.

    Args:
        segments: long segment table from ``segment.build_segments``.
        min_tokens: minimum whitespace-token count to pass the
            too_short check.
        near_duplicate_min_len: normalized texts shorter than this
            (chars) are exempt from the cross-respondent near-identical
            check, since short generic answers ("it was fine") will
            collide legitimately and aren't evidence of fraud.

    Returns:
        QualityResult with per-segment flags + a summary report.
    """
    df = segments.copy()
    df["norm_text"] = df["text"].map(_normalize_for_match)
    df["token_count"] = df["text"].map(_token_count)

    df["flag_empty"] = df["text"].str.len() == 0
    df["flag_too_short"] = (~df["flag_empty"]) & (df["token_count"] < min_tokens)

    # duplicate_verbatim: same respondent gave literally the same
    # answer for >1 ad (copy-paste across the survey).
    dup_mask = pd.Series(False, index=df.index)
    for _, g in df[df["norm_text"].str.len() > 0].groupby(["Respondent ID", "Theme"]):
        dupes = g["norm_text"].duplicated(keep=False)
        dup_mask.loc[g.index] = dupes.values
    df["flag_duplicate_verbatim"] = dup_mask

    # near_identical_cross_respondent: within the same Ad ID + Theme,
    # >1 respondent produced the identical normalized text, and it's
    # long enough that coincidence is implausible.
    near_mask = pd.Series(False, index=df.index)
    eligible = df[
        (df["norm_text"].str.len() >= near_duplicate_min_len) & (~df["flag_empty"])
    ]
    for _, g in eligible.groupby(["Ad ID", "Theme"]):
        dupes = g["norm_text"].duplicated(keep=False)
        near_mask.loc[g.index] = dupes.values
    df["flag_near_identical_cross_respondent"] = near_mask

    df["quarantined"] = (
        df["flag_empty"]
        | df["flag_too_short"]
        | df["flag_duplicate_verbatim"]
        | df["flag_near_identical_cross_respondent"]
    )

    flag_cols = [
        "flag_empty",
        "flag_too_short",
        "flag_duplicate_verbatim",
        "flag_near_identical_cross_respondent",
    ]
    report = (
        df.groupby("Theme")[flag_cols + ["quarantined"]]
        .sum()
        .astype(int)
        .assign(n_segments=df.groupby("Theme").size())
    )
    report["quarantine_rate"] = (report["quarantined"] / report["n_segments"]).round(3)

    n_quarantined = int(df["quarantined"].sum())
    logger.info(
        "Quality: quarantined %d / %d segments (%.1f%%)",
        n_quarantined, len(df), 100 * n_quarantined / max(len(df), 1),
    )
    for theme, row in report.iterrows():
        logger.info(
            "Quality: [%s] empty=%d short=%d dup=%d near_dup=%d -> quarantined=%d/%d",
            theme, row["flag_empty"], row["flag_too_short"],
            row["flag_duplicate_verbatim"], row["flag_near_identical_cross_respondent"],
            row["quarantined"], row["n_segments"],
        )

    return QualityResult(df=df, report=report.reset_index())
