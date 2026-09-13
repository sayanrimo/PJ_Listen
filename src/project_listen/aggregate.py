"""Stage 6 — Aggregate.

Rolls respondent x ad x theme feature rows up to one row per Ad ID —
the grain Kantar LINK scores live at, and the grain model.py trains on.

CHANGE (one-question-one-construct): previously this module pivoted
every anchor construct wide by Theme (anchor_x_Memory_mean,
anchor_x_Meaning_mean, ...) PLUS a theme-collapsed "_overall" average
— because the old scoring step compared every construct against every
theme's text. features.score_anchor_similarity now only ever scores a
construct against its one mapped Theme (features.theme_construct_map),
so every other theme's row for that construct is NaN by construction.
That makes the per-theme pivot unnecessary: a plain groupby-mean/std
on the anchor_<construct> column already skips the NaN rows and lands
on exactly one mean/std per construct per Ad ID — which is also
exactly "one score per construct per respondent x ad" at the
respondent-level grain (see aggregate_to_respondent_level below).

Design note (mean + std), unchanged: a mean-only feature table would
hide the fact that two ads with identical average "Enjoyability" can
have very different *consensus* — one where every respondent agrees,
one where half loved it and half hated it. We keep both mean and std
per construct at ad level.
"""
from __future__ import annotations

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def aggregate_to_ad_level(
    segments: pd.DataFrame,
    anchor_scores: pd.DataFrame,
    topic_share_by_ad: pd.DataFrame | None = None,
    exclude_quarantined: bool = True,
) -> pd.DataFrame:
    """Roll respondent x ad x theme rows up to one row per Ad ID.

    Args:
        segments: long segment table, must include 'segment_id',
            'Ad ID', 'Theme', 'Order Shown', and (if
            exclude_quarantined) a 'quarantined' bool column from
            quality.run_quality_gate.
        anchor_scores: output of features.score_anchor_similarity
            (already rescaled to [0, 1], and calibrated if enabled),
            keyed by 'segment_id', one column per `anchor_<construct>`
            — non-NaN only on the rows belonging to that construct's
            mapped Theme.
        topic_share_by_ad: optional output of
            features.discover_topics(...).topic_share_by_ad, already
            at Ad ID grain — merged in directly (no mean/std needed,
            it's already a share).
        exclude_quarantined: drop quarantined segments before
            aggregating (recommended; quarantined text was flagged as
            unscorable/suspicious in quality.py).

    Returns:
        One row per Ad ID: anchor_<construct>_mean/std (across
        respondents, for that construct's single mapped theme),
        order_shown_mean/std, n_conversations, plus topic_share_*
        columns if provided.
    """
    df = segments.merge(anchor_scores, on="segment_id", how="left")

    n_before = len(df)
    if exclude_quarantined and "quarantined" in df.columns:
        df = df[~df["quarantined"]]
        logger.info(
            "aggregate: excluded %d/%d quarantined segments before rollup",
            n_before - len(df), n_before,
        )

    anchor_cols = [c for c in anchor_scores.columns if c.startswith("anchor_")]
    if not anchor_cols:
        logger.warning("aggregate: no anchor_<construct> columns found in anchor_scores.")
    elif df[anchor_cols].isna().all(axis=None):
        logger.warning("aggregate: all anchor scores are NaN — did embed/features run?")

    # groupby.mean()/std() skip NaN by default (skipna=True), so each
    # construct's stats are computed only over the respondents whose
    # segment matched that construct's mapped theme — no pivot needed.
    stats = df.groupby("Ad ID")[anchor_cols].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    ad_features = stats

    # Order Shown control feature (position/order-bias). One value per
    # respondent x ad, not per-theme, so aggregate on the original
    # (deduplicated) respondent x ad grain.
    order_shown = (
        df.drop_duplicates(subset=["Respondent ID", "Ad ID"])
        .groupby("Ad ID")["Order Shown"]
        .agg(order_shown_mean="mean", order_shown_std="std")
    )
    ad_features = ad_features.join(order_shown, how="left")

    # Sample-size transparency: how many *unquarantined* respondent
    # conversations backed each ad's numbers.
    n_conversations = (
        df.drop_duplicates(subset=["Respondent ID", "Ad ID"])
        .groupby("Ad ID")
        .size()
        .rename("n_conversations")
    )
    ad_features = ad_features.join(n_conversations, how="left")

    if topic_share_by_ad is not None and len(topic_share_by_ad):
        ad_features = ad_features.merge(
            topic_share_by_ad.set_index("Ad ID"), left_index=True, right_index=True, how="left"
        )

    # groupby was on "Ad ID", so the index is already named "Ad ID" —
    # reset_index() alone recovers it as a column with no further rename.
    ad_features = ad_features.reset_index()
    logger.info(
        "aggregate: built ad-level table with %d ads x %d features",
        len(ad_features), ad_features.shape[1] - 1,
    )
    return ad_features


def aggregate_to_respondent_level(
    segments: pd.DataFrame,
    anchor_scores: pd.DataFrame,
    exclude_quarantined: bool = True,
) -> pd.DataFrame:
    """Roll respondent x ad x theme rows up to one row per
    (Respondent ID, Ad ID) — one row per individual conversation, with
    exactly ONE score per construct (not per theme, not a cross-theme
    average — a plain, direct score, per your "one score for enjoyment,
    not each question" requirement).

    Because each construct is only ever scored against its single
    mapped Theme (features.theme_construct_map), a given
    (Respondent ID, Ad ID) group has at most one non-NaN value per
    anchor_<construct> column across its (up to 4) theme rows. Taking
    the group mean therefore just recovers that one value — no pivot,
    no "_overall" average across themes needed, since there was never
    more than one theme contributing to that construct to begin with.

    Args:
        segments: long segment table, must include 'segment_id',
            'Respondent ID', 'Ad ID', 'Theme', 'Order Shown', and (if
            exclude_quarantined) a 'quarantined' bool column from
            quality.run_quality_gate.
        anchor_scores: output of features.score_anchor_similarity,
            keyed by 'segment_id', one column per `anchor_<construct>`.
        exclude_quarantined: drop quarantined segments before
            rolling up (recommended, matches aggregate_to_ad_level's
            default).

    Returns:
        One row per (Respondent ID, Ad ID): anchor_<construct> (one
        score, [0, 1] by default or calibrated range if enabled),
        Order Shown, Brand, Ad / Platform, and n_themes_scored (how
        many of the active themes were NOT quarantined for this
        respondent x ad — useful to spot partial/low-quality rows).
    """
    df = segments.merge(anchor_scores, on="segment_id", how="left")

    n_before = len(df)
    if exclude_quarantined and "quarantined" in df.columns:
        n_themes_scored = (
            df[~df["quarantined"]]
            .groupby(["Respondent ID", "Ad ID"])
            .size()
            .rename("n_themes_scored")
        )
        df = df[~df["quarantined"]]
        logger.info(
            "aggregate: excluded %d/%d quarantined segments before respondent-level rollup",
            n_before - len(df), n_before,
        )
    else:
        n_themes_scored = (
            df.groupby(["Respondent ID", "Ad ID"]).size().rename("n_themes_scored")
        )

    anchor_cols = [c for c in anchor_scores.columns if c.startswith("anchor_")]
    if anchor_cols and df[anchor_cols].isna().all(axis=None):
        logger.warning("aggregate: all anchor scores are NaN — did embed/features run?")

    # One score per construct per (Respondent, Ad) — see docstring.
    respondent_features = df.groupby(["Respondent ID", "Ad ID"])[anchor_cols].mean()
    respondent_features = respondent_features.join(n_themes_scored, how="left")

    # Carry through metadata that's constant per respondent x ad
    # (Order Shown, Brand, Ad / Platform) rather than dropping it.
    meta_cols = [c for c in ("Order Shown", "Brand", "Ad / Platform") if c in df.columns]
    if meta_cols:
        meta = df.drop_duplicates(subset=["Respondent ID", "Ad ID"]).set_index(
            ["Respondent ID", "Ad ID"]
        )[meta_cols]
        respondent_features = respondent_features.join(meta, how="left")

    respondent_features = respondent_features.reset_index()
    logger.info(
        "aggregate: built respondent-level table with %d respondent x ad rows x %d features",
        len(respondent_features), respondent_features.shape[1] - 2,
    )
    return respondent_features