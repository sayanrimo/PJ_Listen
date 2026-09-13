"""Stage 7 — Model (train mode only).

Trains on the ad-level feature table against a user-supplied Ad ID ->
LINK-band labels file. Three complementary framings of the same
underlying signal:

  - 3-class: Strong / Average / Weak (LINK's native banding).
  - binary: top third vs bottom third (drops the ambiguous middle —
    often the more decision-useful framing for "kill/keep" calls).
  - regression: continuous LINK component score, if supplied.

GroupKFold-by-Ad-ID leakage safeguard: because features are aggregated
to Ad ID grain there is only one row per ad already, so the "leakage"
risk isn't within-fold — it's that with N=~ads this small, a single
random train/test split has high variance. GroupKFold by Ad ID
(equivalent to plain KFold at this grain, but explicit and future-
proof if this ever runs on a respondent-level or multi-market feature
table where several rows share an Ad ID) is a guardrail against a
future refactor accidentally training and testing on rows derived from
the same ad's conversations.

Accuracy-vs-N-conversations curve: subsamples respondents *per ad*
before the aggregate.py rollup, at decreasing N, to answer "how many
conversations per ad do we actually need to get a usable ad-level
feature vector" — a directly actionable operational question for
survey sample-size planning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from .logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_EXCLUDE_COLS = {"Ad ID", "n_conversations"}


@dataclass(frozen=True)
class TrainResult:
    task: str                       # "3class" | "binary" | "regression"
    model_name: str
    fold_scores: list[float]
    mean_score: float
    metric_name: str
    fitted_model: object            # final model, refit on all data
    feature_names: list[str]
    shap_values: Optional[np.ndarray] = None


def load_labels(path: str, ad_id_col: str = "Ad ID", band_col: str = "LINK_band",
                 score_col: Optional[str] = None) -> pd.DataFrame:
    """Load the user-supplied ad-level labels file and join key.

    Args:
        path: csv or xlsx with at minimum an Ad ID and LINK band column.
        ad_id_col: column name for the Ad ID join key.
        band_col: column with Strong/Average/Weak labels.
        score_col: optional continuous LINK component score column.
    """
    if path.endswith((".xlsx", ".xls")):
        labels = pd.read_excel(path)
    else:
        labels = pd.read_csv(path)

    required = [ad_id_col, band_col]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Labels file {path} is missing required column(s): {missing}")

    keep = [ad_id_col, band_col] + ([score_col] if score_col and score_col in labels.columns else [])
    labels = labels[keep].rename(columns={ad_id_col: "Ad ID", band_col: "LINK_band"})
    if score_col:
        labels = labels.rename(columns={score_col: "LINK_score"})

    valid_bands = {"Strong", "Average", "Weak"}
    bad = set(labels["LINK_band"].dropna().unique()) - valid_bands
    if bad:
        raise ValueError(f"Labels file has unexpected LINK_band value(s): {bad}; expected {valid_bands}")

    return labels


def _prep_xy(features: pd.DataFrame, labels: pd.DataFrame, target_col: str):
    merged = features.merge(labels, on="Ad ID", how="inner")
    if len(merged) < len(features):
        logger.warning(
            "model: %d/%d ads in the feature table had no matching label and were dropped",
            len(features) - len(merged), len(features),
        )
    feature_cols = [c for c in features.columns if c not in FEATURE_EXCLUDE_COLS]
    X = merged[feature_cols].fillna(0.0)
    y = merged[target_col]
    groups = merged["Ad ID"]
    return X, y, groups, feature_cols


def _make_classifier(name: str, random_state: int):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, random_state=random_state, eval_metric="mlogloss",
            use_label_encoder=False,
        )
    raise ValueError(f"Unknown model_name: {name}")


def train_classifier(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    task: str = "3class",
    model_name: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
) -> TrainResult:
    """Train + GroupKFold-evaluate a classifier for the 3-class or
    binary framing.

    Args:
        features: ad-level feature table from aggregate.aggregate_to_ad_level.
        labels: output of load_labels.
        task: "3class" (Strong/Average/Weak) or "binary" (top vs
            bottom third by LINK_score if available, else Strong vs
            Weak with Average dropped).
        model_name: "random_forest" | "gradient_boosting" | "xgboost".
        n_splits: GroupKFold folds. Since Ad ID is unique per row post-
            aggregation, effective n_splits is capped at n_ads.
        random_state: seed.
    """
    if task == "3class":
        target_col = "LINK_band"
        working_labels = labels
    elif task == "binary":
        if "LINK_score" in labels.columns:
            median = labels["LINK_score"].median()
            working_labels = labels.copy()
            working_labels["LINK_binary"] = np.where(working_labels["LINK_score"] >= median, "Top", "Bottom")
            target_col = "LINK_binary"
        else:
            working_labels = labels[labels["LINK_band"].isin(["Strong", "Weak"])].copy()
            target_col = "LINK_band"
    else:
        raise ValueError(f"Unknown classification task: {task}")

    X, y_raw, groups, feature_cols = _prep_xy(features, working_labels, target_col)

    n_ads = groups.nunique()
    effective_splits = min(n_splits, n_ads)
    if effective_splits < 2:
        raise ValueError(
            f"Need at least 2 distinct Ad IDs with labels to cross-validate; got {n_ads}."
        )
    if effective_splits < n_splits:
        logger.warning(
            "model: only %d labeled ads available; reducing GroupKFold n_splits from %d to %d",
            n_ads, n_splits, effective_splits,
        )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    gkf = GroupKFold(n_splits=effective_splits)
    fold_scores: list[float] = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        clf = _make_classifier(model_name, random_state)
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        fold_scores.append(acc)
        logger.info("model: [%s/%s fold %d] group-held-out accuracy = %.3f", task, model_name, fold_i, acc)

    final_model = _make_classifier(model_name, random_state)
    final_model.fit(X, y)

    shap_values = _compute_shap(final_model, X)

    result = TrainResult(
        task=task, model_name=model_name, fold_scores=fold_scores,
        mean_score=float(np.mean(fold_scores)), metric_name="accuracy",
        fitted_model=final_model, feature_names=feature_cols, shap_values=shap_values,
    )
    logger.info(
        "model: [%s/%s] mean GroupKFold accuracy = %.3f +/- %.3f (n_ads=%d, splits=%d)",
        task, model_name, result.mean_score, float(np.std(fold_scores)), n_ads, effective_splits,
    )
    return result


def train_regressor(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_name: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
) -> TrainResult:
    """Train + GroupKFold-evaluate a regressor on the continuous
    LINK_score column (requires labels to include it)."""
    if "LINK_score" not in labels.columns:
        raise ValueError("Regression requires a 'LINK_score' (or configured score_col) in the labels file.")

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    X, y, groups, feature_cols = _prep_xy(features, labels, "LINK_score")
    n_ads = groups.nunique()
    effective_splits = min(n_splits, n_ads)
    if effective_splits < 2:
        raise ValueError(f"Need at least 2 distinct labeled Ad IDs to cross-validate; got {n_ads}.")

    def make_reg():
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=random_state)
        if model_name == "gradient_boosting":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, random_state=random_state)
        raise ValueError(f"Unknown model_name: {model_name}")

    gkf = GroupKFold(n_splits=effective_splits)
    fold_scores: list[float] = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        reg = make_reg()
        reg.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = reg.predict(X.iloc[test_idx])
        r2 = r2_score(y.iloc[test_idx], preds)
        fold_scores.append(r2)
        logger.info("model: [regression fold %d] R^2 = %.3f", fold_i, r2)

    final_model = make_reg()
    final_model.fit(X, y)
    shap_values = _compute_shap(final_model, X)

    result = TrainResult(
        task="regression", model_name=model_name, fold_scores=fold_scores,
        mean_score=float(np.mean(fold_scores)), metric_name="r2",
        fitted_model=final_model, feature_names=feature_cols, shap_values=shap_values,
    )
    logger.info("model: [regression/%s] mean GroupKFold R^2 = %.3f", model_name, result.mean_score)
    return result


def _compute_shap(fitted_model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Best-effort SHAP driver attribution; returns None (with a
    warning) rather than crashing the pipeline if shap isn't
    installed or the model type isn't tree-based-explainable."""
    try:
        import shap
        explainer = shap.TreeExplainer(fitted_model)
        return explainer.shap_values(X)
    except Exception as exc:  # broad: SHAP has many model-specific failure modes
        logger.warning("model: SHAP explanation skipped (%s)", exc)
        return None


def top_shap_drivers(result: TrainResult, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean |SHAP value| for driver-analysis reporting."""
    if result.shap_values is None:
        raise RuntimeError("No SHAP values available on this TrainResult.")
    sv = result.shap_values
    if isinstance(sv, list):  # multi-class: list of (n_samples, n_features) per class
        importance = np.mean([np.abs(c).mean(axis=0) for c in sv], axis=0)
    else:
        importance = np.abs(sv).mean(axis=0)
    ranked = (
        pd.DataFrame({"feature": result.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return ranked


def accuracy_vs_n_conversations(
    build_features_fn,
    labels: pd.DataFrame,
    n_conversations_grid: list[int],
    n_repeats: int = 5,
    task: str = "3class",
    model_name: str = "random_forest",
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample respondents-per-ad at decreasing N, rebuild ad-level
    features each time, and report mean GroupKFold accuracy — answers
    "how many conversations per ad do we need".

    Args:
        build_features_fn: callable(n_respondents_per_ad: int, seed:
            int) -> ad-level feature DataFrame. Wraps the earlier
            pipeline stages (segment/quality/embed/features/aggregate)
            with respondent subsampling injected before segmentation;
            supplied by pipeline.py, which owns the raw respondent
            table this needs to subsample from.
        labels: output of load_labels.
        n_conversations_grid: e.g. [5, 10, 20, 30, 50, 100].
        n_repeats: repeats per grid point (different random respondent
            subsets) to average out sampling variance.
        task, model_name, random_state: passed through to train_classifier.

    Returns:
        DataFrame: n_conversations, mean_accuracy, std_accuracy.
    """
    rows = []
    for n in n_conversations_grid:
        accs = []
        for rep in range(n_repeats):
            seed = random_state + rep
            feats = build_features_fn(n, seed)
            try:
                result = train_classifier(feats, labels, task=task, model_name=model_name, random_state=seed)
                accs.append(result.mean_score)
            except ValueError as exc:
                logger.warning("model: skipping n=%d rep=%d (%s)", n, rep, exc)
        if accs:
            rows.append({"n_conversations": n, "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs))})
            logger.info("model: N=%d -> accuracy %.3f +/- %.3f (over %d reps)", n, np.mean(accs), np.std(accs), len(accs))
    return pd.DataFrame(rows)
