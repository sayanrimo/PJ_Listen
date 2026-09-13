"""Pipeline — CLI entry point.

    python pipeline.py --config config.yaml --mode predict
    python pipeline.py --config config.yaml --mode train

Wires ingest -> segment -> quality -> embed -> features -> aggregate
-> (train mode only) model, driven entirely by config.yaml so no code
changes are needed to point at a new xlsx, swap embedding backends,
toggle which theme questions were fielded, toggle which constructs are
scored, or edit anchor phrases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from CWD (or nearest parent) if present; no-op if missing
except ImportError:
    pass  # python-dotenv is optional — env vars can also be exported manually

from . import aggregate, embed, features, ingest, model, quality, segment
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_feature_pipeline(cfg: dict, raw_df: pd.DataFrame | None = None) -> dict:
    """Runs ingest(optional)->segment->quality->embed->features->aggregate.

    Args:
        cfg: parsed config.yaml.
        raw_df: if given, skips ingest.load_raw and uses this frame
            directly (used by the accuracy-vs-N-conversations curve to
            inject respondent-subsampled data without re-reading disk).

    Returns:
        dict with keys: segments, quality, anchor_scores, discovery,
        ad_features.
    """
    theme_config = cfg["segment"]["themes"]

    if raw_df is None:
        ingest_result = ingest.load_raw(
            cfg["ingest"]["input_path"], cfg["ingest"]["sheet_name"], theme_config=theme_config,
        )
        raw_df = ingest_result.df

    seg_result = segment.build_segments(raw_df, theme_config)
    active_themes = set(seg_result.active_themes)

    qc = quality.run_quality_gate(
        seg_result.df,
        min_tokens=cfg["quality"]["min_tokens"],
        near_duplicate_min_len=cfg["quality"]["near_duplicate_min_len"],
    )

    embed_cfg = embed.EmbedConfig(
        backend=cfg["embed"]["backend"],
        model_name=cfg["embed"]["model_name"],
        api_base=cfg["embed"].get("api_base", ""),
        api_key_env=cfg["embed"].get("api_key_env", "QWEN_API_KEY"),
        output_dim=cfg["embed"]["output_dim"],
        batch_size=cfg["embed"]["batch_size"],
        max_retries=cfg["embed"]["max_retries"],
        backoff_base_s=cfg["embed"]["backoff_base_s"],
        cache_path=cfg["embed"]["cache_path"],
        # AZURE_EMBEDDING_DEPLOYMENT env var, if set, overrides the YAML
        # value; the YAML value remains a usable fallback either way.
        azure_deployment=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", cfg["embed"].get("azure_deployment", "")),
        azure_api_version=cfg["embed"].get("azure_api_version", "2024-06-01"),
        azure_endpoint_env=cfg["embed"].get("azure_endpoint_env", "AZURE_EMBEDDING_ENDPOINT"),
        azure_api_key_env=cfg["embed"].get("azure_api_key_env", "AZURE_EMBEDDING_API_KEY"),
        azure_api_style=cfg["embed"].get("azure_api_style", "deployment"),
    )
    embed.log_embedding_diagnostics(embed_cfg)
    backend = embed._build_backend(embed_cfg)
    cache = embed.EmbeddingCache(embed_cfg.cache_path)

    scoreable = qc.df[~qc.df["quarantined"]].reset_index(drop=True)
    seg_embeddings = embed.embed_texts(
        scoreable["text"].tolist(), embed_cfg, instruction=None, backend=backend, cache=cache,
    )

    anchor_sets = features.load_anchor_sets(
        cfg["features"]["anchors"], cfg["features"]["theme_construct_map"], active_themes,
    )
    centroids = features.compute_anchor_centroids(anchor_sets, embed_cfg, backend=backend, cache=cache)
    anchor_scores = features.score_anchor_similarity(scoreable, seg_embeddings, anchor_sets, centroids)

    anchor_cols = [c for c in anchor_scores.columns if c.startswith("anchor_")]
    # Fixed default rescale: cosine [-1, 1] -> [0, 1]. Always applied.
    anchor_scores = features.rescale_cosine_to_unit(anchor_scores, anchor_cols)
    # Optional further rescale to a client-facing range. Off by default.
    anchor_scores = features.apply_calibration(
        anchor_scores, anchor_cols, cfg.get("scoring", {}).get("calibration", {"enabled": False}),
    )

    discovery_result = None
    if cfg["features"].get("run_discovery", True):
        try:
            discovery_result = features.discover_topics(
                scoreable, seg_embeddings,
                n_neighbors=cfg["features"]["discovery"]["n_neighbors"],
                min_dist=cfg["features"]["discovery"]["min_dist"],
                umap_dim=cfg["features"]["discovery"]["umap_dim"],
                min_cluster_size=cfg["features"]["discovery"]["min_cluster_size"],
                random_state=cfg["random_seed"],
            )
        except ImportError as exc:
            logger.warning("pipeline: discovery skipped, missing dependency: %s", exc)

    topic_share = discovery_result.topic_share_by_ad if discovery_result else None
    ad_features = aggregate.aggregate_to_ad_level(
        qc.df, anchor_scores, topic_share_by_ad=topic_share,
        exclude_quarantined=cfg["aggregate"]["exclude_quarantined"],
    )
    respondent_features = aggregate.aggregate_to_respondent_level(
        qc.df, anchor_scores, exclude_quarantined=cfg["aggregate"]["exclude_quarantined"],
    )

    llm_calibration_results: list[dict] = []
    if cfg["features"].get("run_llm_calibration", False):
        calib_cfg = cfg["features"]["llm_calibration"]
        for construct in calib_cfg.get("constructs", []):
            try:
                corr = features.calibrate_with_llm_rubric(
                    scoreable, anchor_scores, construct,
                    construct_definition=construct.replace("_", " "),
                    sample_size=calib_cfg["sample_size"],
                    model=calib_cfg.get("model", "gpt-4o"),
                    random_state=cfg["random_seed"],
                    use_azure=calib_cfg.get("use_azure", False),
                    azure_deployment=calib_cfg.get("azure_deployment", ""),
                    azure_api_version=calib_cfg.get("azure_api_version", "2024-06-01"),
                    azure_endpoint_env=calib_cfg.get("azure_endpoint_env", "AZURE_GPT_ENDPOINT"),
                    azure_api_key_env=calib_cfg.get("azure_api_key_env", "AZURE_GPT_API_KEY"),
                )
                logger.info("pipeline: LLM calibration [%s] -> %s", construct, corr)
                llm_calibration_results.append({"construct": construct, **corr})
            except (RuntimeError, ImportError) as exc:
                logger.warning("pipeline: LLM calibration skipped for '%s': %s", construct, exc)
                llm_calibration_results.append({"construct": construct, "error": str(exc)})
    else:
        logger.info("pipeline: run_llm_calibration=false — GPT client not initialized, GPT credentials not required")

    return {
        "segments": seg_result.df,
        "quality": qc,
        "anchor_scores": anchor_scores,
        "discovery": discovery_result,
        "ad_features": ad_features,
        "respondent_features": respondent_features,
        "llm_calibration": llm_calibration_results,
    }


def _write_outputs(cfg: dict, results: dict, model_report: dict | None = None) -> None:
    out_dir = Path(cfg["output"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_features_path = out_dir / cfg["output"]["ad_features_filename"]
    results["ad_features"].to_csv(ad_features_path, index=False)
    logger.info("pipeline: wrote %s", ad_features_path)

    respondent_features_filename = cfg["output"].get(
        "respondent_features_filename", "respondent_level_features.csv"
    )
    respondent_features_path = out_dir / respondent_features_filename
    results["respondent_features"].to_csv(respondent_features_path, index=False)
    logger.info("pipeline: wrote %s", respondent_features_path)

    qc_path = out_dir / cfg["output"]["qc_report_filename"]
    results["quality"].report.to_csv(qc_path, index=False)
    logger.info("pipeline: wrote %s", qc_path)

    if results["discovery"] is not None:
        topic_path = out_dir / cfg["output"]["topic_labels_filename"]
        with open(topic_path, "w") as f:
            json.dump(results["discovery"].topic_labels, f, indent=2)
        logger.info("pipeline: wrote %s", topic_path)

    if results.get("llm_calibration"):
        calib_json_filename = cfg["output"].get(
            "llm_calibration_filename_json", "llm_calibration_report.json"
        )
        calib_csv_filename = cfg["output"].get(
            "llm_calibration_filename_csv", "llm_calibration_report.csv"
        )
        calib_json_path = out_dir / calib_json_filename
        with open(calib_json_path, "w") as f:
            json.dump(results["llm_calibration"], f, indent=2)
        logger.info("pipeline: wrote %s", calib_json_path)

        calib_csv_path = out_dir / calib_csv_filename
        pd.DataFrame(results["llm_calibration"]).to_csv(calib_csv_path, index=False)
        logger.info("pipeline: wrote %s", calib_csv_path)

    if model_report is not None:
        model_path = out_dir / cfg["output"]["model_report_filename"]
        with open(model_path, "w") as f:
            json.dump(model_report, f, indent=2, default=str)
        logger.info("pipeline: wrote %s", model_path)


def run_predict_mode(cfg: dict) -> None:
    logger.info("pipeline: running in PREDICT mode (no labels, feature table + diagnostics only)")
    results = run_feature_pipeline(cfg)
    _write_outputs(cfg, results)
    logger.info("pipeline: predict mode complete.")


def run_train_mode(cfg: dict) -> None:
    logger.info("pipeline: running in TRAIN mode")
    results = run_feature_pipeline(cfg)

    labels = model.load_labels(
        cfg["model"]["labels_path"],
        ad_id_col=cfg["model"]["ad_id_col"],
        band_col=cfg["model"]["band_col"],
        score_col=cfg["model"].get("score_col"),
    )

    ad_features = results["ad_features"]
    model_report: dict = {}

    for task in ("3class", "binary"):
        try:
            result = model.train_classifier(
                ad_features, labels, task=task, model_name=cfg["model"]["model_name"],
                n_splits=cfg["model"]["n_splits"], random_state=cfg["random_seed"],
            )
            report = {
                "mean_score": result.mean_score, "metric": result.metric_name,
                "fold_scores": result.fold_scores,
            }
            if result.shap_values is not None:
                report["top_drivers"] = model.top_shap_drivers(result).to_dict(orient="records")
            model_report[task] = report
        except ValueError as exc:
            logger.warning("pipeline: skipping task '%s': %s", task, exc)

    if "LINK_score" in labels.columns:
        try:
            reg_result = model.train_regressor(
                ad_features, labels, model_name=cfg["model"]["model_name"],
                n_splits=cfg["model"]["n_splits"], random_state=cfg["random_seed"],
            )
            model_report["regression"] = {
                "mean_score": reg_result.mean_score, "metric": reg_result.metric_name,
                "fold_scores": reg_result.fold_scores,
            }
        except ValueError as exc:
            logger.warning("pipeline: skipping regression: %s", exc)

    # Accuracy-vs-N-conversations curve. build_features_fn subsamples
    # respondents per Ad ID from the *raw* ingested table, then reruns
    # the full segment->...->aggregate chain.
    raw_df = ingest.load_raw(
        cfg["ingest"]["input_path"], cfg["ingest"]["sheet_name"], theme_config=cfg["segment"]["themes"],
    ).df

    def build_features_fn(n_per_ad: int, seed: int) -> pd.DataFrame:
        # Sample row indices per Ad ID group rather than doing
        # groupby(...).apply(lambda g: g.sample(...)) directly on the
        # raw frame — the latter triggers a pandas DeprecationWarning
        # ("operating on the grouping columns") on recent pandas
        # versions when the callable's return value still contains the
        # group key column. Selecting on the index sidesteps it while
        # producing the identical subsampled frame.
        sampled_idx = (
            raw_df.groupby("Ad ID")
            .apply(lambda g: g.sample(n=min(n_per_ad, len(g)), random_state=seed).index)
        )
        flat_idx = np.concatenate(sampled_idx.values) if len(sampled_idx) else np.array([], dtype=int)
        subsampled = raw_df.loc[flat_idx]
        return run_feature_pipeline(cfg, raw_df=subsampled)["ad_features"]

    try:
        curve = model.accuracy_vs_n_conversations(
            build_features_fn, labels, n_conversations_grid=cfg["model"]["n_conversations_grid"],
            n_repeats=cfg["model"]["n_repeats"], task="3class",
            model_name=cfg["model"]["model_name"], random_state=cfg["random_seed"],
        )
        model_report["accuracy_vs_n_conversations"] = curve.to_dict(orient="records")
    except Exception as exc:  # broad: this is a diagnostic extra, don't fail the whole run
        logger.warning("pipeline: accuracy-vs-N-conversations curve skipped: %s", exc)

    _write_outputs(cfg, results, model_report=model_report)
    logger.info("pipeline: train mode complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project LISTEN pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--mode", required=True, choices=["predict", "train"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(log_dir=cfg["output"]["output_dir"])
    set_seeds(cfg["random_seed"])

    if args.mode == "predict":
        run_predict_mode(cfg)
    else:
        run_train_mode(cfg)


if __name__ == "__main__":
    main()