# Design decisions & trade-offs

## One-question-one-construct scoring

Every construct in `config.yaml`'s `features.anchors` is scored against
exactly one Theme, declared in `features.theme_construct_map`. A construct
mapped to the Emotion theme is only ever compared against Emotion-theme
segments — never averaged across every theme's text. This keeps "one score
per construct per respondent x ad" literally true rather than producing a
per-theme matrix plus a cross-theme "_overall" average, and it avoids
comparing an Optimisation answer against an Enjoyability anchor set that
was never meant to score it.

The trade-off: this makes `theme_construct_map` a piece of domain judgement
someone has to get right (which question does "authenticity" really belong
to?), and `features.load_anchor_sets` deliberately raises at startup if an
enabled construct has no mapping — rather than silently defaulting to some
theme — so a missing mapping is caught immediately, not discovered as a
gap in the output months later.

## Dynamic theme enable/disable

`segment.themes` in `config.yaml` lets a study turn off a theme it never
fielded (e.g. no Optimisation question this wave). `ingest.py` stops
requiring that theme's source column(s), `segment.py` stops emitting rows
for it, and any construct mapped to it is skipped with a warning (not an
error) rather than crashing the run. At least one theme must remain
enabled, or the pipeline has nothing to score.

## Why anchors vs. clustering vs. supervised

Three feature families are implemented, deliberately layered rather than
picking one:

- **Concept-anchor cosine similarity (primary).** Zero-shot, interpretable,
  and directly nameable against LINK's own vocabulary (Enjoyability,
  Relevance, Persuasion, Branding) plus market-research diagnostics
  (scepticism, authenticity, nostalgia, ...). This is what a driver-analysis
  readout should lean on, because "Ad X scored low on Emotion_authenticity"
  is directly actionable for a creative team in a way a cluster ID isn't.
  Weakness: anchors encode *our* hypotheses about what matters — if a real
  driver of LINK performance isn't anchored, this family is blind to it.

- **UMAP -> HDBSCAN discovery (secondary).** Unsupervised, so it can surface
  diagnostics nobody thought to write an anchor for — e.g. a recurring
  objection specific to one brand. Trade-off: cluster IDs aren't inherently
  interpretable (mitigated with c-TF-IDF labelling, but a human still has
  to read the label and sanity-check it), and cluster stability/count is
  sensitive to `min_cluster_size` and sample size — with a few hundred
  segments split across up to 4 themes, expect broad rather than
  fine-grained topics unless segment volume grows.

- **Supervised model on top of both feature families (model.py).** Only
  runs in train mode, only once ad-level features exist. This is
  intentionally the *last* layer, not a replacement for the anchor/
  discovery features — it consumes them as inputs rather than learning
  directly from raw embeddings, which keeps SHAP driver attribution
  pointed at human-readable feature names instead of opaque embedding
  dimensions.

- **GPT rubric calibration (validation only, not a feature).** Runs on a
  small (default 30) sample, per construct, purely to report a
  Pearson/Spearman correlation between anchor-cosine and an LLM judge. If
  correlation is low for a construct, that's a signal to revise the anchor
  phrases in `config.yaml` — not a reason to swap production scoring to LLM
  calls, which would be far more expensive and slower to run at full
  respondent volume across every future wave. It is never called when
  `features.run_llm_calibration` is false, and its credentials
  (`AZURE_GPT_*`) are entirely separate from the embedding backend's
  (`AZURE_EMBEDDING_*`) — neither falls back to the other.

## Why mean *and* std for every feature

An ad-level mean hides consensus. Two ads can have an identical mean
Enjoyability score while one has near-unanimous respondent agreement and
the other is bimodal (half loved it, half hated it) — those are different
creative diagnoses ("broadly liked" vs. "polarizing") that call for
different fixes, and LINK's own effectiveness bands are sensitive to this
kind of variance, not just central tendency. Emitting both `_mean` and
`_std` per construct at ad level keeps that signal available to both the
driver-analysis readout and the supervised model, rather than deciding
upfront that it isn't predictive.

## GroupKFold-by-Ad-ID leakage safeguard

Features are already aggregated to one row per Ad ID before model.py runs,
so there's no respondent-level leakage risk at that grain — but
`GroupKFold(groups=Ad ID)` is used anyway, for two reasons:

1. **It's the correct evaluation unit regardless.** With a small number of
   ads, plain KFold or a random train/test split has high variance
   run-to-run; grouping makes the "one row = one ad, never split across
   train/test" invariant explicit and impossible to violate even if the
   code is refactored to add engineered features that temporarily exist
   below Ad ID grain.
2. **It documents intent.** A future engineer reading `model.py` sees
   `GroupKFold(groups=groups)` and immediately understands "no row sharing
   an Ad ID may appear in both train and test," which is the actual
   constraint LINK-band prediction cares about, rather than having to infer
   it from the shape of the feature table.

## Predict vs. train mode

The respondent xlsx has no LINK target column, and the pipeline must never
fabricate one. Splitting into two modes keeps that invariant structural
rather than relying on a human remembering not to invent labels:

- **predict mode** runs ingest -> ... -> aggregate and stops. Output is the
  ad-level feature table plus QC/topic diagnostics — usable standalone for
  exploratory analysis, or as the scoring input to a model trained in a
  separate run/dataset where real LINK labels exist.
- **train mode** requires an explicit `labels_path` config pointing at a
  user-supplied Ad ID -> LINK-band (+ optional continuous score) file,
  joined on `Ad ID`. If that file is missing or malformed, the pipeline
  fails loudly at `model.load_labels` rather than silently proceeding
  without a target.

This also means the exact same feature pipeline code path
(`pipeline.run_feature_pipeline`) backs both modes — train mode is predict
mode plus a join and a model step, not a separate implementation that
could drift out of sync with what predict mode actually produces.

## Azure credential and endpoint separation (why two of everything)

Two independent Azure OpenAI deployments are in play — an embedding
deployment and a (optional) GPT chat deployment for calibration — and both
share one Azure resource/API key in this project's setup, but the code
treats them as fully independent to prevent two failure modes seen during
development:

1. **Credential-into-config-label mistakes.** Env var *names* like
   `AZURE_EMBEDDING_API_KEY` are label strings in `config.yaml`; the actual
   secret values only ever live in `.env`. The embedding backend and the
   GPT calibration function each read only their own named env var and
   never fall back to the other's — enforced by
   `test_embed_credentials.py` / `test_gpt_credentials.py`.
2. **Endpoint-vs-full-URL mistakes.** `embed.py`'s
   `_validate_deployment_style_endpoint` / `_validate_v1_style_endpoint`
   fail loudly, before any HTTP request, if the endpoint contains path
   segments like `/openai/v1`, `/responses`, `/deployments/`, or a query
   string — these indicate a full playground/testing URL was pasted in
   place of the bare resource endpoint from Azure Portal's "Keys and
   Endpoint" page.