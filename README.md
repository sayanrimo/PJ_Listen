# Project LISTEN

Turns consumer ad-conversation transcripts into interpretable features that
predict / explain Kantar LINK-style creative-effectiveness scores.

```
Raw xlsx (respondent x ad)
   -> ingest        validate schema (dynamic per enabled theme), normalize text/dtypes
   -> segment        melt to respondent x ad x theme (long format, up to 4 themes)
   -> quality        rule-based QC flags, quarantine (don't score)
   -> embed          Azure OpenAI (text-embedding-3-large) embeddings, cached to parquet
   -> features        (a) concept-anchor cosine similarity  [primary — one score per construct]
                       (b) UMAP -> HDBSCAN topic discovery   [discovery]
                       (c) GPT rubric calibration (small n)  [optional validation only]
   -> aggregate       roll up to AD LEVEL and RESPONDENT LEVEL, mean + std
   -> model            (train mode only) classify/regress LINK band, SHAP
```

## Schema

Input file: a respondent-level xlsx, sheet "Sheet1". Required columns:

- `Respondent ID`, `Ad ID`, `Brand`, `Ad / Platform`, `Order Shown` — always required.
- One or more **Themes**, each mapped to one or more source text columns via
  `config.yaml`'s `segment.themes` block:
  - `Memory` <- "Memory response - what happened"
  - `Meaning` <- "Meaning response - what the ad was saying"
  - `Emotion` <- "Emotion response - feeling and why"
  - `Optimisation` <- "Optimisation response - one improvement" + "Why improvement would help" (concatenated)

Any theme can be turned off for a wave that didn't field that question by
setting `enabled: false` in `config.yaml` — ingest then stops requiring its
source column(s), segment.py stops emitting rows for it, and any construct
mapped to that theme (see `features.theme_construct_map`) is auto-skipped
with a warning rather than erroring. At least one theme must stay enabled.

There is **no separate ad-catalog file** and **no fuzzy brand-recall or
per-ad "intended message" comprehension scoring** in this pipeline — every
construct (including Branding) is scored the same way, via anchor-cosine
similarity against its one mapped theme's text.

## One-question-one-construct scoring

Every construct in `config.yaml`'s `features.anchors` is scored against
exactly **one** Theme, declared in `features.theme_construct_map` — e.g.
`enjoyability` is only ever compared against the Emotion-theme answer, never
averaged across all four themes. This means each construct yields exactly
one raw score per (Respondent ID, Ad ID), not a per-theme matrix.

Scores default to a `[0, 1]` scale (rescaled from raw cosine `[-1, 1]`).
An optional linear `scoring.calibration` stretch to a client-facing range
(e.g. 0-10) is off by default.

## No LINK target in the input file

There is no LINK score/band column in the respondent xlsx. The pipeline
runs in two modes:

- **predict mode**: no labels; produces the feature table + diagnostics only.
- **train mode**: requires a separate ad-level labels file (`Ad ID` ->
  `LINK_band` Strong/Average/Weak, optional `LINK_score`), joined on `Ad ID`.

The pipeline never fabricates a target.

## New: synthetic dummy data generator

`generate_dummy_data.py` generates a respondent-level xlsx matching this
exact 4-theme schema, plus an internally-consistent `labels.csv` (each ad
gets a hidden "quality latent" driving both the verbatims' sentiment mix
and the LINK score/band, so `model.py`'s classifier/SHAP has real signal to
find).

```bash
python generate_dummy_data.py --n-ads 10 --n-respondents-per-ad 20 --seed 42
```

## New: respondent chatbot

`chatbot.py` is a CLI chat flow that walks one respondent through the
enabled themes (per `config.yaml`) for one ad, and appends the answers
directly into the schema `ingest.py` expects:

```bash
python chatbot.py --config config/config.yaml --out data/dummy_data.xlsx --ad-id AD01
```

Question order: Memory first (unaided, before anything else could prime the
answer), Optimisation last (most effortful/reflective).

It does not play the ad itself — it assumes the respondent just watched it
(in person, shared screen, or a link sent separately).

## Folder structure

```
project_listen/
  pipeline.py                 # root shim: `python pipeline.py --config ... --mode ...`
  chatbot.py                  # respondent chat CLI
  generate_dummy_data.py      # synthetic dataset generator
  config/
    config.yaml                # all knobs + anchor phrases live here
  src/project_listen/
    ingest.py
    segment.py
    quality.py
    embed.py
    features.py
    aggregate.py
    model.py
    pipeline.py                 # wires all stages together
    logging_utils.py
  data/
    dummy_data.xlsx              # input (respondent x ad table)
    labels.csv                    # optional, train mode only (Ad ID -> LINK band)
    cache/embeddings.parquet       # auto-created embedding cache
    outputs/                        # auto-created: features, QC report, model report
  tests/
    test_smoke.py                 # runs on first 15 rows, no network/API needed
    test_embed_credentials.py
    test_gpt_credentials.py
    test_dynamic_scoring.py
  scripts/
    check_embedding_quality.py
    test_embedding_connection.py
    test_gpt_connection.py
  requirements.txt
  .env.example
  README.md
  DESIGN_DECISIONS.md
```

## Setup (Azure)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials —
found in Azure Portal under your Azure OpenAI resource -> **Keys and
Endpoint**. `.env` is loaded automatically (via `python-dotenv`).

Note the credential separation: `AZURE_EMBEDDING_ENDPOINT` /
`AZURE_EMBEDDING_API_KEY` for the embedding deployment, and
`AZURE_GPT_ENDPOINT` / `AZURE_GPT_API_KEY` for the (optional) LLM
calibration deployment. These never fall back to each other.

Then in `config/config.yaml`, confirm `embed.azure_deployment` and
`features.llm_calibration.azure_deployment` are the exact Azure deployment
names (not model names), and that `ingest.input_path` points at the right
file.

## Run

```bash
# 1. Generate (or regenerate) the synthetic dataset
python generate_dummy_data.py --n-ads 10 --n-respondents-per-ad 20 --seed 42

# 2. Predict mode (no labels — just the feature table + diagnostics)
python pipeline.py --config config/config.yaml --mode predict

# 3. Train mode (uses data/labels.csv, generated alongside the dataset)
python pipeline.py --config config/config.yaml --mode train

# 4. Collect a real respondent session via chat
python chatbot.py --config config/config.yaml --out data/dummy_data.xlsx --ad-id AD01
```

Outputs land in `data/outputs/`:
- `ad_level_features.csv` — one row per Ad ID, `anchor_<construct>_mean/std` columns
- `respondent_level_features.csv` — one row per (Respondent ID, Ad ID)
- `qc_report.csv` — per-theme QC flag counts
- `topic_labels.json`
- `model_report.json` — train mode only
- `pipeline.log`

## Smoke test

```bash
pytest tests/ -v
```

Runs ingest -> segment -> quality on the first 15 rows, plus embedding-
and GPT-credential-separation tests and the dynamic-scoring unit tests.
No embedding API calls, no network — safe for CI on every commit.

## Editing anchor phrases

Anchor phrases live in `config.yaml` under `features.anchors`, 5-10
paraphrases per construct. Set `enabled: false` on a construct to drop it
from scoring without deleting its phrase list. Every enabled construct must
have exactly one entry in `features.theme_construct_map`.

## Notes / assumptions

- Input file must match the column-name contract implied by
  `config.yaml`'s enabled themes (`ingest.required_columns`). The pipeline
  fails loudly if a required column is missing or renamed.
- `dummy_data.xlsx` generated by `generate_dummy_data.py` has no real LINK
  target — `labels.csv` here is synthetic (generated alongside it,
  internally consistent for testing purposes) — train mode against real
  ads still requires a real labels file.