"""Project LISTEN — respondent chatbot.

An actual interactive CLI chat flow that asks a respondent the
open-ended questions this pipeline is built to score (Memory, Meaning,
Emotion, and the two-part Optimisation question), for one ad — and
writes the answers straight into the exact schema ``ingest.py``
expects, so the output can be appended directly to your respondent
xlsx (e.g. ``data/dummy_data.xlsx``) with no manual reformatting.

Question order and rationale:
  1. Memory        — spontaneous recall, no priming, asked first
                      before any other question can bias it.
  2. Meaning        — what the respondent thinks the ad was saying.
  3. Emotion        — feeling and why.
  4. Optimisation   — two-part (what to change / why), asked last
                      since it's the most effortful, reflective
                      question.

This targets the ACTUAL 4-theme schema implemented in ingest.py /
segment.py (Memory, Meaning, Emotion, Optimisation) — there is no
ad-catalog file, brand-recall fuzzy matching, or per-ad "intended
message" comprehension scoring in this pipeline; those were an earlier
9-theme design that was never implemented in ingest.py/features.py and
have been removed from this script to avoid drift between the tool
and the docs.

Question set and required columns are driven by config.yaml's
segment.themes block, exactly like ingest.py itself — so a theme that
is disabled there (e.g. Optimisation turned off for a wave that didn't
field it) is skipped here too, and the respondent isn't asked a
question whose column ingest won't require.

Usage:
    python chatbot.py --config config/config.yaml --out data/dummy_data.xlsx --ad-id AD01
    python chatbot.py --config config/config.yaml --out data/dummy_data.xlsx --ad-id AD02 --respondent-id R101 --brand Acme --platform YouTube

Notes:
  - This does NOT play the ad itself — it assumes the respondent has
    just watched it (in person, on a shared screen, or via a link sent
    separately). Wiring in actual video playback is a separate
    front-end concern outside this CLI's scope.
  - The session time budget is tracked and shown to the respondent,
    but not hard-enforced — hard-stopping mid-question would just
    produce a blank/short response that gets quarantined anyway.
  - Every answer is re-prompted once if left completely blank, since
    an empty respondent-level cell fails ingest's near-duplicate /
    too-short checks anyway and produces a wasted respondent row.
  - There's no ad catalog in this schema, so Brand / Ad-Platform for
    a session are supplied by the operator via --brand / --platform
    (or interactively), not looked up automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from project_listen import ingest  # noqa: E402

TOTAL_BUDGET_SECONDS = 6 * 60

# Prompt text per Theme. Only themes enabled in config.yaml's
# segment.themes are actually asked (see build_questions).
THEME_PROMPTS: dict[str, list[dict]] = {
    "Memory": [
        {
            "column": "Memory response - what happened",
            "prompt": "To start, in your own words — what do you remember happening in the ad you just watched?",
        },
    ],
    "Meaning": [
        {
            "column": "Meaning response - what the ad was saying",
            "prompt": "In your own words, what was the ad trying to tell you? What was its main message?",
        },
    ],
    "Emotion": [
        {
            "column": "Emotion response - feeling and why",
            "prompt": "How did watching this ad make you feel, and why?",
        },
    ],
    "Optimisation": [
        {
            "column": "Optimisation response - one improvement",
            "prompt": "If you could change ONE thing about this ad, what would it be?",
        },
        {
            "column": "Why improvement would help",
            "prompt": "Why do you think that change would help?",
        },
    ],
}

# Order themes are asked in, regardless of dict order in config.yaml.
THEME_ORDER = ["Memory", "Meaning", "Emotion", "Optimisation"]


def build_questions(theme_config: dict[str, dict]) -> list[dict]:
    """Flatten enabled themes (per config.yaml's segment.themes) into
    an ordered list of {column, prompt} dicts, in THEME_ORDER."""
    active = {name for name, cfg in theme_config.items() if cfg.get("enabled", True)}
    questions: list[dict] = []
    for theme in THEME_ORDER:
        if theme in active and theme in THEME_PROMPTS:
            questions.extend(THEME_PROMPTS[theme])
    if not questions:
        raise ValueError(
            "No enabled theme in config.yaml's segment.themes has a known chatbot "
            "prompt — nothing to ask."
        )
    return questions


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _ask(prompt: str, elapsed_fn) -> str:
    remaining = TOTAL_BUDGET_SECONDS - elapsed_fn()
    print(f"\n[{_fmt_mmss(remaining)} left in this session]")
    print(prompt)
    answer = input("> ").strip()
    if not answer:
        print("(Just a quick note — even a short answer helps. Go ahead and share what comes to mind.)")
        answer = input("> ").strip()
    return answer


def load_theme_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["segment"]["themes"]


def run_session(
    questions: list[dict], ad_id: str, respondent_id: str, brand: str,
    platform: str, order_shown: int,
) -> dict:
    start = time.monotonic()
    elapsed_fn = lambda: time.monotonic() - start  # noqa: E731

    print("=" * 60)
    print("Thanks for watching that ad. There are no right or wrong")
    print("answers, just tell us what you think.")
    print("=" * 60)

    answers: dict[str, str] = {}
    for q in questions:
        answers[q["column"]] = _ask(q["prompt"], elapsed_fn)

    total_elapsed = elapsed_fn()
    print(f"\nAll done — thank you! (session took {_fmt_mmss(total_elapsed)})")
    if total_elapsed > TOTAL_BUDGET_SECONDS * 1.5:
        print("(Ran noticeably over the target time — worth reviewing question length/order.)")

    row = {
        "Respondent ID": respondent_id,
        "Ad ID": ad_id,
        "Brand": brand,
        "Ad / Platform": platform,
        "Order Shown": order_shown,
        **answers,
    }
    return row


def append_row(out_path: str, row: dict, theme_config: dict) -> None:
    """Append one respondent row, validating against the SAME dynamic
    required-columns contract ingest.py itself uses — so a session
    collected here always matches what the pipeline will later accept.
    """
    required = ingest.required_columns(theme_config)
    out_path = Path(out_path)
    new_row_df = pd.DataFrame([row])[required]

    if out_path.exists():
        existing = pd.read_excel(out_path, sheet_name="Sheet1", engine="openpyxl")
        ingest.validate_columns(existing, "Sheet1", required)
        combined = pd.concat([existing[required], new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_excel(out_path, index=False, sheet_name="Sheet1")
    print(f"Saved response to {out_path} ({len(combined)} total respondent rows).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project LISTEN respondent chatbot")
    parser.add_argument("--config", default="config/config.yaml",
                         help="Path to config.yaml — determines which themes are asked "
                              "(segment.themes) and the required-columns contract.")
    parser.add_argument("--out", default="data/chat_sessions.xlsx",
                         help="Where to append this session's row. Point this at your main "
                              "respondent xlsx to merge directly, or a separate file to review "
                              "chat-collected responses before merging.")
    parser.add_argument("--ad-id", required=True, help="Ad ID this session is for, e.g. AD01.")
    parser.add_argument("--respondent-id", default=None, help="Defaults to a timestamp-based ID.")
    parser.add_argument("--brand", default="", help="Value for the 'Brand' column.")
    parser.add_argument("--platform", default="Chatbot Live Session",
                         help="Value for the 'Ad / Platform' column.")
    parser.add_argument("--order-shown", type=int, default=1)
    args = parser.parse_args()

    theme_config = load_theme_config(args.config)
    questions = build_questions(theme_config)

    respondent_id = args.respondent_id or f"CHAT{int(time.time())}"
    brand = args.brand or input("Brand for this ad (used in the 'Brand' column): ").strip()

    row = run_session(questions, args.ad_id, respondent_id, brand, args.platform, args.order_shown)
    append_row(args.out, row, theme_config)


if __name__ == "__main__":
    main()