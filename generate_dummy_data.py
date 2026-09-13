"""Generates a synthetic respondent-level dataset matching the ACTUAL
4-theme schema implemented in ingest.py / segment.py:

    Respondent ID, Ad ID, Brand, Ad / Platform, Order Shown,
    Memory response - what happened,
    Meaning response - what the ad was saying,
    Emotion response - feeling and why,
    Optimisation response - one improvement,
    Why improvement would help

There is no ad catalog, no brand-recall column, and no per-ad
"intended message" comprehension scoring in this pipeline — an earlier
draft of this generator built a 9-theme/electricals/ad-catalog dataset
that was never matched by ingest.py/features.py's actual implementation.
This version generates exactly the columns ingest.required_columns()
will ask for (given config.yaml's segment.themes), plus a synthetic
labels.csv for train-mode testing.

Design note — combinatorial phrase generation: verbatims are composed
from 3 independently-sampled fragment slots (opener x detail x closer)
plus a shared filler-sentence pool appended with ~45% probability, to
keep quality.py's near_identical_cross_respondent quarantine check
from flagging most of the dataset once more than one respondent per
Ad ID + Theme is generated.

Design note — latent ad quality: each ad gets a hidden
quality_latent in [0, 1], which drives both the sentiment mix of the
generated verbatims AND the LINK_score/LINK_band in labels.csv, so the
ad-level features this pipeline computes actually correlate with the
labels (useful for testing model.py's classifier/SHAP meaningfully,
rather than against independent noise).

Usage:
    python generate_dummy_data.py --n-ads 10 --n-respondents-per-ad 10 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

BRANDS = [
    "Eveready", "Philips", "Havells", "Bajaj Electricals", "Orient Electric",
    "Crompton", "Usha", "V-Guard", "Anchor by Panasonic", "Polycab",
]

AD_PLATFORMS = [
    "TVC 30s", "TVC 20s", "Digital Video 15s", "YouTube Pre-roll 10s",
    "Social Media Reel 15s",
]

SENTIMENTS = ["positive", "neutral", "negative"]

# Shared filler sentences appended at random — purely to multiply the
# combination space and cut near-duplicate collisions; not
# semantically load-bearing for any construct.
FILLERS = [
    "I watched this on my phone.",
    "I saw this during a TV break at home.",
    "This came up while I was scrolling.",
    "I was half-distracted when I saw it, to be honest.",
    "My family was around when this played.",
    "I've seen a few ads like this recently.",
    "I watched it twice before answering.",
    "It played before a video I was watching.",
    "It was on in the background while I was doing chores.",
    "I paused what I was doing to watch it properly.",
    "",
    "",
    "",
]


def _compose(rng: random.Random, *slots: list[str], filler_prob: float = 0.45) -> str:
    parts = [rng.choice(slot) for slot in slots if slot]
    if rng.random() < filler_prob:
        parts.append(rng.choice(FILLERS))
    return " ".join(p for p in parts if p).strip()


# --- Memory ---
MEMORY_OPENERS = [
    "I remember it showed", "From what I recall, the ad had", "The ad opened with",
    "What stuck with me was", "I saw a scene with", "It started off showing",
    "The clip I remember had", "As far as I remember, there was",
]
MEMORY_DETAILS = [
    "a family using the product at home", "someone comparing it with an older version",
    "a demonstration of how it works", "a shop scene with a price shown on screen",
    "a close-up of the product's design", "people looking happy while using it",
    "a before-and-after kind of scene", "a voiceover explaining the main benefit",
]
MEMORY_CLOSERS = [
    "and then the brand name appeared at the end.", "with some background music playing throughout.",
    "and a voiceover explaining the product.", "before it cut to the price and where to buy it.",
    "and it ended with a call to action.", "",
]


def gen_memory(rng: random.Random) -> str:
    return _compose(rng, MEMORY_OPENERS, MEMORY_DETAILS, MEMORY_CLOSERS)


# --- Meaning (message understanding, generic — not scored against a
# per-ad ground truth in this schema, just an anchor-cosine construct
# like every other theme) ---
MEANING_FRAGMENTS = {
    "positive": {
        "opener": ["I think the ad was clearly saying that", "The main message was pretty clear:", "It was trying to tell me that"],
        "detail": ["this brand is reliable and worth choosing.", "this product solves a real everyday problem.", "you get good value for the price."],
        "closer": ["That came through clearly.", "It made sense to me.", ""],
    },
    "neutral": {
        "opener": ["I think the ad was saying something like", "It was sort of saying that"],
        "detail": ["the product is decent.", "it's an okay option to consider."],
        "closer": ["Not totally sure though.", ""],
    },
    "negative": {
        "opener": ["Honestly I'm not sure what the ad was really trying to say.", "I couldn't quite tell what the main point was.", "It wasn't very clear to me what message they wanted to give."],
        "detail": ["Maybe something about the product being good, but it wasn't clear.", "I got a bit lost on what exactly they were promoting."],
        "closer": ["I'd have to watch it again.", ""],
    },
}


def gen_meaning(sentiment: str, rng: random.Random) -> str:
    pool = MEANING_FRAGMENTS[sentiment]
    return _compose(rng, pool["opener"], pool["detail"], pool["closer"])


# --- Emotion ---
EMOTION_FRAGMENTS = {
    "positive": {
        "opener": ["I really enjoyed watching this ad.", "This ad was fun to watch.", "I liked it a lot.", "It made me feel good."],
        "detail": ["The tone felt upbeat and pleasant.", "It made me smile a couple of times.", "It had a nice, cheerful energy."],
        "closer": ["I'd happily watch it again.", "It left a good impression.", ""],
    },
    "neutral": {
        "opener": ["It was okay, nothing special.", "I felt neutral about it.", "Fine, I guess."],
        "detail": ["It didn't really stand out to me.", "It felt like a fairly standard ad."],
        "closer": ["Nothing more to add really.", ""],
    },
    "negative": {
        "opener": ["I didn't really enjoy this ad.", "It felt a bit boring to me.", "I wasn't a fan of it."],
        "detail": ["It felt repetitive.", "It dragged on a bit too long.", "The tone felt flat."],
        "closer": ["I probably wouldn't watch it again.", ""],
    },
}


def gen_emotion(sentiment: str, rng: random.Random) -> str:
    pool = EMOTION_FRAGMENTS[sentiment]
    return _compose(rng, pool["opener"], pool["detail"], pool["closer"])


# --- Optimisation (compound: improvement + why) ---
OPTIMISATION_IMPROVEMENTS = [
    "show a real customer testimonial", "explain the price more clearly", "make the ad a bit shorter",
    "show a direct comparison with competitor products", "highlight the warranty or after-sales support",
    "show the product being used in different situations", "add clearer information about where to buy it",
    "make the brand name more visible earlier in the ad", "show more of the actual product in use",
    "use a more relatable spokesperson",
]
OPTIMISATION_WHYS = [
    "so I could trust the claims more.", "because I wasn't sure about the exact cost.",
    "because it felt a bit long for the message it was giving.",
    "so I could see how it's actually better than other options.",
    "since that matters a lot when I'm deciding what to buy.",
    "to make the ad feel more real and relatable.", "so I'd know how to actually get the product.",
    "because I almost forgot which brand it was for.",
]


def gen_optimisation(rng: random.Random) -> tuple[str, str]:
    return rng.choice(OPTIMISATION_IMPROVEMENTS), rng.choice(OPTIMISATION_WHYS)


def sentiment_for_respondent(quality_latent: float, rng: random.Random) -> str:
    p_pos = 0.15 + 0.65 * quality_latent
    p_neg = 0.60 - 0.45 * quality_latent
    p_neu = max(0.0, 1.0 - p_pos - p_neg)
    return rng.choices(SENTIMENTS, weights=[p_pos, p_neu, p_neg], k=1)[0]


def generate_respondents_for_ad(
    ad_id: str, brand: str, platform: str, quality_latent: float,
    n_respondents: int, rng: random.Random, resp_counter: list[int],
) -> list[dict]:
    rows = []
    for _ in range(n_respondents):
        resp_counter[0] += 1
        respondent_id = f"R{resp_counter[0]:03d}"
        sentiment = sentiment_for_respondent(quality_latent, rng)
        improvement, why = gen_optimisation(rng)
        rows.append({
            "Respondent ID": respondent_id,
            "Ad ID": ad_id,
            "Brand": brand,
            "Ad / Platform": platform,
            "Order Shown": rng.randint(1, 3),
            "Memory response - what happened": gen_memory(rng),
            "Meaning response - what the ad was saying": gen_meaning(sentiment, rng),
            "Emotion response - feeling and why": gen_emotion(sentiment, rng),
            "Optimisation response - one improvement": improvement,
            "Why improvement would help": why,
        })
    return rows


def build_labels(ads: list[dict], rng: random.Random) -> pd.DataFrame:
    """LINK_score derived from the same quality_latent that drove the
    verbatims, plus noise — so the feature table and the labels are
    genuinely related, not independently random."""
    rows = []
    for ad in ads:
        latent = ad["quality_latent"]
        noise = rng.gauss(0, 8)
        score = max(1, min(99, round(latent * 100 + noise)))
        if score >= 70:
            band = "Strong"
        elif score >= 31:
            band = "Average"
        else:
            band = "Weak"
        rows.append({"Ad ID": ad["ad_id"], "LINK_band": band, "LINK_score": int(score)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Project LISTEN dummy data (4-theme schema)")
    parser.add_argument("--n-ads", type=int, default=3)
    parser.add_argument("--n-respondents-per-ad", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ads = []
    for i in range(args.n_ads):
        ads.append({
            "ad_id": f"AD{i + 1:02d}",
            "brand": rng.choice(BRANDS),
            "platform": rng.choice(AD_PLATFORMS),
            "quality_latent": rng.random(),
        })

    resp_counter = [0]
    all_rows = []
    for ad in ads:
        all_rows.extend(
            generate_respondents_for_ad(
                ad["ad_id"], ad["brand"], ad["platform"], ad["quality_latent"],
                args.n_respondents_per_ad, rng, resp_counter,
            )
        )

    dummy_data = pd.DataFrame(all_rows)
    labels = build_labels(ads, rng)

    dummy_data_path = out_dir / "dummy_data.xlsx"
    labels_path = out_dir / "labels.csv"

    dummy_data.to_excel(dummy_data_path, index=False, sheet_name="Sheet1")
    labels.to_csv(labels_path, index=False)

    print(f"Wrote {dummy_data_path} ({len(dummy_data)} respondent rows across "
          f"{args.n_ads} ads, 4 themes each)")
    print(f"Wrote {labels_path} ({len(labels)} labeled ads)")
    print(labels["LINK_band"].value_counts().to_string())


if __name__ == "__main__":
    main()