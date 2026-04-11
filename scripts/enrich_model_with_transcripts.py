#!/usr/bin/env python3
"""Enrich ZBA model training data with transcript-derived features.

Takes the parsed transcript data (parsed_cases.jsonl) and creates
features that can be merged into the main ZBA dataset for model training.

New features derived from transcripts:
- hearing_word_count: Total words in the hearing (proxy for case complexity)
- hearing_support_mentions: Count of support/favor mentions
- hearing_opposition_mentions: Count of opposition/concern mentions
- hearing_sentiment_ratio: support / (support + opposition)
- hearing_has_attorney: Whether an attorney was mentioned
- hearing_variance_count: Number of distinct variance types discussed
- hearing_neighborhood_count: Neighborhoods covered in hearing
- hearing_approval_rate: Ratio of approve to deny mentions in hearing

Usage:
    python scripts/enrich_model_with_transcripts.py
"""

import json
import os
import pandas as pd
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).parent.parent / "data" / "zba_transcripts"
PARSED_FILE = TRANSCRIPTS_DIR / "parsed_cases.jsonl"
OUTPUT_FILE = TRANSCRIPTS_DIR / "transcript_features.csv"
SUMMARY_FILE = TRANSCRIPTS_DIR / "transcript_summary.json"


def load_parsed_cases() -> list:
    cases = []
    with open(PARSED_FILE) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def build_date_features(cases: list) -> pd.DataFrame:
    """Build per-hearing-date feature rows."""
    rows = []
    for case in cases:
        date = case.get("hearing_date", "unknown")
        if date == "unknown":
            continue

        support = case.get("support_mentions", 0)
        opposition = case.get("opposition_mentions", 0)
        total_sentiment = support + opposition

        approved = case.get("total_approved", 0)
        denied = case.get("total_denied", 0)
        total_decisions = approved + denied

        rows.append({
            "hearing_date": date,
            "hearing_word_count": case.get("word_count", 0),
            "hearing_support_mentions": support,
            "hearing_opposition_mentions": opposition,
            "hearing_sentiment_ratio": round(support / total_sentiment, 3) if total_sentiment > 0 else 0.5,
            "hearing_has_attorney": 1 if case.get("attorneys") else 0,
            "hearing_attorney_count": len(case.get("attorneys", [])),
            "hearing_variance_count": len(case.get("variances_mentioned", [])),
            "hearing_neighborhood_count": len(case.get("neighborhoods", [])),
            "hearing_address_count": len(case.get("addresses", [])),
            "hearing_approval_mentions": approved,
            "hearing_denial_mentions": denied,
            "hearing_approval_rate": round(approved / total_decisions, 3) if total_decisions > 0 else 0.5,
            "hearing_deferred_count": case.get("total_deferred", 0),
        })

    return pd.DataFrame(rows)


def build_summary(cases: list, df: pd.DataFrame) -> dict:
    """Build summary statistics for the transcript corpus."""
    total_words = sum(c.get("word_count", 0) for c in cases)
    total_hearings = len(cases)
    date_range = sorted(set(c["hearing_date"] for c in cases if c.get("hearing_date") != "unknown"))

    all_neighborhoods = set()
    all_variances = set()
    all_attorneys = set()
    all_addresses = set()

    for c in cases:
        all_neighborhoods.update(c.get("neighborhoods", []))
        all_variances.update(c.get("variances_mentioned", []))
        all_attorneys.update(c.get("attorneys", []))
        all_addresses.update(c.get("addresses", []))

    return {
        "total_hearings": total_hearings,
        "total_words": total_words,
        "date_range": f"{date_range[0]} to {date_range[-1]}" if date_range else "unknown",
        "unique_dates": len(date_range),
        "unique_neighborhoods": len(all_neighborhoods),
        "neighborhoods": sorted(all_neighborhoods),
        "unique_variance_types": len(all_variances),
        "variance_types": sorted(all_variances),
        "unique_attorneys": len(all_attorneys),
        "top_attorneys": sorted(all_attorneys)[:20],
        "unique_addresses": len(all_addresses),
        "avg_words_per_hearing": round(total_words / total_hearings) if total_hearings > 0 else 0,
        "total_approval_mentions": int(df["hearing_approval_mentions"].sum()),
        "total_denial_mentions": int(df["hearing_denial_mentions"].sum()),
        "avg_sentiment_ratio": round(float(df["hearing_sentiment_ratio"].mean()), 3),
    }


def main():
    if not PARSED_FILE.exists():
        print("No parsed cases found. Run parse_zba_transcripts.py first.")
        return

    cases = load_parsed_cases()
    print(f"Loaded {len(cases)} hearing records")

    # Build features
    df = build_date_features(cases)
    print(f"Built {len(df)} feature rows")

    # Save features CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved features to {OUTPUT_FILE}")

    # Build and save summary
    summary = build_summary(cases, df)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {SUMMARY_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"TRANSCRIPT CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"  Hearings: {summary['total_hearings']}")
    print(f"  Total words: {summary['total_words']:,}")
    print(f"  Date range: {summary['date_range']}")
    print(f"  Avg words/hearing: {summary['avg_words_per_hearing']:,}")
    print(f"  Unique addresses: {summary['unique_addresses']}")
    print(f"  Unique attorneys: {summary['unique_attorneys']}")
    print(f"  Neighborhoods: {summary['unique_neighborhoods']}")
    print(f"  Variance types: {summary['unique_variance_types']}")
    print(f"  Approval mentions: {summary['total_approval_mentions']:,}")
    print(f"  Denial mentions: {summary['total_denial_mentions']:,}")
    print(f"  Avg sentiment ratio: {summary['avg_sentiment_ratio']}")
    print(f"{'='*60}")

    # Show how to merge with ZBA dataset
    print(f"\nTo merge with ZBA dataset, join on hearing_date column.")
    print(f"Example: zba_df.merge(transcript_features, left_on='date', right_on='hearing_date', how='left')")


if __name__ == "__main__":
    main()
