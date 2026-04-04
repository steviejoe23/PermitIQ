#!/usr/bin/env python3
"""
normalize_decisions.py — Normalize decision labels in zba_cases_cleaned.csv

Re-derives decision_clean from the raw 'decision' column using the same logic as
rebuild_dataset.py, then recovers additional decisions from raw_text using regex
pattern matching. This ensures all approval/denial variants are mapped to canonical
APPROVED/DENIED labels.

Mapping from raw decision:
  APPROVED ← GRANTED, APPEAL SUSTAINED
  DENIED   ← DENIED, REFUSED, REJECTED, APPEAL DENIED

Mapping from tracker decision (via 'decision' column after merge):
  APPROVED ← AppProv, Approved
  DENIED   ← DeniedPrej, Denied

Usage:
  python3 normalize_decisions.py
"""

import pandas as pd
import re
import sys

CSV_PATH = "zba_cases_cleaned.csv"

# Combined mapping: raw parser + tracker decision values → canonical label
DECISION_MAP = {
    # From parser (rebuild_dataset.py Step 3a)
    "GRANTED": "APPROVED",
    "APPEAL SUSTAINED": "APPROVED",
    "DENIED": "DENIED",
    "REFUSED": "DENIED",
    "REJECTED": "DENIED",
    "APPEAL DENIED": "DENIED",
    # From tracker (rebuild_dataset.py Step 3b)
    "AppProv": "APPROVED",
    "Approved": "APPROVED",
    "DeniedPrej": "DENIED",
    "Denied": "DENIED",
    # Already-canonical values (idempotent)
    "APPROVED": "APPROVED",
}

# Regex patterns for raw_text recovery (from rebuild_dataset.py Step 3a)
APPROVE_PATTERNS = [
    r'voted\s+to\s+(?:approve|grant|sustain)',
    r'(?:appeal|petition|application)\s+(?:is\s+)?(?:hereby\s+)?(?:sustained|granted|approved)',
    r'board\s+(?:hereby\s+)?(?:votes?|voted)\s+(?:to\s+)?(?:approve|grant|sustain)',
    r'relief\s+(?:is\s+)?(?:hereby\s+)?granted',
    r'approved\s+with\s+(?:conditions|provisos|the\s+following)',
    r'(?:the\s+)?(?:zoning\s+)?(?:board|bza|zba)\s+(?:voted\s+to\s+)?(?:approv|grant|sustain)',
]

DENY_PATTERNS = [
    r'voted\s+to\s+(?:deny|refuse|reject|dismiss)',
    r'(?:appeal|petition|application)\s+(?:is\s+)?(?:hereby\s+)?(?:denied|refused|rejected|dismissed)',
    r'board\s+(?:hereby\s+)?(?:votes?|voted)\s+(?:to\s+)?(?:deny|refuse|reject)',
    r'relief\s+(?:is\s+)?(?:hereby\s+)?denied',
    r'(?:the\s+)?(?:zoning\s+)?(?:board|bza|zba)\s+(?:voted\s+to\s+)?(?:deny|refuse|reject)',
]


def recover_from_text(text):
    """Try to determine decision from raw case text."""
    text = str(text).lower()
    for pattern in APPROVE_PATTERNS:
        if re.search(pattern, text):
            return "APPROVED"
    for pattern in DENY_PATTERNS:
        if re.search(pattern, text):
            return "DENIED"
    return None


def main():
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Total rows: {len(df)}")

    # --- BEFORE ---
    print("\n=== BEFORE normalization ===")
    print("decision_clean value counts:")
    print(df["decision_clean"].value_counts(dropna=False).to_string())
    if "decision" in df.columns:
        print("\nRaw decision value counts:")
        print(df["decision"].value_counts(dropna=False).to_string())

    before_approved = (df["decision_clean"] == "APPROVED").sum()
    before_denied = (df["decision_clean"] == "DENIED").sum()
    before_nan = df["decision_clean"].isna().sum()

    # --- STEP 1: Map from raw decision column ---
    if "decision" in df.columns:
        df["decision_clean"] = df["decision"].map(DECISION_MAP)
        mapped = df["decision_clean"].notna().sum()
        print(f"\nStep 1: Mapped {mapped} decisions from raw 'decision' column")
    else:
        print("WARNING: No raw 'decision' column. Normalizing decision_clean in place.")
        df["decision_clean"] = df["decision_clean"].map(DECISION_MAP)

    # --- STEP 2: Recover from raw_text for still-missing rows ---
    null_mask = df["decision_clean"].isna()
    if "raw_text" in df.columns and null_mask.sum() > 0:
        recovered = 0
        for idx in df.index[null_mask]:
            result = recover_from_text(df.at[idx, "raw_text"])
            if result:
                df.at[idx, "decision_clean"] = result
                recovered += 1
        print(f"Step 2: Recovered {recovered} decisions from raw_text")
    else:
        print("Step 2: No raw_text column or no missing decisions to recover")

    # --- STEP 3: Try tracker CSV for remaining missing ---
    null_mask = df["decision_clean"].isna()
    tracker_path = "zba_tracker.csv"
    import os
    if null_mask.sum() > 0 and os.path.exists(tracker_path):
        tracker = pd.read_csv(tracker_path, low_memory=False)
        tracker_decision_map = {
            "AppProv": "APPROVED",
            "Approved": "APPROVED",
            "DeniedPrej": "DENIED",
            "Denied": "DENIED",
        }
        tracker["decision_mapped"] = tracker["decision"].map(tracker_decision_map)
        tracker_with_dec = tracker[tracker["decision_mapped"].notna()].copy()

        # Clean case numbers for matching
        tracker_with_dec = tracker_with_dec.copy()
        tracker_with_dec["_case"] = tracker_with_dec["boa_apno"].fillna("").astype(str).str.strip().str.replace("-", "", regex=False)
        df["_case"] = df["case_number"].fillna("").astype(str).str.strip().str.replace("-", "", regex=False)

        tracker_lookup = tracker_with_dec.drop_duplicates(subset="_case", keep="first").set_index("_case")["decision_mapped"].to_dict()

        recovered_tracker = 0
        for idx in df.index[null_mask]:
            case = df.at[idx, "_case"]
            if case in tracker_lookup:
                df.at[idx, "decision_clean"] = tracker_lookup[case]
                recovered_tracker += 1

        if "_case" in df.columns:
            df.drop(columns=["_case"], inplace=True)

        print(f"Step 3: Recovered {recovered_tracker} decisions from ZBA tracker")
    else:
        print("Step 3: No tracker recovery needed or tracker not found")

    # --- AFTER ---
    after_approved = (df["decision_clean"] == "APPROVED").sum()
    after_denied = (df["decision_clean"] == "DENIED").sum()
    after_nan = df["decision_clean"].isna().sum()

    print("\n=== AFTER normalization ===")
    print("decision_clean value counts:")
    print(df["decision_clean"].value_counts(dropna=False).to_string())

    print(f"\n=== CHANGES ===")
    print(f"APPROVED: {before_approved} → {after_approved} ({after_approved - before_approved:+d})")
    print(f"DENIED:   {before_denied} → {after_denied} ({after_denied - before_denied:+d})")
    print(f"NaN:      {before_nan} → {after_nan} ({after_nan - before_nan:+d})")

    # Verify only canonical values
    unexpected = df["decision_clean"].dropna().unique()
    unexpected = [v for v in unexpected if v not in ("APPROVED", "DENIED")]
    if unexpected:
        print(f"\nERROR: Unexpected decision values found: {unexpected}")
        sys.exit(1)

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved normalized data to {CSV_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
