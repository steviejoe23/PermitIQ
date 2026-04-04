"""
Extract attorney and developer leads from ZBA cases dataset for sales outreach.

Outputs:
  leads/attorneys.csv  — Top attorneys ranked by ZBA case volume
  leads/developers.csv — Top developers/applicants ranked by case volume

Usage:
  python3 extract_leads.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "zba_cases_cleaned.csv")
LEADS_DIR = os.path.join(BASE_DIR, "leads")
os.makedirs(LEADS_DIR, exist_ok=True)

MIN_CASES = 3  # Minimum cases to be considered a "serious player"

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  {len(df):,} total cases loaded")

# Normalize decision
df["is_approved"] = df["decision_clean"].str.upper().eq("APPROVED").astype(int)

# Parse filing dates to get most recent case info
def parse_date(val):
    """Try multiple date formats."""
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return pd.NaT

df["filing_date_parsed"] = df["filing_date"].apply(parse_date)


def top_items(series, n=3):
    """Return top N most common items as comma-separated string."""
    counts = Counter(series.dropna())
    return ", ".join([item for item, _ in counts.most_common(n)])


def top_variance_types(variance_col, n=5):
    """Extract and count individual variance types across cases."""
    all_types = []
    for val in variance_col.dropna():
        all_types.extend(str(val).split(","))
    counts = Counter(t.strip() for t in all_types if t.strip())
    return ", ".join([t for t, _ in counts.most_common(n)])


# ---------------------------------------------------------------------------
# 1. ATTORNEYS / CONTACTS
# ---------------------------------------------------------------------------
print("\n--- Extracting Attorneys/Contacts ---")

# Clean contact names
contacts = df[df["contact"].notna()].copy()
contacts["contact_clean"] = (
    contacts["contact"]
    .str.strip()
    .str.title()
    # Normalize common OCR/formatting issues
    .str.replace(r"\s+", " ", regex=True)
)

# Filter out obvious non-person entries (companies acting as contacts are fine — they're still leads)
# But filter garbage entries
contacts = contacts[contacts["contact_clean"].str.len() > 2]
contacts = contacts[~contacts["contact_clean"].str.match(r"^\d+$")]  # pure numbers

# Group by contact
attorney_groups = contacts.groupby("contact_clean")

attorney_records = []
for name, group in attorney_groups:
    total_cases = len(group)
    if total_cases < MIN_CASES:
        continue

    wins = group["is_approved"].sum()
    win_rate = wins / total_cases if total_cases > 0 else 0

    # Most common wards
    wards = group["ward"].dropna().astype(int, errors="ignore")
    ward_str = top_items(wards.astype(str), n=3)

    # Most common variance types
    var_str = top_variance_types(group["variance_types"], n=5)

    # Most recent case
    most_recent = group["filing_date_parsed"].max()
    most_recent_str = most_recent.strftime("%Y-%m-%d") if pd.notna(most_recent) else ""

    # Sample addresses (for context)
    sample_addresses = ", ".join(
        group["address"].dropna().unique()[:3]
    )

    # Is this person likely an attorney? (appears in has_attorney=1 cases)
    attorney_cases = group["has_attorney"].sum()
    likely_attorney = attorney_cases > 0

    attorney_records.append({
        "name": name,
        "total_cases": total_cases,
        "wins": int(wins),
        "losses": total_cases - int(wins),
        "win_rate": round(win_rate * 100, 1),
        "likely_attorney": likely_attorney,
        "top_wards": ward_str,
        "top_variance_types": var_str,
        "most_recent_case": most_recent_str,
        "sample_addresses": sample_addresses,
    })

attorneys_df = pd.DataFrame(attorney_records)
attorneys_df = attorneys_df.sort_values("total_cases", ascending=False).reset_index(drop=True)

out_path = os.path.join(LEADS_DIR, "attorneys.csv")
attorneys_df.to_csv(out_path, index=False)
print(f"  {len(attorneys_df)} contacts with {MIN_CASES}+ cases saved to {out_path}")

# ---------------------------------------------------------------------------
# 2. DEVELOPERS / APPLICANTS
# ---------------------------------------------------------------------------
print("\n--- Extracting Developers/Applicants ---")

applicants = df[df["applicant_name"].notna()].copy()
applicants["applicant_clean"] = (
    applicants["applicant_name"]
    .str.strip()
    .str.title()
    .str.replace(r"\s+", " ", regex=True)
)

# Filter garbage
applicants = applicants[applicants["applicant_clean"].str.len() > 2]
applicants = applicants[~applicants["applicant_clean"].str.match(r"^\d+$")]

applicant_groups = applicants.groupby("applicant_clean")

dev_records = []
for name, group in applicant_groups:
    total_cases = len(group)
    if total_cases < MIN_CASES:
        continue

    wins = group["is_approved"].sum()
    success_rate = wins / total_cases if total_cases > 0 else 0

    ward_str = top_items(group["ward"].dropna().astype(int, errors="ignore").astype(str), n=3)
    var_str = top_variance_types(group["variance_types"], n=5)

    most_recent = group["filing_date_parsed"].max()
    most_recent_str = most_recent.strftime("%Y-%m-%d") if pd.notna(most_recent) else ""

    sample_addresses = ", ".join(group["address"].dropna().unique()[:3])

    # Check if they typically use an attorney
    uses_attorney_pct = group["has_attorney"].mean() * 100

    # Common contact/attorney they use
    attorney_used = top_items(group["contact"].dropna(), n=2)

    dev_records.append({
        "name": name,
        "total_cases": total_cases,
        "wins": int(wins),
        "losses": total_cases - int(wins),
        "success_rate": round(success_rate * 100, 1),
        "uses_attorney_pct": round(uses_attorney_pct, 0),
        "common_attorneys": attorney_used,
        "top_wards": ward_str,
        "top_variance_types": var_str,
        "most_recent_filing": most_recent_str,
        "sample_addresses": sample_addresses,
    })

devs_df = pd.DataFrame(dev_records)
devs_df = devs_df.sort_values("total_cases", ascending=False).reset_index(drop=True)

out_path = os.path.join(LEADS_DIR, "developers.csv")
devs_df.to_csv(out_path, index=False)
print(f"  {len(devs_df)} developers with {MIN_CASES}+ cases saved to {out_path}")

# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("LEAD EXTRACTION SUMMARY")
print("=" * 70)

print(f"\nAttorneys/Contacts with {MIN_CASES}+ cases: {len(attorneys_df)}")
print(f"Developers/Applicants with {MIN_CASES}+ cases: {len(devs_df)}")

print(f"\n--- TOP 10 ATTORNEYS BY VOLUME ---")
print(f"{'Name':<30} {'Cases':>6} {'Win%':>6} {'Top Wards':<20} {'Recent':>12}")
print("-" * 80)
for _, row in attorneys_df.head(10).iterrows():
    print(f"{row['name']:<30} {row['total_cases']:>6} {row['win_rate']:>5.1f}% {row['top_wards']:<20} {row['most_recent_case']:>12}")

print(f"\n--- TOP 10 DEVELOPERS BY VOLUME ---")
print(f"{'Name':<30} {'Cases':>6} {'Win%':>6} {'Atty%':>6} {'Top Wards':<20} {'Recent':>12}")
print("-" * 90)
for _, row in devs_df.head(10).iterrows():
    print(f"{row['name']:<30} {row['total_cases']:>6} {row['success_rate']:>5.1f}% {row['uses_attorney_pct']:>5.0f}% {row['top_wards']:<20} {row['most_recent_filing']:>12}")

print(f"\nFiles saved:")
print(f"  leads/attorneys.csv")
print(f"  leads/developers.csv")
