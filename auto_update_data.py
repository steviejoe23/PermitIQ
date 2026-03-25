"""
PermitIQ — Auto-Update Data from Boston Open Data Portal
Pulls fresh ZBA decisions, building permits, and property assessments
from data.boston.gov's CKAN API, then re-integrates into the dataset.

Run manually:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 auto_update_data.py

Or set up a cron job to run weekly:
    crontab -e
    0 3 * * 1 cd ~/Desktop/Boston\ Zoning\ Project && source zoning-env/bin/activate && python3 auto_update_data.py >> auto_update.log 2>&1

Requires: pip install requests pandas
"""

import requests
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta

print("=" * 60)
print("  PermitIQ Auto-Update — Boston Open Data")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ========================================
# CONFIGURATION — Boston CKAN API
# ========================================

BASE_URL = "https://data.boston.gov/api/3/action/datastore_search"

# Resource IDs from data.boston.gov
RESOURCES = {
    "zba_tracker": {
        "resource_id": "0f0fa8c2-87ba-45d6-a876-0d177dd02512",
        "local_file": "zba_tracker.csv",
        "description": "ZBA Appeals Tracker",
    },
    "building_permits": {
        "resource_id": "6ddcd912-32a0-43df-9908-63574f8c7e77",
        "local_file": "building_permits.csv",
        "description": "Approved Building Permits",
    },
    "property_assessment": {
        "resource_id": "ee73430d-96c0-423e-ad21-c4cfb54c8961",
        "local_file": "property_assessment_fy2026.csv",
        "description": "Property Assessment FY2026",
    },
}

# How many records to pull per API call (CKAN default max is 32000)
BATCH_SIZE = 32000


def fetch_dataset(resource_id, description, limit=None):
    """Fetch a dataset from Boston's CKAN API with pagination."""
    print(f"\n--- Fetching: {description} ---")

    all_records = []
    offset = 0

    while True:
        params = {
            "resource_id": resource_id,
            "limit": BATCH_SIZE,
            "offset": offset,
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  ❌ API error at offset {offset}: {e}")
            break

        if not data.get("success"):
            print(f"  ❌ API returned error: {data.get('error', 'unknown')}")
            break

        records = data["result"]["records"]
        total = data["result"].get("total", 0)

        if not records:
            break

        all_records.extend(records)
        offset += BATCH_SIZE

        print(f"  Fetched {len(all_records)}/{total} records...")

        if limit and len(all_records) >= limit:
            all_records = all_records[:limit]
            break

        if len(all_records) >= total:
            break

    if all_records:
        df = pd.DataFrame(all_records)
        # Drop CKAN internal columns
        df = df.drop(columns=["_id", "_full_text"], errors="ignore")
        print(f"  ✅ Got {len(df)} records, {len(df.columns)} columns")
        return df
    else:
        print("  ⚠️ No records fetched")
        return None


def incremental_update_zba(new_df, existing_path):
    """Merge new ZBA tracker records with existing data."""
    if not os.path.exists(existing_path):
        print(f"  No existing file at {existing_path}, saving full dataset")
        new_df.to_csv(existing_path, index=False)
        return new_df

    existing = pd.read_csv(existing_path, low_memory=False)
    print(f"  Existing records: {len(existing)}")

    # Match on boa_apno (BOA case number)
    key_col = "boa_apno"
    if key_col not in new_df.columns:
        # Try alternate column names
        for alt in ["boa_apno", "BOA_APNO", "case_number"]:
            if alt in new_df.columns:
                key_col = alt
                break

    if key_col in new_df.columns and key_col in existing.columns:
        existing_keys = set(existing[key_col].astype(str).str.strip().str.upper())
        new_df["_key"] = new_df[key_col].astype(str).str.strip().str.upper()
        truly_new = new_df[~new_df["_key"].isin(existing_keys)].drop(columns=["_key"])

        if len(truly_new) > 0:
            print(f"  🆕 {len(truly_new)} new cases found!")
            combined = pd.concat([existing, truly_new], ignore_index=True)
            combined.to_csv(existing_path, index=False)
            print(f"  Saved: {len(combined)} total records")
            return combined
        else:
            print(f"  No new cases (all {len(new_df)} records already exist)")
            return existing
    else:
        print(f"  ⚠️ Key column '{key_col}' not found, replacing file")
        new_df.to_csv(existing_path, index=False)
        return new_df


def incremental_update_permits(new_df, existing_path):
    """Merge new building permits with existing data."""
    if not os.path.exists(existing_path):
        new_df.to_csv(existing_path, index=False)
        return new_df

    existing = pd.read_csv(existing_path, low_memory=False)
    print(f"  Existing records: {len(existing)}")

    key_col = "permitnumber"
    if key_col in new_df.columns and key_col in existing.columns:
        existing_keys = set(existing[key_col].astype(str).str.strip())
        new_df["_key"] = new_df[key_col].astype(str).str.strip()
        truly_new = new_df[~new_df["_key"].isin(existing_keys)].drop(columns=["_key"])

        if len(truly_new) > 0:
            print(f"  🆕 {len(truly_new)} new permits found!")
            combined = pd.concat([existing, truly_new], ignore_index=True)
            combined.to_csv(existing_path, index=False)
            print(f"  Saved: {len(combined)} total records")
            return combined
        else:
            print(f"  No new permits")
            return existing
    else:
        new_df.to_csv(existing_path, index=False)
        return new_df


# ========================================
# MAIN UPDATE FLOW
# ========================================

updated_any = False

# 1. ZBA Tracker — always fetch full (only ~16K records)
print("\n" + "=" * 60)
print("  1. ZBA TRACKER")
print("=" * 60)
zba_new = fetch_dataset(
    RESOURCES["zba_tracker"]["resource_id"],
    RESOURCES["zba_tracker"]["description"],
)
if zba_new is not None:
    result = incremental_update_zba(zba_new, RESOURCES["zba_tracker"]["local_file"])
    if result is not None:
        updated_any = True

# 2. Building Permits — large dataset, fetch all
print("\n" + "=" * 60)
print("  2. BUILDING PERMITS")
print("=" * 60)
bp_new = fetch_dataset(
    RESOURCES["building_permits"]["resource_id"],
    RESOURCES["building_permits"]["description"],
)
if bp_new is not None:
    result = incremental_update_permits(bp_new, RESOURCES["building_permits"]["local_file"])
    if result is not None:
        updated_any = True

# 3. Property Assessment — replace yearly (data changes with each FY)
print("\n" + "=" * 60)
print("  3. PROPERTY ASSESSMENT")
print("=" * 60)
pa_new = fetch_dataset(
    RESOURCES["property_assessment"]["resource_id"],
    RESOURCES["property_assessment"]["description"],
)
if pa_new is not None:
    pa_new.to_csv(RESOURCES["property_assessment"]["local_file"], index=False)
    print(f"  ✅ Saved {len(pa_new)} property records")
    updated_any = True

# ========================================
# RE-INTEGRATE IF UPDATED
# ========================================
if updated_any:
    print("\n" + "=" * 60)
    print("  RE-INTEGRATING DATA")
    print("=" * 60)
    print("  Running integration pipeline...")

    # Run the integration scripts in order
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "integrate_external_data.py"],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"  ⚠️ Integration warning: {result.stderr[-300:]}")
    except Exception as e:
        print(f"  ❌ Integration failed: {e}")
        print("  You can run manually: python3 integrate_external_data.py")

    try:
        result = subprocess.run(
            ["python3", "fuzzy_match_properties.py"],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"  ⚠️ Fuzzy match warning: {result.stderr[-300:]}")
    except Exception as e:
        print(f"  ❌ Fuzzy match failed: {e}")

    print("\n  ⚠️ Remember to retrain the model after data update:")
    print("    python3 train_model_v2.py")
else:
    print("\n  No updates found. Data is current.")

print("\n" + "=" * 60)
print(f"  AUTO-UPDATE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
