"""
PermitIQ — Clean Dataset Rebuild
Rebuilds zba_cases_cleaned.csv from the base zba_cases_dataset.csv,
deduplicates, then runs the full pipeline: reextract → integrate → fuzzy match.

Run this whenever the dataset gets corrupted or you want a clean rebuild:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 rebuild_dataset.py
"""

import pandas as pd
import numpy as np
import subprocess
import os
from datetime import datetime

print("=" * 60)
print("  PermitIQ — Clean Dataset Rebuild")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ========================================
# STEP 1: Load base dataset
# ========================================
print("\n--- Step 1: Loading base dataset ---")
base_file = 'zba_cases_dataset.csv'
if not os.path.exists(base_file):
    print(f"❌ {base_file} not found! Run the OCR pipeline first.")
    exit(1)

df = pd.read_csv(base_file, low_memory=False)
print(f"Base rows: {len(df)}")
print(f"Unique case numbers: {df['case_number'].nunique()}")
print(f"Source PDFs: {df['source_pdf'].nunique()}")

# ========================================
# STEP 2: Deduplicate — keep best version of each case
# ========================================
print("\n--- Step 2: Deduplicating ---")

# For cases that appear in multiple PDFs, keep the one with the longest raw_text
# (most complete OCR extraction)
df['text_length'] = df['raw_text'].fillna('').str.len()
df = df.sort_values('text_length', ascending=False)
df = df.drop_duplicates(subset=['case_number'], keep='first')

print(f"After dedup: {len(df)} unique cases")

# ========================================
# STEP 3: Clean decisions
# ========================================
print("\n--- Step 3: Cleaning decisions ---")

# Normalize decisions
decision_map = {
    'GRANTED': 'APPROVED',
    'APPEAL SUSTAINED': 'APPROVED',
    'DENIED': 'DENIED',
    'REFUSED': 'DENIED',
    'REJECTED': 'DENIED',
    'APPEAL DENIED': 'DENIED',
}

df['decision_clean'] = df['decision'].map(decision_map)
has_decision = df['decision_clean'].notna().sum()
print(f"Cases with clean decisions: {has_decision} ({has_decision/len(df):.1%})")
print(f"  APPROVED: {(df['decision_clean'] == 'APPROVED').sum()}")
print(f"  DENIED: {(df['decision_clean'] == 'DENIED').sum()}")

# ========================================
# STEP 4: Save clean base
# ========================================
print("\n--- Step 4: Saving clean base ---")
df.to_csv('zba_cases_cleaned.csv', index=False)
print(f"Saved: {len(df)} rows, {len(df.columns)} columns")

# ========================================
# STEP 5: Run reextract_features.py
# ========================================
print("\n--- Step 5: Re-extracting features ---")
result = subprocess.run(
    ["python3", "reextract_features.py"],
    capture_output=True, text=True, timeout=600
)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    for line in lines[-8:]:
        print(f"  {line}")
else:
    print(f"  ⚠️ Error: {result.stderr[-300:]}")

# Verify row count hasn't changed
df2 = pd.read_csv('zba_cases_cleaned.csv', usecols=['case_number'], low_memory=False)
print(f"\n  Rows after reextract: {len(df2)}")

# ========================================
# STEP 6: Run integrate_external_data.py
# ========================================
print("\n--- Step 6: Integrating external data ---")
result = subprocess.run(
    ["python3", "integrate_external_data.py"],
    capture_output=True, text=True, timeout=600
)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    for line in lines[-8:]:
        print(f"  {line}")
else:
    print(f"  ⚠️ Error: {result.stderr[-300:]}")

# Check row count — this is where expansion can happen
df3 = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"\n  Rows after integration: {len(df3)}")
if len(df3) > len(df2) * 1.5:
    print(f"  ⚠️ Row explosion detected! Expected ~{len(df2)}, got {len(df3)}")
    print(f"  Auto-fixing: deduplicating by case_number...")
    df3['_text_len'] = df3['raw_text'].fillna('').str.len()
    df3 = df3.sort_values('_text_len', ascending=False).drop_duplicates(subset=['case_number'], keep='first')
    df3 = df3.drop(columns=['_text_len'])
    df3.to_csv('zba_cases_cleaned.csv', index=False)
    print(f"  Fixed: {len(df3)} rows after auto-dedup")

# ========================================
# STEP 7: Run fuzzy_match_properties.py
# ========================================
print("\n--- Step 7: Fuzzy matching properties ---")
result = subprocess.run(
    ["python3", "fuzzy_match_properties.py"],
    capture_output=True, text=True, timeout=600
)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"  {line}")
else:
    print(f"  ⚠️ Error: {result.stderr[-300:]}")

# ========================================
# FINAL CHECK
# ========================================
print("\n" + "=" * 60)
df_final = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"  FINAL DATASET: {len(df_final)} rows, {len(df_final.columns)} columns")
print(f"  Unique case numbers: {df_final['case_number'].nunique()}")
print(f"  Decisions: {df_final['decision_clean'].notna().sum()}")
print(f"  Approval rate: {(df_final['decision_clean'] == 'APPROVED').mean():.1%}")
print(f"{'='*60}")
print(f"\n  Now retrain: python3 train_model_v2.py")
