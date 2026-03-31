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
print(f"Cases with decisions from parser: {has_decision} ({has_decision/len(df):.1%})")

# --- AGGRESSIVE DECISION RECOVERY from raw_text ---
# Many cases have decisions in the raw text that the parser missed
import re
null_mask = df['decision_clean'].isna()
recovered = 0
if null_mask.sum() > 0:
    rt = df.loc[null_mask, 'raw_text'].fillna('').str.lower()

    # Approval patterns (order matters — check most specific first)
    approve_patterns = [
        r'voted\s+to\s+(?:approve|grant|sustain)',
        r'(?:appeal|petition|application)\s+(?:is\s+)?(?:hereby\s+)?(?:sustained|granted|approved)',
        r'board\s+(?:hereby\s+)?(?:votes?|voted)\s+(?:to\s+)?(?:approve|grant|sustain)',
        r'relief\s+(?:is\s+)?(?:hereby\s+)?granted',
        r'approved\s+with\s+(?:conditions|provisos|the\s+following)',
        r'(?:the\s+)?(?:zoning\s+)?(?:board|bza|zba)\s+(?:voted\s+to\s+)?(?:approv|grant|sustain)',
    ]

    deny_patterns = [
        r'voted\s+to\s+(?:deny|refuse|reject|dismiss)',
        r'(?:appeal|petition|application)\s+(?:is\s+)?(?:hereby\s+)?(?:denied|refused|rejected|dismissed)',
        r'board\s+(?:hereby\s+)?(?:votes?|voted)\s+(?:to\s+)?(?:deny|refuse|reject)',
        r'relief\s+(?:is\s+)?(?:hereby\s+)?denied',
        r'(?:the\s+)?(?:zoning\s+)?(?:board|bza|zba)\s+(?:voted\s+to\s+)?(?:deny|refuse|reject)',
    ]

    for idx in df.index[null_mask]:
        text = str(df.loc[idx, 'raw_text']).lower()
        found = False

        for pattern in approve_patterns:
            if re.search(pattern, text):
                df.loc[idx, 'decision_clean'] = 'APPROVED'
                recovered += 1
                found = True
                break

        if not found:
            for pattern in deny_patterns:
                if re.search(pattern, text):
                    df.loc[idx, 'decision_clean'] = 'DENIED'
                    recovered += 1
                    break

    print(f"  Recovered from raw_text: {recovered} additional decisions")

has_decision = df['decision_clean'].notna().sum()
print(f"Cases with clean decisions (total): {has_decision} ({has_decision/len(df):.1%})")
print(f"  APPROVED: {(df['decision_clean'] == 'APPROVED').sum()}")
print(f"  DENIED: {(df['decision_clean'] == 'DENIED').sum()}")
print(f"  Still missing: {df['decision_clean'].isna().sum()}")

# ========================================
# STEP 3b: Merge ZBA Tracker decisions for cases still missing
# ========================================
print("\n--- Step 3b: Merging ZBA Tracker decisions ---")
import os
tracker_path = 'zba_tracker.csv'
if os.path.exists(tracker_path):
    tracker = pd.read_csv(tracker_path, low_memory=False)
    print(f"  ZBA Tracker: {len(tracker)} records, {tracker['decision'].notna().sum()} with decisions")

    # Map tracker decisions to our format
    tracker_decision_map = {
        'AppProv': 'APPROVED',
        'Approved': 'APPROVED',
        'DeniedPrej': 'DENIED',
        'Denied': 'DENIED',
    }
    tracker['decision_mapped'] = tracker['decision'].map(tracker_decision_map)
    tracker_with_dec = tracker[tracker['decision_mapped'].notna()].copy()

    # Clean case numbers for matching
    tracker_with_dec['_case'] = tracker_with_dec['boa_apno'].fillna('').astype(str).str.strip().str.replace('-', '')
    df['_case'] = df['case_number'].fillna('').astype(str).str.strip().str.replace('-', '')

    # Fill missing decisions from tracker
    still_missing = df['decision_clean'].isna()
    before_tracker = still_missing.sum()

    tracker_lookup = tracker_with_dec.drop_duplicates(subset='_case', keep='first').set_index('_case')['decision_mapped'].to_dict()
    for idx in df.index[still_missing]:
        case = df.loc[idx, '_case']
        if case in tracker_lookup:
            df.loc[idx, 'decision_clean'] = tracker_lookup[case]

    # Also fill missing addresses, wards, and contact from tracker
    tracker_addr_lookup = tracker_with_dec.drop_duplicates(subset='_case', keep='first').set_index('_case')
    for idx in df.index:
        case = df.loc[idx, '_case']
        if case in tracker_addr_lookup.index:
            tr = tracker_addr_lookup.loc[case]
            if pd.isna(df.loc[idx, 'address']) or str(df.loc[idx, 'address']).strip() == '':
                addr = str(tr.get('address', '')).split(' Boston')[0].strip()
                if addr:
                    df.loc[idx, 'address'] = addr
            if pd.isna(df.loc[idx].get('ward')):
                w = tr.get('ward')
                if pd.notna(w):
                    df.loc[idx, 'ward'] = w
            # Fill contact from tracker (for attorney detection)
            if 'contact' not in df.columns or pd.isna(df.loc[idx].get('contact', pd.NA)):
                c = tr.get('contact')
                if pd.notna(c):
                    if 'contact' not in df.columns:
                        df['contact'] = pd.NA
                    df.loc[idx, 'contact'] = c
            # Fill appeal_type from tracker (Building vs Zoning — strong predictor)
            if 'appeal_type' not in df.columns:
                df['appeal_type'] = pd.NA
            if pd.isna(df.loc[idx].get('appeal_type', pd.NA)):
                at = tr.get('appeal_type')
                if pd.notna(at):
                    df.loc[idx, 'appeal_type'] = at

    after_tracker = df['decision_clean'].isna().sum()
    recovered_from_tracker = before_tracker - after_tracker
    print(f"  Recovered from tracker: {recovered_from_tracker} decisions")
    print(f"  Still missing: {after_tracker}")

    df = df.drop(columns=['_case'], errors='ignore')

    # ========================================
    # STEP 3c: Import tracker-only cases (not in our OCR data)
    # ========================================
    print("\n--- Step 3c: Importing tracker-only cases ---")
    our_cases = set(df['case_number'].fillna('').astype(str).str.strip().str.replace('-', ''))
    tracker_with_dec['_case'] = tracker_with_dec['boa_apno'].fillna('').astype(str).str.strip().str.replace('-', '')

    # Find cases in tracker that we DON'T have
    new_mask = ~tracker_with_dec['_case'].isin(our_cases)
    new_cases = tracker_with_dec[new_mask].drop_duplicates(subset='_case', keep='first').copy()
    print(f"  Tracker cases not in our data: {len(new_cases)}")

    if len(new_cases) > 0:
        # Build rows in our format from tracker data
        new_rows = []
        for _, row in new_cases.iterrows():
            addr_raw = str(row.get('address', ''))
            addr_clean = addr_raw.split(' Boston')[0].strip() if addr_raw else ''

            new_row = {
                'case_number': row['_case'],
                'address': addr_clean,
                'decision': row.get('decision', ''),
                'decision_clean': row['decision_mapped'],
                'ward': row.get('ward', ''),
                'raw_text': str(row.get('project_description', '')),  # Use description as raw_text
                'source_pdf': 'zba_tracker',
                'zoning_district': row.get('zoning_district', ''),
                'filing_date': row.get('submitted_date', ''),
                'final_decision_date': row.get('final_decision_date', ''),
                'contact': row.get('contact', ''),
                'appeal_type': row.get('appeal_type', ''),
            }
            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)
        # Align columns — add missing columns as NaN
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA
        # Keep only columns in our dataset
        new_df = new_df[[c for c in df.columns if c in new_df.columns]]

        before_import = len(df)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"  Imported: {len(new_df)} tracker-only cases")
        print(f"  Dataset: {before_import} → {len(df)} rows")
        print(f"  New decisions: {new_df['decision_clean'].value_counts().to_dict()}")
else:
    print("  ZBA Tracker not found — skipping")

has_decision_final = df['decision_clean'].notna().sum()
print(f"\nFinal decision counts:")
print(f"  APPROVED: {(df['decision_clean'] == 'APPROVED').sum()}")
print(f"  DENIED: {(df['decision_clean'] == 'DENIED').sum()}")
print(f"  Missing: {df['decision_clean'].isna().sum()}")
print(f"  Total with decisions: {has_decision_final} ({has_decision_final/len(df):.1%})")

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
