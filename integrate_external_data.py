"""
PermitIQ — Integrate External Data Sources
Merges ZBA Tracker, Property Assessment, and Building Permits into the ZBA dataset.
"""

import pandas as pd
import numpy as np
import re
import os

print("=" * 60)
print("  PERMITIQ DATA INTEGRATION")
print("=" * 60)

# ========================================
# LOAD CORE DATASET
# ========================================
print("\n--- Loading core ZBA dataset ---")
df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"  {len(df)} cases, {len(df.columns)} columns")
original_cols = len(df.columns)


def normalize_for_matching(addr):
    if pd.isna(addr):
        return ''
    addr = str(addr).lower().strip()
    addr = re.sub(r',?\s*ward\s*\d+', '', addr)
    addr = re.sub(r'\bstreet\b', 'st', addr)
    addr = re.sub(r'\bavenue\b', 'ave', addr)
    addr = re.sub(r'\broad\b', 'rd', addr)
    addr = re.sub(r'\bdrive\b', 'dr', addr)
    addr = re.sub(r'\bboulevard\b', 'blvd', addr)
    addr = re.sub(r'\blane\b', 'ln', addr)
    addr = re.sub(r'\bcourt\b', 'ct', addr)
    addr = re.sub(r'\bplace\b', 'pl', addr)
    addr = re.sub(r'\bterrace\b', 'ter', addr)
    addr = re.sub(r'\bparkway\b', 'pkwy', addr)
    addr = re.sub(r'\bcircle\b', 'cir', addr)
    addr = re.sub(r'\bhighway\b', 'hwy', addr)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr


# ========================================
# 1. ZBA TRACKER INTEGRATION
# ========================================
print("\n--- Integrating ZBA Tracker (15,932 records) ---")
zba_tracker = pd.read_csv('zba_tracker.csv', low_memory=False)
print(f"  Tracker records: {len(zba_tracker)}")
print(f"  Tracker columns: {list(zba_tracker.columns)}")

# The tracker has BOA case numbers — match on those
# Our dataset has case_number like "BOA1775242"
# Tracker has boa_apno like "BOA1685710"

# Rename tracker columns to avoid conflicts
tracker_cols = {
    'address': 'tracker_address',
    'ward': 'tracker_ward',
    'city': 'tracker_city',
    'zip': 'tracker_zip',
    'zoning_district': 'tracker_zoning_district',
    'status': 'tracker_status',
    'contact': 'tracker_contact',
    'project_description': 'tracker_description',
    'decision': 'tracker_decision',
    'ever_deferred': 'tracker_deferred',
    'num_deferrals': 'tracker_num_deferrals',
    'submitted_date': 'tracker_submitted',
    'received_date': 'tracker_received',
    'hearing_date': 'tracker_hearing',
    'final_decision_date': 'tracker_decision_date',
    'closed_date': 'tracker_closed',
    'parent_apno': 'parent_apno',
    'appeal_type': 'tracker_appeal_type',
}

zba_tracker = zba_tracker.rename(columns=tracker_cols)

# Clean case numbers for matching
df['case_clean'] = df['case_number'].str.strip().str.upper()
zba_tracker['case_clean'] = zba_tracker['boa_apno'].str.strip().str.upper()

# Merge
before_ward = df['ward'].notna().sum()
merged = df.merge(zba_tracker, on='case_clean', how='left', suffixes=('', '_tracker'))

# Fill in missing wards from tracker
mask = merged['ward'].isna() & merged['tracker_ward'].notna()
merged.loc[mask, 'ward'] = merged.loc[mask, 'tracker_ward'].astype(float)
after_ward = merged['ward'].notna().sum()
print(f"  Ward coverage: {before_ward} → {after_ward} (+{after_ward - before_ward})")

# Fill in missing addresses from tracker
if 'address_clean' not in merged.columns:
    merged['address_clean'] = merged.get('address', pd.Series(dtype=str))
before_addr = merged['address_clean'].notna().sum()
mask = merged['address_clean'].isna() & merged['tracker_address'].notna()
merged.loc[mask, 'address_clean'] = merged.loc[mask, 'tracker_address']
after_addr = merged['address_clean'].notna().sum()
print(f"  Address coverage: {before_addr} → {after_addr} (+{after_addr - before_addr})")

# Add new features from tracker
merged['has_deferrals'] = (merged['tracker_num_deferrals'].fillna(0) > 0).astype(int)
merged['num_deferrals'] = merged['tracker_num_deferrals'].fillna(0).astype(int)

# Extract "ESQ" from contact = has attorney
# Ensure has_attorney column exists (may not if reextract didn't create it)
if 'has_attorney' not in merged.columns:
    merged['has_attorney'] = 0
mask = merged['has_attorney'] == 0
esq_mask = merged['tracker_contact'].fillna('').str.contains(r'ESQ|Esq|Attorney|Law', na=False, regex=True)
merged.loc[mask & esq_mask, 'has_attorney'] = 1
new_attorneys = (mask & esq_mask).sum()
print(f"  New attorney detections from tracker: {new_attorneys}")

# Extract zoning district
mask = merged['tracker_zoning_district'].notna()
merged['zoning_district'] = ''
merged.loc[mask, 'zoning_district'] = merged.loc[mask, 'tracker_zoning_district']
print(f"  Cases with zoning district: {(merged['zoning_district'] != '').sum()}")

# Matched count
matched = merged['case_clean'].isin(zba_tracker['case_clean']).sum()
print(f"  Cases matched to tracker: {matched}/{len(df)}")

# Drop tracker temp columns, keep useful ones
keep_tracker = ['tracker_status', 'tracker_description', 'tracker_zoning_district',
                'tracker_city', 'tracker_zip', 'has_deferrals', 'num_deferrals',
                'parent_apno', 'tracker_appeal_type', 'zoning_district']
drop_cols = [c for c in merged.columns if c.startswith('tracker_') and c not in keep_tracker]
drop_cols.append('case_clean')
merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors='ignore')

df = merged

# ========================================
# 2. PROPERTY ASSESSMENT INTEGRATION
# ========================================
print("\n--- Integrating Property Assessment FY2026 (184,552 records) ---")
pa = pd.read_csv('property_assessment_fy2026.csv', low_memory=False,
                 usecols=['PID', 'ST_NUM', 'ST_NAME', 'CITY', 'ZIP_CODE',
                         'LAND_SF', 'TOTAL_VALUE', 'LAND_VALUE', 'BLDG_VALUE',
                         'YR_BUILT', 'RES_UNITS', 'LIVING_AREA', 'BLDG_TYPE',
                         'LU_DESC', 'NUM_PARKING', 'OVERALL_COND', 'GROSS_AREA',
                         'BED_RMS', 'TT_RMS', 'NUM_BLDGS'])

print(f"  Property records: {len(pa)}")

# Build address lookup from property assessment
# Format: "195 Lexington ST" → match against our addresses
# Fix ST_NUM: remove ".0" suffix from float conversion
pa['ST_NUM_CLEAN'] = pa['ST_NUM'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
pa['pa_address'] = pa['ST_NUM_CLEAN'] + ' ' + pa['ST_NAME'].astype(str).str.strip()
pa['pa_address_lower'] = pa['pa_address'].apply(normalize_for_matching)
# Remove rows with nan street numbers
pa = pa[pa['ST_NUM_CLEAN'] != 'nan']

# Clean numeric fields
for col in ['LAND_SF', 'TOTAL_VALUE', 'LAND_VALUE', 'BLDG_VALUE', 'LIVING_AREA', 'GROSS_AREA']:
    pa[col] = pd.to_numeric(pa[col].astype(str).str.replace(',', ''), errors='coerce')

# Also create a parcel_id mapping (PID → zero-padded format)
pa['parcel_id_padded'] = pa['PID'].astype(str).str.zfill(10)

# Create address-based lookup (dedup to one record per address)
pa_dedup = pa.sort_values('TOTAL_VALUE', ascending=False).drop_duplicates(subset='pa_address_lower')
pa_lookup = pa_dedup.set_index('pa_address_lower')

# Match our ZBA cases to property data by address
df['match_addr'] = df['address_clean'].apply(normalize_for_matching)

# Try to match
matched_count = 0
property_features = []

for idx, row in df.iterrows():
    addr = row['match_addr']
    if not addr:
        property_features.append({})
        continue

    # Exact match
    if addr in pa_lookup.index:
        rec = pa_lookup.loc[addr]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        property_features.append({
            'lot_size_sf': rec.get('LAND_SF', np.nan),
            'total_value': rec.get('TOTAL_VALUE', np.nan),
            'land_value': rec.get('LAND_VALUE', np.nan),
            'bldg_value': rec.get('BLDG_VALUE', np.nan),
            'yr_built': rec.get('YR_BUILT', np.nan),
            'living_area': rec.get('LIVING_AREA', np.nan),
            'gross_area': rec.get('GROSS_AREA', np.nan),
            'existing_units': rec.get('RES_UNITS', np.nan),
            'existing_parking': rec.get('NUM_PARKING', np.nan),
            'bedrooms': rec.get('BED_RMS', np.nan),
            'total_rooms': rec.get('TT_RMS', np.nan),
            'land_use': rec.get('LU_DESC', ''),
            'bldg_type': rec.get('BLDG_TYPE', ''),
            'overall_cond': rec.get('OVERALL_COND', ''),
            'pa_parcel_id': rec.get('parcel_id_padded', ''),
        })
        matched_count += 1
    else:
        property_features.append({})

prop_df = pd.DataFrame(property_features)
print(f"  Matched to property data: {matched_count}/{len(df)} ({matched_count/len(df):.1%})")

# Add property features to main df
for col in prop_df.columns:
    df[col] = prop_df[col].values

# Create derived features
if 'yr_built' in df.columns:
    df['property_age'] = 2026 - pd.to_numeric(df['yr_built'], errors='coerce')
else:
    df['property_age'] = np.nan
if 'total_value' in df.columns and 'lot_size_sf' in df.columns:
    df['value_per_sqft'] = pd.to_numeric(df['total_value'], errors='coerce') / pd.to_numeric(df['lot_size_sf'], errors='coerce').replace(0, np.nan)
    df['is_high_value'] = (pd.to_numeric(df['total_value'], errors='coerce') > 1_000_000).astype(int)
else:
    df['value_per_sqft'] = np.nan
    df['is_high_value'] = 0

for col_name, label in [('lot_size_sf', 'lot size'), ('total_value', 'total value'), ('yr_built', 'year built')]:
    if col_name in df.columns:
        print(f"  With {label}: {df[col_name].notna().sum()}")
    else:
        print(f"  With {label}: 0 (column not created)")
if 'is_high_value' in df.columns:
    print(f"  High-value properties (>$1M): {df['is_high_value'].sum()}")


# ========================================
# 3. BUILDING PERMITS INTEGRATION
# ========================================
print("\n--- Integrating Building Permits (718,721 records) ---")

# This is a big file, read just what we need
bp = pd.read_csv('building_permits.csv', low_memory=False,
                 usecols=['address', 'ward', 'parcel_id', 'declared_valuation',
                         'worktype', 'status', 'sq_feet'])

print(f"  Permit records: {len(bp)}")

# Count permits per address
bp['bp_addr_lower'] = bp['address'].astype(str).str.lower().str.strip()

# Aggregate permits per address
bp_agg = bp.groupby('bp_addr_lower').agg(
    total_permits=('worktype', 'count'),
    total_permit_value=('declared_valuation', lambda x: pd.to_numeric(
        x.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce'
    ).sum()),
    bp_ward=('ward', 'first'),
).reset_index()

# Match to our dataset
bp_lookup = bp_agg.set_index('bp_addr_lower')

matched_bp = 0
permit_features = []

for idx, row in df.iterrows():
    addr = row['match_addr']
    if not addr:
        permit_features.append({})
        continue

    if addr in bp_lookup.index:
        rec = bp_lookup.loc[addr]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        permit_features.append({
            'prior_permits': rec.get('total_permits', 0),
            'prior_permit_value': rec.get('total_permit_value', 0),
        })
        matched_bp += 1

        # Also fill in ward if missing
        if pd.isna(df.at[idx, 'ward']) and pd.notna(rec.get('bp_ward')):
            df.at[idx, 'ward'] = float(rec['bp_ward'])
    else:
        permit_features.append({})

bp_df = pd.DataFrame(permit_features)
print(f"  Matched to permit data: {matched_bp}/{len(df)} ({matched_bp/len(df):.1%})")

for col in bp_df.columns:
    df[col] = bp_df[col].values

df['has_prior_permits'] = (pd.to_numeric(df['prior_permits'], errors='coerce').fillna(0) > 0).astype(int)
print(f"  Cases with prior permits: {df['has_prior_permits'].sum()}")

# Drop temp columns
df = df.drop(columns=['match_addr'], errors='ignore')

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 60)
print("  INTEGRATION COMPLETE")
print("=" * 60)
print(f"  Total cases: {len(df)}")
print(f"  Total columns: {len(df.columns)} (was {original_cols})")
print(f"  Addresses: {df['address_clean'].notna().sum()} ({df['address_clean'].notna().mean():.1%})")
print(f"  Wards: {df['ward'].notna().sum()} ({df['ward'].notna().mean():.1%})")
if 'total_value' in df.columns:
    print(f"  With property data: {df['total_value'].notna().sum()} ({df['total_value'].notna().mean():.1%})")
if 'has_prior_permits' in df.columns:
    print(f"  With permit history: {(df['has_prior_permits']==1).sum()} ({(df['has_prior_permits']==1).mean():.1%})")
if 'has_attorney' in df.columns:
    print(f"  With attorney: {(df['has_attorney']==1).sum()} ({(df['has_attorney']==1).mean():.1%})")

# Save
df.to_csv('zba_cases_cleaned.csv', index=False)
print(f"\n✅ Saved to zba_cases_cleaned.csv ({len(df.columns)} columns)")
