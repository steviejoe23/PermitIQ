"""
PermitIQ — Fuzzy Address Matching for Property Assessment
Handles: range addresses, suffix letters, city/zip in address,
         directional prefixes, and partial street name matching.
"""

import pandas as pd
import numpy as np
import re

print("=" * 60)
print("  FUZZY PROPERTY MATCHING")
print("=" * 60)


def norm(addr):
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


def generate_variants(addr):
    """Generate multiple matching variants for a ZBA address."""
    variants = [addr]

    parts = addr.split()
    if len(parts) < 2:
        return variants

    num = parts[0]
    rest = ' '.join(parts[1:])

    # 1. Strip city/zip: "38 fenway boston 02115" → "38 fenway"
    cleaned = re.sub(r'\b(?:boston|roxbury|dorchester|south boston|east boston|charlestown|brighton|allston|jamaica plain|hyde park|mattapan|roslindale|west roxbury)\b.*$', '', addr).strip()
    cleaned = re.sub(r'\b0\d{4}\b.*$', '', cleaned).strip()
    if cleaned != addr and len(cleaned) > 5:
        variants.append(cleaned)

    # 2. Range address: "85-99 berkeley st" → try "85 berkeley st" and "99 berkeley st"
    range_match = re.match(r'^(\d+)\s*[-to]+\s*(\d+[a-zA-Z]?)\s+(.+)', addr)
    if range_match:
        num1 = range_match.group(1)
        num2 = range_match.group(2)
        street = range_match.group(3)
        variants.extend([f'{num1} {street}', f'{num2} {street}'])

    # 3. Suffix letter: "18r robeson st" → "18 robeson st"
    suffix_match = re.match(r'^(\d+)[rRaAbBcC]\s*(.+)', addr)
    if suffix_match:
        variants.append(f'{suffix_match.group(1)} {suffix_match.group(2)}')

    # Also handle "18rrobeson st" (no space)
    nospace_match = re.match(r'^(\d+[rRaAbBcC])(\w+.+)', addr)
    if nospace_match:
        num_part = re.match(r'(\d+)', nospace_match.group(1)).group(1)
        street_part = nospace_match.group(2).strip()
        variants.append(f'{num_part} {street_part}')

    # 4. Directional prefix: "562 east fifth st" → "562 e fifth st"
    dir_cleaned = re.sub(r'\beast\b', 'e', addr)
    dir_cleaned = re.sub(r'\bwest\b', 'w', dir_cleaned)
    dir_cleaned = re.sub(r'\bnorth\b', 'n', dir_cleaned)
    dir_cleaned = re.sub(r'\bsouth\b', 's', dir_cleaned)
    if dir_cleaned != addr:
        variants.append(dir_cleaned)
    # Also try without direction at all
    no_dir = re.sub(r'\b(?:east|west|north|south|e|w|n|s)\b\s*', '', addr).strip()
    no_dir = re.sub(r'\s+', ' ', no_dir)
    if no_dir != addr:
        variants.append(no_dir)

    # 5. "to" in range: "49 to 49a linwood st" → "49 linwood st"
    to_match = re.match(r'^(\d+)\s+to\s+\d+[a-zA-Z]?\s+(.+)', addr)
    if to_match:
        variants.append(f'{to_match.group(1)} {to_match.group(2)}')

    # Apply city/zip stripping to all variants
    final_variants = []
    for v in variants:
        v_clean = re.sub(r'\b(?:boston|roxbury|dorchester|south boston|east boston|charlestown|brighton|allston|jamaica plain|hyde park|mattapan|roslindale|west roxbury)\b.*$', '', v).strip()
        v_clean = re.sub(r'\b0\d{4}\b.*$', '', v_clean).strip()
        if v_clean and len(v_clean) > 3:
            final_variants.append(v_clean)
        if v != v_clean and v and len(v) > 3:
            final_variants.append(v)

    return list(set(final_variants))


# ========================================
# LOAD DATA
# ========================================
print("\nLoading datasets...")
df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"  ZBA cases: {len(df)}")

pa = pd.read_csv('property_assessment_fy2026.csv', low_memory=False,
                 usecols=['PID', 'ST_NUM', 'ST_NAME', 'CITY', 'ZIP_CODE',
                         'LAND_SF', 'TOTAL_VALUE', 'LAND_VALUE', 'BLDG_VALUE',
                         'YR_BUILT', 'RES_UNITS', 'LIVING_AREA', 'BLDG_TYPE',
                         'LU_DESC', 'NUM_PARKING', 'OVERALL_COND', 'GROSS_AREA',
                         'BED_RMS', 'TT_RMS', 'NUM_BLDGS'])

pa['ST_NUM_CLEAN'] = pa['ST_NUM'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
pa = pa[pa['ST_NUM_CLEAN'] != 'nan']
pa['pa_address'] = pa['ST_NUM_CLEAN'] + ' ' + pa['ST_NAME'].astype(str).str.strip()
pa['pa_address_lower'] = pa['pa_address'].apply(norm)

# Clean numeric fields
for col in ['LAND_SF', 'TOTAL_VALUE', 'LAND_VALUE', 'BLDG_VALUE', 'LIVING_AREA', 'GROSS_AREA']:
    pa[col] = pd.to_numeric(pa[col].astype(str).str.replace(',', ''), errors='coerce')

pa['parcel_id_padded'] = pa['PID'].astype(str).str.zfill(10)

# Build lookup (dedup by address, keep highest value)
pa_dedup = pa.sort_values('TOTAL_VALUE', ascending=False).drop_duplicates(subset='pa_address_lower')
pa_lookup = pa_dedup.set_index('pa_address_lower')
pa_set = set(pa_lookup.index)

print(f"  PA unique addresses: {len(pa_set)}")

# ========================================
# FUZZY MATCHING
# ========================================
print("\nRunning fuzzy matching...")

# Check which cases already have property data
already_matched = df['total_value'].notna() if 'total_value' in df.columns else pd.Series(False, index=df.index)
print(f"  Already matched: {already_matched.sum()}")

new_matches = 0
match_types = {'exact': 0, 'variant': 0}

prop_cols = ['lot_size_sf', 'total_value', 'land_value', 'bldg_value',
             'yr_built', 'living_area', 'gross_area', 'existing_units',
             'existing_parking', 'bedrooms', 'total_rooms', 'land_use',
             'bldg_type', 'overall_cond', 'pa_parcel_id']

# Ensure columns exist
for col in prop_cols:
    if col not in df.columns:
        df[col] = np.nan

PA_COL_MAP = {
    'lot_size_sf': 'LAND_SF', 'total_value': 'TOTAL_VALUE',
    'land_value': 'LAND_VALUE', 'bldg_value': 'BLDG_VALUE',
    'yr_built': 'YR_BUILT', 'living_area': 'LIVING_AREA',
    'gross_area': 'GROSS_AREA', 'existing_units': 'RES_UNITS',
    'existing_parking': 'NUM_PARKING', 'bedrooms': 'BED_RMS',
    'total_rooms': 'TT_RMS', 'land_use': 'LU_DESC',
    'bldg_type': 'BLDG_TYPE', 'overall_cond': 'OVERALL_COND',
    'pa_parcel_id': 'parcel_id_padded',
}

for idx, row in df.iterrows():
    # Skip if already has property data
    if pd.notna(row.get('total_value')) and row.get('total_value', 0) > 0:
        continue

    addr = row.get('address_clean')
    if pd.isna(addr):
        continue

    addr_norm = norm(addr)
    if not addr_norm or len(addr_norm) < 5 or not addr_norm[0].isdigit():
        continue

    # Generate variants
    variants = generate_variants(addr_norm)

    matched = False
    for variant in variants:
        if variant in pa_set:
            rec = pa_lookup.loc[variant]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]

            for our_col, pa_col in PA_COL_MAP.items():
                val = rec.get(pa_col)
                if pd.notna(val):
                    df.at[idx, our_col] = val

            new_matches += 1
            if variant == addr_norm:
                match_types['exact'] += 1
            else:
                match_types['variant'] += 1
            matched = True
            break

    if idx % 5000 == 0 and idx > 0:
        print(f"  Processed {idx}/{len(df)} — new matches: {new_matches}")

print(f"\n  New matches: {new_matches}")
print(f"    Exact: {match_types['exact']}")
print(f"    Variant: {match_types['variant']}")

# Update derived features
df['property_age'] = 2026 - pd.to_numeric(df['yr_built'], errors='coerce')
df['value_per_sqft'] = pd.to_numeric(df['total_value'], errors='coerce') / pd.to_numeric(df['lot_size_sf'], errors='coerce').replace(0, np.nan)
df['is_high_value'] = (pd.to_numeric(df['total_value'], errors='coerce') > 1_000_000).astype(int)

total_with_property = df['total_value'].notna().sum()
print(f"\n  TOTAL with property data: {total_with_property}/{len(df)} ({total_with_property/len(df):.1%})")
print(f"  With lot size: {df['lot_size_sf'].notna().sum()}")
print(f"  With year built: {df['yr_built'].notna().sum()}")
print(f"  High-value (>$1M): {df['is_high_value'].sum()}")

# Save
df.to_csv('zba_cases_cleaned.csv', index=False)
print(f"\n✅ Saved ({len(df.columns)} columns)")
