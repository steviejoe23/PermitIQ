#!/usr/bin/env python3
"""
Precompute development difficulty risk scores for all ~98,510 Boston parcels.

Scoring components (0-100 scale):
  - Ward denial rate (30%) — higher denial rate = higher risk
  - Zoning district denial rate (25%)
  - Ward case density (15%) — more ZBA cases = more scrutiny
  - Lot size relative to zoning norms (15%) — undersized lots need more variances
  - Zoning restrictiveness (15%) — residential > commercial in difficulty

Output: data/parcel_risk_scores.csv
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

GEOJSON_PATH = PROJECT_ROOT / "boston_parcels_zoning.geojson"
ZBA_PATH = PROJECT_ROOT / "zba_cases_cleaned.csv"
ASSESSMENT_PATH = PROJECT_ROOT / "property_assessment_fy2026.csv"
OUTPUT_PATH = DATA_DIR / "parcel_risk_scores.csv"

# Weights for each scoring component
W_WARD_DENIAL = 0.30
W_DISTRICT_DENIAL = 0.25
W_CASE_DENSITY = 0.15
W_LOT_SIZE = 0.15
W_ZONING_RESTRICT = 0.15

# District-to-ward mapping derived from ZBA case data (mode ward per district)
DISTRICT_TO_WARD = {
    'Allston/Brighton Neighborhood': 22,
    'Audbon Circle Neighborhood': 21,
    'Bay Village Neighborhood': 5,
    'Boston Proper': 5,
    'Bulfinch Triangle': 3,
    'Bulfinch Triangle, Central Artery Special': 3,
    'Cambridge Street North': 3,
    'Charlestown Neighborhood': 2,
    'Chinatown': 3,
    'City Square Neighborhood': 2,
    'Dorchester Neighborhood': 16,
    'East Boston Neighborhood': 1,
    'Fenway Neighborhood': 5,
    'Government Center/Markets': 3,
    'Government Center/Markets, Central Artery Special': 3,
    'Greater Mattapan Neighborhood': 14,
    'Harborpark: Charlestown Waterfront': 2,
    'Harborpark: Dorchester Bay/Neponset River Waterfront': 16,
    'Harborpark: North End Waterfront': 3,
    'Huntington Avenue/Prudential Center': 4,
    'Hyde Park Neighborhood': 18,
    'Jamaica Plain Neighborhood': 19,
    'Leather District': 3,
    'Midtown Cultural': 3,
    'Mission Hill Neighborhood': 10,
    'Newmarket Industrial Commercial Neighborhood District': 8,
    'North End Neighborhood': 3,
    'North End Neighborhood, Central Artery Special': 3,
    'North Station Economic Development Area': 3,
    'Roslindale Neighborhood': 20,
    'Roxbury Neighborhood': 12,
    'South Boston': 7,
    'South Boston Neighborhood': 6,
    'South End Neighborhood': 4,
    'South Station Economic Development Area': 3,
    'Stuart Street District': 4,
    'West Roxbury Neighborhood': 20,
}

# Zoning restrictiveness by subdistrict_use keyword
# Higher = more restrictive = harder to develop
RESTRICTIVENESS_MAP = {
    # Most restrictive — single-family residential
    'single family': 90,
    '1-family': 90,
    '1f': 90,
    # High restrictive — low-density residential
    'low residential': 80,
    'residential-1': 75,
    # Medium-high — general residential
    'medium residential': 65,
    'residential-2': 65,
    'residential-3': 60,
    'residential': 60,
    'multi-family': 55,
    # Medium — mixed
    'mixed use': 45,
    'neighborhood shopping': 40,
    'local convenience': 40,
    # Lower — commercial
    'general commercial': 30,
    'commercial': 30,
    'community commercial': 35,
    'neighborhood commercial': 35,
    # Low — industrial/institutional
    'industrial': 20,
    'manufacturing': 20,
    'waterfront manufacturing': 20,
    'institutional': 25,
    'open space': 15,
    'recreation': 15,
}


def classify_restrictiveness(subdistrict_use: str) -> float:
    """Score 0-100 based on subdistrict use type. Higher = more restrictive."""
    if not subdistrict_use or pd.isna(subdistrict_use):
        return 50.0  # default mid-range for unknown

    use_lower = str(subdistrict_use).lower().strip()

    # Check for exact or substring matches
    for keyword, score in RESTRICTIVENESS_MAP.items():
        if keyword in use_lower:
            return float(score)

    # Fallback heuristics
    if 'resid' in use_lower:
        return 65.0
    if 'commerc' in use_lower:
        return 30.0
    if 'industr' in use_lower or 'manufact' in use_lower:
        return 20.0

    return 50.0  # unknown


def load_parcel_data():
    """Load parcel properties from geojson (skip geometry for speed)."""
    print("Loading parcel geojson...")
    t0 = time.time()
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)

    records = []
    for feat in gj['features']:
        p = feat['properties']
        records.append({
            'parcel_id': p.get('parcel_id'),
            'neighborhood_district': p.get('neighborhood_district'),
            'primary_zoning': p.get('primary_zoning'),
            'zoning_subdistrict': p.get('zoning_subdistrict'),
            'subdistrict_use': p.get('subdistrict_use'),
            'max_height_ft': p.get('max_height_ft'),
            'max_floors': p.get('max_floors'),
            'max_far': p.get('max_far'),
        })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} parcels in {time.time()-t0:.1f}s")
    return df


def load_zba_stats():
    """Compute per-ward and per-district denial rates and case volumes."""
    print("Computing ZBA statistics...")
    zba = pd.read_csv(
        ZBA_PATH,
        usecols=['ward', 'decision_clean', 'zoning_district'],
        low_memory=False,
    )

    # Only look at APPROVED/DENIED decisions
    zba = zba[zba['decision_clean'].isin(['APPROVED', 'DENIED'])].copy()
    zba['is_denied'] = (zba['decision_clean'] == 'DENIED').astype(int)

    # Per-ward stats
    ward_stats = zba.groupby('ward').agg(
        ward_denial_rate=('is_denied', 'mean'),
        ward_case_volume=('is_denied', 'count'),
    ).reset_index()
    ward_stats['ward'] = ward_stats['ward'].astype(int)

    # Per-district stats
    district_stats = zba.groupby('zoning_district').agg(
        district_denial_rate=('is_denied', 'mean'),
        district_case_volume=('is_denied', 'count'),
    ).reset_index()

    print(f"  Ward stats: {len(ward_stats)} wards, "
          f"denial rate range {ward_stats['ward_denial_rate'].min():.1%}-{ward_stats['ward_denial_rate'].max():.1%}")
    print(f"  District stats: {len(district_stats)} districts, "
          f"denial rate range {district_stats['district_denial_rate'].min():.1%}-{district_stats['district_denial_rate'].max():.1%}")

    return ward_stats, district_stats


def load_lot_sizes():
    """Load lot sizes from property assessment, keyed by parcel_id."""
    print("Loading lot sizes from assessment data...")
    assess = pd.read_csv(
        ASSESSMENT_PATH,
        usecols=['GIS_ID', 'LAND_SF'],
        low_memory=False,
    )
    # LAND_SF has commas — clean it
    assess['LAND_SF'] = (
        assess['LAND_SF']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    assess['lot_area_sqft'] = pd.to_numeric(assess['LAND_SF'], errors='coerce')
    # GIS_ID is numeric; parcel_id in geojson is zero-padded 10-char string
    assess['parcel_id'] = assess['GIS_ID'].astype(str).str.zfill(10)
    # Take first (or max) lot size per parcel for multi-unit buildings
    lot_sizes = assess.groupby('parcel_id')['lot_area_sqft'].max().reset_index()
    print(f"  Loaded lot sizes for {len(lot_sizes):,} parcels")
    return lot_sizes


def compute_risk_scores(parcels, ward_stats, district_stats, lot_sizes):
    """Compute the composite risk score for each parcel."""
    print("Computing risk scores...")
    t0 = time.time()

    # Map parcels to wards via neighborhood_district
    parcels['ward'] = parcels['neighborhood_district'].map(DISTRICT_TO_WARD)

    # Merge ward stats
    parcels = parcels.merge(
        ward_stats[['ward', 'ward_denial_rate', 'ward_case_volume']],
        on='ward', how='left',
    )

    # Merge district stats
    parcels = parcels.merge(
        district_stats[['zoning_district', 'district_denial_rate', 'district_case_volume']],
        left_on='neighborhood_district', right_on='zoning_district', how='left',
    )

    # Merge lot sizes
    parcels = parcels.merge(lot_sizes, on='parcel_id', how='left')

    # --- Component 1: Ward denial rate (already 0-1, scale to 0-100) ---
    parcels['ward_denial_score'] = parcels['ward_denial_rate'].fillna(0.1) * 100

    # --- Component 2: District denial rate (already 0-1, scale to 0-100) ---
    parcels['district_denial_score'] = parcels['district_denial_rate'].fillna(0.1) * 100

    # --- Component 3: Case density (percentile-based normalization) ---
    # Normalize ward case volume to 0-100 using rank percentile
    vol = parcels['ward_case_volume'].fillna(0)
    vol_min, vol_max = vol.min(), vol.max()
    if vol_max > vol_min:
        parcels['case_density_score'] = ((vol - vol_min) / (vol_max - vol_min) * 100)
    else:
        parcels['case_density_score'] = 50.0

    # --- Component 4: Lot size score ---
    # Undersized lots (< median for their zoning district) get higher scores
    # Compute median lot size per zoning subdistrict
    district_median_lot = parcels.groupby('zoning_subdistrict')['lot_area_sqft'].transform('median')
    # Ratio: if lot is half the median, ratio = 0.5 -> high risk
    lot_ratio = parcels['lot_area_sqft'] / district_median_lot.replace(0, np.nan)
    # Clamp ratio to [0.1, 3.0] to avoid extremes
    lot_ratio = lot_ratio.clip(0.1, 3.0)
    # Invert: smaller lots = higher score. ratio=0.1 -> 100, ratio=1.0 -> ~50, ratio=3.0 -> ~0
    # Use: score = 100 * (1 - (ratio - 0.1) / (3.0 - 0.1))
    parcels['lot_size_score'] = (100 * (1 - (lot_ratio - 0.1) / 2.9)).fillna(50.0)

    # --- Component 5: Zoning restrictiveness ---
    parcels['zoning_restrictiveness'] = parcels['subdistrict_use'].apply(classify_restrictiveness)

    # --- Composite score ---
    parcels['risk_score_raw'] = (
        W_WARD_DENIAL * parcels['ward_denial_score']
        + W_DISTRICT_DENIAL * parcels['district_denial_score']
        + W_CASE_DENSITY * parcels['case_density_score']
        + W_LOT_SIZE * parcels['lot_size_score']
        + W_ZONING_RESTRICT * parcels['zoning_restrictiveness']
    )

    # Normalize to 0-100 using min-max on the raw composite
    raw_min = parcels['risk_score_raw'].min()
    raw_max = parcels['risk_score_raw'].max()
    if raw_max > raw_min:
        parcels['risk_score'] = (
            (parcels['risk_score_raw'] - raw_min) / (raw_max - raw_min) * 100
        ).round(1)
    else:
        parcels['risk_score'] = 50.0

    # Risk level labels
    parcels['risk_level'] = pd.cut(
        parcels['risk_score'],
        bins=[-1, 25, 50, 75, 101],
        labels=['Easy', 'Moderate', 'Difficult', 'Very Difficult'],
    )

    print(f"  Computed scores in {time.time()-t0:.1f}s")
    return parcels


def main():
    print("=" * 60)
    print("Parcel Risk Score Precomputation")
    print("=" * 60)
    t_start = time.time()

    # Load all data
    parcels = load_parcel_data()
    ward_stats, district_stats = load_zba_stats()
    lot_sizes = load_lot_sizes()

    # Compute scores
    result = compute_risk_scores(parcels, ward_stats, district_stats, lot_sizes)

    # Select output columns and round
    output_cols = [
        'parcel_id',
        'risk_score',
        'risk_level',
        'ward_denial_rate',
        'district_denial_rate',
        'case_density_score',
        'lot_size_score',
        'zoning_restrictiveness',
    ]
    output = result[output_cols].copy()
    output['ward_denial_rate'] = output['ward_denial_rate'].round(4)
    output['district_denial_rate'] = output['district_denial_rate'].round(4)
    output['case_density_score'] = output['case_density_score'].round(1)
    output['lot_size_score'] = output['lot_size_score'].round(1)
    output['zoning_restrictiveness'] = output['zoning_restrictiveness'].round(1)

    # Write output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Parcels scored: {len(output):,}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print()
    print("Risk level distribution:")
    print(output['risk_level'].value_counts().sort_index().to_string())
    print()
    print("Score statistics:")
    print(output['risk_score'].describe().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
