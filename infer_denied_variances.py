#!/usr/bin/env python3
"""
infer_denied_variances.py — Infer variance types for denied ZBA cases
that are missing variance_types, using tracker project descriptions
and parcel zoning dimensional data.

Steps:
1. Load zba_cases_cleaned.csv + boston_parcels_zoning.geojson
2. For denied cases missing variance_types:
   a. Extract project details from tracker_description
   b. Look up parcel zoning limits via pa_parcel_id → GeoJSON
   c. Infer which variances are needed by comparing project vs limits
   d. Flag use changes, new construction, parking/height mentions
3. Update variance_types and num_variances
4. Save updated CSV and report results
"""

import pandas as pd
import numpy as np
import json
import re
import sys
from collections import defaultdict


def load_parcel_zoning(geojson_path):
    """Build parcel_id → zoning dimensions lookup from GeoJSON."""
    print("Loading parcel zoning data...")
    with open(geojson_path) as f:
        gj = json.load(f)

    lookup = {}
    for feat in gj['features']:
        props = feat['properties']
        pid = props.get('parcel_id', '')
        lookup[pid] = {
            'max_far': props.get('max_far'),
            'max_height_ft': props.get('max_height_ft'),
            'max_floors': props.get('max_floors'),
            'front_setback_ft': props.get('front_setback_ft'),
            'side_setback_ft': props.get('side_setback_ft'),
            'rear_setback_ft': props.get('rear_setback_ft'),
            'max_du_per_area': props.get('max_du_per_area'),
            'primary_zoning': props.get('primary_zoning'),
            'subdistrict_use': props.get('subdistrict_use'),
            'subdistrict_type': props.get('subdistrict_type'),
        }

    print(f"  Loaded {len(lookup)} parcels with zoning dimensions")
    return lookup


def pa_parcel_to_geojson_id(pa_id):
    """Convert PA parcel ID (e.g., 100001000 or 100001000.0) to GeoJSON format (0100001000)."""
    if pd.isna(pa_id):
        return None
    pid_str = str(int(float(pa_id)))
    # GeoJSON IDs are 10-digit zero-padded
    return pid_str.zfill(10)


def extract_stories_from_desc(desc):
    """Extract number of stories from tracker description."""
    if not isinstance(desc, str):
        return None
    # "4 story", "4-story", "three story", "3 stories"
    word_nums = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                 'eleven': 11, 'twelve': 12}

    patterns = [
        r'(\d+)[\s-]*stor(?:y|ies)',
        r'(\d+)[\s-]*floor',
        r'(\d+)[\s-]*level',
    ]
    for pat in patterns:
        m = re.search(pat, desc, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Word numbers
    for word, num in word_nums.items():
        if re.search(rf'{word}[\s-]*stor(?:y|ies)', desc, re.IGNORECASE):
            return num

    return None


def extract_units_from_desc(desc):
    """Extract number of units from tracker description."""
    if not isinstance(desc, str):
        return None

    patterns = [
        r'(\d+)\s*(?:dwelling\s+)?units?',
        r'(\d+)\s*(?:residential\s+)?(?:condo|apartment|flat)s?',
        r'(\d+)\s*(?:family|fam)\b',
        r'comprising\s+(?:of\s+)?(\d+)\s+units',
    ]
    for pat in patterns:
        m = re.search(pat, desc, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if val > 0 and val < 500:  # sanity check
                return val
    return None


def extract_height_from_desc(desc):
    """Extract height in feet from description."""
    if not isinstance(desc, str):
        return None
    m = re.search(r'(\d+)[\s-]*(?:foot|feet|ft|\')\s*(?:tall|high|height)', desc, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def detect_use_change(desc):
    """Detect if description mentions a use change."""
    if not isinstance(desc, str):
        return False
    patterns = [
        r'change\s+(?:the\s+)?(?:occupancy|use|usage)',
        r'convert\s+(?:from|the|existing)',
        r'conversion\s+(?:of|from)',
        r'(?:from|change)\s+\w+\s+to\s+\w+',
        r'change\s+of\s+occupancy',
        r'change\s+of\s+use',
    ]
    for pat in patterns:
        if re.search(pat, desc, re.IGNORECASE):
            return True
    return False


def detect_new_construction(desc):
    """Detect new construction from description."""
    if not isinstance(desc, str):
        return False
    patterns = [
        r'\berect\b',
        r'\bnew\s+(?:construction|building|residential|commercial|structure)',
        r'\bconstruct\s+(?:a\s+)?new\b',
        r'\bbuild\s+(?:a\s+)?new\b',
        r'\bdemolish\s+(?:and|&)\s+(?:re)?build\b',
        r'\braze\s+and\s+(?:re)?build\b',
    ]
    for pat in patterns:
        if re.search(pat, desc, re.IGNORECASE):
            return True
    return False


def detect_parking_mention(desc):
    """Detect parking mentioned in description."""
    if not isinstance(desc, str):
        return False
    return bool(re.search(r'\bparking\b|\bgarage\b|\bvehicle\b|\bcar\s*space', desc, re.IGNORECASE))


def detect_addition(desc):
    """Detect addition/expansion from description."""
    if not isinstance(desc, str):
        return False
    patterns = [
        r'\baddition\b',
        r'\bextension\b',
        r'\bexpand\b',
        r'\badd\s+(?:a\s+)?(?:\d+|new)\s+(?:story|stories|floor|unit|room|deck)',
        r'\broof\s*deck\b',
        r'\bdormer\b',
    ]
    for pat in patterns:
        if re.search(pat, desc, re.IGNORECASE):
            return True
    return False


def detect_residential(desc):
    """Detect residential use from description."""
    if not isinstance(desc, str):
        return False
    return bool(re.search(
        r'\bresidential\b|\bdwelling\b|\bunit[s]?\b|\bfamily\b|\bcondo\b|\bapartment\b|\bhousing\b',
        desc, re.IGNORECASE
    ))


def detect_commercial(desc):
    """Detect commercial use from description."""
    if not isinstance(desc, str):
        return False
    return bool(re.search(
        r'\bcommercial\b|\bretail\b|\brestaurant\b|\boffice\b|\bstore\b|\bshop\b|\bbusiness\b|\bbar\b|\btavern\b',
        desc, re.IGNORECASE
    ))


def infer_variances(row, parcel_zoning):
    """
    Infer which variance types a denied case likely needed.

    Returns: list of inferred variance type strings
    """
    desc = row.get('tracker_description', '')
    if not isinstance(desc, str) or not desc.strip():
        return []

    # Get parcel zoning dimensions
    pa_id = row.get('pa_parcel_id')
    geo_id = pa_parcel_to_geojson_id(pa_id)
    zoning = parcel_zoning.get(geo_id, {}) if geo_id else {}

    inferred = []
    reasons = []

    # Extract project details
    stories = extract_stories_from_desc(desc)
    units = extract_units_from_desc(desc)
    height_ft = extract_height_from_desc(desc)
    is_use_change = detect_use_change(desc)
    is_new_const = detect_new_construction(desc)
    is_addition = detect_addition(desc)
    has_parking = detect_parking_mention(desc)
    is_residential = detect_residential(desc)
    is_commercial = detect_commercial(desc)

    max_floors = zoning.get('max_floors')
    max_height = zoning.get('max_height_ft')
    max_far = zoning.get('max_far')
    max_du = zoning.get('max_du_per_area')
    subdistrict_use = zoning.get('subdistrict_use', '')

    # --- Height variance ---
    if stories and max_floors and stories > max_floors:
        inferred.append('height')
        reasons.append(f'{stories} stories > {max_floors} max floors')
    elif height_ft and max_height and height_ft > max_height:
        inferred.append('height')
        reasons.append(f'{height_ft}ft > {max_height}ft max height')
    elif stories and stories >= 4 and not max_floors:
        # 4+ stories often needs height variance in Boston residential zones
        inferred.append('height')
        reasons.append(f'{stories} stories, likely exceeds limit (no zoning data)')

    # --- FAR variance ---
    # New multi-unit construction often exceeds FAR
    if is_new_const and units and units >= 4 and max_far and max_far < 2.0:
        inferred.append('far')
        reasons.append(f'new construction with {units} units, max FAR {max_far}')

    # --- Parking variance ---
    if has_parking and is_new_const:
        # New construction mentioning parking often means insufficient parking
        inferred.append('parking')
        reasons.append('new construction with parking mention')
    elif units and units >= 3 and not has_parking:
        # Multi-unit without parking mention = likely parking variance
        inferred.append('parking')
        reasons.append(f'{units} units with no parking mentioned')

    # --- Conditional use / use variance ---
    if is_use_change:
        inferred.append('conditional_use')
        reasons.append('change of use/occupancy')

    # --- Use conflict with zoning subdistrict ---
    if subdistrict_use and isinstance(subdistrict_use, str):
        sub_lower = subdistrict_use.lower()
        if is_commercial and 'residential' in sub_lower and 'commercial' not in sub_lower:
            if 'conditional_use' not in inferred:
                inferred.append('conditional_use')
                reasons.append(f'commercial use in {subdistrict_use} zone')
        if is_residential and 'commercial' in sub_lower and 'residential' not in sub_lower:
            if 'conditional_use' not in inferred:
                inferred.append('conditional_use')
                reasons.append(f'residential use in {subdistrict_use} zone')

    # --- Lot area / frontage (for new construction, typically needed) ---
    if is_new_const and zoning:
        # Most new construction in Boston needs lot area variance
        inferred.append('lot_area')
        reasons.append('new construction typically needs lot area relief')

    # --- Setback variances for additions ---
    if is_addition and zoning:
        front_sb = zoning.get('front_setback_ft')
        side_sb = zoning.get('side_setback_ft')
        rear_sb = zoning.get('rear_setback_ft')
        # Additions and roof decks commonly need setback relief
        if re.search(r'roof\s*deck|dormer|rear|front|side', desc, re.IGNORECASE):
            if rear_sb is not None:
                inferred.append('rear_setback')
                reasons.append('addition likely needs setback relief')

    # --- Roof deck (common denied type, often needs height + setback) ---
    if re.search(r'roof\s*deck|deck\s+on\s+(?:the\s+)?roof', desc, re.IGNORECASE):
        if 'height' not in inferred:
            inferred.append('height')
            reasons.append('roof deck typically exceeds height limit')
        if 'rear_setback' not in inferred:
            inferred.append('rear_setback')
            reasons.append('roof deck typically needs setback relief')

    # --- Signage / billboard ---
    if re.search(r'\bsign(?:age)?\b|\bbillboard\b|\bbanner\b|\bawning\b', desc, re.IGNORECASE):
        inferred.append('signage')
        reasons.append('signage/billboard project')

    # --- Mixed-use projects ---
    if re.search(r'mixed[\s-]*use', desc, re.IGNORECASE) or (is_residential and is_commercial):
        if 'conditional_use' not in inferred:
            inferred.append('conditional_use')
            reasons.append('mixed-use project')
        if 'far' not in inferred:
            inferred.append('far')
            reasons.append('mixed-use typically exceeds FAR')
        if 'height' not in inferred:
            inferred.append('height')
            reasons.append('mixed-use typically exceeds height')
        if 'parking' not in inferred:
            inferred.append('parking')
            reasons.append('mixed-use typically needs parking relief')

    # --- Townhouses ---
    if re.search(r'\btownhouse[s]?\b|\btown\s*home[s]?\b|\brow\s*house[s]?\b', desc, re.IGNORECASE):
        if 'lot_area' not in inferred:
            inferred.append('lot_area')
            reasons.append('townhouse project needs lot area relief')
        if 'parking' not in inferred:
            inferred.append('parking')
            reasons.append('townhouse project typically needs parking')
        if 'lot_frontage' not in inferred:
            inferred.append('lot_frontage')
            reasons.append('townhouse project needs frontage relief')

    # --- Multi-family without explicit "erect/new" but still construction ---
    if re.search(r'(\d+)[\s-]*(?:family\s+dwelling|residential\s+units|apts?|apartments)', desc, re.IGNORECASE) and not inferred:
        m = re.search(r'(\d+)[\s-]*(?:family|residential\s+units|apts?|apartments)', desc, re.IGNORECASE)
        if m:
            fam = int(m.group(1))
            if fam >= 3:
                inferred.append('parking')
                reasons.append(f'{fam}-unit dwelling likely needs parking')
                inferred.append('use')
                reasons.append(f'{fam}-unit dwelling in likely non-conforming lot')

    # --- Standalone "N residential units" or "N units" ---
    if not inferred and re.search(r'(\d+)\s+(?:residential\s+)?units', desc, re.IGNORECASE):
        m = re.search(r'(\d+)\s+(?:residential\s+)?units', desc, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if n >= 3:
                inferred.append('parking')
                reasons.append(f'{n} units likely needs parking')

    # --- Brewery / restaurant expansion ---
    if re.search(r'\bbrewery\b|\bbrew(?:ing|house)\b', desc, re.IGNORECASE):
        if 'conditional_use' not in inferred:
            inferred.append('conditional_use')
            reasons.append('brewery use')

    if re.search(r'add\s+(?:\d+\s+)?(?:new\s+)?seat(?:ing|s)', desc, re.IGNORECASE):
        if 'conditional_use' not in inferred:
            inferred.append('conditional_use')
            reasons.append('restaurant seating expansion')

    # --- Wireless / telecom ---
    if re.search(r'\bwireless\b|\bantenna\b|\btelecom\b|\bcell\s*tower\b', desc, re.IGNORECASE):
        if 'conditional_use' not in inferred:
            inferred.append('conditional_use')
            reasons.append('wireless/telecom installation')

    # --- Curb cut / driveway ---
    if re.search(r'\bcurb\s*cut\b|\bdriveway\b', desc, re.IGNORECASE):
        if 'parking' not in inferred:
            inferred.append('parking')
            reasons.append('curb cut/driveway')

    # --- Basement conversion / extension ---
    if re.search(r'(?:extend|convert|finish)\s+.*basement|basement\s+(?:conversion|extension|living)', desc, re.IGNORECASE):
        if 'far' not in inferred:
            inferred.append('far')
            reasons.append('basement conversion adds FAR')

    # --- Deck / porch (not roof deck, already handled) ---
    if re.search(r'\b(?:rear\s+)?deck\b|\bporch\b', desc, re.IGNORECASE) and 'rear_setback' not in inferred:
        if not re.search(r'roof\s*deck', desc, re.IGNORECASE):
            inferred.append('rear_setback')
            reasons.append('deck/porch likely needs setback relief')

    # --- Large projects (square footage mentioned) ---
    sqft_match = re.search(r'([\d,]+)\s*(?:square\s*f(?:oo|ee)t|sf|sq\.?\s*ft)', desc, re.IGNORECASE)
    if sqft_match and not inferred:
        sqft = int(sqft_match.group(1).replace(',', ''))
        if sqft > 10000:
            inferred.append('far')
            reasons.append(f'large project ({sqft:,} sqft) likely exceeds FAR')
            inferred.append('height')
            reasons.append(f'large project likely exceeds height')
            inferred.append('parking')
            reasons.append(f'large project likely needs parking relief')

    # --- Hotel ---
    if re.search(r'\bhotel\b|\blodging\b|\binn\b', desc, re.IGNORECASE):
        if 'conditional_use' not in inferred:
            inferred.append('conditional_use')
            reasons.append('hotel/lodging use')
        if 'parking' not in inferred:
            inferred.append('parking')
            reasons.append('hotel typically needs parking relief')

    return inferred, reasons


def flag_basic_attributes(desc):
    """For cases where full inference isn't possible, flag basic attributes."""
    flags = {}
    if not isinstance(desc, str) or not desc.strip():
        return flags

    flags['is_use_change'] = 1 if detect_use_change(desc) else 0
    flags['is_new_construction'] = 1 if detect_new_construction(desc) else 0
    flags['has_parking_mention'] = 1 if detect_parking_mention(desc) else 0
    flags['has_height_mention'] = 1 if extract_stories_from_desc(desc) is not None else 0
    flags['is_addition'] = 1 if detect_addition(desc) else 0
    flags['is_residential'] = 1 if detect_residential(desc) else 0
    flags['is_commercial'] = 1 if detect_commercial(desc) else 0

    return flags


def main():
    print("=" * 70)
    print("INFER DENIED VARIANCES FROM TRACKER DESCRIPTIONS")
    print("=" * 70)

    # Load data
    print("\nLoading zba_cases_cleaned.csv...")
    df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
    print(f"  Total cases: {len(df)}")

    parcel_zoning = load_parcel_zoning('boston_parcels_zoning.geojson')

    # Identify target cases: denied with no variance_types
    denied_mask = df['decision_clean'] == 'DENIED'
    no_var_mask = df['variance_types'].isna() | (df['variance_types'] == '')
    target_mask = denied_mask & no_var_mask

    target_cases = df[target_mask].copy()
    print(f"\nDenied cases: {denied_mask.sum()}")
    print(f"Denied without variance_types: {len(target_cases)}")

    has_desc_mask = target_cases['tracker_description'].notna() & (target_cases['tracker_description'] != '')
    print(f"  with tracker_description: {has_desc_mask.sum()}")

    has_parcel_mask = target_cases['pa_parcel_id'].notna()
    print(f"  with pa_parcel_id: {has_parcel_mask.sum()}")

    has_both = has_desc_mask & has_parcel_mask
    print(f"  with BOTH description + parcel: {has_both.sum()}")

    # Process each target case
    enriched_full = 0      # Got inferred variances from desc + zoning comparison
    enriched_desc_only = 0  # Got inferred variances from description alone (no parcel match)
    flagged_only = 0        # Only got basic flags
    no_info = 0             # No description at all

    variance_counts = defaultdict(int)
    all_reasons = []

    for idx in target_cases.index:
        desc = df.at[idx, 'tracker_description']

        if not isinstance(desc, str) or not desc.strip():
            no_info += 1
            continue

        # Try full inference
        inferred, reasons = infer_variances(df.loc[idx], parcel_zoning)

        if inferred:
            # Deduplicate
            inferred = list(dict.fromkeys(inferred))

            # Update the dataframe
            df.at[idx, 'variance_types'] = ','.join(inferred)
            df.at[idx, 'num_variances'] = len(inferred)

            # Set individual variance columns
            all_var_types = ['height', 'far', 'lot_area', 'lot_frontage', 'lot_width',
                           'front_setback', 'rear_setback', 'side_setback',
                           'parking', 'conditional_use', 'use', 'signage', 'open_space']
            for vt in all_var_types:
                col = f'var_{vt}'
                if col in df.columns:
                    df.at[idx, col] = 1 if vt in inferred else df.at[idx, col]

            pa_id = df.at[idx, 'pa_parcel_id']
            geo_id = pa_parcel_to_geojson_id(pa_id)
            if geo_id and geo_id in parcel_zoning:
                enriched_full += 1
            else:
                enriched_desc_only += 1

            for v in inferred:
                variance_counts[v] += 1
            all_reasons.extend(reasons)
        else:
            # Just flag basic attributes
            flags = flag_basic_attributes(desc)
            if any(flags.values()):
                flagged_only += 1
                # For cases with use changes, we can still infer conditional_use
                if flags.get('is_use_change'):
                    df.at[idx, 'variance_types'] = 'conditional_use'
                    df.at[idx, 'num_variances'] = 1
                    variance_counts['conditional_use'] += 1
                    enriched_desc_only += 1
                    flagged_only -= 1
            else:
                flagged_only += 1

    # Report results
    total_enriched = enriched_full + enriched_desc_only
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Target cases (denied, no variances): {len(target_cases)}")
    print(f"  Enriched with full inference (desc + zoning data): {enriched_full}")
    print(f"  Enriched from description alone:                   {enriched_desc_only}")
    print(f"  Total enriched with inferred variances:            {total_enriched}")
    print(f"  Flagged only (had desc, no variances inferred):    {flagged_only}")
    print(f"  No tracker description:                            {no_info}")

    print(f"\nInferred variance type distribution:")
    for vt, count in sorted(variance_counts.items(), key=lambda x: -x[1]):
        print(f"  {vt:25s}: {count}")

    print(f"\nSample inference reasons (first 20):")
    for r in all_reasons[:20]:
        print(f"  - {r}")

    # Verify: how many denied cases now have variance_types?
    denied_after = df[df['decision_clean'] == 'DENIED']
    has_var_after = denied_after['variance_types'].notna() & (denied_after['variance_types'] != '')
    print(f"\nDenied cases with variance_types:")
    print(f"  Before: {denied_mask.sum() - len(target_cases)} / {denied_mask.sum()}")
    print(f"  After:  {has_var_after.sum()} / {denied_mask.sum()}")
    print(f"  Improvement: +{total_enriched} cases ({total_enriched/len(target_cases)*100:.1f}% of missing)")

    # Show some examples
    print(f"\nSample enriched cases:")
    enriched_examples = df[(df['decision_clean'] == 'DENIED') & (df.index.isin(target_cases.index)) & (df['variance_types'].notna()) & (df['variance_types'] != '')]
    for _, row in enriched_examples.head(10).iterrows():
        print(f"  {row['case_number']}: {row['variance_types']}")
        desc = str(row.get('tracker_description', ''))[:100]
        print(f"    Desc: {desc}")

    # Save
    print(f"\nSaving updated zba_cases_cleaned.csv...")
    df.to_csv('zba_cases_cleaned.csv', index=False)
    print("Done!")

    return total_enriched


if __name__ == '__main__':
    enriched = main()
    sys.exit(0 if enriched > 0 else 1)
