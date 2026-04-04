"""
PermitIQ — Improved Variance Extraction for Denied Cases

Targets denied cases with raw_text but no variance_types.
Uses broader regex patterns + inference from project descriptions + tracker descriptions.

Strategy:
1. DIRECT match: Broader regex for explicit variance/violation mentions (OCR-tolerant)
2. SECTION match: Extract from "Article X, Section Y" patterns + known section meanings
3. INFERRED match: Infer likely variance types from project descriptions
4. TRACKER match: Extract from tracker_description column
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

print("=" * 70)
print("PermitIQ — Improved Variance Extraction for Denied Cases")
print("=" * 70)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"Total cases: {len(df)}")

# Target: denied cases with raw_text but no variance_types
mask = (
    (df['decision_clean'] == 'DENIED') &
    (df['variance_types'].isna()) &
    (df['raw_text'].notna())
)
target_indices = df[mask].index.tolist()
print(f"Denied cases with no variance_types but has raw_text: {len(target_indices)}")

# Also count how many already have variance_types for comparison
denied_with_var = ((df['decision_clean'] == 'DENIED') & (df['variance_types'].notna())).sum()
print(f"Denied cases already with variance_types: {denied_with_var}")


# ============================================================
# LAYER 1: BROAD DIRECT REGEX (OCR-tolerant)
# ============================================================

# These patterns are much broader than the originals in reextract_features.py
# They handle OCR garbling, alternative phrasings, and different document formats
BROAD_VARIANCE_PATTERNS = {
    'height': [
        r'h\s*e\s*i\s*g\s*h\s*t',           # OCR spaced: "h e i g h t"
        r'hei[gq]ht',                          # OCR: "heiqht"
        r'helght',                              # OCR misspelling
        r'height\s*(?:variance|relief|limit|exceed|restrict|regulat|violat)',
        r'exceed\w*\s+(?:the\s+)?(?:maximum\s+)?(?:allowable\s+)?height',
        r'over\s*(?:the\s+)?(?:allowable\s+)?height',
        r'(?:maximum|max\.?)\s+(?:building\s+)?height',
        r'taller\s+than',
        r'(?:building|structure)\s+height',
        r'roof\s+(?:height|line)\s+(?:exceed|violat)',
        r'stories?\s+(?:exceed|over|above|more\s+than)',
        r'(?:number|#)\s*(?:of\s+)?stories?\s+(?:exceed|violat)',
        r'(?:4|5|6|7|8|9|10)\s*(?:-\s*)?stor(?:y|ies)\s+(?:where|when|in\s+a)',
        r'headhouse',                           # roof structure = height issue
        r'roof\s+structure\s+restrict',
    ],
    'far': [
        r'f\s*\.?\s*a\s*\.?\s*r\s*\.?',       # F.A.R. with OCR spacing
        r'floor\s*area\s*ratio',
        r'floor\s*area\s*(?:exceed|violat|excessive)',
        r'excessive\s*(?:floor\s*area|f\.?a\.?r)',
        r'exceeds?\s+(?:the\s+)?(?:maximum\s+)?(?:allowable\s+)?(?:floor\s+area|f\.?a\.?r)',
        r'(?:maximum|max\.?)\s+floor\s*area',
        r'gross\s+floor\s+area\s+(?:exceed|ratio)',
    ],
    'lot_area': [
        r'lot\s*area',
        r'lot\s*size',
        r'(?:insufficient|undersized|substandard|deficient)\s*lot',
        r'minimum\s*lot\s*(?:area|size)',
        r'lot\s*(?:is\s+)?(?:too\s+)?small',
        r'lot\s*(?:does\s+)?not\s+(?:meet|comply|conform)',
        r'(?:square\s+)?(?:foot|feet|footage|sf)\s+(?:lot\s+)?(?:where|when|required)',
    ],
    'lot_frontage': [
        r'(?:lot\s*)?frontage',
        r'lot\s*width',
        r'(?:insufficient|deficient)\s*frontage',
        r'minimum\s*(?:lot\s+)?frontage',
        r'(?:street\s+)?frontage\s+(?:required|insufficient|less)',
    ],
    'front_setback': [
        r'front\s*(?:yard|setback|set\s*back)',
        r'(?:insufficient|deficient)\s*front\s*(?:yard|setback)',
        r'front\s+(?:yard\s+)?(?:depth|setback)\s+(?:required|insufficient|less)',
    ],
    'rear_setback': [
        r'rear\s*(?:yard|setback|set\s*back)',
        r'(?:insufficient|deficient)\s*rear\s*(?:yard|setback)',
        r'rear\s+(?:yard\s+)?(?:depth|setback)\s+(?:required|insufficient|less)',
    ],
    'side_setback': [
        r'side\s*(?:yard|setback|set\s*back)',
        r'(?:insufficient|deficient)\s*side\s*(?:yard|setback)',
        r'side\s+(?:yard\s+)?(?:width|setback)\s+(?:required|insufficient|less)',
    ],
    'parking': [
        r'(?:off[\s-]*street\s+)?parking\s*(?:variance|relief|space|spot|stall)',
        r'(?:insufficient|deficient|inadequate)\s*parking',
        r'parking\s*(?:required|requirement|regulation|restrict)',
        r'(?:number|#)\s*(?:of\s+)?parking',
        r'(?:\d+)\s+parking\s+space',
        r'parking\s+(?:spaces?\s+)?(?:where|when)',
        r'no\s+(?:off[\s-]*street\s+)?parking',
        r'waiv\w+\s+(?:of\s+)?parking',
        r'parking\s+(?:waiv|exempt|relief)',
    ],
    'conditional_use': [
        r'conditional\s*use',
        r'change\s*(?:of\s+)?(?:the\s+)?(?:use|occupancy)',
        r'change\s+(?:the\s+)?occupancy',
        r'(?:proposed|new)\s+use',
        r'(?:forbidden|prohibited|not\s+(?:a\s+)?(?:permitted|allowed))\s+use',
        r'use\s*(?:variance|permit|not\s+(?:permitted|allowed))',
        r'(?:use|occupancy)\s+(?:is\s+)?(?:not\s+)?(?:permitted|allowed)',
        r'use\s*(?::?\s*)regs?\s+in\s+',         # "Use: Regs in Business"
    ],
    'open_space': [
        r'(?:usable\s+)?open\s+space',
        r'(?:insufficient|deficient)\s*(?:usable\s+)?open\s*space',
        r'open\s+space\s+(?:required|requirement|ratio)',
        r'landscap\w+\s+(?:open\s+)?space',
    ],
    'density': [
        r'density',
        r'(?:dwelling\s+)?units?\s*per\s*(?:acre|lot|sf|square)',
        r'too\s+many\s+(?:units|dwellings)',
        r'(?:maximum|max\.?)\s+(?:number\s+of\s+)?(?:units|dwellings)',
        r'(?:exceeds?\s+)?(?:the\s+)?(?:maximum|allowable)\s+(?:number\s+of\s+)?(?:units|density)',
    ],
    'nonconforming': [
        r'nonconform',
        r'non[\s-]*conform',
        r'pre[\s-]*existing',
        r'(?:extension|expansion|enlargement)\s+(?:of\s+)?(?:a\s+)?(?:nonconform|non[\s-]*conform)',
        r'(?:legal(?:ly)?|existing)\s+nonconform',
        r'(?:nonconform|non[\s-]*conform)\w*\s+(?:use|structure|building)',
        r'extend\w*\s+(?:the\s+)?(?:nonconform|non[\s-]*conform)',
    ],
}


# ============================================================
# LAYER 2: SECTION-BASED INFERENCE
# Known Boston Zoning Code article/section meanings
# ============================================================

# Article -> likely variance type mapping
ARTICLE_VARIANCE_MAP = {
    # Article 7 = dimensional requirements (height, setbacks, FAR, lot area)
    7: ['height', 'far', 'lot_area', 'front_setback', 'rear_setback', 'side_setback'],
    # Article 8 = use regulations
    8: ['conditional_use'],
    # Specific neighborhood articles with dimensional requirements
    25: ['height', 'far', 'lot_area'],
    32: ['height', 'far', 'lot_area'],
    39: ['height', 'far', 'lot_area'],
    51: ['height', 'far', 'lot_area'],
    53: ['height', 'far', 'lot_area'],
    55: ['height', 'far', 'lot_area'],
    56: ['height', 'far', 'lot_area'],
    60: ['height', 'far', 'lot_area'],
    62: ['height', 'far', 'lot_area'],
    63: ['height', 'far', 'lot_area'],
    64: ['height', 'far', 'lot_area'],
    65: ['height', 'far', 'lot_area'],
    66: ['conditional_use'],  # Use regs in business districts
    67: ['height', 'far', 'lot_area'],
    68: ['height', 'far', 'lot_area'],
    69: ['height', 'far', 'lot_area'],
    80: ['height', 'far'],   # Large project review
}

# Section keywords that map to variance types
SECTION_KEYWORD_MAP = {
    r'roof\s+structure': ['height'],
    r'(?:height|stor(?:y|ies))\s+(?:limit|restrict|regulat|maximum)': ['height'],
    r'floor\s+area\s+ratio': ['far'],
    r'lot\s+(?:area|size)\s+(?:minimum|regulat)': ['lot_area'],
    r'frontage': ['lot_frontage'],
    r'front\s+yard': ['front_setback'],
    r'rear\s+yard': ['rear_setback'],
    r'side\s+yard': ['side_setback'],
    r'(?:off[\s-]*street\s+)?parking': ['parking'],
    r'open\s+space': ['open_space'],
    r'use\s*(?::|regulat|regs?)': ['conditional_use'],
    r'(?:flood|gcod|coastal)': ['nonconforming'],
    r'density|dwelling\s+unit': ['density'],
}


# ============================================================
# LAYER 3: INFERENCE FROM PROJECT DESCRIPTIONS
# When text has no explicit variance mention, infer from project type
# These are WEAKER signals, only used when no direct match found
# ============================================================

INFERRED_PATTERNS = {
    'conditional_use': [
        r'change\s+(?:of\s+)?occupancy',
        r'(?:from|to)\s+(?:\w+\s+){0,3}(?:restaurant|retail|commercial|office|clinic|daycare|school|church|bar|nightclub|club|store|shop|salon|spa|gym|fitness|medical|dental|veterinar)',
        r'(?:establish|convert|erect)\s+(?:a\s+)?(?:restaurant|retail|commercial|daycare|school|church|bar|nightclub|club)',
        r'(?:new|proposed)\s+(?:restaurant|retail|commercial|daycare|school|church|bar|nightclub|club|store)',
        r'liquor\s+(?:store|license)',
        r'(?:adult|marijuana|cannabis|dispensary)',
        r'billboard',
        r'wireless\s+(?:communication|facility|antenna|tower)',
        r'private\s+(?:club|membership)',
        r'(?:lodging|rooming|boarding)\s+house',
        r'increase\s+(?:the\s+)?(?:capacity|occupancy|lodgers|persons)',
    ],
    'nonconforming': [
        r'legaliz\w+',
        r'confirm\s+(?:the\s+)?occupancy',
        r'existing\s+(?:condition|use)',
        r'(?:work|construction)\s+(?:already|previously)\s+(?:done|completed|performed)',
        r'correct\s+violation',
        r'(?:pre[\s-]*existing|existing)\s+(?:nonconform|non[\s-]*conform)',
    ],
    'height': [
        r'(?:add|build|construct|erect|install)\s+(?:\w+\s+){0,3}(?:roof\s+deck|penthouse|headhouse)',
        r'(?:add|build|construct|erect)\s+(?:\w+\s+){0,3}(?:(?:3rd|third|4th|fourth|5th|fifth)\s+(?:floor|story|storey|level))',
        r'(?:add|new|construct)\s+(?:\w+\s+){0,3}dormer',
        r'vertical\s+(?:addition|expansion|extension)',
        r'extend\s+(?:living\s+)?space\s+(?:to|onto)\s+(?:the\s+)?(?:attic|roof|3rd|third|4th|fourth)',
    ],
    'lot_area': [
        r'(?:subdivid|subdivision)',
        r'(?:newly\s+created|new)\s+(?:\d+\s+(?:square\s+foot|sf|sq\.?\s*ft))\s+lot',
        r'(?:vacant|empty)\s+lot',
    ],
    'parking': [
        r'(?:pave|establish|create)\s+(?:\w+\s+){0,3}(?:driveway|parking|curb\s+cut)',
    ],
}


def extract_variances_broad(text, tracker_desc=None):
    """
    Multi-layer variance extraction.
    Returns (list_of_variance_types, method_used)
    """
    if pd.isna(text) or str(text).strip() == '' or str(text).strip() == 'nan':
        return [], 'no_text'

    text_str = str(text)
    text_lower = text_str.lower()

    # Combine with tracker description if available
    combined_lower = text_lower
    if tracker_desc and not pd.isna(tracker_desc) and str(tracker_desc).strip() != 'nan':
        combined_lower = text_lower + ' ' + str(tracker_desc).lower()

    found = set()
    method = 'none'

    # --- LAYER 1: Direct broad regex ---
    for var_type, patterns in BROAD_VARIANCE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined_lower):
                found.add(var_type)
                if method == 'none':
                    method = 'direct'
                break  # one match per type is enough

    # --- LAYER 2: Article/Section-based inference ---
    # Extract article numbers
    article_matches = re.findall(r'article\s*(\d+)', combined_lower)
    for art_str in article_matches:
        art_num = int(art_str)
        if art_num in ARTICLE_VARIANCE_MAP:
            # Only add if we don't have any direct matches for this category
            for vtype in ARTICLE_VARIANCE_MAP[art_num]:
                found.add(vtype)
                if method in ('none',):
                    method = 'article'

    # Extract section-level keywords
    for pattern, vtypes in SECTION_KEYWORD_MAP.items():
        if re.search(pattern, combined_lower):
            for vtype in vtypes:
                found.add(vtype)
                if method in ('none',):
                    method = 'section'

    # --- LAYER 3: Inferred from project description (only if no direct/article match) ---
    if not found:
        for var_type, patterns in INFERRED_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_lower):
                    found.add(var_type)
                    if method == 'none':
                        method = 'inferred'
                    break

    return sorted(found), method


# ============================================================
# APPLY TO TARGET CASES
# ============================================================

print("\nExtracting variances from denied cases...")

updated = 0
method_counts = Counter()
type_counts = Counter()

for idx in target_indices:
    row = df.loc[idx]
    text = row['raw_text']
    tracker_desc = row.get('tracker_description', None)

    variances, method = extract_variances_broad(text, tracker_desc)

    if variances:
        df.at[idx, 'variance_types'] = ','.join(variances)
        df.at[idx, 'num_variances'] = len(variances)
        updated += 1
        method_counts[method] += 1
        for v in variances:
            type_counts[v] += 1

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Target denied cases: {len(target_indices)}")
print(f"Cases updated with variance_types: {updated} ({updated/len(target_indices)*100:.1f}%)")
print(f"Cases still missing: {len(target_indices) - updated}")

print(f"\nExtraction method breakdown:")
for method, cnt in method_counts.most_common():
    print(f"  {method}: {cnt}")

print(f"\nVariance type distribution (newly extracted):")
for vtype, cnt in type_counts.most_common():
    print(f"  {vtype}: {cnt}")

# Overall stats
total_denied = (df['decision_clean'] == 'DENIED').sum()
denied_with_var_after = ((df['decision_clean'] == 'DENIED') & (df['variance_types'].notna())).sum()
print(f"\nDenied cases with variance_types: {denied_with_var} -> {denied_with_var_after} (of {total_denied} total)")

# Also update num_variances for consistency
df.loc[df['variance_types'].notna() & (df['num_variances'].isna() | (df['num_variances'] == 0)), 'num_variances'] = \
    df.loc[df['variance_types'].notna() & (df['num_variances'].isna() | (df['num_variances'] == 0)), 'variance_types'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

# Also run on ALL cases (not just denied) that are missing variance_types
print(f"\n{'='*60}")
print(f"BONUS: Running on ALL cases missing variance_types...")
print(f"{'='*60}")

all_missing = df[(df['variance_types'].isna()) & (df['raw_text'].notna())].index.tolist()
print(f"All cases missing variance_types with raw_text: {len(all_missing)}")

bonus_updated = 0
bonus_method_counts = Counter()
bonus_type_counts = Counter()

for idx in all_missing:
    # Skip already updated denied cases
    if idx in target_indices:
        continue

    row = df.loc[idx]
    text = row['raw_text']
    tracker_desc = row.get('tracker_description', None)

    variances, method = extract_variances_broad(text, tracker_desc)

    if variances:
        df.at[idx, 'variance_types'] = ','.join(variances)
        df.at[idx, 'num_variances'] = len(variances)
        bonus_updated += 1
        bonus_method_counts[method] += 1
        for v in variances:
            bonus_type_counts[v] += 1

print(f"Additional non-denied cases updated: {bonus_updated}")
if bonus_method_counts:
    print(f"Method breakdown:")
    for method, cnt in bonus_method_counts.most_common():
        print(f"  {method}: {cnt}")

# Final summary
total_with_var = df['variance_types'].notna().sum()
total_num_var = (df['num_variances'] > 0).sum()
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"Total cases: {len(df)}")
print(f"Cases with variance_types: {total_with_var} ({total_with_var/len(df)*100:.1f}%)")
print(f"Cases with num_variances > 0: {total_num_var} ({total_num_var/len(df)*100:.1f}%)")

# Save
print(f"\nSaving to zba_cases_cleaned.csv...")
df.to_csv('zba_cases_cleaned.csv', index=False)
print(f"Done. Saved {len(df)} cases with {len(df.columns)} columns.")
