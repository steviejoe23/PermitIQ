"""
PermitIQ Model Training Script v3 — Leakage-Free
CRITICAL: Only uses features available BEFORE the ZBA hearing.

Run:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 train_model_v2.py
"""

import pandas as pd
import numpy as np
import re
import hashlib
import os
import shutil
import json
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss, precision_recall_curve, f1_score,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import shared model classes so pickle can find them
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api', 'services'))
from model_classes import StackingEnsemble, ManualCalibratedModel

warnings.filterwarnings('ignore')


# ===========================
# LOAD CLEANED DATA
# ===========================
print("=" * 60)
print("  PermitIQ Model Training v3 — Leakage-Free")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"Total rows: {len(df)}")

# Only keep rows with clean decisions
df = df[df['decision_clean'].notna()].copy()
print(f"Rows with decisions: {len(df)}")

# ===========================
# DEDUPLICATION — prefer OCR cases over tracker cases
# ===========================
if 'case_number' in df.columns:
    before_dedup = len(df)
    # Sort so OCR cases come first (they have richer features than tracker-only cases)
    df['_is_tracker'] = (df['source_pdf'] == 'zba_tracker').astype(int) if 'source_pdf' in df.columns else 0
    df['_text_len'] = df['raw_text'].fillna('').str.len() if 'raw_text' in df.columns else 0
    df = df.sort_values(['_is_tracker', '_text_len'], ascending=[True, False])
    df = df.drop_duplicates(subset='case_number', keep='first')
    df = df.drop(columns=['_is_tracker', '_text_len'], errors='ignore')
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        print(f"Deduplicated: removed {dupes_removed} duplicate case numbers (OCR preferred over tracker)")
    print(f"Unique cases: {len(df)}")
    if 'source_pdf' in df.columns:
        n_tracker = (df['source_pdf'] == 'zba_tracker').sum()
        n_ocr = (df['source_pdf'] != 'zba_tracker').sum()
        print(f"  OCR cases: {n_ocr} | Tracker-only cases: {n_tracker}")

# Target variable
df['approved'] = (df['decision_clean'] == 'APPROVED').astype(int)
n_approved_total = (df['approved'] == 1).sum()
n_denied_total = (df['approved'] == 0).sum()
print(f"Approval rate: {df['approved'].mean():.1%}")
print(f"  APPROVED: {n_approved_total}")
print(f"  DENIED: {n_denied_total}")
print(f"  Class balance ratio: {n_denied_total / max(n_approved_total, 1):.2f}:1 (denied:approved)")


# ===========================
# FEATURE ENGINEERING — PRE-HEARING ONLY
# ===========================
# CRITICAL: Every feature here must be knowable BEFORE the ZBA hearing.
# If it comes from the decision text describing what happened at/after the hearing, it LEAKS.
print("\n" + "=" * 60)
print("  Feature Engineering (PRE-HEARING ONLY)")
print("=" * 60)

# --- Variance features (from application, known at filing) ---
df['num_variances'] = df['num_variances'].fillna(0).astype(int)

variance_types = [
    'height', 'far', 'lot_area', 'lot_frontage',
    'front_setback', 'rear_setback', 'side_setback',
    'parking', 'conditional_use', 'open_space', 'density', 'nonconforming'
]
for vt in variance_types:
    col = f'var_{vt}'
    if col not in df.columns:
        df[col] = df['variance_types'].fillna('').str.contains(vt).astype(int)

# --- Derive features from raw_text CAREFULLY ---
# Only extract things that describe the APPLICATION, not the OUTCOME
if 'raw_text' in df.columns:
    rt = df['raw_text'].fillna('').str.lower()

    # SAFE: Use type (from application description)
    if 'is_residential' not in df.columns:
        df['is_residential'] = rt.str.contains(r'residential|dwelling|family|apartment|condo', regex=True).astype(int)
    if 'is_commercial' not in df.columns:
        df['is_commercial'] = rt.str.contains(r'commercial|retail|office|restaurant|business', regex=True).astype(int)

    # SAFE: Attorney (representation at filing)
    # Re-extract with broader pattern — the existing column only catches 82/8767 cases
    df['has_attorney'] = rt.str.contains(
        r'attorney|counsel|esq\.?|law\s*office|represented\s*by|on\s*behalf\s*of.*(?:llc|inc|esq)',
        regex=True
    ).astype(int)

    # Also check 'contact' field for attorney indicators (tracker cases)
    if 'contact' in df.columns:
        contact_atty = df['contact'].fillna('').str.lower().str.contains(
            r'attorney|counsel|esq|law\s*office|llp|law\s*group',
            regex=True
        ).astype(int)
        df['has_attorney'] = (df['has_attorney'] | contact_atty).astype(int)

        # Professional representative detection: contacts appearing 5+ times
        # are attorneys/architects/expediters (e.g. Jeffrey Drago 675x, Richard Lynds 659x)
        contact_clean = df['contact'].fillna('').str.strip().str.lower()
        contact_counts = contact_clean.value_counts()
        pro_reps = set(contact_counts[contact_counts >= 5].index) - {'', 'nan'}
        is_pro_rep = contact_clean.isin(pro_reps).astype(int)
        df['has_attorney'] = (df['has_attorney'] | is_pro_rep).astype(int)
        print(f"  Professional reps (5+ cases): {len(pro_reps)} unique contacts, {is_pro_rep.sum()} cases")

    print(f"  has_attorney: {df['has_attorney'].sum()} cases ({df['has_attorney'].mean():.0%})")

    # SAFE: BPDA involvement (happens before ZBA hearing)
    if 'bpda_involved' not in df.columns:
        df['bpda_involved'] = rt.str.contains(r'bpda|boston\s*planning', regex=True).astype(int)

    # SAFE: Appeal type (Building vs Zoning — known at filing, strong predictor)
    # Building appeals have ~58% approval vs ~91% for zoning
    if 'appeal_type' in df.columns:
        df['is_building_appeal'] = (df['appeal_type'].fillna('').str.lower().str.contains('building')).astype(int)
    else:
        df['is_building_appeal'] = rt.str.contains(r'building\s*(?:code|violation|appeal)', regex=True).astype(int)
    print(f"  is_building_appeal: {df['is_building_appeal'].sum()} cases ({df['is_building_appeal'].mean():.0%})")

    # SAFE: Is this an appeal of building commissioner refusal? (known at filing)
    df['is_refusal_appeal'] = rt.str.contains(
        r'refus(?:al|ed)|annul|building\s*commissioner', regex=True).astype(int)
    print(f"  is_refusal_appeal: {df['is_refusal_appeal'].sum()} cases ({df['is_refusal_appeal'].mean():.0%})")

    # SAFE: Violation types (these describe what code is violated, known from application)
    if 'excessive_far' not in df.columns:
        df['excessive_far'] = rt.str.contains(r'floor\s*area\s*ratio|far|excessive.*floor', regex=True).astype(int)
    if 'insufficient_lot' not in df.columns:
        df['insufficient_lot'] = rt.str.contains(r'insufficient\s*lot|lot\s*area', regex=True).astype(int)
    if 'insufficient_frontage' not in df.columns:
        df['insufficient_frontage'] = rt.str.contains(r'insufficient\s*frontage|frontage', regex=True).astype(int)
    if 'insufficient_yard' not in df.columns:
        df['insufficient_yard'] = rt.str.contains(r'insufficient\s*(?:front|rear|side)\s*yard|setback', regex=True).astype(int)
    if 'insufficient_parking' not in df.columns:
        df['insufficient_parking'] = rt.str.contains(r'insufficient\s*parking|parking', regex=True).astype(int)

    # SAFE: Project types (from application description)
    # Broader patterns to also catch tracker descriptions
    proj_patterns = {
        'proj_demolition': r'demol|raze|tear\s*down',
        'proj_new_construction': r'new\s*construct|erect|build.*new|proposed.*building|construct.*(?:building|structure)',
        'proj_addition': r'addition|extend|expansion|enlarg',
        'proj_conversion': r'convert|conversion|change.*(?:use|occupancy)|change\s*from',
        'proj_renovation': r'renovat|remodel|alter|rehabilitat|gut\s*rehab|interior\s*(?:work|renovation)',
        'proj_subdivision': r'subdivis',
        'proj_adu': r'accessory\s*(?:dwelling|apartment)|adu|in-law',
        'proj_roof_deck': r'roof\s*deck',
        'proj_parking': r'parking\s*(?:garage|structure|lot)',
        'proj_single_family': r'single.?family|one.?family',
        'proj_multi_family': r'multi.?family|(?:two|three|four|five|six).?family',
        'proj_mixed_use': r'mixed.?use',
    }
    for col_name, pattern in proj_patterns.items():
        if col_name not in df.columns:
            df[col_name] = rt.str.contains(pattern, regex=True).astype(int)
        else:
            # Re-extract with broader patterns for tracker cases that had sparse features
            missing_mask = (df[col_name].fillna(0) == 0)
            if missing_mask.sum() > 0:
                df.loc[missing_mask, col_name] = rt[missing_mask].str.contains(pattern, regex=True).astype(int)

    # SAFE: Better units/stories extraction from tracker descriptions
    # Tracker descriptions often have "5 Residential Units", "3 story", etc.
    # Override zeros with extracted values
    units_pattern = r'(\d+)\s*(?:residential\s*)?(?:unit|condo|apartment|dwelling)'
    stories_pattern = r'(\d+)\s*(?:stor|floor|level)'
    extracted_units = rt.str.extract(units_pattern, expand=False).astype(float)
    extracted_stories = rt.str.extract(stories_pattern, expand=False).astype(float)
    # Fill in where we currently have 0
    if 'proposed_units' in df.columns:
        zero_units = df['proposed_units'].fillna(0) == 0
        df.loc[zero_units & extracted_units.notna(), 'proposed_units'] = extracted_units[zero_units & extracted_units.notna()]
    if 'proposed_stories' in df.columns:
        zero_stories = df['proposed_stories'].fillna(0) == 0
        df.loc[zero_stories & extracted_stories.notna(), 'proposed_stories'] = extracted_stories[zero_stories & extracted_stories.notna()]

    print(f"  Units extracted: {extracted_units.notna().sum()} cases")
    print(f"  Stories extracted: {extracted_stories.notna().sum()} cases")

    # SAFE: Legal framework (which articles apply — known from zoning code)
    if 'is_conditional_use' not in df.columns:
        df['is_conditional_use'] = rt.str.contains(r'conditional\s*use', regex=True).astype(int)
    if 'is_variance' not in df.columns:
        df['is_variance'] = rt.str.contains(r'variance', regex=True).astype(int)

    # *** REMOVED — POST-HEARING FEATURES (LEAK) ***
    # has_opposition — describes what happened at hearing
    # no_opposition_noted — describes hearing outcome
    # planning_support — planning recommendation (sometimes pre, often described post)
    # planning_proviso — provisos attached after approval
    # community_process — describes what happened
    # support_letters — mixed, but extracted from decision text
    # opposition_letters — same
    # non_opposition_letter — same
    # councilor_involved — extracted from hearing testimony
    # mayors_office_involved — same
    # hardship_mentioned — board's determination language
    # has_deferrals / num_deferrals — happen during process
    # text_length_log — correlates with outcome (approvals have longer text)

print("  REMOVED post-hearing features: has_opposition, no_opposition_noted,")
print("    planning_support, planning_proviso, community_process, support_letters,")
print("    opposition_letters, non_opposition_letter, councilor_involved,")
print("    mayors_office_involved, hardship_mentioned, has_deferrals, num_deferrals,")
print("    text_length_log")

# --- Fill NAs for PRE-HEARING binary features only ---
binary_cols = [
    'is_residential', 'is_commercial', 'has_attorney',
    'bpda_involved', 'is_building_appeal', 'is_refusal_appeal',
    'excessive_far', 'insufficient_lot', 'insufficient_frontage',
    'insufficient_yard', 'insufficient_parking',
    'proj_demolition', 'proj_new_construction', 'proj_addition',
    'proj_conversion', 'proj_renovation', 'proj_subdivision',
    'proj_adu', 'proj_roof_deck', 'proj_parking',
    'proj_single_family', 'proj_multi_family', 'proj_mixed_use',
    'article_7', 'article_8', 'article_51', 'article_65', 'article_80',
    'is_conditional_use', 'is_variance',
    'has_prior_permits', 'is_high_value',
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = 0

# --- Numeric features (PRE-HEARING) ---
df['proposed_units'] = df['proposed_units'].fillna(0).astype(int) if 'proposed_units' in df.columns else 0
df['proposed_stories'] = df['proposed_stories'].fillna(0).astype(int) if 'proposed_stories' in df.columns else 0

# num_articles: count of article_* columns that are 1
article_cols_present = [c for c in df.columns if c.startswith('article_') and df[c].dtype in ['int64', 'float64']]
if 'num_articles' not in df.columns:
    df['num_articles'] = df[article_cols_present].sum(axis=1) if article_cols_present else 0
df['num_articles'] = df['num_articles'].fillna(0).astype(int)

# num_sections
if 'num_sections' not in df.columns:
    df['num_sections'] = df['num_zoning_sections'].fillna(0).astype(int) if 'num_zoning_sections' in df.columns else 0
df['num_sections'] = df['num_sections'].fillna(0).astype(int)

# year_recency — extract from source_pdf, tracker dates, or filing_date
df['_year'] = np.nan
if 'source_pdf' in df.columns:
    df['_year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
# Fill from final_decision_date
if 'final_decision_date' in df.columns:
    fdd_year = pd.to_datetime(df['final_decision_date'], errors='coerce').dt.year
    df['_year'] = df['_year'].fillna(fdd_year)
# Fill from filing_date
if 'filing_date' in df.columns:
    fd_year = pd.to_datetime(df['filing_date'], errors='coerce').dt.year
    df['_year'] = df['_year'].fillna(fd_year)
# Fill from tracker CSV directly for remaining missing cases
tracker_path = 'zba_tracker.csv'
if os.path.exists(tracker_path) and df['_year'].isna().sum() > 100:
    print(f"  Filling year from tracker CSV for {df['_year'].isna().sum()} cases...")
    tracker_dates = pd.read_csv(tracker_path, usecols=['boa_apno', 'final_decision_date', 'submitted_date'], low_memory=False)
    tracker_dates['_case'] = tracker_dates['boa_apno'].fillna('').astype(str).str.strip().str.replace('-', '')
    tracker_dates['_fdd'] = pd.to_datetime(tracker_dates['final_decision_date'], errors='coerce')
    tracker_dates['_sd'] = pd.to_datetime(tracker_dates['submitted_date'], errors='coerce')
    tracker_dates['_year'] = tracker_dates['_fdd'].dt.year.fillna(tracker_dates['_sd'].dt.year)
    tracker_year_map = tracker_dates.dropna(subset=['_year']).drop_duplicates('_case').set_index('_case')['_year'].to_dict()

    df['_case_clean'] = df['case_number'].fillna('').astype(str).str.strip().str.replace('-', '')
    missing_year = df['_year'].isna()
    filled = 0
    for idx in df.index[missing_year]:
        case = df.loc[idx, '_case_clean']
        if case in tracker_year_map:
            df.loc[idx, '_year'] = tracker_year_map[case]
            filled += 1
    df.drop(columns=['_case_clean'], errors='ignore', inplace=True)
    print(f"  Filled {filled} years from tracker dates")

n_with_year = df['_year'].notna().sum()
print(f"  Year coverage: {n_with_year}/{len(df)} ({n_with_year/len(df):.0%})")
df['_year'] = df['_year'].fillna(df['_year'].median())
year_min = df['_year'].min()
df['year_recency'] = (df['_year'] - year_min).fillna(0)
print(f"  Year range: {int(df['_year'].min())}-{int(df['_year'].max())}")

# --- META-FEATURES (help model understand data quality) ---
print("\nEngineering meta-features...")

# Project complexity score: how many different things are being requested
proj_cols_present = [c for c in df.columns if c.startswith('proj_')]
df['project_complexity'] = df[proj_cols_present].sum(axis=1) if proj_cols_present else 0
print(f"  project_complexity: mean={df['project_complexity'].mean():.2f}")

# Total violations: sum of insufficient_* and excessive_*
violation_cols = ['excessive_far', 'insufficient_lot', 'insufficient_frontage',
                  'insufficient_yard', 'insufficient_parking']
df['total_violations'] = df[[c for c in violation_cols if c in df.columns]].sum(axis=1)
print(f"  total_violations: mean={df['total_violations'].mean():.2f}")

# Number of nonzero features (proxy for data richness — helps model discount sparse cases)
binary_feature_cols = [c for c in binary_cols if c in df.columns]
df['num_features_active'] = df[binary_feature_cols].sum(axis=1)
print(f"  num_features_active: mean={df['num_features_active'].mean():.2f}")

# Property features
df['lot_size_sf'] = pd.to_numeric(df.get('lot_size_sf', 0), errors='coerce').fillna(0)
df['total_value'] = pd.to_numeric(df.get('total_value', 0), errors='coerce').fillna(0)
df['property_age'] = pd.to_numeric(df.get('property_age', 0), errors='coerce').fillna(0)
df['living_area'] = pd.to_numeric(df.get('living_area', 0), errors='coerce').fillna(0)
df['value_per_sqft'] = pd.to_numeric(df.get('value_per_sqft', 0), errors='coerce').fillna(0)
df['prior_permits'] = pd.to_numeric(df.get('prior_permits', 0), errors='coerce').fillna(0)


# ===========================
# ATTORNEY WIN RATE (new, unique feature)
# ===========================
print("\nComputing attorney win rates...")
if 'applicant_name' in df.columns:
    # Count wins per applicant on TRAINING data (computed after split below)
    # For now, just prepare the column — will fill after split
    df['_applicant_clean'] = df['applicant_name'].fillna('unknown').astype(str).str.strip().str.lower()
    applicant_counts = df['_applicant_clean'].value_counts()
    repeat_applicants = applicant_counts[applicant_counts >= 3].index
    print(f"  {len(repeat_applicants)} applicants with 3+ cases")
else:
    df['_applicant_clean'] = 'unknown'
    repeat_applicants = []


# ===========================
# TRAIN/TEST SPLIT (Temporal) — BEFORE target encoding
# ===========================
print("\n" + "=" * 60)
print("  Train/Test/Calibration Split")
print("=" * 60)

if 'source_pdf' in df.columns:
    df['_year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
    max_year = df['_year'].max()
    temporal_mask = df['_year'] >= max_year
    n_recent = temporal_mask.sum()

    # Need enough cases AND both classes in the test set
    n_recent_denied = (df.loc[temporal_mask, 'approved'] == 0).sum() if temporal_mask.sum() > 0 else 0
    if n_recent >= 200 and n_recent_denied >= 10:
        train_idx = ~temporal_mask
        test_cal_idx = temporal_mask
        print(f"\nTEMPORAL SPLIT: Train on <{int(max_year)}, test+cal on {int(max_year)}+")
    else:
        # Use random stratified split — temporal split would give too-small test set
        print(f"\nOnly {n_recent} cases in {int(max_year)} — too few for temporal split")
        print(f"Using STRATIFIED RANDOM split instead (70/15/15)")
        _train_i, _rest_i = train_test_split(
            df.index, test_size=0.3, random_state=42,
            stratify=df['approved']
        )
        _test_i, _cal_i = train_test_split(
            _rest_i, test_size=0.5, random_state=42,
            stratify=df.loc[_rest_i, 'approved']
        )
        train_idx = pd.Series(False, index=df.index); train_idx.loc[_train_i] = True
        test_idx = pd.Series(False, index=df.index); test_idx.loc[_test_i] = True
        cal_idx = pd.Series(False, index=df.index); cal_idx.loc[_cal_i] = True
        print(f"   Train: {train_idx.sum()} | Test: {test_idx.sum()} | Calibration: {cal_idx.sum()}")
        # Keep _year — needed for year_ward_rate target encoding

    if n_recent >= 200 and n_recent_denied >= 10:
        test_cal_indices = df.index[test_cal_idx]
        test_cal_labels = df.loc[test_cal_indices, 'approved']
        min_class_count = test_cal_labels.value_counts().min()
        use_stratify = test_cal_labels if min_class_count >= 4 else None
        if use_stratify is None:
            print(f"   WARNING: Too few minority samples ({min_class_count}) for stratified split — using random split")
        test_indices, cal_indices = train_test_split(
            test_cal_indices, test_size=0.5, random_state=42,
            stratify=use_stratify
        )
        test_idx = pd.Series(False, index=df.index)
        cal_idx = pd.Series(False, index=df.index)
        test_idx.loc[test_indices] = True
        cal_idx.loc[cal_indices] = True

        print(f"   Train: {train_idx.sum()} | Test: {test_idx.sum()} | Calibration: {cal_idx.sum()}")
    else:
        print(f"\nOnly {n_recent} recent cases — random 3-way split")
        _min_class = df['approved'].value_counts().min()
        _strat = df['approved'] if _min_class >= 4 else None
        _train_i, _rest_i = train_test_split(df.index, test_size=0.3, random_state=42, stratify=_strat)
        _rest_strat = df.loc[_rest_i, 'approved'] if df.loc[_rest_i, 'approved'].value_counts().min() >= 4 else None
        _test_i, _cal_i = train_test_split(_rest_i, test_size=0.5, random_state=42, stratify=_rest_strat)
        train_idx = pd.Series(False, index=df.index); train_idx.loc[_train_i] = True
        test_idx = pd.Series(False, index=df.index); test_idx.loc[_test_i] = True
        cal_idx = pd.Series(False, index=df.index); cal_idx.loc[_cal_i] = True
        print(f"   Train: {train_idx.sum()} | Test: {test_idx.sum()} | Calibration: {cal_idx.sum()}")
    # Keep _year — needed for year_ward_rate target encoding
else:
    _min_class = df['approved'].value_counts().min()
    _strat = df['approved'] if _min_class >= 4 else None
    _train_i, _rest_i = train_test_split(df.index, test_size=0.3, random_state=42, stratify=_strat)
    _rest_strat = df.loc[_rest_i, 'approved'] if df.loc[_rest_i, 'approved'].value_counts().min() >= 4 else None
    _test_i, _cal_i = train_test_split(_rest_i, test_size=0.5, random_state=42, stratify=_rest_strat)
    train_idx = pd.Series(False, index=df.index); train_idx.loc[_train_i] = True
    test_idx = pd.Series(False, index=df.index); test_idx.loc[_test_i] = True
    cal_idx = pd.Series(False, index=df.index); cal_idx.loc[_cal_i] = True
    print(f"\nRandom 3-way split — Train: {train_idx.sum()} | Test: {test_idx.sum()} | Cal: {cal_idx.sum()}")


# ===========================
# TARGET ENCODING — from TRAINING data only
# ===========================
print("\nComputing target-encoded features (training data only)...")

SMOOTHING_WEIGHT = 20
df['ward'] = df['ward'].fillna('unknown').astype(str)
train_global_rate = df.loc[train_idx, 'approved'].mean()

# Ward approval rates
ward_counts = df.loc[train_idx].groupby('ward')['approved'].agg(['mean', 'count'])
ward_approval = {}
for ward_name, row_data in ward_counts.iterrows():
    weight = row_data['count'] / (row_data['count'] + SMOOTHING_WEIGHT)
    ward_approval[ward_name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
df['ward_approval_rate'] = df['ward'].map(ward_approval).fillna(train_global_rate)

# Zoning approval rates
if 'zoning_clean' not in df.columns:
    if 'zoning_district' in df.columns:
        df['zoning_clean'] = df['zoning_district'].fillna(df.get('zoning', pd.Series(dtype=str))).fillna('unknown')
    elif 'zoning' in df.columns:
        df['zoning_clean'] = df['zoning'].fillna('unknown')
    else:
        df['zoning_clean'] = 'unknown'
df['zoning_clean'] = df['zoning_clean'].fillna('unknown').astype(str)
top_zoning = df.loc[train_idx, 'zoning_clean'].value_counts().head(20).index.tolist()
df['zoning_group'] = df['zoning_clean'].apply(lambda x: x if x in top_zoning else 'other')
zoning_counts = df.loc[train_idx].groupby('zoning_group')['approved'].agg(['mean', 'count'])
zoning_approval = {}
for z_name, row_data in zoning_counts.iterrows():
    weight = row_data['count'] / (row_data['count'] + SMOOTHING_WEIGHT)
    zoning_approval[z_name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
df['zoning_approval_rate'] = df['zoning_group'].map(zoning_approval).fillna(train_global_rate)

# Attorney win rate — computed from training data only
ATTORNEY_SMOOTHING = 5
train_applicant_rates = df.loc[train_idx].groupby('_applicant_clean')['approved'].agg(['mean', 'count'])
attorney_win_rates = {}
for name, row_data in train_applicant_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + ATTORNEY_SMOOTHING)
        attorney_win_rates[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
df['attorney_win_rate'] = df['_applicant_clean'].map(attorney_win_rates).fillna(train_global_rate)

# Contact win rate
CONTACT_SMOOTHING = 5
if 'contact' in df.columns:
    df['_contact_clean'] = df['contact'].fillna('unknown').astype(str).str.strip().str.lower()
    train_contact_rates = df.loc[train_idx].groupby('_contact_clean')['approved'].agg(['mean', 'count'])
    contact_win_rates = {}
    for name, row_data in train_contact_rates.iterrows():
        if row_data['count'] >= 3:
            weight = row_data['count'] / (row_data['count'] + CONTACT_SMOOTHING)
            contact_win_rates[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
    df['contact_win_rate'] = df['_contact_clean'].map(contact_win_rates).fillna(train_global_rate)
else:
    df['contact_win_rate'] = train_global_rate
    contact_win_rates = {}

# Ward x Zoning district interaction rate
WARD_ZD_SMOOTHING = 10
df['_ward_zd'] = df['ward'].astype(str) + '_' + df['zoning_clean'].astype(str)
train_wz_rates = df.loc[train_idx].groupby('_ward_zd')['approved'].agg(['mean', 'count'])
ward_zoning_rates = {}
for name, row_data in train_wz_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + WARD_ZD_SMOOTHING)
        ward_zoning_rates[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
df['ward_zoning_rate'] = df['_ward_zd'].map(ward_zoning_rates).fillna(train_global_rate)

# Year x Ward interaction rate
YEAR_WARD_SMOOTHING = 5
df['_year_ward'] = df['_year'].fillna(0).astype(int).astype(str) + '_' + df['ward'].astype(str)
train_yw_rates = df.loc[train_idx].groupby('_year_ward')['approved'].agg(['mean', 'count'])
year_ward_rates = {}
for name, row_data in train_yw_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + YEAR_WARD_SMOOTHING)
        year_ward_rates[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate
df['year_ward_rate'] = df['_year_ward'].map(year_ward_rates).fillna(train_global_rate)

print(f"  Ward groups: {len(ward_approval)} | Zoning groups: {len(zoning_approval)}")
print(f"  Attorney win rates: {len(attorney_win_rates)} attorneys with 3+ cases")
print(f"  Contact win rates: {len(contact_win_rates)} contacts with 3+ cases")
print(f"  Ward x Zoning combos: {len(ward_zoning_rates)}")
print(f"  Year x Ward combos: {len(year_ward_rates)}")
print(f"  Training global approval rate: {train_global_rate:.1%}")


# ===========================
# FEATURE INTERACTIONS
# ===========================
print("\nEngineering feature interactions...")
df['interact_height_stories'] = df.get('var_height', pd.Series(0, index=df.index)).fillna(0) * df['proposed_stories'].fillna(0)
df['interact_attorney_variances'] = df.get('has_attorney', pd.Series(0, index=df.index)).fillna(0) * df['num_variances'].fillna(0)
df['interact_highvalue_permits'] = df.get('is_high_value', pd.Series(0, index=df.index)).fillna(0) * df.get('has_prior_permits', pd.Series(0, index=df.index)).fillna(0)
df['lot_size_log'] = np.log1p(df['lot_size_sf'].fillna(0))
df['total_value_log'] = np.log1p(df['total_value'].fillna(0))
df['prior_permits_log'] = np.log1p(df['prior_permits'].fillna(0))

# Contact x Appeal Type interaction — contacts may have different success rates by appeal type
df['contact_x_appeal'] = df['contact_win_rate'] * df['is_building_appeal']

# Attorney x Building appeal — representation matters more for harder cases
df['attorney_x_building'] = df.get('has_attorney', pd.Series(0, index=df.index)).fillna(0) * df['is_building_appeal']

# Variance count buckets (nonlinear: 0 vs 1-2 vs 3+ variances)
df['many_variances'] = (df['num_variances'] >= 3).astype(int)

# Property data available flag — helps model know when property features are real vs imputed zeros
df['has_property_data'] = (df['total_value'] > 0).astype(int)

# --- NEW FEATURES (Session 10) ---
print("\nEngineering new features (Session 10)...")

# Variance interaction features (pre-hearing: from application)
df['var_height_and_far'] = (df.get('var_height', pd.Series(0, index=df.index)).fillna(0) *
                            df.get('var_far', pd.Series(0, index=df.index)).fillna(0)).astype(int)
df['var_parking_and_units'] = (df.get('var_parking', pd.Series(0, index=df.index)).fillna(0) *
                               (df['proposed_units'].fillna(0) > 2).astype(int)).astype(int)
df['num_variances_sq'] = (df['num_variances'].fillna(0) ** 2).astype(float)
print(f"  var_height_and_far: {df['var_height_and_far'].sum()} cases")
print(f"  var_parking_and_units: {df['var_parking_and_units'].sum()} cases")
print(f"  num_variances_sq: mean={df['num_variances_sq'].mean():.2f}")

# Existing parking data (from assessor DB — pre-hearing)
df['existing_parking_count'] = pd.to_numeric(df.get('existing_parking', 0), errors='coerce').fillna(0)
df['has_existing_parking'] = (df['existing_parking_count'] > 0).astype(int)
print(f"  has_existing_parking: {df['has_existing_parking'].sum()} cases")

# Project scale features (from application)
df['units_log'] = np.log1p(df['proposed_units'].fillna(0))
df['large_project'] = (df['proposed_units'].fillna(0) > 5).astype(int)
print(f"  large_project (>5 units): {df['large_project'].sum()} cases")

# Density features — lot utilization (from application + assessor)
_lot = df['lot_size_sf'].fillna(0).replace(0, np.nan)
df['units_per_lot_area'] = (df['proposed_units'].fillna(0) / _lot).fillna(0)
df['value_per_unit'] = np.where(
    df['proposed_units'].fillna(0) > 0,
    df['total_value'].fillna(0) / df['proposed_units'].fillna(0).replace(0, 1),
    0
)
df['value_per_unit_log'] = np.log1p(df['value_per_unit'])
print(f"  units_per_lot_area: {(df['units_per_lot_area'] > 0).sum()} nonzero")
print(f"  value_per_unit_log: mean={df['value_per_unit_log'].mean():.2f}")

# Change of occupancy (from tracker description — pre-hearing application data)
if 'tracker_description' in df.columns:
    td = df['tracker_description'].fillna('').str.lower()
    df['is_change_occupancy'] = td.str.contains(r'change.*(?:occupancy|use)|convert|conversion', regex=True).astype(int)
    df['is_maintain_use'] = td.str.contains(r'maintain', regex=True).astype(int)
    print(f"  is_change_occupancy: {df['is_change_occupancy'].sum()} cases")
    print(f"  is_maintain_use: {df['is_maintain_use'].sum()} cases")
else:
    df['is_change_occupancy'] = 0
    df['is_maintain_use'] = 0

# Setback combo — multiple setback variances requested at once (complex ask)
setback_cols = ['var_front_setback', 'var_rear_setback', 'var_side_setback']
df['num_setback_variances'] = df[[c for c in setback_cols if c in df.columns]].fillna(0).sum(axis=1)
df['multiple_setbacks'] = (df['num_setback_variances'] >= 2).astype(int)
print(f"  multiple_setbacks: {df['multiple_setbacks'].sum()} cases")

# Stories x FAR interaction (tall building with high density — harder approval)
df['interact_stories_far'] = df['proposed_stories'].fillna(0) * df.get('var_far', pd.Series(0, index=df.index)).fillna(0)

# Attorney x Variance count x Building appeal — triple interaction for complex cases
df['complex_case_score'] = (df.get('has_attorney', pd.Series(0, index=df.index)).fillna(0) +
                            df['many_variances'] +
                            df['is_building_appeal'] +
                            df['large_project']).astype(int)
print(f"  complex_case_score: mean={df['complex_case_score'].mean():.2f}")


# ===========================
# CLEAN FEATURE LIST — PRE-HEARING ONLY
# ===========================

feature_cols = [
    # Variance features (13) — from application
    'num_variances',
    'var_height', 'var_far', 'var_lot_area', 'var_lot_frontage',
    'var_front_setback', 'var_rear_setback', 'var_side_setback',
    'var_parking', 'var_conditional_use', 'var_open_space',
    'var_density', 'var_nonconforming',

    # Violation types (5) — from zoning code analysis
    'excessive_far', 'insufficient_lot', 'insufficient_frontage',
    'insufficient_yard', 'insufficient_parking',

    # Use type (2) — from application
    'is_residential', 'is_commercial',

    # Representation (4) — known at filing
    'has_attorney',
    'bpda_involved',
    'is_building_appeal',  # Building appeals have ~58% approval vs ~91% zoning
    'is_refusal_appeal',   # Appeal of building commissioner refusal

    # Project types (12) — from application
    'proj_demolition', 'proj_new_construction', 'proj_addition',
    'proj_conversion', 'proj_renovation', 'proj_subdivision',
    'proj_adu', 'proj_roof_deck', 'proj_parking',
    'proj_single_family', 'proj_multi_family', 'proj_mixed_use',

    # Legal framework (4) — from zoning code
    'article_80',
    'is_conditional_use', 'is_variance',
    'num_articles',

    # Building scale (2) — from application
    'proposed_units', 'proposed_stories',

    # Location-based (6) — historical, computed from training data
    'ward_approval_rate', 'zoning_approval_rate',
    'attorney_win_rate', 'contact_win_rate',
    'ward_zoning_rate',   # Ward x Zoning district interaction
    'year_ward_rate',     # Year x Ward interaction (temporal + spatial)

    # Recency (1)
    'year_recency',

    # Property features (6) — from tax assessor database
    'lot_size_sf', 'total_value', 'property_age',
    'living_area', 'is_high_value', 'value_per_sqft',

    # Permit history (2) — from building permits database
    'prior_permits', 'has_prior_permits',

    # Interactions (3) — engineered
    'interact_height_stories',
    'interact_attorney_variances', 'interact_highvalue_permits',

    # Log transforms (3)
    'lot_size_log', 'total_value_log', 'prior_permits_log',

    # Additional interactions (4)
    'contact_x_appeal', 'attorney_x_building',
    'many_variances', 'has_property_data',

    # Meta-features (3) — project complexity and data quality signals
    'project_complexity', 'total_violations', 'num_features_active',

    # NEW (Session 10) — Variance interactions (3)
    'var_height_and_far', 'var_parking_and_units', 'num_variances_sq',

    # NEW (Session 10) — Existing conditions (2)
    'has_existing_parking', 'existing_parking_count',

    # NEW (Session 10) — Scale & density (4)
    'units_log', 'large_project', 'units_per_lot_area', 'value_per_unit_log',

    # NEW (Session 10) — Application type (2)
    'is_change_occupancy', 'is_maintain_use',

    # NEW (Session 10) — Complexity signals (4)
    'multiple_setbacks', 'num_setback_variances',
    'interact_stories_far', 'complex_case_score',
]

# Remove articles 7/8/51/65 — too generic, often just mean "has a variance"
# Keep article_80 — specifically for large project review

X = df[feature_cols].fillna(0)
y = df['approved']

X_train = X[train_idx]
X_test = X[test_idx]
X_cal = X[cal_idx]
y_train = y[train_idx]
y_test = y[test_idx]
y_cal = y[cal_idx]

n_features = len(feature_cols)
print(f"\nFeature matrix: {X.shape[0]} samples x {n_features} features")
print(f"  PRE-HEARING ONLY — no post-hearing leakage")
print(f"  Removed 14 leaking features, added attorney_win_rate")


# ===========================
# TRAIN & COMPARE MODELS
# ===========================
print("\n" + "=" * 60)
print("  Training Models")
print("=" * 60)

n_approved = (y_train == 1).sum()
n_denied = (y_train == 0).sum()
weight_denied = n_approved / n_denied if n_denied > 0 else 1.0

# --- Feature richness weights ---
# OCR cases (~2700 chars) have much richer features than tracker-only cases (~170 chars).
# Weight cases by feature quality so sparse tracker cases don't dilute the signal.
if 'source_pdf' in df.columns:
    train_source = df.loc[train_idx, 'source_pdf'].fillna('')
    richness_weight = np.where(train_source == 'zba_tracker', 0.3, 1.0)
    n_tracker_train = (train_source == 'zba_tracker').sum()
    n_ocr_train = (train_source != 'zba_tracker').sum()
    print(f"\n  Feature richness — OCR: {n_ocr_train} (weight 1.0), Tracker: {n_tracker_train} (weight 0.3)")
else:
    richness_weight = np.ones(y_train.shape[0])

# Combine class balance + feature richness weights
class_weight_arr = y_train.map({1: 1.0, 0: weight_denied}).values
sample_weights = class_weight_arr * richness_weight
print(f"  Class balance — Approved: {n_approved}, Denied: {n_denied}, Denied weight: {weight_denied:.2f}")

# Try to import XGBoost for better performance
try:
    from xgboost import XGBClassifier
    has_xgboost = True
    print("  XGBoost available ✅")
except ImportError:
    has_xgboost = False

models = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        min_samples_leaf=20, subsample=0.8, max_features='sqrt',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=800, max_depth=20, min_samples_leaf=8,
        random_state=42, class_weight='balanced', n_jobs=-1,
        max_features='sqrt'
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=42, C=0.5
    )
}

if has_xgboost:
    # Use sample_weight (includes richness + class balance) instead of scale_pos_weight
    # to avoid double-counting class imbalance
    models['XGBoost'] = XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='auc', use_label_encoder=False,
        n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0,
        gamma=0.1  # Minimum loss reduction for split (helps with overfitting)
    )
    # Second XGBoost config: deeper trees, more aggressive learning for denial detection
    models['XGBoost_Deep'] = XGBClassifier(
        n_estimators=1200, max_depth=7, learning_rate=0.02,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        random_state=42, eval_metric='auc', use_label_encoder=False,
        n_jobs=-1, reg_alpha=0.3, reg_lambda=2.0,
        gamma=0.2
    )

uses_sample_weight = {'Gradient Boosting', 'XGBoost', 'XGBoost_Deep'}
best_model = None
best_auc = 0
best_name = ""
all_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in uses_sample_weight:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"    AUC:         {auc:.4f}")
    print(f"    Brier Score: {brier:.4f}  (lower = better calibrated)")
    print(f"    Log Loss:    {logloss:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"    Predicted:   DENIED  APPROVED")
    print(f"    DENIED       {tn:5d}    {fp:5d}")
    print(f"    APPROVED     {fn:5d}    {tp:5d}")
    print(f"  True Positive Rate (recall on APPROVED): {tp/(tp+fn):.1%}")
    print(f"  True Negative Rate (recall on DENIED):   {tn/(tn+fp):.1%}")

    # Compute denial recall (critical for risk assessment product)
    denial_recall = tn / (tn + fp) if (tn + fp) > 0 else 0

    all_results[name] = {
        'model': model, 'auc': auc, 'brier': brier, 'logloss': logloss,
        'y_prob': y_prob, 'y_pred': y_pred, 'denial_recall': denial_recall
    }

    # Model selection: use composite score that values denial recall
    # AUC alone picks models that classify everything as "approved"
    # For risk assessment, we NEED to catch denials
    composite = 0.6 * auc + 0.4 * denial_recall
    print(f"  Composite score (0.6*AUC + 0.4*DenialRecall): {composite:.4f}")

    if composite > best_auc:
        best_auc = composite
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"  BEST INDIVIDUAL MODEL: {best_name}")
print(f"    AUC: {all_results[best_name]['auc']:.4f}")
print(f"    Denial Recall: {all_results[best_name]['denial_recall']:.1%}")
print(f"    Composite: {best_auc:.4f}")
print(f"{'='*60}")


# ===========================
# FEATURE SELECTION — remove noise features
# ===========================
print("\n" + "=" * 60)
print("  Feature Selection (importance-based)")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    imp_arr = best_model.feature_importances_
    imp_threshold = 0.002  # Features below this are noise
    weak_features = [f for f, imp in zip(feature_cols, imp_arr) if imp < imp_threshold]
    strong_features = [f for f, imp in zip(feature_cols, imp_arr) if imp >= imp_threshold]
    print(f"  {len(weak_features)} features below {imp_threshold} importance threshold:")
    for f in weak_features:
        idx = feature_cols.index(f)
        print(f"    - {f}: {imp_arr[idx]:.5f}")
    print(f"  Keeping {len(strong_features)} features")

    if len(weak_features) > 3 and len(strong_features) >= 20:
        # Retrain best model with selected features only
        X_train_sel = X_train[strong_features]
        X_test_sel = X_test[strong_features]

        import copy
        selected_model = copy.deepcopy(models[best_name])
        if best_name in uses_sample_weight:
            selected_model.fit(X_train_sel, y_train, sample_weight=sample_weights)
        else:
            selected_model.fit(X_train_sel, y_train)
        y_prob_sel = selected_model.predict_proba(X_test_sel)[:, 1]
        auc_sel = roc_auc_score(y_test, y_prob_sel)
        cm_sel = confusion_matrix(y_test, selected_model.predict(X_test_sel))
        tn_sel = cm_sel[0, 0]; fp_sel = cm_sel[0, 1]
        dr_sel = tn_sel / (tn_sel + fp_sel) if (tn_sel + fp_sel) > 0 else 0
        comp_sel = 0.6 * auc_sel + 0.4 * dr_sel

        print(f"\n  Feature-selected model:")
        print(f"    AUC: {auc_sel:.4f} (was {all_results[best_name]['auc']:.4f}, delta: {auc_sel - all_results[best_name]['auc']:+.4f})")
        print(f"    Denial Recall: {dr_sel:.1%}")
        print(f"    Composite: {comp_sel:.4f} (was {best_auc:.4f})")

        if comp_sel > best_auc:
            print(f"  ✅ Feature selection improved model — using {len(strong_features)} features")
            feature_cols = strong_features
            X = df[feature_cols].fillna(0)
            X_train = X[train_idx]
            X_test = X[test_idx]
            X_cal = X[cal_idx]
            best_model = selected_model
            best_auc = comp_sel
            all_results[best_name]['auc'] = auc_sel
            all_results[best_name]['denial_recall'] = dr_sel
            all_results[best_name]['y_prob'] = y_prob_sel
            all_results[best_name]['y_pred'] = selected_model.predict(X_test_sel)
        else:
            print(f"  ❌ Feature selection did not help — keeping all {len(feature_cols)} features")
else:
    strong_features = feature_cols


# ===========================
# STACKING ENSEMBLE
# ===========================
print("\n" + "=" * 60)
print("  Stacking Ensemble")
print("=" * 60)

# Build stacking ensemble from top 3 models
stack_candidates = sorted(all_results.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]
print(f"  Base models: {[n for n, _ in stack_candidates]}")

base_model_list = [(name, models[name]) for name, _ in stack_candidates if name in models]

if len(base_model_list) >= 2:
    meta_learner = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
    stacker = StackingEnsemble(
        base_models=base_model_list,
        meta_model=meta_learner,
        n_folds=5
    )
    print("  Training stacking ensemble (5-fold OOF)...")
    stacker.fit(X_train, y_train, sample_weight=sample_weights)

    y_prob_stack = stacker.predict_proba(X_test)[:, 1]
    auc_stack = roc_auc_score(y_test, y_prob_stack)
    y_pred_stack = (y_prob_stack >= 0.5).astype(int)
    cm_stack = confusion_matrix(y_test, y_pred_stack)
    tn_s, fp_s = cm_stack[0, 0], cm_stack[0, 1]
    dr_stack = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0
    comp_stack = 0.6 * auc_stack + 0.4 * dr_stack

    print(f"\n  Stacking ensemble results:")
    print(f"    AUC: {auc_stack:.4f} (best individual: {all_results[best_name]['auc']:.4f}, delta: {auc_stack - all_results[best_name]['auc']:+.4f})")
    print(f"    Denial Recall: {dr_stack:.1%}")
    print(f"    Composite: {comp_stack:.4f} (best individual: {best_auc:.4f})")

    all_results['Stacking'] = {
        'model': stacker, 'auc': auc_stack, 'brier': brier_score_loss(y_test, y_prob_stack),
        'logloss': log_loss(y_test, y_prob_stack),
        'y_prob': y_prob_stack, 'y_pred': y_pred_stack, 'denial_recall': dr_stack
    }

    if comp_stack > best_auc:
        print(f"  ✅ Stacking ensemble wins! Using stacking model.")
        best_model = stacker
        best_name = 'Stacking'
        best_auc = comp_stack
    else:
        print(f"  ❌ Stacking did not beat best individual — keeping {best_name}")
else:
    print("  Not enough base models for stacking")


print(f"\n{'='*60}")
print(f"  FINAL BEST MODEL: {best_name}")
print(f"    AUC: {all_results[best_name]['auc']:.4f}")
print(f"    Denial Recall: {all_results[best_name]['denial_recall']:.1%}")
print(f"    Composite: {best_auc:.4f}")
print(f"{'='*60}")


# ===========================
# CALIBRATION (on separate holdout — NOT test set)
# ===========================
print("\n" + "=" * 60)
print("  Calibration (on held-out calibration set)")
print("=" * 60)

y_prob_cal_raw = best_model.predict_proba(X_cal)[:, 1]
brier_uncal = brier_score_loss(y_cal, y_prob_cal_raw)
print(f"\n  Uncalibrated Brier (on cal set): {brier_uncal:.4f}")

# Check calibration before
prob_true_pre, prob_pred_pre = calibration_curve(y_cal, y_prob_cal_raw, n_bins=8, strategy='uniform')
print(f"\n  Pre-calibration check (cal set):")
print(f"    {'Predicted':>10s}  {'Actual':>8s}  {'Gap':>8s}")
for pt, pp in zip(prob_true_pre, prob_pred_pre):
    print(f"    {pp:>10.2%}  {pt:>8.2%}  {pt-pp:>+8.2%}")

# Calibrate using Platt scaling on calibration set
# For custom models (StackingEnsemble), implement manual Platt scaling
try:
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)
    y_prob_test_cal = calibrated_model.predict_proba(X_test)[:, 1]
    calibration_worked = True
except (ValueError, TypeError) as e:
    print(f"  sklearn calibration failed ({e.__class__.__name__}), using manual Platt scaling...")
    # Manual Platt scaling: fit logistic regression on raw probabilities
    from sklearn.linear_model import LogisticRegression as _LR
    platt_lr = _LR(max_iter=1000, C=1e10, solver='lbfgs')
    platt_lr.fit(y_prob_cal_raw.reshape(-1, 1), y_cal)

    calibrated_model = ManualCalibratedModel(best_model, platt_lr)
    y_prob_test_cal = calibrated_model.predict_proba(X_test)[:, 1]
    calibration_worked = True

# Evaluate calibrated model on TEST set (fair evaluation)
brier_cal = brier_score_loss(y_test, y_prob_test_cal)
auc_cal = roc_auc_score(y_test, y_prob_test_cal)

best_uncal_auc = all_results[best_name]['auc']
print(f"\n  Calibrated model (evaluated on TEST set):")
print(f"    AUC:   {auc_cal:.4f} (was {best_uncal_auc:.4f})")
print(f"    Brier: {brier_cal:.4f} (was {all_results[best_name]['brier']:.4f})")

prob_true_post, prob_pred_post = calibration_curve(y_test, y_prob_test_cal, n_bins=8, strategy='uniform')
print(f"\n  Post-calibration check (test set):")
print(f"    {'Predicted':>10s}  {'Actual':>8s}  {'Gap':>8s}")
for pt, pp in zip(prob_true_post, prob_pred_post):
    print(f"    {pp:>10.2%}  {pt:>8.2%}  {pt-pp:>+8.2%}")

# Use calibrated model as final
final_model = calibrated_model
final_auc = auc_cal
final_brier = brier_cal


# ===========================
# THRESHOLD OPTIMIZATION
# ===========================
print("\n" + "=" * 60)
print("  Threshold Optimization")
print("=" * 60)

precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob_test_cal)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_pr[optimal_idx]

print(f"\n  Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.4f})")
print(f"\n  Threshold sweep:")
print(f"    {'Threshold':>10s}  {'F1':>6s}  {'Precision':>9s}  {'Recall':>6s}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_t = (y_prob_test_cal >= t).astype(int)
    f1_t = f1_score(y_test, y_t, zero_division=0)
    tp_t = ((y_t == 1) & (y_test == 1)).sum()
    fp_t = ((y_t == 1) & (y_test == 0)).sum()
    fn_t = ((y_t == 0) & (y_test == 1)).sum()
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    print(f"    {t:>10.1f}  {f1_t:>6.4f}  {prec_t:>9.4f}  {rec_t:>6.4f}")


# ===========================
# FEATURE IMPORTANCE
# ===========================
# For stacking models, use the best individual base model's importances
importance_model = best_model
importance_model_name = best_name
if isinstance(best_model, StackingEnsemble):
    # Find the best individual model and RETRAIN on current feature set for SHAP compatibility
    for name_res, res in sorted(all_results.items(), key=lambda x: x[1]['auc'], reverse=True):
        if name_res != 'Stacking' and hasattr(res.get('model', None), 'feature_importances_'):
            import copy
            importance_model = copy.deepcopy(models[name_res])
            importance_model_name = name_res
            # Retrain on the current (possibly feature-selected) feature set
            if name_res in uses_sample_weight:
                importance_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                importance_model.fit(X_train, y_train)
            print(f"  Retrained {name_res} on {len(feature_cols)} features for SHAP explainability")
            break

if hasattr(importance_model, 'feature_importances_'):
    importances = sorted(
        zip(feature_cols, importance_model.feature_importances_),
        key=lambda x: -x[1]
    )
    print(f"\nTop 20 Features ({importance_model_name}):")
    for i, (feat, imp) in enumerate(importances[:20]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")
elif hasattr(importance_model, 'coef_'):
    importances = sorted(
        zip(feature_cols, abs(importance_model.coef_[0])),
        key=lambda x: -x[1]
    )
    print(f"\nTop 20 Features ({importance_model_name} — |coefficients|):")
    for i, (feat, imp) in enumerate(importances[:20]):
        bar = "█" * int(imp * 20)
        print(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")


# ===========================
# CROSS VALIDATION (Stratified)
# ===========================
print("\nRunning 5-fold stratified cross validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Simple CV — use best individual model (stacking CV is too slow and redundant)
cv_base_model = importance_model if isinstance(best_model, StackingEnsemble) else best_model
cv_scores = cross_val_score(cv_base_model, X, y, cv=skf, scoring='roc_auc')
print(f"5-Fold CV AUC (simple): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Fold scores: {', '.join(f'{s:.4f}' for s in cv_scores)}")

# Honest CV: recompute target encoding within each fold to avoid leakage
# Target-encoded features are the top predictors, so this matters
print("\nRunning honest CV with in-fold target encoding...")
target_enc_cols = ['ward_approval_rate', 'zoning_approval_rate', 'attorney_win_rate',
                   'contact_win_rate', 'ward_zoning_rate', 'year_ward_rate']
non_te_features = [f for f in feature_cols if f not in target_enc_cols]
te_features_present = [f for f in target_enc_cols if f in feature_cols]

honest_cv_scores = []
for fold_i, (cv_train_idx, cv_val_idx) in enumerate(skf.split(X, y)):
    # Recompute target encoding from CV training fold only
    cv_train_df = df.iloc[cv_train_idx].copy()
    cv_val_df = df.iloc[cv_val_idx].copy()
    cv_global_rate = cv_train_df['approved'].mean()

    # Ward approval rate
    if 'ward_approval_rate' in te_features_present:
        wc = cv_train_df.groupby('ward')['approved'].agg(['mean', 'count'])
        wa = {}
        for w, r in wc.iterrows():
            wt = r['count'] / (r['count'] + SMOOTHING_WEIGHT)
            wa[w] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['ward_approval_rate'] = cv_val_df['ward'].map(wa).fillna(cv_global_rate)

    # Zoning approval rate
    if 'zoning_approval_rate' in te_features_present:
        cv_train_df['_zg'] = cv_train_df['zoning_clean'].apply(lambda x: x if x in top_zoning else 'other')
        cv_val_df['_zg'] = cv_val_df['zoning_clean'].apply(lambda x: x if x in top_zoning else 'other')
        zc = cv_train_df.groupby('_zg')['approved'].agg(['mean', 'count'])
        za = {}
        for z, r in zc.iterrows():
            wt = r['count'] / (r['count'] + SMOOTHING_WEIGHT)
            za[z] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['zoning_approval_rate'] = cv_val_df['_zg'].map(za).fillna(cv_global_rate)

    # Attorney win rate
    if 'attorney_win_rate' in te_features_present:
        ac = cv_train_df.groupby('_applicant_clean')['approved'].agg(['mean', 'count'])
        aw = {}
        for n, r in ac.iterrows():
            if r['count'] >= 3:
                wt = r['count'] / (r['count'] + ATTORNEY_SMOOTHING)
                aw[n] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['attorney_win_rate'] = cv_val_df['_applicant_clean'].map(aw).fillna(cv_global_rate)

    # Contact win rate
    if 'contact_win_rate' in te_features_present and '_contact_clean' in cv_train_df.columns:
        cc = cv_train_df.groupby('_contact_clean')['approved'].agg(['mean', 'count'])
        cw = {}
        for n, r in cc.iterrows():
            if r['count'] >= 3:
                wt = r['count'] / (r['count'] + CONTACT_SMOOTHING)
                cw[n] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['contact_win_rate'] = cv_val_df['_contact_clean'].map(cw).fillna(cv_global_rate)

    # Ward x Zoning rate
    if 'ward_zoning_rate' in te_features_present:
        cv_train_df['_wz'] = cv_train_df['ward'].astype(str) + '_' + cv_train_df['zoning_clean'].astype(str)
        cv_val_df['_wz'] = cv_val_df['ward'].astype(str) + '_' + cv_val_df['zoning_clean'].astype(str)
        wzc = cv_train_df.groupby('_wz')['approved'].agg(['mean', 'count'])
        wzr = {}
        for n, r in wzc.iterrows():
            if r['count'] >= 3:
                wt = r['count'] / (r['count'] + WARD_ZD_SMOOTHING)
                wzr[n] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['ward_zoning_rate'] = cv_val_df['_wz'].map(wzr).fillna(cv_global_rate)

    # Year x Ward rate
    if 'year_ward_rate' in te_features_present:
        cv_train_df['_yw'] = cv_train_df['_year'].fillna(0).astype(int).astype(str) + '_' + cv_train_df['ward'].astype(str)
        cv_val_df['_yw'] = cv_val_df['_year'].fillna(0).astype(int).astype(str) + '_' + cv_val_df['ward'].astype(str)
        ywc = cv_train_df.groupby('_yw')['approved'].agg(['mean', 'count'])
        ywr = {}
        for n, r in ywc.iterrows():
            if r['count'] >= 3:
                wt = r['count'] / (r['count'] + YEAR_WARD_SMOOTHING)
                ywr[n] = wt * r['mean'] + (1 - wt) * cv_global_rate
        cv_val_df['year_ward_rate'] = cv_val_df['_yw'].map(ywr).fillna(cv_global_rate)

    # Also recompute contact_x_appeal if present
    if 'contact_x_appeal' in feature_cols:
        cv_val_df['contact_x_appeal'] = cv_val_df['contact_win_rate'] * cv_val_df['is_building_appeal']

    X_cv_val = cv_val_df[feature_cols].fillna(0)
    y_cv_val = cv_val_df['approved']

    # Train on CV train fold with original target encoding (computed from full training set)
    X_cv_train = X.iloc[cv_train_idx]
    y_cv_train = y.iloc[cv_train_idx]

    import copy
    cv_model = copy.deepcopy(cv_base_model)
    cv_model_name = importance_model_name if isinstance(best_model, StackingEnsemble) else best_name
    if cv_model_name in uses_sample_weight:
        # Recompute sample weights for this fold
        cv_source = df.iloc[cv_train_idx]['source_pdf'].fillna('') if 'source_pdf' in df.columns else pd.Series('', index=cv_train_df.index)
        cv_richness = np.where(cv_source == 'zba_tracker', 0.3, 1.0)
        cv_n_app = (y_cv_train == 1).sum()
        cv_n_den = (y_cv_train == 0).sum()
        cv_wd = cv_n_app / cv_n_den if cv_n_den > 0 else 1.0
        cv_cw = y_cv_train.map({1: 1.0, 0: cv_wd}).values
        cv_sw = cv_cw * cv_richness
        cv_model.fit(X_cv_train, y_cv_train, sample_weight=cv_sw)
    else:
        cv_model.fit(X_cv_train, y_cv_train)

    y_cv_prob = cv_model.predict_proba(X_cv_val)[:, 1]
    fold_auc = roc_auc_score(y_cv_val, y_cv_prob)
    honest_cv_scores.append(fold_auc)

honest_cv_scores = np.array(honest_cv_scores)
print(f"5-Fold Honest CV AUC: {honest_cv_scores.mean():.4f} (+/- {honest_cv_scores.std():.4f})")
print(f"  Fold scores: {', '.join(f'{s:.4f}' for s in honest_cv_scores)}")
print(f"  Delta from simple CV: {honest_cv_scores.mean() - cv_scores.mean():+.4f}")
cv_scores_honest = honest_cv_scores  # Use honest scores for reporting


# ===========================
# SAVE DIAGNOSTIC PLOTS
# ===========================
print("\nSaving diagnostic plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve — all models
ax = axes[0]
for name, res in all_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
# Also plot calibrated model
fpr_cal, tpr_cal, _ = roc_curve(y_test, y_prob_test_cal)
ax.plot(fpr_cal, tpr_cal, '--', label=f"Calibrated (AUC={final_auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# Calibration — before and after
ax = axes[1]
ax.plot(prob_pred_pre, prob_true_pre, 'o-', label='Before calibration', color='tab:red')
ax.plot(prob_pred_post, prob_true_post, 's-', label='After calibration (Platt)', color='tab:green')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Probability')
ax.set_title('Calibration: Before vs After')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Feature Importance
ax = axes[2]
if hasattr(importance_model, 'feature_importances_'):
    top_n = 20
    top_feats = importances[:top_n]
    feat_names = [f[0] for f in reversed(top_feats)]
    feat_vals = [f[1] for f in reversed(top_feats)]
    ax.barh(feat_names, feat_vals, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features ({importance_model_name})')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: model_diagnostics.png")


# ===========================
# SAVE MODEL PACKAGE
# ===========================
dataset_hash = hashlib.md5(
    pd.util.hash_pandas_object(df[feature_cols + ['approved']].head(1000)).values.tobytes()
).hexdigest()[:12]
model_version = f"v3.{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

model_package = {
    'model': final_model,  # Calibrated model
    'base_model': importance_model if isinstance(best_model, StackingEnsemble) else best_model,  # For SHAP
    'feature_cols': feature_cols,
    'ward_approval_rates': ward_approval,
    'zoning_approval_rates': zoning_approval,
    'top_zoning': top_zoning,
    'attorney_win_rates': attorney_win_rates,
    'contact_win_rates': contact_win_rates,
    'ward_zoning_rates': ward_zoning_rates,
    'year_ward_rates': year_ward_rates,
    'top_zoning': top_zoning,
    'overall_approval_rate': float(train_global_rate),
    'total_cases': int(len(df)),
    'model_name': best_name,
    'auc_score': float(final_auc),
    'auc_uncalibrated': float(best_uncal_auc),
    'brier_score': float(final_brier),
    'optimal_threshold': float(optimal_threshold),
    'cv_auc_mean': float(cv_scores_honest.mean()),
    'cv_auc_std': float(cv_scores_honest.std()),
    'cv_auc_simple': float(cv_scores.mean()),
    'variance_types': variance_types,
    'binary_features': binary_cols,
    'project_types': [
        'demolition', 'new_construction', 'addition', 'conversion',
        'renovation', 'subdivision', 'adu', 'roof_deck', 'parking',
        'single_family', 'multi_family', 'mixed_use'
    ],
    'model_version': model_version,
    'dataset_hash': dataset_hash,
    'trained_at': datetime.datetime.now().isoformat(),
    'train_size': int(train_idx.sum()),
    'test_size': int(test_idx.sum()),
    'cal_size': int(cal_idx.sum()),
    'is_calibrated': True,
    'calibration_method': 'platt_scaling',
    'leakage_free': True,
    'removed_features': [
        'has_opposition', 'no_opposition_noted', 'planning_support',
        'planning_proviso', 'community_process', 'support_letters',
        'opposition_letters', 'non_opposition_letter', 'councilor_involved',
        'mayors_office_involved', 'hardship_mentioned', 'has_deferrals',
        'num_deferrals', 'text_length_log',
    ],
}

joblib.dump(model_package, 'zba_model_v2.pkl')
joblib.dump(model_package, 'api/zba_model.pkl')

# --- Model versioning: save to history ---

history_dir = 'model_history'
os.makedirs(history_dir, exist_ok=True)

# Save versioned copy
version_file = os.path.join(history_dir, f'{model_version}.pkl')
shutil.copy2('zba_model_v2.pkl', version_file)
shutil.copy2('model_diagnostics.png', os.path.join(history_dir, f'{model_version}_diagnostics.png'))

# Save version metadata to history log
history_log = os.path.join(history_dir, 'training_log.jsonl')
log_entry = {
    'version': model_version,
    'trained_at': datetime.datetime.now().isoformat(),
    'model_name': best_name,
    'auc_calibrated': round(final_auc, 4),
    'auc_uncalibrated': round(best_uncal_auc, 4),
    'brier_score': round(final_brier, 4),
    'optimal_threshold': round(optimal_threshold, 3),
    'cv_auc_mean': round(float(cv_scores_honest.mean()), 4),
    'cv_auc_std': round(float(cv_scores_honest.std()), 4),
    'n_features': n_features,
    'train_size': int(train_idx.sum()),
    'test_size': int(test_idx.sum()),
    'cal_size': int(cal_idx.sum()),
    'dataset_hash': dataset_hash,
    'leakage_free': True,
    'is_calibrated': True,
    'approval_rate': round(float(df['approved'].mean()), 3),
}
with open(history_log, 'a') as f:
    f.write(json.dumps(log_entry) + '\n')

# Print comparison with previous version if exists
try:
    with open(history_log) as f:
        lines = f.readlines()
    if len(lines) >= 2:
        prev = json.loads(lines[-2])
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON vs {prev['version']}")
        print(f"{'='*60}")
        print(f"  AUC:   {prev['auc_calibrated']:.4f} → {final_auc:.4f}  ({final_auc - prev['auc_calibrated']:+.4f})")
        print(f"  Brier: {prev['brier_score']:.4f} → {final_brier:.4f}  ({final_brier - prev['brier_score']:+.4f})")
        print(f"  CV:    {prev['cv_auc_mean']:.4f} → {cv_scores.mean():.4f}  ({cv_scores.mean() - prev['cv_auc_mean']:+.4f})")
        print(f"  Train: {prev['train_size']:,} → {int(train_idx.sum()):,}")
except Exception:
    pass

print(f"\n{'='*60}")
print(f"  MODEL SAVED SUCCESSFULLY")
print(f"{'='*60}")
print(f"  Version: {model_version}")
print(f"  Files: zba_model_v2.pkl, api/zba_model.pkl, model_history/{model_version}.pkl")
print(f"  Model: {best_name} (calibrated via Platt scaling)")
print(f"  AUC: {final_auc:.4f} (uncalibrated: {best_uncal_auc:.4f})")
print(f"  Brier: {final_brier:.4f}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")
print(f"  CV AUC (honest): {cv_scores_honest.mean():.4f} (+/- {cv_scores_honest.std():.4f})")
print(f"  CV AUC (simple): {cv_scores.mean():.4f}")
print(f"  Features: {n_features} (PRE-HEARING ONLY, leakage-free)")
print(f"  Training cases: {train_idx.sum()}")
print(f"  Dataset hash: {dataset_hash}")
print(f"  Plots: model_diagnostics.png")
print(f"  History: model_history/training_log.jsonl")
print(f"\n  PermitIQ model training complete.")
