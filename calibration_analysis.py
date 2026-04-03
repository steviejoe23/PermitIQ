"""
PermitIQ Calibration Analysis
=============================
Answers: When we say "90% chance of approval", how often does the case actually get approved?

This is critical — developers make $30-100K financial decisions based on our probability outputs.
We need to verify whether those numbers are trustworthy.

Run:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 calibration_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve

# Import model classes so pickle can deserialize custom models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api', 'services'))
from model_classes import StackingEnsemble, ManualCalibratedModel

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. LOAD MODEL AND DATASET
# ============================================================
print("=" * 70)
print("  PermitIQ Calibration Analysis")
print("  Can developers trust our probability numbers?")
print("=" * 70)

model_path = os.path.join(BASE_DIR, 'api', 'zba_model.pkl')
print(f"\nLoading model from {model_path}...")
pkg = joblib.load(model_path)
model = pkg['model']
feature_cols = pkg['feature_cols']
print(f"  Model type: {type(model).__name__}")
print(f"  Features: {len(feature_cols)}")
if 'auc' in pkg:
    print(f"  Reported AUC: {pkg['auc']:.4f}")
if 'brier_score' in pkg:
    print(f"  Reported Brier: {pkg['brier_score']:.4f}")

# Load rate dictionaries from model package
ward_approval_rates = pkg.get('ward_approval_rates', {})
zoning_approval_rates = pkg.get('zoning_approval_rates', {})
contact_win_rates = pkg.get('contact_win_rates', {})
attorney_win_rates = {}  # Will compute from training data
ward_zoning_rates = pkg.get('ward_zoning_rates', {})
year_ward_rates = pkg.get('year_ward_rates', {})
train_global_rate = pkg.get('global_approval_rate', 0.85)

print(f"\nLoading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, 'zba_cases_cleaned.csv'), low_memory=False)
print(f"  Total rows: {len(df)}")

# Keep rows with decisions
df = df[df['decision_clean'].notna()].copy()
print(f"  Rows with decisions: {len(df)}")

# Dedup — same logic as train_model_v2.py
if 'case_number' in df.columns:
    before = len(df)
    df['_is_tracker'] = (df['source_pdf'] == 'zba_tracker').astype(int) if 'source_pdf' in df.columns else 0
    df['_text_len'] = df['raw_text'].fillna('').str.len() if 'raw_text' in df.columns else 0
    df = df.sort_values(['_is_tracker', '_text_len'], ascending=[True, False])
    df = df.drop_duplicates(subset='case_number', keep='first')
    df = df.drop(columns=['_is_tracker', '_text_len'], errors='ignore')
    print(f"  After dedup: {len(df)} (removed {before - len(df)})")

df['approved'] = (df['decision_clean'] == 'APPROVED').astype(int)
print(f"  Approval rate: {df['approved'].mean():.1%}")


# ============================================================
# 2. REPLICATE FEATURE ENGINEERING (exact same as train_model_v2.py)
# ============================================================
print("\nEngineering features (replicating training pipeline)...")

# Variance features
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

# Raw text features
if 'raw_text' in df.columns:
    rt = df['raw_text'].fillna('').str.lower()
    if 'is_residential' not in df.columns:
        df['is_residential'] = rt.str.contains(r'residential|dwelling|family|apartment|condo', regex=True).astype(int)
    if 'is_commercial' not in df.columns:
        df['is_commercial'] = rt.str.contains(r'commercial|retail|office|restaurant|business', regex=True).astype(int)
    df['has_attorney'] = rt.str.contains(
        r'attorney|counsel|esq\.?|law\s*office|represented\s*by|on\s*behalf\s*of.*(?:llc|inc|esq)', regex=True
    ).astype(int)
    if 'contact' in df.columns:
        contact_atty = df['contact'].fillna('').str.lower().str.contains(
            r'attorney|counsel|esq|law\s*office|llp|law\s*group', regex=True
        ).astype(int)
        df['has_attorney'] = (df['has_attorney'] | contact_atty).astype(int)
        contact_clean = df['contact'].fillna('').str.strip().str.lower()
        contact_counts = contact_clean.value_counts()
        pro_reps = set(contact_counts[contact_counts >= 5].index) - {'', 'nan'}
        is_pro_rep = contact_clean.isin(pro_reps).astype(int)
        df['has_attorney'] = (df['has_attorney'] | is_pro_rep).astype(int)
    if 'bpda_involved' not in df.columns:
        df['bpda_involved'] = rt.str.contains(r'bpda|boston\s*planning', regex=True).astype(int)
    if 'appeal_type' in df.columns:
        df['is_building_appeal'] = (df['appeal_type'].fillna('').str.lower().str.contains('building')).astype(int)
    else:
        df['is_building_appeal'] = rt.str.contains(r'building\s*(?:code|violation|appeal)', regex=True).astype(int)
    df['is_refusal_appeal'] = rt.str.contains(
        r'refus(?:al|ed)|annul|building\s*commissioner', regex=True).astype(int)
    # Violation types
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
    # Project types
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
            missing_mask = (df[col_name].fillna(0) == 0)
            if missing_mask.sum() > 0:
                df.loc[missing_mask, col_name] = rt[missing_mask].str.contains(pattern, regex=True).astype(int)
    # Units/stories extraction
    units_pattern = r'(\d+)\s*(?:residential\s*)?(?:unit|condo|apartment|dwelling)'
    stories_pattern = r'(\d+)\s*(?:stor|floor|level)'
    extracted_units = rt.str.extract(units_pattern, expand=False).astype(float)
    extracted_stories = rt.str.extract(stories_pattern, expand=False).astype(float)
    if 'proposed_units' in df.columns:
        zero_units = df['proposed_units'].fillna(0) == 0
        df.loc[zero_units & extracted_units.notna(), 'proposed_units'] = extracted_units[zero_units & extracted_units.notna()]
    if 'proposed_stories' in df.columns:
        zero_stories = df['proposed_stories'].fillna(0) == 0
        df.loc[zero_stories & extracted_stories.notna(), 'proposed_stories'] = extracted_stories[zero_stories & extracted_stories.notna()]
    # Legal framework
    if 'is_conditional_use' not in df.columns:
        df['is_conditional_use'] = rt.str.contains(r'conditional\s*use', regex=True).astype(int)
    if 'is_variance' not in df.columns:
        df['is_variance'] = rt.str.contains(r'variance', regex=True).astype(int)

# Binary columns fill
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

# Numeric features
df['proposed_units'] = df['proposed_units'].fillna(0).astype(int) if 'proposed_units' in df.columns else 0
df['proposed_stories'] = df['proposed_stories'].fillna(0).astype(int) if 'proposed_stories' in df.columns else 0
article_cols_present = [c for c in df.columns if c.startswith('article_') and df[c].dtype in ['int64', 'float64']]
if 'num_articles' not in df.columns:
    df['num_articles'] = df[article_cols_present].sum(axis=1) if article_cols_present else 0
df['num_articles'] = df['num_articles'].fillna(0).astype(int)
if 'num_sections' not in df.columns:
    df['num_sections'] = df['num_zoning_sections'].fillna(0).astype(int) if 'num_zoning_sections' in df.columns else 0
df['num_sections'] = df['num_sections'].fillna(0).astype(int)

# Year / recency
df['_year'] = np.nan
if 'source_pdf' in df.columns:
    df['_year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
if 'final_decision_date' in df.columns:
    fdd_year = pd.to_datetime(df['final_decision_date'], errors='coerce').dt.year
    df['_year'] = df['_year'].fillna(fdd_year)
if 'filing_date' in df.columns:
    fd_year = pd.to_datetime(df['filing_date'], errors='coerce').dt.year
    df['_year'] = df['_year'].fillna(fd_year)
tracker_path = os.path.join(BASE_DIR, 'zba_tracker.csv')
if os.path.exists(tracker_path) and df['_year'].isna().sum() > 100:
    tracker_dates = pd.read_csv(tracker_path, usecols=['boa_apno', 'final_decision_date', 'submitted_date'], low_memory=False)
    tracker_dates['_case'] = tracker_dates['boa_apno'].fillna('').astype(str).str.strip().str.replace('-', '')
    tracker_dates['_fdd'] = pd.to_datetime(tracker_dates['final_decision_date'], errors='coerce')
    tracker_dates['_sd'] = pd.to_datetime(tracker_dates['submitted_date'], errors='coerce')
    tracker_dates['_year'] = tracker_dates['_fdd'].dt.year.fillna(tracker_dates['_sd'].dt.year)
    tracker_year_map = tracker_dates.dropna(subset=['_year']).drop_duplicates('_case').set_index('_case')['_year'].to_dict()
    df['_case_clean'] = df['case_number'].fillna('').astype(str).str.strip().str.replace('-', '')
    missing_year = df['_year'].isna()
    for idx in df.index[missing_year]:
        case = df.loc[idx, '_case_clean']
        if case in tracker_year_map:
            df.loc[idx, '_year'] = tracker_year_map[case]
    df.drop(columns=['_case_clean'], errors='ignore', inplace=True)

df['_year'] = df['_year'].fillna(df['_year'].median())
year_min = df['_year'].min()
df['year_recency'] = (df['_year'] - year_min).fillna(0)

# Meta-features
proj_cols_present = [c for c in df.columns if c.startswith('proj_')]
df['project_complexity'] = df[proj_cols_present].sum(axis=1) if proj_cols_present else 0
violation_cols = ['excessive_far', 'insufficient_lot', 'insufficient_frontage', 'insufficient_yard', 'insufficient_parking']
df['total_violations'] = df[[c for c in violation_cols if c in df.columns]].sum(axis=1)
binary_feature_cols = [c for c in binary_cols if c in df.columns]
df['num_features_active'] = df[binary_feature_cols].sum(axis=1)

# Property features
for col in ['lot_size_sf', 'total_value', 'property_age', 'living_area', 'value_per_sqft', 'prior_permits']:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

# Applicant name for attorney win rate
if 'applicant_name' in df.columns:
    df['_applicant_clean'] = df['applicant_name'].fillna('unknown').astype(str).str.strip().str.lower()
else:
    df['_applicant_clean'] = 'unknown'


# ============================================================
# 3. TRAIN/TEST SPLIT — SAME LOGIC AS train_model_v2.py
# ============================================================
print("\nSplitting data (same method as training)...")

if 'source_pdf' in df.columns:
    df['_year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
    # Refill from other sources
    if 'final_decision_date' in df.columns:
        fdd_year = pd.to_datetime(df['final_decision_date'], errors='coerce').dt.year
        df['_year'] = df['_year'].fillna(fdd_year)
    if 'filing_date' in df.columns:
        fd_year = pd.to_datetime(df['filing_date'], errors='coerce').dt.year
        df['_year'] = df['_year'].fillna(fd_year)
    df['_year'] = df['_year'].fillna(df['_year'].median())

    max_year = df['_year'].max()
    temporal_mask = df['_year'] >= max_year
    n_recent = temporal_mask.sum()
    n_recent_denied = (df.loc[temporal_mask, 'approved'] == 0).sum() if temporal_mask.sum() > 0 else 0

    if n_recent >= 200 and n_recent_denied >= 10:
        train_idx = ~temporal_mask
        test_cal_idx = temporal_mask
        split_type = f"TEMPORAL: Train <{int(max_year)}, Test/Cal {int(max_year)}+"
        test_cal_indices = df.index[test_cal_idx]
        test_cal_labels = df.loc[test_cal_indices, 'approved']
        min_class_count = test_cal_labels.value_counts().min()
        use_stratify = test_cal_labels if min_class_count >= 4 else None
        test_indices, cal_indices = train_test_split(
            test_cal_indices, test_size=0.5, random_state=42, stratify=use_stratify
        )
        test_idx = pd.Series(False, index=df.index)
        cal_idx = pd.Series(False, index=df.index)
        test_idx.loc[test_indices] = True
        cal_idx.loc[cal_indices] = True
    else:
        split_type = "STRATIFIED RANDOM 70/15/15"
        _strat = df['approved'] if df['approved'].value_counts().min() >= 4 else None
        _train_i, _rest_i = train_test_split(df.index, test_size=0.3, random_state=42, stratify=_strat)
        _rest_strat = df.loc[_rest_i, 'approved'] if df.loc[_rest_i, 'approved'].value_counts().min() >= 4 else None
        _test_i, _cal_i = train_test_split(_rest_i, test_size=0.5, random_state=42, stratify=_rest_strat)
        train_idx = pd.Series(False, index=df.index); train_idx.loc[_train_i] = True
        test_idx = pd.Series(False, index=df.index); test_idx.loc[_test_i] = True
        cal_idx = pd.Series(False, index=df.index); cal_idx.loc[_cal_i] = True
else:
    split_type = "STRATIFIED RANDOM 70/15/15 (no source_pdf)"
    _strat = df['approved'] if df['approved'].value_counts().min() >= 4 else None
    _train_i, _rest_i = train_test_split(df.index, test_size=0.3, random_state=42, stratify=_strat)
    _rest_strat = df.loc[_rest_i, 'approved'] if df.loc[_rest_i, 'approved'].value_counts().min() >= 4 else None
    _test_i, _cal_i = train_test_split(_rest_i, test_size=0.5, random_state=42, stratify=_rest_strat)
    train_idx = pd.Series(False, index=df.index); train_idx.loc[_train_i] = True
    test_idx = pd.Series(False, index=df.index); test_idx.loc[_test_i] = True
    cal_idx = pd.Series(False, index=df.index); cal_idx.loc[_cal_i] = True

print(f"  Split type: {split_type}")
print(f"  Train: {train_idx.sum()} | Test: {test_idx.sum()} | Cal: {cal_idx.sum()}")


# ============================================================
# 4. TARGET ENCODING — from training data only (same as training)
# ============================================================
print("\nComputing target-encoded features from training data...")

SMOOTHING_WEIGHT = 20
df['ward'] = df['ward'].fillna('unknown').astype(str)
train_global_rate_computed = df.loc[train_idx, 'approved'].mean()

# Ward approval rates
ward_counts = df.loc[train_idx].groupby('ward')['approved'].agg(['mean', 'count'])
ward_approval = {}
for ward_name, row_data in ward_counts.iterrows():
    weight = row_data['count'] / (row_data['count'] + SMOOTHING_WEIGHT)
    ward_approval[ward_name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
df['ward_approval_rate'] = df['ward'].map(ward_approval).fillna(train_global_rate_computed)

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
    zoning_approval[z_name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
df['zoning_approval_rate'] = df['zoning_group'].map(zoning_approval).fillna(train_global_rate_computed)

# Attorney win rate
ATTORNEY_SMOOTHING = 5
train_applicant_rates = df.loc[train_idx].groupby('_applicant_clean')['approved'].agg(['mean', 'count'])
attorney_win_rates_computed = {}
for name, row_data in train_applicant_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + ATTORNEY_SMOOTHING)
        attorney_win_rates_computed[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
df['attorney_win_rate'] = df['_applicant_clean'].map(attorney_win_rates_computed).fillna(train_global_rate_computed)

# Contact win rate
CONTACT_SMOOTHING = 5
if 'contact' in df.columns:
    df['_contact_clean'] = df['contact'].fillna('unknown').astype(str).str.strip().str.lower()
    train_contact_rates = df.loc[train_idx].groupby('_contact_clean')['approved'].agg(['mean', 'count'])
    contact_win_rates_computed = {}
    for name, row_data in train_contact_rates.iterrows():
        if row_data['count'] >= 3:
            weight = row_data['count'] / (row_data['count'] + CONTACT_SMOOTHING)
            contact_win_rates_computed[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
    df['contact_win_rate'] = df['_contact_clean'].map(contact_win_rates_computed).fillna(train_global_rate_computed)
else:
    df['contact_win_rate'] = train_global_rate_computed

# Ward x Zoning
WARD_ZD_SMOOTHING = 10
df['_ward_zd'] = df['ward'].astype(str) + '_' + df['zoning_clean'].astype(str)
train_wz_rates = df.loc[train_idx].groupby('_ward_zd')['approved'].agg(['mean', 'count'])
ward_zoning_rates_computed = {}
for name, row_data in train_wz_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + WARD_ZD_SMOOTHING)
        ward_zoning_rates_computed[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
df['ward_zoning_rate'] = df['_ward_zd'].map(ward_zoning_rates_computed).fillna(train_global_rate_computed)

# Year x Ward
YEAR_WARD_SMOOTHING = 5
df['_year_ward'] = df['_year'].fillna(0).astype(int).astype(str) + '_' + df['ward'].astype(str)
train_yw_rates = df.loc[train_idx].groupby('_year_ward')['approved'].agg(['mean', 'count'])
year_ward_rates_computed = {}
for name, row_data in train_yw_rates.iterrows():
    if row_data['count'] >= 3:
        weight = row_data['count'] / (row_data['count'] + YEAR_WARD_SMOOTHING)
        year_ward_rates_computed[name] = weight * row_data['mean'] + (1 - weight) * train_global_rate_computed
df['year_ward_rate'] = df['_year_ward'].map(year_ward_rates_computed).fillna(train_global_rate_computed)

# Feature interactions
df['interact_height_stories'] = df.get('var_height', pd.Series(0, index=df.index)).fillna(0) * df['proposed_stories'].fillna(0)
df['interact_attorney_variances'] = df.get('has_attorney', pd.Series(0, index=df.index)).fillna(0) * df['num_variances'].fillna(0)
df['interact_highvalue_permits'] = df.get('is_high_value', pd.Series(0, index=df.index)).fillna(0) * df.get('has_prior_permits', pd.Series(0, index=df.index)).fillna(0)
df['lot_size_log'] = np.log1p(df['lot_size_sf'].fillna(0))
df['total_value_log'] = np.log1p(df['total_value'].fillna(0))
df['prior_permits_log'] = np.log1p(df['prior_permits'].fillna(0))
df['contact_x_appeal'] = df['contact_win_rate'] * df['is_building_appeal']
df['attorney_x_building'] = df.get('has_attorney', pd.Series(0, index=df.index)).fillna(0) * df['is_building_appeal']
df['many_variances'] = (df['num_variances'] >= 3).astype(int)
df['has_property_data'] = (df['total_value'] > 0).astype(int)


# ============================================================
# 5. BUILD FEATURE MATRIX AND PREDICT
# ============================================================
print("\nBuilding feature matrix...")

# Ensure all feature columns exist
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols].fillna(0)
y = df['approved']

X_test = X[test_idx]
y_test = y[test_idx]
X_cal = X[cal_idx]
y_cal = y[cal_idx]

# Also evaluate on combined test + calibration for maximum sample size
X_held_out = X[test_idx | cal_idx]
y_held_out = y[test_idx | cal_idx]

print(f"  Test set: {len(X_test)} cases")
print(f"  Calibration set: {len(X_cal)} cases")
print(f"  Combined held-out: {len(X_held_out)} cases")

print("\nRunning predictions...")
y_prob_test = model.predict_proba(X_test)[:, 1]
y_prob_cal = model.predict_proba(X_cal)[:, 1]
y_prob_held = model.predict_proba(X_held_out)[:, 1]

print(f"  Prediction range: [{y_prob_held.min():.3f}, {y_prob_held.max():.3f}]")
print(f"  Mean predicted probability: {y_prob_held.mean():.3f}")
print(f"  Actual approval rate (held-out): {y_held_out.mean():.3f}")


# ============================================================
# 6. CALIBRATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  CALIBRATION ANALYSIS RESULTS")
print("=" * 70)

# Overall metrics
brier = brier_score_loss(y_held_out, y_prob_held)
auc = roc_auc_score(y_held_out, y_prob_held)
ll = log_loss(y_held_out, y_prob_held)

print(f"\n  Overall Metrics (on {len(y_held_out)} held-out cases):")
print(f"    AUC-ROC:     {auc:.4f}")
print(f"    Brier Score: {brier:.4f}  (0 = perfect, 0.25 = random)")
print(f"    Log Loss:    {ll:.4f}")

# Custom calibration buckets as requested
bucket_edges = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
bucket_labels = ['0-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

print(f"\n  {'Bucket':<12} {'Count':>7} {'Avg Predicted':>15} {'Actual Approval':>17} {'Gap':>8} {'Assessment'}")
print(f"  {'-'*12} {'-'*7} {'-'*15} {'-'*17} {'-'*8} {'-'*25}")

bucket_data = []
for i in range(len(bucket_labels)):
    lo, hi = bucket_edges[i], bucket_edges[i+1]
    mask = (y_prob_held >= lo) & (y_prob_held < hi)
    count = mask.sum()
    if count > 0:
        avg_pred = y_prob_held[mask].mean()
        actual_rate = y_held_out.values[mask].mean()
        gap = actual_rate - avg_pred
        gap_pct = gap * 100

        # Assessment
        abs_gap = abs(gap_pct)
        if abs_gap <= 3:
            assessment = "WELL CALIBRATED"
        elif abs_gap <= 7:
            assessment = "SLIGHTLY OFF"
        elif abs_gap <= 15:
            assessment = "CAUTION"
        else:
            assessment = "UNRELIABLE"

        if gap > 0:
            direction = "(conservative)"
        else:
            direction = "(overconfident)"

        print(f"  {bucket_labels[i]:<12} {count:>7} {avg_pred:>14.1%} {actual_rate:>16.1%} {gap_pct:>+7.1f}pp  {assessment} {direction}")
        bucket_data.append({
            'label': bucket_labels[i], 'count': count,
            'avg_predicted': avg_pred, 'actual_rate': actual_rate,
            'gap': gap, 'assessment': assessment
        })
    else:
        print(f"  {bucket_labels[i]:<12} {0:>7} {'N/A':>15} {'N/A':>17} {'N/A':>8}")
        bucket_data.append({
            'label': bucket_labels[i], 'count': 0,
            'avg_predicted': None, 'actual_rate': None,
            'gap': None, 'assessment': 'NO DATA'
        })

# Expected Calibration Error (ECE) — weighted by bucket size
total_held = len(y_held_out)
ece = 0.0
for b in bucket_data:
    if b['count'] > 0 and b['gap'] is not None:
        ece += (b['count'] / total_held) * abs(b['gap'])
print(f"\n  Expected Calibration Error (ECE): {ece:.4f} ({ece*100:.1f}%)")
print(f"  (ECE < 0.05 is good, < 0.02 is excellent)")

# Maximum Calibration Error (MCE)
mce = max([abs(b['gap']) for b in bucket_data if b['gap'] is not None], default=0)
print(f"  Maximum Calibration Error (MCE): {mce:.4f} ({mce*100:.1f}%)")

# Finer-grained calibration curve using sklearn
print("\n  Fine-grained calibration curve (10 bins):")
prob_true_10, prob_pred_10 = calibration_curve(y_held_out, y_prob_held, n_bins=10, strategy='uniform')
print(f"  {'Bin Center':>12} {'Predicted':>12} {'Actual':>12} {'Gap':>8}")
for pt, pp in zip(prob_true_10, prob_pred_10):
    gap = pt - pp
    print(f"  {pp:>11.1%} {pp:>11.1%} {pt:>11.1%} {gap*100:>+7.1f}pp")


# ============================================================
# 7. DISTRIBUTION ANALYSIS — Where are predictions concentrated?
# ============================================================
print(f"\n  Prediction Distribution:")
pct_below_50 = (y_prob_held < 0.50).mean() * 100
pct_50_70 = ((y_prob_held >= 0.50) & (y_prob_held < 0.70)).mean() * 100
pct_70_90 = ((y_prob_held >= 0.70) & (y_prob_held < 0.90)).mean() * 100
pct_above_90 = (y_prob_held >= 0.90).mean() * 100
print(f"    Below 50%:    {pct_below_50:5.1f}% of predictions")
print(f"    50-70%:       {pct_50_70:5.1f}% of predictions")
print(f"    70-90%:       {pct_70_90:5.1f}% of predictions")
print(f"    Above 90%:    {pct_above_90:5.1f}% of predictions")


# ============================================================
# 8. SUBGROUP ANALYSIS — Are certain case types miscalibrated?
# ============================================================
print(f"\n  Subgroup Calibration (are some case types miscalibrated?):")

subgroups = {
    'Building Appeals': df.loc[test_idx | cal_idx, 'is_building_appeal'] == 1,
    'Zoning Appeals': df.loc[test_idx | cal_idx, 'is_building_appeal'] == 0,
    'With Attorney': df.loc[test_idx | cal_idx, 'has_attorney'] == 1,
    'Without Attorney': df.loc[test_idx | cal_idx, 'has_attorney'] == 0,
    'High Variance (3+)': df.loc[test_idx | cal_idx, 'num_variances'] >= 3,
    'Low Variance (0-2)': df.loc[test_idx | cal_idx, 'num_variances'] < 3,
}

print(f"  {'Subgroup':<22} {'N':>6} {'Avg Pred':>10} {'Actual':>10} {'Brier':>8} {'Gap':>8}")
print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
for name, mask in subgroups.items():
    if mask.sum() > 10:
        sub_prob = y_prob_held[mask.values]
        sub_actual = y_held_out.values[mask.values]
        sub_brier = brier_score_loss(sub_actual, sub_prob)
        gap = (sub_actual.mean() - sub_prob.mean()) * 100
        print(f"  {name:<22} {mask.sum():>6} {sub_prob.mean():>9.1%} {sub_actual.mean():>9.1%} {sub_brier:>7.4f} {gap:>+7.1f}pp")


# ============================================================
# 9. THE KEY QUESTION — Can developers trust these numbers?
# ============================================================
print("\n" + "=" * 70)
print("  VERDICT: Can a developer trust these probability numbers?")
print("=" * 70)

# Determine verdict based on metrics
issues = []
strengths = []

if ece < 0.02:
    strengths.append(f"Excellent ECE ({ece:.3f}) — probabilities closely match reality")
elif ece < 0.05:
    strengths.append(f"Good ECE ({ece:.3f}) — probabilities reasonably match reality")
else:
    issues.append(f"ECE of {ece:.3f} means probabilities are off by ~{ece*100:.0f}% on average")

if brier < 0.10:
    strengths.append(f"Good Brier score ({brier:.4f}) — predictions are accurate")
elif brier < 0.15:
    strengths.append(f"Acceptable Brier score ({brier:.4f})")
else:
    issues.append(f"High Brier score ({brier:.4f}) — predictions are inaccurate")

# Check the high-confidence bucket specifically (90-100%)
high_conf = [b for b in bucket_data if b['label'] == '90-100%' and b['count'] > 0]
if high_conf:
    hc = high_conf[0]
    if abs(hc['gap']) <= 0.05:
        strengths.append(f"90-100% bucket well-calibrated: says {hc['avg_predicted']:.0%}, actual {hc['actual_rate']:.0%}")
    else:
        direction = "overconfident" if hc['gap'] < 0 else "conservative"
        issues.append(f"90-100% bucket is {direction}: says {hc['avg_predicted']:.0%}, actual {hc['actual_rate']:.0%}")

# Check low-confidence bucket
low_conf = [b for b in bucket_data if b['label'] == '0-50%' and b['count'] > 0]
if low_conf:
    lc = low_conf[0]
    if abs(lc['gap']) <= 0.10:
        strengths.append(f"Low-risk flag works: says {lc['avg_predicted']:.0%}, actual {lc['actual_rate']:.0%}")
    else:
        issues.append(f"Low-confidence bucket off: says {lc['avg_predicted']:.0%}, actual {lc['actual_rate']:.0%}")

print("\n  STRENGTHS:")
for s in strengths:
    print(f"    + {s}")
if not strengths:
    print(f"    (none identified)")

print("\n  CONCERNS:")
for i in issues:
    print(f"    - {i}")
if not issues:
    print(f"    (none identified)")

# Per-bucket trust recommendations
print("\n  TRUST RECOMMENDATIONS BY PROBABILITY RANGE:")
for b in bucket_data:
    if b['count'] > 0 and b['gap'] is not None:
        abs_gap = abs(b['gap']) * 100
        if abs_gap <= 5:
            trust = "TRUST — well calibrated, numbers are reliable"
        elif abs_gap <= 10:
            trust = "TRUST WITH CAUTION — slight systematic bias"
        elif abs_gap <= 20:
            trust = "USE AS DIRECTIONAL ONLY — do not rely on exact %"
        else:
            trust = "DO NOT TRUST — significant miscalibration"
        direction = ""
        if b['gap'] > 0.02:
            direction = f" (model is conservative — reality is {abs_gap:.0f}pp better)"
        elif b['gap'] < -0.02:
            direction = f" (model is overconfident — reality is {abs_gap:.0f}pp worse)"
        print(f"    {b['label']:<12} (n={b['count']:>5}): {trust}{direction}")

print(f"\n  BOTTOM LINE:")
if ece < 0.05 and brier < 0.12:
    print(f"    The model is reasonably well-calibrated for a real-estate risk tool.")
    print(f"    Developers can use the probabilities as meaningful risk indicators,")
    print(f"    but should treat them as estimates (+/- {max(ece, mce)*100:.0f}pp), not guarantees.")
elif ece < 0.10:
    print(f"    The model has moderate calibration. Probabilities indicate directional")
    print(f"    risk (high vs low), but exact percentages should not be taken literally.")
    print(f"    A '90% chance' might really be anywhere from {90-mce*100:.0f}% to {min(100, 90+mce*100):.0f}%.")
else:
    print(f"    WARNING: The model has poor calibration (ECE={ece:.3f}).")
    print(f"    Probabilities should be treated as ordinal rankings only (higher = better),")
    print(f"    NOT as accurate percentage chances. Retraining with better calibration is needed.")


# ============================================================
# 10. GENERATE CALIBRATION CHART
# ============================================================
print(f"\nGenerating calibration_report.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('PermitIQ Model Calibration Report', fontsize=16, fontweight='bold', y=0.98)
fig.patch.set_facecolor('#f8f9fa')

# --- Panel 1: Reliability Diagram ---
ax1 = axes[0, 0]
prob_true_fine, prob_pred_fine = calibration_curve(y_held_out, y_prob_held, n_bins=10, strategy='uniform')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration', linewidth=1.5)
ax1.plot(prob_pred_fine, prob_true_fine, 's-', color='#2563eb', linewidth=2.5,
         markersize=10, markerfacecolor='white', markeredgewidth=2, label=f'Model (ECE={ece:.3f})')
# Shade the "good" zone
ax1.fill_between([0, 1], [0-0.05, 1-0.05], [0+0.05, 1+0.05], alpha=0.1, color='green', label='+/- 5% zone')
ax1.set_xlabel('Predicted Probability', fontsize=12)
ax1.set_ylabel('Actual Approval Rate', fontsize=12)
ax1.set_title('Reliability Diagram', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# --- Panel 2: Calibration Buckets (bar chart) ---
ax2 = axes[0, 1]
valid_buckets = [b for b in bucket_data if b['count'] > 0 and b['avg_predicted'] is not None]
labels = [b['label'] for b in valid_buckets]
predicted = [b['avg_predicted'] for b in valid_buckets]
actual = [b['actual_rate'] for b in valid_buckets]
x_pos = np.arange(len(labels))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, [p * 100 for p in predicted], width, label='Predicted', color='#93c5fd', edgecolor='#2563eb', linewidth=1.5)
bars2 = ax2.bar(x_pos + width/2, [a * 100 for a in actual], width, label='Actual', color='#86efac', edgecolor='#16a34a', linewidth=1.5)
# Add count labels on top
for i, b in enumerate(valid_buckets):
    ax2.text(i, max(b['avg_predicted'], b['actual_rate']) * 100 + 2, f'n={b["count"]}',
             ha='center', fontsize=9, color='#666')
ax2.set_xlabel('Probability Bucket', fontsize=12)
ax2.set_ylabel('Approval Rate (%)', fontsize=12)
ax2.set_title('Predicted vs Actual by Bucket', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=30, ha='right')
ax2.legend(fontsize=10)
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

# --- Panel 3: Prediction Distribution ---
ax3 = axes[1, 0]
ax3.hist(y_prob_held[y_held_out == 1], bins=30, alpha=0.6, color='#16a34a', label='Actually Approved', density=True, edgecolor='white')
ax3.hist(y_prob_held[y_held_out == 0], bins=30, alpha=0.6, color='#dc2626', label='Actually Denied', density=True, edgecolor='white')
ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
ax3.set_xlabel('Predicted Probability of Approval', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Prediction Distribution by Outcome', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Summary Statistics ---
ax4 = axes[1, 1]
ax4.axis('off')

summary_lines = [
    ('CALIBRATION METRICS', '', True),
    ('AUC-ROC', f'{auc:.4f}'),
    ('Brier Score', f'{brier:.4f}'),
    ('ECE', f'{ece:.4f} ({ece*100:.1f}%)'),
    ('MCE', f'{mce:.4f} ({mce*100:.1f}%)'),
    ('Log Loss', f'{ll:.4f}'),
    ('', '', False),
    ('DATASET', '', True),
    ('Held-out cases', f'{len(y_held_out):,}'),
    ('Actual approval rate', f'{y_held_out.mean():.1%}'),
    ('Mean predicted prob', f'{y_prob_held.mean():.1%}'),
    ('Split type', split_type[:35]),
]

y_start = 0.95
for i, item in enumerate(summary_lines):
    if len(item) == 3 and item[2] == True:
        ax4.text(0.05, y_start - i * 0.07, item[0], fontsize=12, fontweight='bold',
                transform=ax4.transAxes, family='monospace')
    elif len(item) == 3 and item[2] == False:
        continue
    else:
        ax4.text(0.08, y_start - i * 0.07, f'{item[0]}:', fontsize=11,
                transform=ax4.transAxes, family='monospace', color='#444')
        ax4.text(0.60, y_start - i * 0.07, item[1], fontsize=11,
                transform=ax4.transAxes, family='monospace', fontweight='bold')

# Trust summary at bottom
trust_y = 0.15
if ece < 0.05:
    verdict_text = "VERDICT: Probabilities are trustworthy as risk estimates"
    verdict_color = '#16a34a'
elif ece < 0.10:
    verdict_text = "VERDICT: Use as directional risk indicator, not exact %"
    verdict_color = '#d97706'
else:
    verdict_text = "VERDICT: Probabilities are unreliable — use rankings only"
    verdict_color = '#dc2626'
ax4.text(0.05, trust_y, verdict_text, fontsize=11, fontweight='bold',
        transform=ax4.transAxes, color=verdict_color,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=verdict_color, alpha=0.1))

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(BASE_DIR, 'calibration_report.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved to: {chart_path}")


# ============================================================
# 11. SAVE TEXT REPORT
# ============================================================
report_path = os.path.join(BASE_DIR, 'calibration_report.txt')
with open(report_path, 'w') as f:
    f.write("PermitIQ Model Calibration Report\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"MODEL: {type(model).__name__}\n")
    f.write(f"FEATURES: {len(feature_cols)}\n")
    f.write(f"HELD-OUT CASES: {len(y_held_out)}\n")
    f.write(f"SPLIT: {split_type}\n\n")

    f.write("OVERALL METRICS\n")
    f.write(f"  AUC-ROC:     {auc:.4f}\n")
    f.write(f"  Brier Score: {brier:.4f}\n")
    f.write(f"  ECE:         {ece:.4f} ({ece*100:.1f}%)\n")
    f.write(f"  MCE:         {mce:.4f} ({mce*100:.1f}%)\n")
    f.write(f"  Log Loss:    {ll:.4f}\n\n")

    f.write("CALIBRATION BUCKETS\n")
    f.write(f"  {'Bucket':<12} {'Count':>7} {'Predicted':>12} {'Actual':>12} {'Gap':>8}\n")
    for b in bucket_data:
        if b['count'] > 0:
            f.write(f"  {b['label']:<12} {b['count']:>7} {b['avg_predicted']:>11.1%} {b['actual_rate']:>11.1%} {b['gap']*100:>+7.1f}pp\n")

    f.write(f"\nTRUST RECOMMENDATIONS\n")
    for b in bucket_data:
        if b['count'] > 0 and b['gap'] is not None:
            abs_gap = abs(b['gap']) * 100
            if abs_gap <= 5:
                trust = "TRUST"
            elif abs_gap <= 10:
                trust = "CAUTION"
            elif abs_gap <= 20:
                trust = "DIRECTIONAL ONLY"
            else:
                trust = "DO NOT TRUST"
            f.write(f"  {b['label']:<12}: {trust} (gap: {b['gap']*100:+.1f}pp)\n")

    f.write(f"\n{verdict_text}\n")

print(f"  Saved text report to: {report_path}")
print("\nDone.")
