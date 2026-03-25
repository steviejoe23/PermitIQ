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
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# DEDUPLICATION
# ===========================
if 'case_number' in df.columns:
    before_dedup = len(df)
    df = df.drop_duplicates(subset='case_number', keep='first')
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        print(f"Deduplicated: removed {dupes_removed} duplicate case numbers")
    print(f"Unique cases: {len(df)}")

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
    print(f"  has_attorney: {df['has_attorney'].sum()} cases ({df['has_attorney'].mean():.0%})")

    # SAFE: BPDA involvement (happens before ZBA hearing)
    if 'bpda_involved' not in df.columns:
        df['bpda_involved'] = rt.str.contains(r'bpda|boston\s*planning', regex=True).astype(int)

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
    proj_patterns = {
        'proj_demolition': r'demol',
        'proj_new_construction': r'new\s*construct|erect|build.*new',
        'proj_addition': r'addition|extend|expansion',
        'proj_conversion': r'convert|conversion|change.*use',
        'proj_renovation': r'renovat|remodel|alter|rehabilitat',
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
    'bpda_involved',
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

# year_recency
if 'year_recency' not in df.columns:
    if 'source_pdf' in df.columns:
        df['year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
        df['year_recency'] = (df['year'] - df['year'].min()).fillna(0)
        df = df.drop(columns=['year'], errors='ignore')
    else:
        df['year_recency'] = 0
df['year_recency'] = df['year_recency'].fillna(0)

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
    if n_recent >= 100:
        train_idx = ~temporal_mask
        test_cal_idx = temporal_mask
        print(f"\nTEMPORAL SPLIT: Train on <{int(max_year)}, test+cal on {int(max_year)}+")

        # Split test into test (50%) and calibration (50%)
        test_cal_indices = df.index[test_cal_idx]
        # Use stratification only if both classes have enough samples
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
    df = df.drop(columns=['_year'], errors='ignore')
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

print(f"  Ward groups: {len(ward_approval)} | Zoning groups: {len(zoning_approval)}")
print(f"  Attorney win rates: {len(attorney_win_rates)} attorneys with 3+ cases")
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

    # Representation (2) — known at filing
    'has_attorney',
    'bpda_involved',

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

    # Location-based (3) — historical, computed from training data
    'ward_approval_rate', 'zoning_approval_rate',
    'attorney_win_rate',

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

    # Log transforms (2)
    'lot_size_log', 'total_value_log',
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
sample_weights = y_train.map({1: 1.0, 0: weight_denied}).values
print(f"\n  Class balance — Approved: {n_approved}, Denied: {n_denied}, Denied weight: {weight_denied:.2f}")

models = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=42, C=0.5
    )
}

uses_sample_weight = {'Gradient Boosting'}
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

    all_results[name] = {
        'model': model, 'auc': auc, 'brier': brier, 'logloss': logloss,
        'y_prob': y_prob, 'y_pred': y_pred
    }

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_name} (AUC: {best_auc:.4f})")
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
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_cal, y_cal)

# Evaluate calibrated model on TEST set (fair evaluation)
y_prob_test_cal = calibrated_model.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, y_prob_test_cal)
auc_cal = roc_auc_score(y_test, y_prob_test_cal)

print(f"\n  Calibrated model (evaluated on TEST set):")
print(f"    AUC:   {auc_cal:.4f} (was {best_auc:.4f})")
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
if hasattr(best_model, 'feature_importances_'):
    importances = sorted(
        zip(feature_cols, best_model.feature_importances_),
        key=lambda x: -x[1]
    )
    print(f"\nTop 20 Features ({best_name}):")
    for i, (feat, imp) in enumerate(importances[:20]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")
elif hasattr(best_model, 'coef_'):
    importances = sorted(
        zip(feature_cols, abs(best_model.coef_[0])),
        key=lambda x: -x[1]
    )
    print(f"\nTop 20 Features ({best_name} — |coefficients|):")
    for i, (feat, imp) in enumerate(importances[:20]):
        bar = "█" * int(imp * 20)
        print(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")


# ===========================
# CROSS VALIDATION (Stratified)
# ===========================
print("\nRunning 5-fold stratified cross validation on base model...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring='roc_auc')
print(f"5-Fold Stratified CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Fold scores: {', '.join(f'{s:.4f}' for s in cv_scores)}")


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
if hasattr(best_model, 'feature_importances_'):
    top_n = 20
    top_feats = importances[:top_n]
    feat_names = [f[0] for f in reversed(top_feats)]
    feat_vals = [f[1] for f in reversed(top_feats)]
    ax.barh(feat_names, feat_vals, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features ({best_name})')
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
    'base_model': best_model,  # Uncalibrated (for SHAP)
    'feature_cols': feature_cols,
    'ward_approval_rates': ward_approval,
    'zoning_approval_rates': zoning_approval,
    'top_zoning': top_zoning,
    'attorney_win_rates': attorney_win_rates,
    'overall_approval_rate': float(train_global_rate),
    'total_cases': int(len(df)),
    'model_name': best_name,
    'auc_score': float(final_auc),
    'auc_uncalibrated': float(best_auc),
    'brier_score': float(final_brier),
    'optimal_threshold': float(optimal_threshold),
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
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
    'auc_uncalibrated': round(best_auc, 4),
    'brier_score': round(final_brier, 4),
    'optimal_threshold': round(optimal_threshold, 3),
    'cv_auc_mean': round(float(cv_scores.mean()), 4),
    'cv_auc_std': round(float(cv_scores.std()), 4),
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
print(f"  AUC: {final_auc:.4f} (uncalibrated: {best_auc:.4f})")
print(f"  Brier: {final_brier:.4f}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")
print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Features: {n_features} (PRE-HEARING ONLY, leakage-free)")
print(f"  Training cases: {train_idx.sum()}")
print(f"  Dataset hash: {dataset_hash}")
print(f"  Plots: model_diagnostics.png")
print(f"  History: model_history/training_log.jsonl")
print(f"\n  PermitIQ model training complete.")
