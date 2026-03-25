"""
Market Intelligence Routes — Ward stats, variance rates, attorney leaderboard, trends, etc.
Extracted from main.py to reduce monolith size.

Usage: In main.py, add:
    from routes.market_intel import router as market_router
    app.include_router(market_router)
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

router = APIRouter(tags=["Market Intelligence"])


# These get set by main.py after data loads
_zba_df = None
_VARIANCE_TYPES = []
_PROJECT_TYPES = []


def init(zba_df, variance_types, project_types):
    """Called by main.py after data loads."""
    global _zba_df, _VARIANCE_TYPES, _PROJECT_TYPES
    _zba_df = zba_df
    _VARIANCE_TYPES = variance_types
    _PROJECT_TYPES = project_types


def _require_data():
    if _zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")
    return _zba_df


@router.get("/wards/all")
def all_ward_stats():
    """All ward statistics in a single call."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()
    df['_ward_clean'] = df['ward'].apply(lambda w: str(int(float(w))) if pd.notna(w) else None)
    df = df[df['_ward_clean'].notna()]

    grouped = df.groupby('_ward_clean').agg(
        total=('decision_clean', 'count'),
        approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    )

    wards = []
    for ward_name, row in grouped.iterrows():
        total = int(row['total'])
        approved = int(row['approved'])
        wards.append({
            "ward": ward_name,
            "total_cases": total,
            "approved": approved,
            "denied": total - approved,
            "approval_rate": round(float(approved / total), 3) if total > 0 else 0,
        })

    wards.sort(key=lambda x: -x['approval_rate'])
    return {"wards": wards}


@router.get("/wards/{ward_id}/stats")
def ward_stats(ward_id: str):
    """Get historical ZBA statistics for a specific ward."""
    df = _require_data()

    try:
        ward_num = float(ward_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid ward: {ward_id}")

    ward_cases = df[(df['ward'] == ward_num) & (df['decision_clean'].notna())]

    if len(ward_cases) == 0:
        raise HTTPException(status_code=404, detail=f"No cases found for Ward {ward_id}")

    approved = int((ward_cases['decision_clean'] == 'APPROVED').sum())
    denied = int((ward_cases['decision_clean'] == 'DENIED').sum())
    total = approved + denied

    return {
        "ward": ward_id,
        "total_cases": int(total),
        "approved": approved,
        "denied": denied,
        "approval_rate": round(float(approved / total), 3) if total > 0 else 0,
    }


@router.get("/project_type_stats")
def project_type_stats():
    """Approval rates by project type."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()
    results = []

    for pt in _PROJECT_TYPES:
        col = f'proj_{pt}'
        if col in df.columns:
            matched = df[df[col] == 1]
            if len(matched) > 5:
                approved = (matched['decision_clean'] == 'APPROVED').sum()
                total = len(matched)
                results.append({
                    "project_type": pt.replace('_', ' ').title(),
                    "total_cases": int(total),
                    "approved": int(approved),
                    "approval_rate": round(approved / total, 3),
                })

    results.sort(key=lambda x: -x['approval_rate'])
    return {"project_type_stats": results}


@router.get("/variance_stats")
def variance_stats():
    """Approval rates by variance type."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()
    results = []

    for vt in _VARIANCE_TYPES:
        mask = df['variance_types'].fillna('').str.contains(vt, na=False)
        matched = df[mask]
        if len(matched) > 5:
            approved = (matched['decision_clean'] == 'APPROVED').sum()
            total = len(matched)
            results.append({
                "variance_type": vt,
                "total_cases": int(total),
                "approved": int(approved),
                "approval_rate": round(approved / total, 3),
            })

    results.sort(key=lambda x: -x['approval_rate'])
    return {"variance_stats": results}


@router.get("/neighborhoods")
def neighborhood_stats():
    """Approval rates by zoning district/neighborhood."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()

    z_col = 'zoning_district' if 'zoning_district' in df.columns else (
        'zoning_clean' if 'zoning_clean' in df.columns else 'zoning')
    if z_col not in df.columns:
        return {"neighborhoods": []}

    df['_neighborhood'] = df[z_col].fillna('Unknown').astype(str)
    grouped = df.groupby('_neighborhood').agg(
        total=('decision_clean', 'count'),
        approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    )
    grouped['approval_rate'] = grouped['approved'] / grouped['total']
    qualified = grouped[grouped['total'] >= 10].sort_values('approval_rate', ascending=False)

    results = []
    for name, row in qualified.iterrows():
        results.append({
            "neighborhood": str(name),
            "total_cases": int(row['total']),
            "approved": int(row['approved']),
            "denied": int(row['total'] - row['approved']),
            "approval_rate": round(float(row['approval_rate']), 3),
        })

    return {"neighborhoods": results}


@router.get("/attorneys/leaderboard")
def attorney_leaderboard(min_cases: int = 5, limit: int = 20):
    """Top attorneys by ZBA approval rate."""
    df = _require_data()
    df = df[
        (df['decision_clean'].notna()) &
        (df['applicant_name'].notna()) &
        (df['applicant_name'].str.len() > 3)
    ].copy()

    grouped = df.groupby('applicant_name').agg(
        total=('decision_clean', 'count'),
        approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    )
    grouped['approval_rate'] = grouped['approved'] / grouped['total']
    qualified = grouped[grouped['total'] >= min_cases].sort_values(
        ['approval_rate', 'total'], ascending=[False, False]
    ).head(limit)

    results = []
    for name, row in qualified.iterrows():
        results.append({
            "name": str(name),
            "total_cases": int(row['total']),
            "approved": int(row['approved']),
            "denied": int(row['total'] - row['approved']),
            "approval_rate": round(float(row['approval_rate']), 3),
        })

    has_attorney = df[df.get('has_attorney', pd.Series(dtype=int)) == 1] if 'has_attorney' in df.columns else pd.DataFrame()
    no_attorney = df[df.get('has_attorney', pd.Series(dtype=int)) == 0] if 'has_attorney' in df.columns else pd.DataFrame()

    return {
        "attorneys": results,
        "total_unique_applicants": int(df['applicant_name'].nunique()),
        "attorney_approval_rate": round(float((has_attorney['decision_clean'] == 'APPROVED').mean()), 3) if len(has_attorney) > 0 else None,
        "no_attorney_approval_rate": round(float((no_attorney['decision_clean'] == 'APPROVED').mean()), 3) if len(no_attorney) > 0 else None,
    }


@router.get("/trends")
def approval_trends():
    """Approval rate trends over time, by year."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()

    if 'source_pdf' in df.columns:
        df['_year'] = df['source_pdf'].str.extract(r'(20\d{2})').astype(float)
    elif 'filing_date' in df.columns:
        df['_year'] = pd.to_datetime(df['filing_date'], errors='coerce').dt.year
    else:
        return {"years": []}

    df = df[df['_year'].notna()]
    grouped = df.groupby('_year').agg(
        total=('decision_clean', 'count'),
        approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    )
    grouped['approval_rate'] = grouped['approved'] / grouped['total']

    years = []
    for year, row in grouped.sort_index().iterrows():
        years.append({
            "year": int(year),
            "total_cases": int(row['total']),
            "approved": int(row['approved']),
            "denied": int(row['total'] - row['approved']),
            "approval_rate": round(float(row['approval_rate']), 3),
        })

    return {"years": years}


@router.get("/denial_patterns")
def denial_patterns():
    """What distinguishes denied cases from approved ones?"""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()
    approved = df[df['decision_clean'] == 'APPROVED']
    denied = df[df['decision_clean'] == 'DENIED']

    if len(denied) == 0 or len(approved) == 0:
        return {"patterns": []}

    patterns = []

    # Only compare PRE-HEARING features (no leaked features)
    binary_features = {
        'has_attorney': 'Legal Representation',
        'bpda_involved': 'BPDA Involved',
        'is_residential': 'Residential Use',
        'is_commercial': 'Commercial Use',
        'is_variance': 'Variance Request',
        'is_conditional_use': 'Conditional Use',
    }

    for col, label in binary_features.items():
        if col in df.columns:
            app_rate = float(approved[col].fillna(0).mean())
            den_rate = float(denied[col].fillna(0).mean())
            diff = app_rate - den_rate
            if abs(diff) > 0.03:
                patterns.append({
                    "factor": label,
                    "approved_rate": round(app_rate, 3),
                    "denied_rate": round(den_rate, 3),
                    "difference": round(diff, 3),
                    "direction": "favors_approval" if diff > 0 else "favors_denial",
                })

    if 'num_variances' in df.columns:
        avg_app = float(approved['num_variances'].fillna(0).mean())
        avg_den = float(denied['num_variances'].fillna(0).mean())
        patterns.append({
            "factor": "Average Variance Count",
            "approved_rate": round(avg_app, 2),
            "denied_rate": round(avg_den, 2),
            "difference": round(avg_app - avg_den, 2),
            "direction": "more_variances_denied" if avg_den > avg_app else "fewer_variances_denied",
        })

    patterns.sort(key=lambda x: -abs(x['difference']))

    return {
        "total_approved": int(len(approved)),
        "total_denied": int(len(denied)),
        "patterns": patterns,
    }


@router.get("/voting_patterns")
def voting_patterns():
    """Analyze vote distributions across ZBA decisions."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()
    result = {}

    if 'votes_in_favor' in df.columns and 'votes_opposed' in df.columns:
        df['votes_in_favor'] = pd.to_numeric(df['votes_in_favor'], errors='coerce').fillna(0)
        df['votes_opposed'] = pd.to_numeric(df['votes_opposed'], errors='coerce').fillna(0)

        approved = df[df['decision_clean'] == 'APPROVED']
        denied = df[df['decision_clean'] == 'DENIED']

        result["avg_votes_favor_approved"] = round(float(approved['votes_in_favor'].mean()), 2)
        result["avg_votes_favor_denied"] = round(float(denied['votes_in_favor'].mean()), 2)
        result["avg_votes_opposed_approved"] = round(float(approved['votes_opposed'].mean()), 2)
        result["avg_votes_opposed_denied"] = round(float(denied['votes_opposed'].mean()), 2)

        df['is_unanimous'] = (df['votes_opposed'] == 0) & (df['votes_in_favor'] > 0)
        unanimous = df[df['is_unanimous']]
        result["unanimous_total"] = int(len(unanimous))
        result["unanimous_approved"] = int((unanimous['decision_clean'] == 'APPROVED').sum())
        result["unanimous_approval_rate"] = round(float((unanimous['decision_clean'] == 'APPROVED').mean()), 3) if len(unanimous) > 0 else 0

        split = df[(df['votes_opposed'] > 0)]
        result["split_total"] = int(len(split))
        result["split_approved"] = int((split['decision_clean'] == 'APPROVED').sum())
        result["split_approval_rate"] = round(float((split['decision_clean'] == 'APPROVED').mean()), 3) if len(split) > 0 else 0

    return result


@router.get("/proviso_stats")
def proviso_stats():
    """Common conditions/provisos attached to ZBA approvals."""
    df = _require_data()
    df = df[df['decision_clean'] == 'APPROVED'].copy()

    proviso_cols = {
        'proviso_abutter_notice': 'Abutter Notice Required',
        'proviso_time_limit': 'Time Limit on Approval',
        'proviso_community': 'Community Process Condition',
        'proviso_design_review': 'Design Review Required',
        'proviso_bpda_review': 'BPDA Review Required',
        'proviso_traffic': 'Traffic Mitigation',
        'proviso_environmental': 'Environmental Conditions',
        'proviso_other': 'Other Conditions',
    }

    results = []
    for col, label in proviso_cols.items():
        if col in df.columns:
            count = int(df[col].fillna(0).sum())
            rate = round(float(df[col].fillna(0).mean()), 3) if len(df) > 0 else 0
            if count > 0:
                results.append({"condition": label, "count": count, "rate": rate})

    results.sort(key=lambda x: -x['count'])
    return {"total_approvals": int(len(df)), "conditions": results}


@router.get("/timeline_stats")
def timeline_stats():
    """How long ZBA decisions take — filing to decision date."""
    df = _require_data()
    df = df[df['decision_clean'].notna()].copy()

    if 'filing_date' not in df.columns:
        return {"message": "No filing date data available", "stats": {}}

    df['_filing_dt'] = pd.to_datetime(df['filing_date'], errors='coerce')

    if 'source_pdf' in df.columns:
        df['_decision_dt'] = pd.to_datetime(
            df['source_pdf'].str.extract(r'Filed (.+?)\.pdf')[0],
            errors='coerce'
        )
    else:
        df['_decision_dt'] = pd.NaT

    has_both = df[df['_filing_dt'].notna() & df['_decision_dt'].notna()].copy()
    if len(has_both) == 0:
        return {"message": "Insufficient date data for timeline analysis", "stats": {}}

    has_both['_days'] = (has_both['_decision_dt'] - has_both['_filing_dt']).dt.days
    has_both = has_both[(has_both['_days'] > 0) & (has_both['_days'] < 1100)]

    if len(has_both) == 0:
        return {"message": "No valid timelines computed", "stats": {}}

    overall = {
        "cases_with_timeline": int(len(has_both)),
        "median_days": int(has_both['_days'].median()),
        "mean_days": int(has_both['_days'].mean()),
        "p25_days": int(has_both['_days'].quantile(0.25)),
        "p75_days": int(has_both['_days'].quantile(0.75)),
        "min_days": int(has_both['_days'].min()),
        "max_days": int(has_both['_days'].max()),
    }

    by_decision = {}
    for dec in ['APPROVED', 'DENIED']:
        subset = has_both[has_both['decision_clean'] == dec]
        if len(subset) >= 5:
            by_decision[dec.lower()] = {
                "count": int(len(subset)),
                "median_days": int(subset['_days'].median()),
                "mean_days": int(subset['_days'].mean()),
            }

    by_ward = []
    if 'ward' in has_both.columns:
        ward_grp = has_both.groupby('ward')['_days'].agg(['median', 'count'])
        for w, row in ward_grp[ward_grp['count'] >= 5].iterrows():
            try:
                by_ward.append({
                    "ward": str(int(float(w))),
                    "median_days": int(row['median']),
                    "cases": int(row['count']),
                })
            except (ValueError, TypeError):
                pass
        by_ward.sort(key=lambda x: x['median_days'])

    return {"overall": overall, "by_decision": by_decision, "by_ward": by_ward}
