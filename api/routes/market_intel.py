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
import time

router = APIRouter(tags=["Market Intelligence"])


# These get set by main.py after data loads
_zba_df = None
_VARIANCE_TYPES = []
_PROJECT_TYPES = []
_timeline_stats = None  # Pre-computed timeline stats from ZBA tracker

# --- Simple TTL cache for expensive aggregations ---
_cache = {}
CACHE_TTL = 3600  # 1 hour


def _cached(key, compute_fn):
    """Return cached result or compute and cache it."""
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    result = compute_fn()
    _cache[key] = (result, time.time())
    return result


def init(zba_df, variance_types, project_types, timeline_stats=None):
    """Called by main.py after data loads."""
    global _zba_df, _VARIANCE_TYPES, _PROJECT_TYPES, _timeline_stats
    _zba_df = zba_df
    _VARIANCE_TYPES = variance_types
    _PROJECT_TYPES = project_types
    _timeline_stats = timeline_stats
    _cache.clear()  # Invalidate cache on data reload


def _require_data():
    if _zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")
    return _zba_df


@router.get("/wards/all")
def all_ward_stats():
    """All ward statistics in a single call."""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        def _safe_ward(w):
            if pd.isna(w):
                return None
            try:
                return str(int(float(w)))
            except (ValueError, TypeError):
                return None
        df['_ward_clean'] = df['ward'].apply(_safe_ward)
        df = df[df['_ward_clean'].notna()]

        grouped = df.groupby('_ward_clean').agg(
            total=('decision_clean', 'count'),
            approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
        ).reset_index()

        grouped['denied'] = grouped['total'] - grouped['approved']
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        grouped['total_cases'] = grouped['total']
        wards = grouped.rename(columns={'_ward_clean': 'ward'}).to_dict('records')
        wards.sort(key=lambda x: -x['approval_rate'])
        return {"wards": wards}

    return _cached("wards_all", _compute)


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

    # Variance breakdown for this ward
    variance_breakdown = []
    if 'variance_types' in ward_cases.columns:
        vt_series = ward_cases['variance_types'].fillna('')
        for vtype in _VARIANCE_TYPES:
            mask = vt_series.str.contains(vtype, na=False)
            v_cases = ward_cases[mask]
            if len(v_cases) > 0:
                v_approved = int((v_cases['decision_clean'] == 'APPROVED').sum())
                variance_breakdown.append({
                    "variance_type": vtype,
                    "cases": len(v_cases),
                    "approval_rate": round(v_approved / len(v_cases), 3),
                })
        variance_breakdown.sort(key=lambda x: -x['cases'])

    # Attorney effect
    attorney_effect = None
    if 'has_attorney' in ward_cases.columns:
        with_atty = ward_cases[ward_cases['has_attorney'] == 1]
        without_atty = ward_cases[ward_cases['has_attorney'] == 0]
        if len(with_atty) >= 3 and len(without_atty) >= 3:
            rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
            rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
            attorney_effect = {
                "with_attorney_rate": round(rate_with, 3),
                "without_attorney_rate": round(rate_without, 3),
                "difference": round(rate_with - rate_without, 3),
                "cases_with": len(with_atty),
                "cases_without": len(without_atty),
            }

    return {
        "ward": ward_id,
        "total_cases": int(total),
        "approved": approved,
        "denied": denied,
        "approval_rate": round(float(approved / total), 3) if total > 0 else 0,
        "variance_breakdown": variance_breakdown,
        "attorney_effect": attorney_effect,
    }


@router.get("/project_type_stats")
def project_type_stats():
    """Approval rates by project type."""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        results = []

        # First try proj_ columns (available after full OCR retrain)
        has_proj_cols = any(f'proj_{pt}' in df.columns for pt in _PROJECT_TYPES)
        if has_proj_cols:
            for pt in _PROJECT_TYPES:
                col = f'proj_{pt}'
                if col in df.columns:
                    matched = df[df[col] == 1]
                    if len(matched) > 5:
                        approved = int((matched['decision_clean'] == 'APPROVED').sum())
                        total = len(matched)
                        results.append({
                            "project_type": pt.replace('_', ' ').title(),
                            "total_cases": int(total),
                            "approved": approved,
                            "approval_rate": round(approved / total, 3),
                        })
        else:
            # Derive project types from tracker_description and appeal_type columns
            import re
            project_patterns = {
                'New Construction': r'(?i)\b(new\s+construct|erect\s+a?\s*new|build\s+(?:a\s+)?new)\b',
                'Addition': r'(?i)\b(addition|add\s+(?:a\s+)?(?:rear|side|front|second|third|story|deck|porch|dormer))\b',
                'Renovation': r'(?i)\b(renovati|remodel|rehab|gut\s+(?:and\s+)?renovat|interior\s+renovation)\b',
                'Demolition': r'(?i)\b(demoli|tear\s*down|raze)\b',
                'Conversion': r'(?i)\b(convert|conversion|change\s+(?:of\s+)?(?:use|occupancy))\b',
                'Roof Deck': r'(?i)\b(roof\s*deck)\b',
                'Parking': r'(?i)\b(parking\s+(?:lot|garage|space|pad)|off[- ]street\s+parking|driveway)\b',
                'Subdivision': r'(?i)\b(subdivi|lot\s+split)\b',
                'ADU': r'(?i)\b(accessory\s+(?:dwelling|apartment)|adu|in[- ]law\s+(?:unit|apartment))\b',
                'Mixed Use': r'(?i)\b(mixed[- ]use)\b',
            }

            # Combine tracker_description and appeal_type for matching
            desc_col = None
            for col_name in ['tracker_description', 'description', 'raw_text']:
                if col_name in df.columns:
                    desc_col = col_name
                    break

            if desc_col:
                desc_text = df[desc_col].fillna('')
                for ptype, pattern in project_patterns.items():
                    mask = desc_text.str.contains(pattern, na=False)
                    matched = df[mask]
                    if len(matched) > 5:
                        approved = int((matched['decision_clean'] == 'APPROVED').sum())
                        total = len(matched)
                        results.append({
                            "project_type": ptype,
                            "total_cases": int(total),
                            "approved": approved,
                            "approval_rate": round(approved / total, 3),
                        })

            # Also include appeal_type breakdown (Zoning vs Building)
            if 'appeal_type' in df.columns:
                for atype in df['appeal_type'].dropna().unique():
                    atype_str = str(atype).strip()
                    if len(atype_str) > 1:
                        matched = df[df['appeal_type'] == atype]
                        if len(matched) > 5:
                            approved = int((matched['decision_clean'] == 'APPROVED').sum())
                            total = len(matched)
                            results.append({
                                "project_type": f"{atype_str} Appeal",
                                "total_cases": int(total),
                                "approved": approved,
                                "approval_rate": round(approved / total, 3),
                            })

        results.sort(key=lambda x: -x['approval_rate'])
        return {"project_type_stats": results}
    return _cached("project_type_stats", _compute)


@router.get("/variance_stats")
def variance_stats():
    """Approval rates by variance type."""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        results = []
        for vt in _VARIANCE_TYPES:
            mask = df['variance_types'].fillna('').str.contains(vt, na=False)
            matched = df[mask]
            if len(matched) > 5:
                approved = int((matched['decision_clean'] == 'APPROVED').sum())
                total = len(matched)
                results.append({
                    "variance_type": vt,
                    "total_cases": int(total),
                    "approved": approved,
                    "approval_rate": round(approved / total, 3),
                })
        results.sort(key=lambda x: -x['approval_rate'])
        return {"variance_stats": results}
    return _cached("variance_stats", _compute)


@router.get("/neighborhoods")
def neighborhood_stats():
    """Approval rates by zoning district/neighborhood."""
    def _compute():
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
        ).reset_index()
        grouped['denied'] = grouped['total'] - grouped['approved']
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        qualified = grouped[grouped['total'] >= 10].sort_values('approval_rate', ascending=False)
        qualified['total_cases'] = qualified['total']
        results = qualified.rename(columns={'_neighborhood': 'neighborhood'}).to_dict('records')
        return {"neighborhoods": results}
    return _cached("neighborhoods", _compute)


@router.get("/attorneys/leaderboard")
def attorney_leaderboard(min_cases: int = 5, limit: int = 20):
    """Top attorneys by ZBA approval rate."""
    def _compute():
        df = _require_data()
        # Use 'contact' (attorney/representative) not 'applicant_name' (the person applying)
        _atty_col = 'contact' if 'contact' in df.columns else 'applicant_name'
        df = df[
            (df['decision_clean'].notna()) &
            (df[_atty_col].notna()) &
            (df[_atty_col].str.len() > 3)
        ].copy()

        grouped = df.groupby(_atty_col).agg(
            total=('decision_clean', 'count'),
            approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
        ).reset_index()
        grouped['denied'] = grouped['total'] - grouped['approved']
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        qualified = grouped[grouped['total'] >= min_cases].sort_values(
            ['approval_rate', 'total'], ascending=[False, False]
        ).head(limit)

        qualified['total_cases'] = qualified['total']
        results = qualified.rename(columns={_atty_col: 'name'}).to_dict('records')

        has_attorney = df[df['has_attorney'] == 1] if 'has_attorney' in df.columns else pd.DataFrame()
        no_attorney = df[df['has_attorney'] == 0] if 'has_attorney' in df.columns else pd.DataFrame()

        return {
            "attorneys": results,
            "total_unique_attorneys": int(df[_atty_col].nunique()),
            "attorney_approval_rate": round(float((has_attorney['decision_clean'] == 'APPROVED').mean()), 3) if len(has_attorney) > 0 else None,
            "no_attorney_approval_rate": round(float((no_attorney['decision_clean'] == 'APPROVED').mean()), 3) if len(no_attorney) > 0 else None,
        }
    return _cached(f"attorney_leaderboard_{min_cases}_{limit}", _compute)


@router.get("/trends")
def approval_trends():
    """Approval rate trends over time, by year."""
    def _compute():
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
        ).reset_index()
        grouped['denied'] = grouped['total'] - grouped['approved']
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        grouped['year'] = grouped['_year'].astype(int)
        grouped['total_cases'] = grouped['total']
        years = grouped.drop(columns=['_year']).sort_values('year').to_dict('records')
        return {"years": years}
    return _cached("trends", _compute)


@router.get("/denial_patterns")
def denial_patterns():
    """What distinguishes denied cases from approved ones?"""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        return _compute_denial_patterns(df)
    return _cached("denial_patterns", _compute)


def _compute_denial_patterns(df):
    """Core denial patterns logic."""
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
                    "type": "rate",
                })

    if 'num_variances' in df.columns:
        avg_app = float(approved['num_variances'].fillna(0).mean())
        avg_den = float(denied['num_variances'].fillna(0).mean())
        patterns.append({
            "factor": "Average Variance Count",
            "approved_avg": round(avg_app, 2),
            "denied_avg": round(avg_den, 2),
            "approved_rate": round(avg_app, 2),
            "denied_rate": round(avg_den, 2),
            "difference": round(avg_app - avg_den, 2),
            "direction": "more_variances_denied" if avg_den > avg_app else "fewer_variances_denied",
            "type": "average",
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
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        return _compute_voting(df)
    return _cached("voting_patterns", _compute)


def _compute_voting(df):
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


@router.get("/wards/{ward_id}/trends")
def ward_trends(ward_id: str):
    """Yearly approval rate trends for a specific ward."""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        ward_str = str(int(float(ward_id))) if ward_id.replace('.', '').isdigit() else ward_id.strip()
        ward_col_norm = df['ward'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        ward_df = df[ward_col_norm == ward_str]
        if len(ward_df) == 0:
            return {"ward": ward_str, "years": [], "note": "No cases found for this ward"}

        if '_year' not in ward_df.columns:
            date_col = 'filing_date' if 'filing_date' in ward_df.columns else 'date'
            ward_df['_year'] = pd.to_datetime(ward_df.get(date_col, pd.NaT), errors='coerce').dt.year

        grouped = ward_df[ward_df['_year'].notna()].groupby('_year').agg(
            total=('decision_clean', 'count'),
            approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
        ).reset_index()
        grouped['denied'] = grouped['total'] - grouped['approved']
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        grouped['year'] = grouped['_year'].astype(int)

        years = grouped.drop(columns=['_year']).sort_values('year').to_dict('records')
        return {
            "ward": ward_str,
            "total_cases": int(len(ward_df)),
            "years": years,
        }
    return _cached(f"ward_trends_{ward_id}", _compute)


@router.get("/wards/{ward_id}/top_attorneys")
def ward_top_attorneys(ward_id: str, limit: int = 10):
    """Top attorneys by win rate in a specific ward."""
    def _compute():
        df = _require_data()
        df = df[df['decision_clean'].notna()].copy()
        ward_str = str(int(float(ward_id))) if ward_id.replace('.', '').isdigit() else ward_id.strip()
        _atty_col = 'contact' if 'contact' in df.columns else 'applicant_name'
        ward_col_norm = df['ward'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        ward_df = df[
            (ward_col_norm == ward_str) &
            (df[_atty_col].notna()) &
            (df[_atty_col].str.len() > 3)
        ]
        if len(ward_df) == 0:
            return {"ward": ward_str, "attorneys": []}

        grouped = ward_df.groupby(_atty_col).agg(
            total=('decision_clean', 'count'),
            approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
        ).reset_index()
        grouped['approval_rate'] = (grouped['approved'] / grouped['total']).round(3)
        qualified = grouped[grouped['total'] >= 2].sort_values(
            ['approval_rate', 'total'], ascending=[False, False]
        ).head(limit)

        results = qualified.rename(columns={_atty_col: 'name'}).to_dict('records')
        return {"ward": ward_str, "attorneys": results}
    return _cached(f"ward_top_attorneys_{ward_id}_{limit}", _compute)


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
def timeline_stats(ward: str = None, appeal_type: str = None):
    """Real ZBA timeline statistics from 14K+ tracker records.

    Returns median, 25th, and 75th percentile days for each phase:
    - filing_to_hearing: Time from application to first hearing
    - hearing_to_decision: Time from hearing to final vote
    - filing_to_decision: Total time from filing to decision
    - filing_to_closed: Total time from filing to case closure

    Optional filters: ward (1-22), appeal_type (Zoning, Building)
    """
    if _timeline_stats is None:
        raise HTTPException(status_code=503, detail="Timeline data not loaded")

    result = {
        "overall": _timeline_stats.get("overall", {}),
    }

    if ward:
        try:
            ward_str = str(int(float(ward)))
        except (ValueError, TypeError):
            ward_str = str(ward).strip()
        ward_data = _timeline_stats.get("by_ward", {}).get(ward_str)
        if ward_data:
            result["ward"] = {"ward": ward_str, "phases": ward_data}
        else:
            result["ward"] = {"ward": ward_str, "note": "Insufficient data for this ward"}

    if appeal_type:
        atype_data = _timeline_stats.get("by_appeal_type", {}).get(appeal_type)
        if atype_data:
            result["appeal_type"] = {"type": appeal_type, "phases": atype_data}
        else:
            result["appeal_type"] = {"type": appeal_type, "note": "No data for this appeal type"}

    # Include all wards summary for comparison
    result["wards_available"] = sorted(_timeline_stats.get("by_ward", {}).keys(), key=lambda x: int(x))
    result["appeal_types_available"] = sorted(_timeline_stats.get("by_appeal_type", {}).keys())

    return result
