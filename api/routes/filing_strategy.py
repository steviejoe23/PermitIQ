"""
Filing Strategy Routes — Temporal analysis of ZBA approval patterns.

Helps users optimize when to file based on historical seasonal patterns,
agenda size effects, and monthly approval rate trends.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from typing import Optional, List

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/filing_strategy", tags=["Filing Strategy"])

_zba_df = None
_tracker_df = None
_VARIANCE_TYPES = []


def init(zba_df, variance_types, tracker_path=None):
    global _zba_df, _tracker_df, _VARIANCE_TYPES
    _zba_df = zba_df
    _VARIANCE_TYPES = variance_types
    if tracker_path:
        try:
            _tracker_df = pd.read_csv(tracker_path, low_memory=False)
            logger.info("Filing strategy: tracker loaded (%d rows)", len(_tracker_df))
        except Exception as e:
            logger.warning("Filing strategy: could not load tracker: %s", e)


def _require_data():
    if _zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")
    return _zba_df


def _parse_dates(df):
    """Add parsed date columns."""
    out = df.copy()
    for col in ['filing_date', 'hearing_date']:
        if col in out.columns:
            out[f'_{col}_dt'] = pd.to_datetime(out[col], errors='coerce')
    return out


@router.get("/temporal_analysis")
def temporal_analysis():
    """Monthly and seasonal approval rate patterns across all ZBA cases."""
    df = _require_data()
    decided = df[df['decision_clean'].notna()].copy()

    # Try hearing_date first (from tracker merge), fall back to filing_date
    decided['_date'] = pd.to_datetime(decided.get('hearing_date', pd.Series(dtype='object')), errors='coerce')
    if '_date' not in decided.columns or decided['_date'].isna().all():
        decided['_date'] = pd.to_datetime(decided.get('filing_date', pd.Series(dtype='object')), errors='coerce')

    has_date = decided[decided['_date'].notna()].copy()
    has_date['_month'] = has_date['_date'].dt.month
    has_date['_quarter'] = has_date['_date'].dt.quarter
    has_date['_year'] = has_date['_date'].dt.year
    has_date['_approved'] = (has_date['decision_clean'] == 'APPROVED').astype(int)

    # Monthly patterns
    monthly = has_date.groupby('_month').agg(
        total=('_approved', 'count'),
        approved=('_approved', 'sum')
    ).reset_index()
    monthly['approval_rate'] = (monthly['approved'] / monthly['total']).round(3)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = []
    for _, row in monthly.iterrows():
        m = int(row['_month'])
        monthly_data.append({
            "month": m,
            "month_name": month_names[m - 1] if 1 <= m <= 12 else str(m),
            "total_cases": int(row['total']),
            "approved": int(row['approved']),
            "approval_rate": float(row['approval_rate']),
        })

    # Find best/worst months
    if monthly_data:
        best = max(monthly_data, key=lambda x: x['approval_rate'])
        worst = min(monthly_data, key=lambda x: x['approval_rate'])
    else:
        best = worst = None

    # Quarterly patterns
    quarterly = has_date.groupby('_quarter').agg(
        total=('_approved', 'count'),
        approved=('_approved', 'sum')
    ).reset_index()
    quarterly['approval_rate'] = (quarterly['approved'] / quarterly['total']).round(3)
    quarterly_data = [
        {
            "quarter": int(row['_quarter']),
            "label": f"Q{int(row['_quarter'])}",
            "total_cases": int(row['total']),
            "approval_rate": float(row['approval_rate']),
        }
        for _, row in quarterly.iterrows()
    ]

    # Agenda size effect (cases per hearing date)
    if has_date['_date'].notna().any():
        agenda = has_date.groupby('_date').agg(
            agenda_size=('_approved', 'count'),
            approved=('_approved', 'sum')
        ).reset_index()
        agenda['approval_rate'] = agenda['approved'] / agenda['agenda_size']

        # Bin agenda sizes
        bins = [(1, 5, 'Small (1-5)'), (6, 10, 'Medium (6-10)'),
                (11, 20, 'Large (11-20)'), (21, 999, 'Very Large (21+)')]
        agenda_effect = []
        for lo, hi, label in bins:
            subset = agenda[(agenda['agenda_size'] >= lo) & (agenda['agenda_size'] <= hi)]
            if len(subset) >= 3:
                agenda_effect.append({
                    "size_range": label,
                    "hearings": int(len(subset)),
                    "avg_cases": round(float(subset['agenda_size'].mean()), 1),
                    "approval_rate": round(float(subset['approval_rate'].mean()), 3),
                })
    else:
        agenda_effect = []

    # Year-over-year trend
    yearly = has_date.groupby('_year').agg(
        total=('_approved', 'count'),
        approved=('_approved', 'sum')
    ).reset_index()
    yearly['approval_rate'] = (yearly['approved'] / yearly['total']).round(3)
    yearly_data = [
        {
            "year": int(row['_year']),
            "total_cases": int(row['total']),
            "approval_rate": float(row['approval_rate']),
        }
        for _, row in yearly.iterrows()
        if int(row['total']) >= 10
    ]

    return {
        "monthly_patterns": monthly_data,
        "best_month": best,
        "worst_month": worst,
        "quarterly_patterns": quarterly_data,
        "agenda_size_effect": agenda_effect,
        "yearly_trend": yearly_data,
        "total_cases_analyzed": len(has_date),
    }


@router.get("/recommend")
def recommend_timing(
    variance_types: Optional[str] = Query(None, description="Comma-separated variance types"),
    ward: Optional[str] = Query(None, description="Ward number"),
):
    """Recommend optimal filing timing based on variance types and ward."""
    df = _require_data()
    decided = df[df['decision_clean'].notna()].copy()

    # Filter by variance types if provided
    if variance_types:
        vtypes = [v.strip().lower() for v in variance_types.split(',') if v.strip()]
        if vtypes:
            mask = pd.Series(False, index=decided.index)
            vt_col = decided['variance_types'].fillna('')
            for vt in vtypes:
                mask |= vt_col.str.contains(vt, na=False, case=False, regex=False)
            decided = decided[mask]

    # Filter by ward if provided
    if ward:
        try:
            ward_num = float(ward)
            decided = decided[decided['ward'] == ward_num]
        except (ValueError, TypeError):
            pass

    if len(decided) < 10:
        return {
            "recommendation": "Insufficient data for this combination. Try broader criteria.",
            "cases_analyzed": len(decided),
        }

    decided['_date'] = pd.to_datetime(decided.get('hearing_date', pd.Series(dtype='object')), errors='coerce')
    if decided['_date'].isna().all():
        decided['_date'] = pd.to_datetime(decided.get('filing_date', pd.Series(dtype='object')), errors='coerce')

    has_date = decided[decided['_date'].notna()].copy()
    has_date['_month'] = has_date['_date'].dt.month
    has_date['_approved'] = (has_date['decision_clean'] == 'APPROVED').astype(int)

    monthly = has_date.groupby('_month').agg(
        total=('_approved', 'count'),
        approved=('_approved', 'sum')
    ).reset_index()
    monthly['approval_rate'] = monthly['approved'] / monthly['total']

    # Only recommend from months with enough data
    reliable = monthly[monthly['total'] >= 5]
    if reliable.empty:
        return {
            "recommendation": "Not enough seasonal data for a confident recommendation.",
            "cases_analyzed": len(has_date),
        }

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    best_row = reliable.loc[reliable['approval_rate'].idxmax()]
    worst_row = reliable.loc[reliable['approval_rate'].idxmin()]
    overall_rate = float(has_date['_approved'].mean())

    best_month = int(best_row['_month'])
    worst_month = int(worst_row['_month'])
    delta = float(best_row['approval_rate'] - worst_row['approval_rate'])

    filters_desc = []
    if variance_types:
        filters_desc.append(f"variance types: {variance_types}")
    if ward:
        filters_desc.append(f"Ward {ward}")
    filter_str = " | ".join(filters_desc) if filters_desc else "all cases"

    return {
        "recommendation": f"For {filter_str}, hearings in {month_names[best_month - 1]} have the highest approval rate ({best_row['approval_rate']:.0%}). Avoid {month_names[worst_month - 1]} ({worst_row['approval_rate']:.0%}). Spread: {delta:.1%}.",
        "best_month": {"month": best_month, "name": month_names[best_month - 1], "approval_rate": round(float(best_row['approval_rate']), 3), "cases": int(best_row['total'])},
        "worst_month": {"month": worst_month, "name": month_names[worst_month - 1], "approval_rate": round(float(worst_row['approval_rate']), 3), "cases": int(worst_row['total'])},
        "overall_rate": round(overall_rate, 3),
        "seasonal_spread": round(delta, 3),
        "cases_analyzed": len(has_date),
        "filters": {"variance_types": variance_types, "ward": ward},
    }
