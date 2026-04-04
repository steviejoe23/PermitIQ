"""
Prediction endpoints — ML risk assessment, compare scenarios, batch predict.
Includes feature building, similar cases, variance history, recommendations.
"""

import re
import logging
import traceback
import datetime
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from api import state
from api.utils import safe_int, safe_str, safe_float, _clean_case_address, _clean_case_date
from api.constants import VARIANCE_TYPES, PROJECT_TYPES, FEATURE_LABELS, DISCLAIMER
from api.api_models import ProposalInput
from api.services.recommendations import generate_smart_recommendations

logger = logging.getLogger("permitiq")
router = APIRouter()


# Import verify_api_key from main at runtime to avoid circular imports
def _get_api_key_dep():
    try:
        from api.main import verify_api_key
        return Depends(verify_api_key)
    except ImportError:
        return None


def build_features(parcel_row, proposed_use: str, variances: list,
                   project_type: str = None, ward: str = None,
                   has_attorney: bool = False, proposed_units: int = 0,
                   proposed_stories: int = 0):
    """Build feature vector for ML model prediction — pre-hearing features only."""

    overall_rate = 0.76
    if state.model_package:
        overall_rate = state.model_package.get('overall_approval_rate', 0.76)

    district = str(parcel_row.get('districts') or '') if parcel_row is not None else ''

    # Variance features
    num_variances = len(variances)
    var_features = {}
    for vt in VARIANCE_TYPES:
        matched = any(
            vt in v.lower().replace(' ', '_') or v.lower().replace(' ', '_') in vt
            for v in variances
        )
        var_features[f'var_{vt}'] = int(matched)

    # Violation types
    variances_lower = ' '.join(v.lower() for v in variances)
    excessive_far = int('far' in variances_lower or 'floor area' in variances_lower)
    insufficient_lot = int('lot_area' in variances_lower or 'lot area' in variances_lower)
    insufficient_frontage = int('frontage' in variances_lower)
    insufficient_yard = int('setback' in variances_lower or 'yard' in variances_lower)
    insufficient_parking = int('parking' in variances_lower)

    # Use type
    use_lower = proposed_use.lower() if proposed_use else ""
    is_residential = int(bool(re.search(r'residential|dwelling|family|condo|apartment|housing', use_lower)))
    is_commercial = int(bool(re.search(r'commercial|retail|office|restaurant|store|shop', use_lower)))

    # Project type
    proj_features = {}
    for pt in PROJECT_TYPES:
        if project_type:
            proj_features[f'proj_{pt}'] = int(pt in project_type.lower().replace(' ', '_'))
        else:
            proj_features[f'proj_{pt}'] = int(pt.replace('_', ' ') in use_lower)

    # Ward and zoning rates
    ward_rate = overall_rate
    zoning_rate = overall_rate
    zoning = str(parcel_row.get('primary_zoning') or '') if parcel_row is not None else ''

    if state.model_package:
        ward_rates = state.model_package.get('ward_approval_rates', {})
        zoning_rates = state.model_package.get('zoning_approval_rates', {})
        top_zoning_list = state.model_package.get('top_zoning', [])
        ward_rate = ward_rates.get(str(ward or 'unknown'), overall_rate)
        zoning_group = zoning if zoning in top_zoning_list else 'other'
        zoning_rate = zoning_rates.get(zoning_group, overall_rate)

    # Property data + prior permits (single lookup, not two)
    lot_size = total_value = property_age = living_area = is_high_value = value_per_sqft = 0
    prior_permits = has_prior_permits = 0
    if parcel_row is not None and state.zba_df is not None:
        parcel_id = str(parcel_row.get('parcel_id', ''))
        if parcel_id:
            pa_match = state.zba_df[state.zba_df['pa_parcel_id'].astype(str) == parcel_id]
            if not pa_match.empty:
                rec = pa_match.iloc[0]
                lot_size = float(rec.get('lot_size_sf', 0) or 0)
                total_value = float(rec.get('total_value', 0) or 0)
                property_age = float(rec.get('property_age', 0) or 0)
                living_area = float(rec.get('living_area', 0) or 0)
                is_high_value = int(total_value > 1_000_000)
                value_per_sqft = total_value / lot_size if lot_size > 0 else 0
                prior_permits = float(rec.get('prior_permits', 0) or 0)
                has_prior_permits = int(prior_permits > 0)

    # Rate lookups
    attorney_win_rate = overall_rate
    contact_win_rate = overall_rate
    ward_zoning_rate = overall_rate
    year_ward_rate = overall_rate

    if state.model_package:
        wz_rates = state.model_package.get('ward_zoning_rates', {})
        wz_key = f"{ward or 'unknown'}_{zoning}"
        ward_zoning_rate = wz_rates.get(wz_key, overall_rate)

        yw_rates = state.model_package.get('year_ward_rates', {})
        yw_key = f"2026_{ward or 'unknown'}"
        year_ward_rate = yw_rates.get(yw_key, overall_rate)

    features = {
        'num_variances': num_variances,
        **var_features,
        'excessive_far': excessive_far,
        'insufficient_lot': insufficient_lot,
        'insufficient_frontage': insufficient_frontage,
        'insufficient_yard': insufficient_yard,
        'insufficient_parking': insufficient_parking,
        'is_residential': is_residential,
        'is_commercial': is_commercial,
        'has_attorney': int(has_attorney),
        'bpda_involved': 0,
        'is_building_appeal': 0,
        'is_refusal_appeal': 0,
        **proj_features,
        'article_80': int(proposed_units > 15 or proposed_stories > 4),
        'is_conditional_use': var_features.get('var_conditional_use', 0),
        'is_variance': int(num_variances > 0),
        'num_articles': max(1, num_variances),
        'proposed_units': proposed_units,
        'proposed_stories': proposed_stories,
        'ward_approval_rate': ward_rate,
        'zoning_approval_rate': zoning_rate,
        'attorney_win_rate': attorney_win_rate,
        'contact_win_rate': contact_win_rate,
        'ward_zoning_rate': ward_zoning_rate,
        'year_ward_rate': year_ward_rate,
        'year_recency': max(0, datetime.datetime.now().year - 2020),
        'lot_size_sf': lot_size,
        'total_value': total_value,
        'property_age': property_age,
        'living_area': living_area,
        'is_high_value': is_high_value,
        'value_per_sqft': value_per_sqft,
        'prior_permits': prior_permits,
        'has_prior_permits': has_prior_permits,
        'interact_height_stories': var_features.get('var_height', 0) * proposed_stories,
        'interact_attorney_variances': int(has_attorney) * num_variances,
        'interact_highvalue_permits': is_high_value * has_prior_permits,
        'lot_size_log': float(np.log1p(lot_size)),
        'total_value_log': float(np.log1p(total_value)),
        'prior_permits_log': float(np.log1p(prior_permits)),
        'contact_x_appeal': contact_win_rate * 0,
        'attorney_x_building': int(has_attorney) * 0,
        'many_variances': int(num_variances >= 3),
        'has_property_data': int(total_value > 0),
        'project_complexity': sum(v for v in proj_features.values()),
        'total_violations': excessive_far + insufficient_lot + insufficient_frontage + insufficient_yard + insufficient_parking,
        'num_features_active': sum([
            is_residential, is_commercial, int(has_attorney), 0,
            excessive_far, insufficient_lot, insufficient_frontage,
            insufficient_yard, insufficient_parking,
            *proj_features.values(),
            int(num_variances > 0),
            int(prior_permits > 0),
            is_high_value,
        ]),
        # Variance interactions
        'var_height_and_far': var_features.get('var_height', 0) * var_features.get('var_far', 0),
        'var_parking_and_units': var_features.get('var_parking', 0) * proposed_units,
        'num_variances_sq': num_variances ** 2,
        # Existing conditions (from assessor — not available for new proposals)
        'has_existing_parking': 0,
        'existing_parking_count': 0,
        # Scale & density
        'units_log': float(np.log1p(proposed_units)),
        'large_project': int(proposed_units >= 10 or proposed_stories >= 5),
        'units_per_lot_area': proposed_units / lot_size if lot_size > 0 else 0,
        'value_per_unit_log': float(np.log1p(total_value / proposed_units)) if proposed_units > 0 else 0,
        # Application type
        'is_change_occupancy': int(bool(project_type and 'change' in project_type.lower() and 'occupancy' in project_type.lower())) if project_type else 0,
        'is_maintain_use': int(bool(project_type and 'maintain' in project_type.lower())) if project_type else 0,
        # Complexity signals
        'multiple_setbacks': int(sum([var_features.get('var_front_setback', 0), var_features.get('var_rear_setback', 0), var_features.get('var_side_setback', 0)]) >= 2),
        'num_setback_variances': sum([var_features.get('var_front_setback', 0), var_features.get('var_rear_setback', 0), var_features.get('var_side_setback', 0)]),
        'interact_stories_far': proposed_stories * var_features.get('var_far', 0),
        'complex_case_score': num_variances + (excessive_far + insufficient_lot + insufficient_frontage + insufficient_yard + insufficient_parking) + sum(v for v in proj_features.values()),
    }

    return features


def get_similar_cases(ward, variances, project_type=None, limit=5):
    """Find similar historical ZBA cases using relevance scoring."""
    if state.zba_df is None or len(state.zba_df) == 0:
        return [], 0, 0.0

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    df['_relevance'] = 0.0

    if ward:
        try:
            ward_float = float(ward)
            df.loc[df['ward'] == ward_float, '_relevance'] += 3.0
        except (ValueError, TypeError):
            pass

    if variances and 'variance_types' in df.columns:
        vt = df['variance_types'].fillna('').str.lower()
        for v in variances:
            df.loc[vt.str.contains(v.lower(), na=False), '_relevance'] += 2.0

    if project_type:
        pt_col = f'proj_{project_type.lower().replace(" ", "_")}'
        if pt_col in df.columns:
            df.loc[df[pt_col] == 1, '_relevance'] += 2.0

    if 'year_recency' in df.columns:
        max_rec = df['year_recency'].max()
        if max_rec > 0:
            df['_relevance'] += df['year_recency'].fillna(0) / max_rec

    relevant = df[df['_relevance'] > 0].sort_values('_relevance', ascending=False)
    if len(relevant) < 3:
        relevant = df.sort_values('_relevance', ascending=False)

    total = len(relevant[relevant['_relevance'] >= 2])
    approval_rate = float((relevant.head(max(total, 50))['decision_clean'] == 'APPROVED').mean()) if len(relevant) > 0 else 0.0

    approved_pool = relevant[relevant['decision_clean'] == 'APPROVED']
    denied_pool = relevant[relevant['decision_clean'] == 'DENIED']

    denied_to_show = min(2, len(denied_pool), max(1, limit // 3))
    approved_to_show = limit - denied_to_show

    if len(denied_pool) == 0:
        sample = relevant.head(limit)
    else:
        approved_sample = approved_pool.head(approved_to_show)
        denied_sample = denied_pool.head(denied_to_show)
        sample = pd.concat([approved_sample, denied_sample]).sort_values('_relevance', ascending=False)

    cases = []
    for _, row in sample.iterrows():
        addr = _clean_case_address(row)
        if addr == 'Address not available' or not addr.strip():
            continue
        date_str = _clean_case_date(row)
        _case_variances = str(row.get('variance_types', '')) if pd.notna(row.get('variance_types')) else ''
        _case_attorney = str(row.get('contact', '')) if pd.notna(row.get('contact')) else ''
        cases.append({
            "case_number": str(row.get('case_number') or ''),
            "address": addr,
            "decision": str(row.get('decision_clean') or ''),
            "ward": str(int(float(row['ward']))) if pd.notna(row.get('ward')) else '',
            "date": date_str,
            "relevance_score": round(float(row.get('_relevance', 0)), 1),
            "variances": _case_variances,
            "attorney": _case_attorney,
            "has_attorney": bool(row.get('has_attorney', False)),
        })

    return cases, total, approval_rate


def _estimate_timeline(ward=None, appeal_type=None):
    """Estimate ZBA decision timeline using pre-computed tracker statistics."""
    if state.timeline_stats is None:
        return None

    overall = state.timeline_stats.get("overall", {})
    if not overall:
        return None

    ward_specific = False
    ward_str = None
    source = overall

    if ward:
        try:
            ward_str = str(int(float(ward)))
        except (ValueError, TypeError):
            ward_str = str(ward).strip()
        ward_data = state.timeline_stats.get("by_ward", {}).get(ward_str)
        if ward_data and "filing_to_decision" in ward_data:
            source = ward_data
            ward_specific = True

    phases = {}
    for phase_key, label in [
        ("filing_to_hearing", "Filing to First Hearing"),
        ("hearing_to_decision", "Hearing to Decision"),
        ("filing_to_decision", "Filing to Decision (total)"),
        ("filing_to_closed", "Filing to Case Closed"),
    ]:
        phase_data = source.get(phase_key)
        if phase_data:
            phases[phase_key] = {
                "label": label,
                "median_days": phase_data["median_days"],
                "p25_days": phase_data["p25_days"],
                "p75_days": phase_data["p75_days"],
                "cases_used": phase_data["cases_used"],
            }

    primary = source.get("filing_to_decision", overall.get("filing_to_decision", {}))
    result = {
        "median_days": primary.get("median_days"),
        "p25_days": primary.get("p25_days"),
        "p75_days": primary.get("p75_days"),
        "cases_used": primary.get("cases_used", 0),
        "ward_specific": ward_specific,
        "phases": phases,
    }

    if ward_specific and ward_str:
        result["ward"] = ward_str
        overall_primary = overall.get("filing_to_decision")
        if overall_primary:
            result["overall_median_days"] = overall_primary["median_days"]

    if appeal_type:
        atype_data = state.timeline_stats.get("by_appeal_type", {}).get(appeal_type)
        if atype_data and "filing_to_decision" in atype_data:
            result["appeal_type_median_days"] = atype_data["filing_to_decision"]["median_days"]
            result["appeal_type"] = appeal_type

    return result


def _get_variance_history(variances, ward=None, has_attorney=False):
    """Get real historical approval rates for a variance combination."""
    if state.zba_df is None:
        return None

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    vt = df['variance_types'].fillna('')

    overall_rate = float((df['decision_clean'] == 'APPROVED').mean())

    if variances:
        mask = pd.Series(True, index=df.index)
        for v in variances:
            mask = mask & vt.str.contains(v.lower(), na=False)
        combo_cases = df[mask]
    else:
        combo_cases = df

    combo_count = len(combo_cases)
    combo_approved = int((combo_cases['decision_clean'] == 'APPROVED').sum())
    combo_denied = combo_count - combo_approved
    combo_rate = float(combo_approved / combo_count) if combo_count > 0 else overall_rate

    ward_rate = None
    ward_count = 0
    if ward and combo_count > 0:
        try:
            ward_float = float(ward)
            ward_cases = combo_cases[combo_cases['ward'] == ward_float]
            ward_count = len(ward_cases)
            if ward_count >= 3:
                ward_rate = float((ward_cases['decision_clean'] == 'APPROVED').mean())
        except (ValueError, TypeError):
            pass

    attorney_effect = None
    if combo_count > 0 and 'has_attorney' in combo_cases.columns:
        with_atty = combo_cases[combo_cases['has_attorney'] == 1]
        without_atty = combo_cases[combo_cases['has_attorney'] == 0]
        if len(with_atty) >= 3 and len(without_atty) >= 3:
            rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
            rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
            attorney_effect = {
                "with_attorney": round(rate_with, 3),
                "without_attorney": round(rate_without, 3),
                "difference": round(rate_with - rate_without, 3),
                "cases_with": len(with_atty),
                "cases_without": len(without_atty),
            }

    per_variance = {}
    if variances:
        for v in variances:
            v_cases = df[vt.str.contains(v.lower(), na=False)]
            if len(v_cases) > 0:
                per_variance[v] = {
                    "approval_rate": round(float((v_cases['decision_clean'] == 'APPROVED').mean()), 3),
                    "cases": len(v_cases),
                }

    return {
        "combo_rate": round(combo_rate, 3),
        "combo_cases": combo_count,
        "combo_approved": combo_approved,
        "combo_denied": combo_denied,
        "overall_rate": round(overall_rate, 3),
        "ward_rate": round(ward_rate, 3) if ward_rate is not None else None,
        "ward_cases": ward_count,
        "attorney_effect": attorney_effect,
        "per_variance": per_variance,
    }


def _build_key_factors(variances, ward, has_attorney, project_type, proposed_units, vh):
    """Build key_factors from REAL data."""
    factors = []
    if vh is None:
        return ["Insufficient data for detailed analysis"]

    combo_rate = vh['combo_rate']
    combo_cases = vh['combo_cases']
    variance_list = ', '.join(variances) if variances else 'no specific variances'

    if combo_cases >= 10:
        factors.append(f"Based on {combo_cases} ZBA cases with {variance_list}: {combo_rate:.0%} were approved")
    elif combo_cases >= 3:
        factors.append(f"Limited data: {combo_cases} cases with {variance_list} — {combo_rate:.0%} approved (small sample)")
    else:
        factors.append(f"Very few cases ({combo_cases}) match your exact combination — overall rate is {vh['overall_rate']:.0%}")

    ae = vh.get('attorney_effect')
    if ae:
        diff = ae['difference']
        if has_attorney:
            if diff > 0.02:
                factors.append(f"Attorney representation: {ae['with_attorney']:.0%} approval ({ae['cases_with']} cases) vs {ae['without_attorney']:.0%} without — a {diff:.0%} advantage")
            else:
                factors.append(f"Attorney representation: minimal effect for this combination ({ae['with_attorney']:.0%} with vs {ae['without_attorney']:.0%} without)")
        else:
            if diff > 0.05:
                factors.append(f"No attorney: cases with representation have {diff:.0%} higher approval rate ({ae['with_attorney']:.0%} vs {ae['without_attorney']:.0%})")

    if vh.get('ward_rate') is not None and vh['ward_cases'] >= 3:
        ward_rate = vh['ward_rate']
        ward_cases = vh['ward_cases']
        if abs(ward_rate - combo_rate) > 0.05:
            direction = "higher" if ward_rate > combo_rate else "lower"
            factors.append(f"Ward {ward}: {ward_rate:.0%} approval ({ward_cases} cases) — {direction} than citywide")
        else:
            factors.append(f"Ward {ward}: {ward_rate:.0%} approval ({ward_cases} cases)")

    pv = vh.get('per_variance', {})
    for v_name, v_data in pv.items():
        rate = v_data['approval_rate']
        cases = v_data['cases']
        if rate < vh['overall_rate'] - 0.05:
            factors.append(f"{v_name.title()} variance: {rate:.0%} approval ({cases} cases) — below the overall {vh['overall_rate']:.0%} rate")
        elif rate > vh['overall_rate'] + 0.05:
            factors.append(f"{v_name.title()} variance: {rate:.0%} approval ({cases} cases) — above the overall {vh['overall_rate']:.0%} rate")

    if project_type:
        pt_lower = project_type.lower()
        if 'new_construction' in pt_lower or 'new construction' in pt_lower:
            factors.append("New construction typically faces more Board scrutiny than renovations/additions")
        elif 'addition' in pt_lower:
            factors.append("Additions to existing buildings generally have higher approval rates")

    if proposed_units > 5:
        factors.append(f"Large project ({proposed_units} units) — may attract community attention")

    return factors


def _build_recommendations(prob, variances, ward, has_attorney, project_type, proposed_units,
                           proposed_stories, variance_history, top_drivers):
    """Generate actionable recommendations to improve approval odds."""
    recs = []
    vh = variance_history or {}
    per_variance = vh.get('per_variance', {})
    ae = vh.get('attorney_effect')
    overall_rate = vh.get('overall_rate', 0.88)
    shap_lookup = {d.get('feature_name', ''): d.get('shap_value', 0) for d in (top_drivers or [])}

    if not has_attorney:
        atty_diff = 0
        with_rate = without_rate = 0
        if ae and ae.get('difference', 0) > 0.05:
            atty_diff = ae['difference']
            with_rate = ae['with_attorney']
            without_rate = ae['without_attorney']
        elif ae:
            atty_diff = ae.get('difference', 0)
            with_rate = ae.get('with_attorney', 0)
            without_rate = ae.get('without_attorney', 0)

        if atty_diff > 0.05:
            recs.append({"action": "Hire a zoning attorney", "detail": f"Cases like yours have {with_rate:.0%} approval with representation vs {without_rate:.0%} without ({atty_diff:.0%} improvement).", "estimated_impact": "high"})
        elif shap_lookup.get('has_attorney', 0) < -0.01:
            recs.append({"action": "Hire a zoning attorney", "detail": "The model identifies lack of attorney representation as reducing your approval odds.", "estimated_impact": "medium"})

    if len(variances) > 2 and per_variance:
        worst_var = None
        worst_rate = 1.0
        for v_name, v_data in per_variance.items():
            rate = v_data.get('approval_rate', 1.0)
            if rate < worst_rate:
                worst_rate = rate
                worst_var = v_name
        if worst_var and worst_rate < overall_rate - 0.03:
            recs.append({"action": f"Eliminate the {worst_var} variance", "detail": f"Consider redesigning to eliminate {worst_var} — it has the lowest approval rate ({worst_rate:.0%}).", "estimated_impact": "high"})

    units_shap = shap_lookup.get('proposed_units', 0)
    if proposed_units and proposed_units > 3 and units_shap < -0.005:
        suggested = max(1, proposed_units - 2)
        recs.append({"action": "Reduce unit count", "detail": f"Reducing from {proposed_units} to {suggested} units could improve odds.", "estimated_impact": "high" if units_shap < -0.02 else "medium"})

    height_shap = shap_lookup.get('var_height', 0)
    if 'height' in [v.lower() for v in variances] and height_shap < -0.005:
        recs.append({"action": "Reduce building height", "detail": "Consider reducing height to comply with the zoning limit.", "estimated_impact": "high" if height_shap < -0.02 else "medium"})

    if 'parking' in [v.lower() for v in variances]:
        parking_rate = per_variance.get('parking', {}).get('approval_rate')
        detail = "Adding more parking spaces would eliminate the parking variance. "
        if parking_rate is not None:
            detail += f"Parking variances have a {parking_rate:.0%} approval rate historically."
        recs.append({"action": "Add parking to eliminate variance", "detail": detail, "estimated_impact": "medium"})

    nc_shap = shap_lookup.get('proj_new_construction', 0)
    if project_type and 'new_construction' in project_type.lower().replace(' ', '_') and nc_shap < -0.005:
        recs.append({"action": "Consider renovation over new construction", "detail": "Renovations have higher approval rates than new construction.", "estimated_impact": "medium"})

    stories_shap = shap_lookup.get('proposed_stories', 0)
    if (proposed_stories and proposed_stories > 3 and stories_shap < -0.005
            and not any(r['action'] == 'Reduce unit count' for r in recs)):
        recs.append({"action": "Reduce building height/stories", "detail": f"Reducing from {proposed_stories} stories could improve approval odds.", "estimated_impact": "medium"})

    impact_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: impact_order.get(r["estimated_impact"], 9))
    return recs[:4]


@router.post("/analyze_proposal", tags=["Prediction"])
def analyze_proposal(payload: dict):
    """Analyze a development proposal and predict ZBA approval likelihood."""
    if state.gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    try:
        validated = ProposalInput(**payload)
    except Exception:
        validated = None

    if validated:
        parcel_id = validated.parcel_id
        proposed_use = validated.proposed_use
        variances = validated.variances
        project_type = validated.project_type
        ward = validated.ward
        has_attorney = validated.has_attorney
        proposed_units = validated.proposed_units
        proposed_stories = validated.proposed_stories
    else:
        parcel_id = payload.get("parcel_id", "") or ""
        proposed_use = payload.get("proposed_use", payload.get("use_type", payload.get("use", ""))) or ""
        variances = payload.get("variances", [])
        project_type = payload.get("project_type", None)
        ward = payload.get("ward", None)
        has_attorney = payload.get("has_attorney", False)
        proposed_units = safe_int(payload.get("proposed_units", 0))
        proposed_stories = safe_int(payload.get("proposed_stories", 0))

    # Clamp negative values to 0
    proposed_units = max(0, proposed_units)
    proposed_stories = max(0, proposed_stories)

    # Coerce variances: None → [], string → list
    if variances is None:
        variances = []
    elif isinstance(variances, str):
        variances = [v.strip() for v in variances.split(",") if v.strip()]

    parcel_row = None
    if parcel_id:
        match = state.gdf.loc[[str(parcel_id)]] if str(parcel_id) in state.gdf.index else state.gdf.iloc[0:0]
        if not match.empty:
            parcel_row = match.iloc[0]

    zoning = str(parcel_row.get("primary_zoning") or "") if parcel_row is not None else ""
    district = str(parcel_row.get("districts") or "") if parcel_row is not None else ""

    if not ward and district and state.zba_df is not None and 'zoning_district' in state.zba_df.columns:
        ward_lookup = state.zba_df[
            (state.zba_df['zoning_district'] == district) & state.zba_df['ward'].notna()
        ]['ward']
        if not ward_lookup.empty:
            ward = str(int(ward_lookup.mode().iloc[0]))
            logger.info("Auto-detected ward %s from district %s", ward, district)

    features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney,
                              proposed_units, proposed_stories)
    similar, total_similar, approval_rate_similar = get_similar_cases(ward, variances, project_type)
    variance_history = _get_variance_history(variances, ward, has_attorney)

    # ML MODEL PREDICTION
    if state.model_package and 'model' in state.model_package:
        try:
            model = state.model_package['model']
            feature_cols = state.model_package['feature_cols']

            input_df = pd.DataFrame([features])
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]

            prob = float(model.predict_proba(input_df)[0][1])

            # SHAP explainability
            top_drivers = []
            try:
                import shap
                shap_model = state.model_package.get('base_model', model)
                explainer = shap.TreeExplainer(shap_model)
                sv = explainer.shap_values(input_df)
                if isinstance(sv, list):
                    sv = sv[1]
                shap_row = sv[0]
                base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

                _violation_prefixes = ('excessive_', 'insufficient_')
                _violation_label_keywords = ('exceeds', 'insufficient', 'smaller than', 'below minimum', 'fewer')

                contributions = []
                for col, shap_val in zip(feature_cols, shap_row):
                    input_val = float(input_df[col].iloc[0])
                    label = FEATURE_LABELS.get(col, col.replace("_", " ").title())
                    is_violation_feature = (
                        any(col.startswith(p) for p in _violation_prefixes) or
                        any(kw in label.lower() for kw in _violation_label_keywords)
                    )
                    if is_violation_feature:
                        direction = "common in approved cases" if shap_val > 0 else "risk factor"
                    else:
                        direction = "increases approval odds" if shap_val > 0 else "decreases approval odds"
                    contributions.append({
                        "feature": label,
                        "feature_name": col,
                        "shap_value": round(float(shap_val), 4),
                        "input_value": int(input_val) if input_val == int(input_val) else round(float(input_val), 3),
                        "direction": direction,
                    })
                contributions.sort(key=lambda x: -abs(x["shap_value"]))
                top_drivers = contributions[:10]
            except Exception as shap_err:
                logger.warning("SHAP computation failed: %s", shap_err)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    input_vals = input_df.iloc[0]
                    contributions = []
                    for col, imp in zip(feature_cols, importances):
                        val = float(input_vals[col])
                        if imp > 0.01 and val != 0:
                            label = FEATURE_LABELS.get(col, col.replace("_", " ").title())
                            contributions.append({"feature": label, "feature_name": col, "shap_value": round(float(imp), 4), "input_value": round(val, 3) if abs(val) > 1 else int(val), "direction": "important factor"})
                    contributions.sort(key=lambda x: -abs(x["shap_value"]))
                    top_drivers = contributions[:10]

            if total_similar > 100:
                confidence, margin = "high", 0.05
            elif total_similar > 30:
                confidence, margin = "medium", 0.10
            else:
                confidence, margin = "low", 0.18
            prob_low = max(0.0, prob - margin)
            prob_high = min(1.0, prob + margin)

            key_factors = _build_key_factors(variances, ward, has_attorney, project_type, proposed_units, variance_history)

            calibration_warnings = []
            if prob < 0.50:
                calibration_warnings.append("Low-probability predictions (below 50%) have limited calibration data.")
            if features.get('is_building_appeal', 0) == 1:
                calibration_warnings.append("Building appeal cases are historically ~10pp less likely to be approved than predicted.")

            # Generate smart recommendations (counterfactual analysis)
            smart_recs = None
            try:
                proposal_for_recs = {
                    "parcel_id": parcel_id,
                    "proposed_use": proposed_use,
                    "variances": variances,
                    "project_type": project_type,
                    "ward": ward,
                    "has_attorney": has_attorney,
                    "proposed_units": proposed_units,
                    "proposed_stories": proposed_stories,
                }
                smart_recs = generate_smart_recommendations(
                    proposal_for_recs, prob, build_features, parcel_row
                )
            except Exception as smart_err:
                logger.warning("Smart recommendations failed: %s", smart_err)

            # Build warnings list
            _warnings = []
            if parcel_row is None:
                _warnings.append("Parcel not found — property data unavailable, prediction may be less accurate")
            if not top_drivers or (top_drivers and top_drivers[0].get('direction') == 'important factor'):
                _warnings.append("SHAP explainability unavailable — using feature importance as fallback")

            return {
                "parcel_id": parcel_id,
                "zoning": zoning,
                "district": district,
                "ward": ward,
                "proposed_use": proposed_use,
                "project_type": project_type or "not specified",
                "variances": variances,
                "has_attorney": has_attorney,
                "proposed_units": proposed_units,
                "proposed_stories": proposed_stories,
                "approval_probability": round(prob, 3),
                "probability_range": [round(prob_low, 3), round(prob_high, 3)],
                "confidence": confidence,
                "calibration_warnings": calibration_warnings if calibration_warnings else None,
                "warnings": _warnings if _warnings else None,
                "based_on_cases": total_similar,
                "ward_approval_rate": round(approval_rate_similar, 3) if total_similar > 0 else None,
                "key_factors": key_factors,
                "top_drivers": top_drivers,
                "recommendations": _build_recommendations(prob, variances, ward, has_attorney, project_type, proposed_units, proposed_stories, variance_history, top_drivers),
                "smart_recommendations": smart_recs,
                "similar_cases": similar,
                "variance_history": variance_history,
                "estimated_timeline_days": _estimate_timeline(ward),
                "model": state.model_package.get('model_name', 'ml_model'),
                "model_auc": state.model_package.get('auc_score', 0),
                "total_training_cases": state.model_package.get('total_cases', 0),
                "disclaimer": DISCLAIMER,
            }
        except Exception as e:
            logger.error("ML model error: %s, falling back to heuristic", e)

    # HEURISTIC FALLBACK
    df_decided = state.zba_df[state.zba_df['decision_clean'].notna()] if state.zba_df is not None else pd.DataFrame()
    base_rate = float(df_decided['decision_clean'].eq('APPROVED').mean()) if len(df_decided) > 0 else 0.65
    score = base_rate

    if len(df_decided) > 0 and 'variance_types' in df_decided.columns:
        for v in variances:
            v_mask = df_decided['variance_types'].fillna('').str.contains(v.lower(), na=False)
            v_cases = df_decided[v_mask]
            if len(v_cases) > 10:
                v_rate = float(v_cases['decision_clean'].eq('APPROVED').mean())
                score += (v_rate - base_rate) * 0.15
    else:
        if len(variances) > 3:
            score -= 0.15
        elif len(variances) > 2:
            score -= 0.08

    if 'has_attorney' in df_decided.columns and len(df_decided) > 0:
        atty_cases = df_decided[df_decided['has_attorney'] == 1]
        no_atty_cases = df_decided[df_decided['has_attorney'] == 0]
        if len(atty_cases) > 10 and len(no_atty_cases) > 10:
            atty_boost = float(atty_cases['decision_clean'].eq('APPROVED').mean()) - float(no_atty_cases['decision_clean'].eq('APPROVED').mean())
            if has_attorney:
                score += atty_boost * 0.3
            else:
                score -= atty_boost * 0.15
    else:
        if has_attorney:
            score += 0.10

    if approval_rate_similar > 0 and total_similar > 10:
        score = (score * 0.6) + (approval_rate_similar * 0.4)

    score = max(0.05, min(0.95, score))

    key_factors_heuristic = _build_key_factors(variances, ward, has_attorney, project_type, proposed_units, variance_history)
    key_factors_heuristic.insert(0, f"Note: Using historical statistics ({len(df_decided):,} cases) — ML model not loaded")

    return {
        "parcel_id": parcel_id,
        "zoning": zoning,
        "district": district,
        "proposed_use": proposed_use,
        "project_type": project_type or "not specified",
        "variances": variances,
        "has_attorney": has_attorney,
        "approval_probability": round(score, 3),
        "probability_range": [round(max(0, score - 0.15), 3), round(min(1, score + 0.15), 3)],
        "confidence": "low",
        "based_on_cases": total_similar,
        "ward_approval_rate": round(approval_rate_similar, 3) if total_similar > 0 else None,
        "key_factors": key_factors_heuristic,
        "top_drivers": [],
        "recommendations": _build_recommendations(score, variances, ward, has_attorney, project_type, proposed_units, proposed_stories, variance_history, []),
        "similar_cases": similar,
        "variance_history": variance_history,
        "estimated_timeline_days": _estimate_timeline(ward),
        "model": "data_driven_heuristic",
        "disclaimer": DISCLAIMER,
    }


@router.post("/batch_predict", tags=["Prediction"])
def batch_predict(payload: dict):
    """Predict approval likelihood for multiple proposals at once. Max 50."""
    proposals = payload.get("proposals", [])
    if not proposals:
        raise HTTPException(status_code=400, detail="No proposals provided")
    if len(proposals) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 proposals per batch request")

    results = []
    for i, p in enumerate(proposals):
        try:
            result = analyze_proposal(p)
            result["_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"_index": i, "error": str(e)})

    return {"results": results, "total": len(results)}


@router.post("/compare", tags=["Prediction"])
def compare_scenarios(payload: dict):
    """Run prediction under multiple scenarios and return differences."""
    if state.gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    try:
        parcel_id = payload.get("parcel_id", "")
        proposed_use = payload.get("proposed_use", payload.get("use_type", ""))
        variances = payload.get("variances", [])
        project_type = payload.get("project_type", None)
        ward = payload.get("ward", None)
        has_attorney = payload.get("has_attorney", False)
        proposed_units = max(0, safe_int(payload.get("proposed_units", 0)))
        proposed_stories = max(0, safe_int(payload.get("proposed_stories", 0)))

        parcel_row = None
        if parcel_id:
            match = state.gdf.loc[[str(parcel_id)]] if str(parcel_id) in state.gdf.index else state.gdf.iloc[0:0]
            if not match.empty:
                parcel_row = match.iloc[0]

        district = str(parcel_row.get("districts") or "") if parcel_row is not None else ""

        if not ward and district and state.zba_df is not None and 'zoning_district' in state.zba_df.columns:
            ward_lookup = state.zba_df[
                (state.zba_df['zoning_district'] == district) & state.zba_df['ward'].notna()
            ]['ward']
            if not ward_lookup.empty:
                ward = str(int(ward_lookup.mode().iloc[0]))

        def predict_prob(feat_dict):
            if state.model_package and 'model' in state.model_package:
                model = state.model_package['model']
                feature_cols = state.model_package['feature_cols']
                input_df = pd.DataFrame([feat_dict])
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_cols]
                return float(model.predict_proba(input_df)[0][1])
            return None

        base_features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney, proposed_units, proposed_stories)
        base_prob = predict_prob(base_features)

        if base_prob is None:
            raise HTTPException(status_code=503, detail="No ML model loaded")

        scenarios = []

        # Attorney toggle
        alt_features = build_features(parcel_row, proposed_use, variances, project_type, ward, not has_attorney, proposed_units, proposed_stories)
        alt_prob = predict_prob(alt_features)
        diff = alt_prob - base_prob
        scenario_name = "Without attorney" if has_attorney else "With attorney"
        scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(alt_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"{scenario_name} would change probability by {diff:+.1%}"})

        if len(variances) > 1:
            # Try removing each variance individually, pick the one whose removal helps most
            best_removal = None
            best_removal_prob = base_prob
            for i, v in enumerate(variances):
                trial = variances[:i] + variances[i+1:]
                trial_features = build_features(parcel_row, proposed_use, trial, project_type, ward, has_attorney, proposed_units, proposed_stories)
                trial_prob = predict_prob(trial_features)
                if trial_prob > best_removal_prob:
                    best_removal = (v, trial, trial_prob)
                    best_removal_prob = trial_prob
            if best_removal:
                removed_var, fewer, fewer_prob = best_removal
                diff = fewer_prob - base_prob
                scenario_name = f"Remove {removed_var} variance ({len(fewer)} remaining)"
                scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(fewer_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Removing {removed_var} variance changes probability by {diff:+.1%}", "changed": f"removed {removed_var}"})

        if len(variances) >= 3:
            single = variances[:1]
            single_features = build_features(parcel_row, proposed_use, single, project_type, ward, has_attorney, proposed_units, proposed_stories)
            single_prob = predict_prob(single_features)
            diff = single_prob - base_prob
            scenario_name = f"With only 1 variance ({single[0]})"
            scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(single_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Only 1 variance: probability changes by {diff:+.1%}"})

        if project_type and 'addition' not in (project_type or '').lower():
            add_features = build_features(parcel_row, proposed_use, variances, 'addition', ward, has_attorney, proposed_units, proposed_stories)
            add_prob = predict_prob(add_features)
            diff = add_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = "As an Addition/Extension project"
                scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(add_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"As addition: probability changes by {diff:+.1%}"})

        if proposed_units > 1:
            # Scale reduction: -1 for small projects, -30% for large ones
            if proposed_units <= 5:
                fewer_units = max(1, proposed_units - 1)
            else:
                fewer_units = max(1, int(proposed_units * 0.7))
            fewer_u_features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney, fewer_units, proposed_stories)
            fewer_u_prob = predict_prob(fewer_u_features)
            diff = fewer_u_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = f"With {fewer_units} unit(s) instead of {proposed_units}"
                scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(fewer_u_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Fewer units: probability changes by {diff:+.1%}", "changed": f"units {proposed_units} → {fewer_units}"})

        # Article 80 threshold scenario (15 units or 4+ stories triggers BPDA review)
        if proposed_units > 15:
            art80_features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney, 15, min(proposed_stories, 4))
            art80_prob = predict_prob(art80_features)
            diff = art80_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = "Below Article 80 threshold (15 units, 4 stories)"
                scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(art80_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Avoiding BPDA large project review: probability changes by {diff:+.1%}", "changed": f"units {proposed_units} → 15, stories {proposed_stories} → 4"})

        if proposed_stories > 2:
            fewer_s = proposed_stories - 1
            fewer_s_features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney, proposed_units, fewer_s)
            fewer_s_prob = predict_prob(fewer_s_features)
            diff = fewer_s_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = f"With {fewer_s} stories instead of {proposed_stories}"
                scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(fewer_s_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Fewer stories: probability changes by {diff:+.1%}"})

        best_attorney = True
        best_variances = variances[:1] if len(variances) > 1 else variances
        best_features = build_features(parcel_row, proposed_use, best_variances, project_type, ward, best_attorney, proposed_units, proposed_stories)
        best_prob = predict_prob(best_features)
        diff = best_prob - base_prob
        if abs(diff) > 0.001:
            scenario_name = "Best case (attorney + minimal variances)"
            scenarios.append({"scenario": scenario_name, "name": scenario_name, "probability": round(best_prob, 3), "difference": round(diff, 3), "delta": round(diff, 3), "description": f"Best case: {best_prob:.0%} ({diff:+.1%})"})

        return {"base_probability": round(base_prob, 3), "has_attorney": has_attorney, "num_variances": len(variances), "scenarios": scenarios}

    except Exception as e:
        logger.error("Compare endpoint error: %s", traceback.format_exc())
        logger.error("Comparison failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Comparison failed. Please check your input and try again.")


@router.post("/smart_recommendations", tags=["Prediction"])
def get_smart_recommendations(payload: dict):
    """
    Get specific, data-driven recommendations to improve ZBA approval odds.

    Runs counterfactual analysis using the real ML model: for each modifiable
    factor (variances, attorney, units, stories, project type), computes what
    the probability WOULD be if you changed it. Every recommendation is backed
    by real historical data from 7,500+ ZBA cases.

    Returns prioritized recommendations with model-computed probability impacts,
    similar approved/denied cases, and an optimized probability if all top
    recommendations are implemented.
    """
    if state.gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")
    if not state.model_package or 'model' not in state.model_package:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    # Parse input (same logic as analyze_proposal)
    try:
        validated = ProposalInput(**payload)
        parcel_id = validated.parcel_id
        proposed_use = validated.proposed_use
        variances = validated.variances
        project_type = validated.project_type
        ward = validated.ward
        has_attorney = validated.has_attorney
        proposed_units = validated.proposed_units
        proposed_stories = validated.proposed_stories
    except Exception:
        parcel_id = payload.get("parcel_id", "") or ""
        proposed_use = payload.get("proposed_use", payload.get("use_type", payload.get("use", ""))) or ""
        variances = payload.get("variances", [])
        project_type = payload.get("project_type", None)
        ward = payload.get("ward", None)
        has_attorney = payload.get("has_attorney", False)
        proposed_units = max(0, safe_int(payload.get("proposed_units", 0)))
        proposed_stories = max(0, safe_int(payload.get("proposed_stories", 0)))

    if variances is None:
        variances = []
    elif isinstance(variances, str):
        variances = [v.strip() for v in variances.split(",") if v.strip()]

    proposed_units = max(0, proposed_units)
    proposed_stories = max(0, proposed_stories)

    # Get parcel row
    parcel_row = None
    if parcel_id:
        match = state.gdf.loc[[str(parcel_id)]] if str(parcel_id) in state.gdf.index else state.gdf.iloc[0:0]
        if not match.empty:
            parcel_row = match.iloc[0]

    # Auto-detect ward
    district = str(parcel_row.get("districts") or "") if parcel_row is not None else ""
    if not ward and district and state.zba_df is not None and 'zoning_district' in state.zba_df.columns:
        ward_lookup = state.zba_df[
            (state.zba_df['zoning_district'] == district) & state.zba_df['ward'].notna()
        ]['ward']
        if not ward_lookup.empty:
            ward = str(int(ward_lookup.mode().iloc[0]))

    # Compute base prediction
    features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney,
                              proposed_units, proposed_stories)
    model = state.model_package['model']
    feature_cols = state.model_package['feature_cols']
    input_df = pd.DataFrame([features])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]
    base_prob = float(model.predict_proba(input_df)[0][1])

    # Generate smart recommendations
    proposal = {
        "parcel_id": parcel_id,
        "proposed_use": proposed_use,
        "variances": variances,
        "project_type": project_type,
        "ward": ward,
        "has_attorney": has_attorney,
        "proposed_units": proposed_units,
        "proposed_stories": proposed_stories,
    }

    result = generate_smart_recommendations(proposal, base_prob, build_features, parcel_row)
    result["parcel_id"] = parcel_id
    result["ward"] = ward
    result["disclaimer"] = DISCLAIMER
    return result
