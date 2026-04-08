"""
Smart Recommendation Engine for PermitIQ.

Generates specific, data-driven recommendations to improve ZBA approval odds.
Every recommendation is backed by:
  1. Real counterfactual model predictions (not estimates)
  2. Real historical rates from the ZBA dataset
  3. Real similar case evidence
"""

import logging
import numpy as np
import pandas as pd

from api import state
from api.constants import VARIANCE_TYPES, FEATURE_LABELS
from api.utils import safe_float, safe_int, safe_str, _clean_case_address, _clean_case_date

logger = logging.getLogger("permitiq")


def _predict_prob(features_dict):
    """Run the ML model on a feature dict and return P(approved)."""
    if not state.model_package or 'model' not in state.model_package:
        return None
    model = state.model_package['model']
    feature_cols = state.model_package['feature_cols']
    input_df = pd.DataFrame([features_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]
    return float(model.predict_proba(input_df)[0][1])


def _get_top_attorneys_for_ward(ward, limit=3):
    """Find top-performing attorneys in a given ward from real data."""
    if state.zba_df is None:
        return []
    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    if 'contact' not in df.columns:
        return []

    # Filter to ward if provided
    if ward:
        try:
            ward_float = float(ward)
            df = df[df['ward'] == ward_float]
        except (ValueError, TypeError):
            pass

    # Only rows with attorney
    if 'has_attorney' in df.columns:
        df = df[df['has_attorney'] == 1]

    if df.empty or 'contact' not in df.columns:
        return []

    contacts = df['contact'].dropna()
    contacts = contacts[contacts.str.len() > 3]
    if contacts.empty:
        return []

    # Group by attorney
    attorney_stats = []
    for name, group in df.groupby('contact'):
        if pd.isna(name) or not isinstance(name, str) or len(str(name)) < 4:
            continue
        total = len(group)
        if total < 3:
            continue
        approved = int((group['decision_clean'] == 'APPROVED').sum())
        rate = approved / total
        attorney_stats.append({
            "name": str(name),
            "win_rate": round(rate, 3),
            "cases": total,
            "wins": approved,
        })

    attorney_stats.sort(key=lambda x: (-x['win_rate'], -x['cases']))
    return attorney_stats[:limit]


def _get_ward_use_type_rates(ward):
    """Get approval rates by use type (residential vs commercial) for a ward."""
    if state.zba_df is None:
        return {}
    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()

    if ward:
        try:
            ward_float = float(ward)
            df = df[df['ward'] == ward_float]
        except (ValueError, TypeError):
            pass

    result = {}
    for col, label in [('is_residential', 'residential'), ('is_commercial', 'commercial')]:
        if col in df.columns:
            subset = df[df[col] == 1]
            if len(subset) >= 5:
                rate = float((subset['decision_clean'] == 'APPROVED').mean())
                result[label] = {"rate": round(rate, 3), "cases": len(subset)}
    return result


def _find_similar_approved(ward, variances, project_type, has_attorney, proposed_units, limit=5):
    """Find approved cases similar to the proposal, showing what winners did."""
    if state.zba_df is None:
        return []
    df = state.zba_df[state.zba_df['decision_clean'] == 'APPROVED'].copy()
    return _score_and_return_cases(df, ward, variances, project_type, limit)


def _find_similar_denied(ward, variances, project_type, has_attorney, proposed_units, limit=3):
    """Find denied cases similar to the proposal, showing what to avoid."""
    if state.zba_df is None:
        return []
    df = state.zba_df[state.zba_df['decision_clean'] == 'DENIED'].copy()
    return _score_and_return_cases(df, ward, variances, project_type, limit)


def _score_and_return_cases(df, ward, variances, project_type, limit):
    """Score cases by relevance and return formatted results."""
    if df.empty:
        return []

    df = df.copy()
    df['_score'] = 0.0

    # Ward match
    if ward:
        try:
            ward_float = float(ward)
            df.loc[df['ward'] == ward_float, '_score'] += 3.0
        except (ValueError, TypeError):
            pass

    # Variance match
    if variances and 'variance_types' in df.columns:
        vt = df['variance_types'].fillna('').str.lower()
        for v in variances:
            df.loc[vt.str.contains(v.lower(), na=False), '_score'] += 2.0

    # Project type match
    if project_type:
        pt_col = f'proj_{project_type.lower().replace(" ", "_")}'
        if pt_col in df.columns:
            df.loc[df[pt_col] == 1, '_score'] += 2.0

    # Recency bonus
    if 'year_recency' in df.columns:
        max_rec = df['year_recency'].max()
        if max_rec > 0:
            df['_score'] += df['year_recency'].fillna(0) / max_rec * 0.5

    relevant = df[df['_score'] > 0].sort_values('_score', ascending=False)
    if len(relevant) < limit:
        relevant = df.sort_values('_score', ascending=False)

    cases = []
    for _, row in relevant.head(limit * 2).iterrows():
        addr = _clean_case_address(row)
        if addr == 'Address not available' or not addr.strip():
            continue

        # Extract key details for comparison
        case_info = {
            "case_number": safe_str(row.get('case_number')),
            "address": addr,
            "decision": safe_str(row.get('decision_clean')),
            "ward": str(int(float(row['ward']))) if pd.notna(row.get('ward')) and str(row.get('ward', '')).replace('.', '', 1).replace('-', '', 1).isdigit() else '',
            "date": _clean_case_date(row),
            "has_attorney": bool(row.get('has_attorney', 0)),
            "num_variances": safe_int(row.get('num_variances', 0)),
            "proposed_units": safe_int(row.get('proposed_units', 0)),
            "proposed_stories": safe_int(row.get('proposed_stories', 0)),
            "relevance_score": round(float(row.get('_score', 0)), 1),
        }

        # Add variance types if available
        vt_val = safe_str(row.get('variance_types'))
        if vt_val:
            case_info["variance_types"] = vt_val

        # Add attorney name if available
        contact = safe_str(row.get('contact'))
        if contact and len(contact) > 3:
            case_info["attorney"] = contact

        cases.append(case_info)
        if len(cases) >= limit:
            break

    return cases


def _compute_variance_removal_impacts(build_features_fn, parcel_row, proposed_use,
                                       variances, project_type, ward, has_attorney,
                                       proposed_units, proposed_stories, base_prob):
    """For each variance, compute probability without it using the real model."""
    impacts = []
    if len(variances) < 1:
        return impacts

    # Get per-variance historical rates from the dataset
    per_variance_rates = {}
    if state.zba_df is not None:
        df = state.zba_df[state.zba_df['decision_clean'].notna()]
        vt = df['variance_types'].fillna('').str.lower()
        for v in variances:
            v_cases = df[vt.str.contains(v.lower(), na=False)]
            if len(v_cases) > 0:
                per_variance_rates[v] = {
                    "rate": round(float((v_cases['decision_clean'] == 'APPROVED').mean()), 3),
                    "cases": len(v_cases),
                }

    for v in variances:
        remaining = [x for x in variances if x != v]
        alt_features = build_features_fn(
            parcel_row, proposed_use, remaining, project_type,
            ward, has_attorney, proposed_units, proposed_stories
        )
        alt_prob = _predict_prob(alt_features)
        if alt_prob is None:
            continue

        delta = alt_prob - base_prob
        if abs(delta) < 0.001:
            continue

        v_data = per_variance_rates.get(v, {})
        v_rate = v_data.get('rate')
        v_cases = v_data.get('cases', 0)

        evidence_parts = []
        if v_cases > 0:
            evidence_parts.append(f"{v} variance alone has {v_rate:.0%} approval rate ({v_cases:,} cases)")

        # Check combo effect
        if len(variances) > 1:
            combo_mask = pd.Series(True, index=state.zba_df.index)
            df_decided = state.zba_df[state.zba_df['decision_clean'].notna()]
            vt_col = df_decided['variance_types'].fillna('').str.lower()
            for var in variances:
                combo_mask = combo_mask & vt_col.str.contains(var.lower(), na=False)
            combo_cases = df_decided[combo_mask.reindex(df_decided.index, fill_value=False)]
            if len(combo_cases) >= 3:
                combo_rate = float((combo_cases['decision_clean'] == 'APPROVED').mean())
                evidence_parts.append(
                    f"combined with your other variances: {combo_rate:.0%} approval ({len(combo_cases)} cases)"
                )

        evidence = ". ".join(evidence_parts) + "." if evidence_parts else ""

        impacts.append({
            "variance": v,
            "delta": round(delta, 4),
            "new_probability": round(alt_prob, 3),
            "evidence": evidence,
            "historical_rate": v_rate,
            "historical_cases": v_cases,
        })

    # Sort by largest improvement first
    impacts.sort(key=lambda x: -x['delta'])
    return impacts


def _compute_scale_impacts(build_features_fn, parcel_row, proposed_use,
                           variances, project_type, ward, has_attorney,
                           proposed_units, proposed_stories, base_prob):
    """Compute impacts of reducing units and stories using the real model."""
    results = {}

    # --- Units ---
    if proposed_units > 1:
        # Compute median units for ward from real data
        ward_median_units = None
        if state.zba_df is not None and ward:
            df = state.zba_df[state.zba_df['decision_clean'].notna()]
            if 'proposed_units' in df.columns:
                try:
                    ward_float = float(ward)
                    ward_cases = df[(df['ward'] == ward_float) & (df['proposed_units'] > 0)]
                    if len(ward_cases) >= 5:
                        ward_median_units = int(ward_cases['proposed_units'].median())
                except (ValueError, TypeError):
                    pass

        # Try reducing to various levels
        unit_options = set()
        if ward_median_units and ward_median_units < proposed_units:
            unit_options.add(ward_median_units)
        if proposed_units > 4:
            unit_options.add(max(1, proposed_units - 2))
        if proposed_units > 2:
            unit_options.add(max(1, proposed_units - 1))
        # Also try half
        if proposed_units > 3:
            unit_options.add(max(1, proposed_units // 2))

        best_unit_rec = None
        for target_units in sorted(unit_options):
            if target_units >= proposed_units:
                continue
            alt_features = build_features_fn(
                parcel_row, proposed_use, variances, project_type,
                ward, has_attorney, target_units, proposed_stories
            )
            alt_prob = _predict_prob(alt_features)
            if alt_prob is None:
                continue
            delta = alt_prob - base_prob
            if delta > 0.005 and (best_unit_rec is None or delta > best_unit_rec['delta']):
                best_unit_rec = {
                    "current_units": proposed_units,
                    "suggested_units": target_units,
                    "delta": round(delta, 4),
                    "new_probability": round(alt_prob, 3),
                    "ward_median": ward_median_units,
                }

        if best_unit_rec:
            results['units'] = best_unit_rec

    # --- Stories ---
    if proposed_stories > 2:
        alt_features = build_features_fn(
            parcel_row, proposed_use, variances, project_type,
            ward, has_attorney, proposed_units, proposed_stories - 1
        )
        alt_prob = _predict_prob(alt_features)
        if alt_prob is not None:
            delta = alt_prob - base_prob
            if delta > 0.002:
                results['stories'] = {
                    "current_stories": proposed_stories,
                    "suggested_stories": proposed_stories - 1,
                    "delta": round(delta, 4),
                    "new_probability": round(alt_prob, 3),
                }

    return results


def _get_timing_advice(ward):
    """Check for quarterly/seasonal patterns in the data."""
    if state.zba_df is None:
        return None

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()

    # Try to extract quarter from filing_date or hearing_date
    date_col = None
    for col in ('hearing_date', 'filing_date', 'date'):
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return None

    try:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        df = df[dates.notna()].copy()
        df['_quarter'] = dates[dates.notna()].dt.quarter
    except Exception:
        return None

    if df.empty or '_quarter' not in df.columns:
        return None

    # Filter to ward if specified
    ward_df = df
    if ward:
        try:
            ward_float = float(ward)
            ward_subset = df[df['ward'] == ward_float]
            if len(ward_subset) >= 20:
                ward_df = ward_subset
        except (ValueError, TypeError):
            pass

    q_stats = []
    for q in range(1, 5):
        q_cases = ward_df[ward_df['_quarter'] == q]
        if len(q_cases) >= 10:
            rate = float((q_cases['decision_clean'] == 'APPROVED').mean())
            q_stats.append({"quarter": q, "rate": round(rate, 3), "cases": len(q_cases)})

    if len(q_stats) < 2:
        return None

    best_q = max(q_stats, key=lambda x: x['rate'])
    worst_q = min(q_stats, key=lambda x: x['rate'])
    spread = best_q['rate'] - worst_q['rate']

    if spread < 0.03:
        return None  # No meaningful seasonal pattern

    quarter_labels = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}
    return {
        "best_quarter": quarter_labels.get(best_q['quarter'], f"Q{best_q['quarter']}"),
        "best_rate": best_q['rate'],
        "best_cases": best_q['cases'],
        "worst_quarter": quarter_labels.get(worst_q['quarter'], f"Q{worst_q['quarter']}"),
        "worst_rate": worst_q['rate'],
        "worst_cases": worst_q['cases'],
        "spread": round(spread, 3),
        "all_quarters": q_stats,
    }


def _get_variance_combo_analysis(variances):
    """Analyze how specific variance combinations perform vs individually."""
    if state.zba_df is None or not variances or len(variances) < 2:
        return None

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    vt = df['variance_types'].fillna('').str.lower()

    # Full combo rate
    full_mask = pd.Series(True, index=df.index)
    for v in variances:
        full_mask = full_mask & vt.str.contains(v.lower(), na=False)
    full_cases = df[full_mask]
    if len(full_cases) < 3:
        return None
    full_rate = float((full_cases['decision_clean'] == 'APPROVED').mean())

    # All pairs and sub-combos (for combos of 3+)
    sub_combos = []
    if len(variances) >= 3:
        for i, v_remove in enumerate(variances):
            sub = [x for j, x in enumerate(variances) if j != i]
            sub_mask = pd.Series(True, index=df.index)
            for v in sub:
                sub_mask = sub_mask & vt.str.contains(v.lower(), na=False)
            sub_cases = df[sub_mask]
            if len(sub_cases) >= 3:
                sub_rate = float((sub_cases['decision_clean'] == 'APPROVED').mean())
                improvement = sub_rate - full_rate
                if improvement > 0.02:
                    sub_combos.append({
                        "removed": v_remove,
                        "remaining": sub,
                        "rate": round(sub_rate, 3),
                        "cases": len(sub_cases),
                        "improvement": round(improvement, 3),
                    })

    sub_combos.sort(key=lambda x: -x['improvement'])

    return {
        "full_combo_rate": round(full_rate, 3),
        "full_combo_cases": len(full_cases),
        "better_sub_combos": sub_combos[:3],
    }


def generate_smart_recommendations(proposal, prediction_prob, build_features_fn,
                                    parcel_row=None):
    """
    Generate specific, data-backed recommendations to improve approval odds.

    Approach:
    1. For each modifiable factor, compute the counterfactual with the real ML model
    2. Back every recommendation with real historical data
    3. Find similar approved and denied cases for concrete examples
    4. Prioritize by model-computed impact

    Args:
        proposal: dict with parcel_id, proposed_use, variances, project_type,
                  ward, has_attorney, proposed_units, proposed_stories
        prediction_prob: float, current P(approved) from the model
        build_features_fn: the build_features() function from prediction module
        parcel_row: GeoDataFrame row for the parcel (or None)

    Returns:
        dict with recommendations, similar cases, and optimization summary
    """
    variances = proposal.get('variances', [])
    ward = proposal.get('ward')
    has_attorney = proposal.get('has_attorney', False)
    project_type = proposal.get('project_type')
    proposed_units = safe_int(proposal.get('proposed_units', 0))
    proposed_stories = safe_int(proposal.get('proposed_stories', 0))
    proposed_use = proposal.get('proposed_use', '')

    recommendations = []
    cumulative_new_prob = prediction_prob

    # ---- 1. Variance removal impacts ----
    variance_impacts = _compute_variance_removal_impacts(
        build_features_fn, parcel_row, proposed_use,
        variances, project_type, ward, has_attorney,
        proposed_units, proposed_stories, prediction_prob
    )
    for vi in variance_impacts:
        if vi['delta'] > 0.005:
            difficulty = "medium"
            if vi['variance'] in ('parking',):
                difficulty = "medium"  # usually requires design change
            elif vi['variance'] in ('height', 'far'):
                difficulty = "high"  # requires significant redesign
            elif vi['variance'] in ('conditional_use',):
                difficulty = "low"

            action_detail = f"Remove the {vi['variance']} variance"
            if vi['variance'] == 'parking':
                action_detail = f"Provide the required parking to eliminate the parking variance"
            elif vi['variance'] == 'height':
                action_detail = f"Reduce building height to comply with the zoning limit"
            elif vi['variance'] == 'far':
                action_detail = f"Reduce floor area to comply with the FAR limit"

            recommendations.append({
                "action": f"Remove {vi['variance']} variance",
                "current": f"{len(variances)} variances including {vi['variance']}",
                "suggested": action_detail,
                "probability_impact": f"+{vi['delta']:.2%}" if vi['delta'] > 0 else f"{vi['delta']:.2%}",
                "new_probability": vi['new_probability'],
                "evidence": vi['evidence'],
                "difficulty": difficulty,
                "priority": 0,  # Will be re-sorted by impact
                "_delta": vi['delta'],
            })

    # ---- 2. Attorney impact ----
    if not has_attorney:
        # Compute counterfactual with attorney
        alt_features = build_features_fn(
            parcel_row, proposed_use, variances, project_type,
            ward, True, proposed_units, proposed_stories
        )
        alt_prob = _predict_prob(alt_features)

        # Get real attorney data
        attorney_evidence_parts = []
        if alt_prob is not None:
            delta = alt_prob - prediction_prob

            # Get ward-specific attorney rates
            if state.zba_df is not None:
                df = state.zba_df[state.zba_df['decision_clean'].notna()]
                ward_filter = df
                ward_label = "citywide"
                if ward:
                    try:
                        ward_float = float(ward)
                        ward_subset = df[df['ward'] == ward_float]
                        if len(ward_subset) >= 10:
                            ward_filter = ward_subset
                            ward_label = f"in Ward {int(ward_float)}"
                    except (ValueError, TypeError):
                        pass

                if 'has_attorney' in ward_filter.columns:
                    with_atty = ward_filter[ward_filter['has_attorney'] == 1]
                    without_atty = ward_filter[ward_filter['has_attorney'] == 0]
                    if len(with_atty) >= 3 and len(without_atty) >= 3:
                        rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
                        rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
                        attorney_evidence_parts.append(
                            f"{ward_label}, cases with attorneys: {rate_with:.0%} approval "
                            f"({len(with_atty):,} cases) vs without: {rate_without:.0%} "
                            f"({len(without_atty):,} cases)"
                        )

                # Top attorneys in this ward
                top_attorneys = _get_top_attorneys_for_ward(ward, limit=3)
                if top_attorneys:
                    names = [f"{a['name']} ({a['win_rate']:.0%})" for a in top_attorneys]
                    attorney_evidence_parts.append(
                        f"Top attorneys in this ward: {', '.join(names)}"
                    )

            if delta > 0.005:
                recommendations.append({
                    "action": "Hire a zoning attorney",
                    "current": "No attorney representation",
                    "suggested": "Retain an experienced ZBA attorney",
                    "probability_impact": f"+{delta:.2%}",
                    "new_probability": round(alt_prob, 3),
                    "evidence": ". ".join(attorney_evidence_parts) + "." if attorney_evidence_parts else "",
                    "difficulty": "low",
                    "priority": 0,
                    "_delta": delta,
                })

    # ---- 3. Scale reduction (units and stories) ----
    scale_impacts = _compute_scale_impacts(
        build_features_fn, parcel_row, proposed_use,
        variances, project_type, ward, has_attorney,
        proposed_units, proposed_stories, prediction_prob
    )

    if 'units' in scale_impacts:
        ui = scale_impacts['units']
        evidence = f"Model predicts {ui['delta']:+.1%} improvement at {ui['suggested_units']} units."
        if ui.get('ward_median') is not None:
            evidence += f" Ward median is {ui['ward_median']} units for approved projects."

        recommendations.append({
            "action": f"Reduce from {proposed_units} to {ui['suggested_units']} units",
            "current": f"{proposed_units} units proposed",
            "suggested": f"Reduce to {ui['suggested_units']} units",
            "probability_impact": f"+{ui['delta']:.2%}",
            "new_probability": ui['new_probability'],
            "evidence": evidence,
            "difficulty": "high",
            "priority": 0,
            "_delta": ui['delta'],
        })

    if 'stories' in scale_impacts:
        si = scale_impacts['stories']
        recommendations.append({
            "action": f"Reduce from {proposed_stories} to {si['suggested_stories']} stories",
            "current": f"{proposed_stories} stories proposed",
            "suggested": f"Reduce to {si['suggested_stories']} stories",
            "probability_impact": f"+{si['delta']:.2%}",
            "new_probability": si['new_probability'],
            "evidence": f"Model predicts {si['delta']:+.1%} improvement at {si['suggested_stories']} stories.",
            "difficulty": "high",
            "priority": 0,
            "_delta": si['delta'],
        })

    # ---- 4. Variance combination analysis ----
    combo_analysis = _get_variance_combo_analysis(variances)
    if combo_analysis and combo_analysis.get('better_sub_combos'):
        best_sub = combo_analysis['better_sub_combos'][0]
        # This is the historical data perspective — model impact was already captured
        # in variance_impacts above. Only add if we haven't already recommended removing this variance.
        already_recommended = {r['action'].replace('Remove ', '').replace(' variance', '') for r in recommendations}
        if best_sub['removed'] not in already_recommended:
            recommendations.append({
                "action": f"Remove {best_sub['removed']} variance (combo analysis)",
                "current": f"All {len(variances)} variances: {combo_analysis['full_combo_rate']:.0%} approval",
                "suggested": f"Without {best_sub['removed']}: {best_sub['rate']:.0%} approval",
                "probability_impact": f"+{best_sub['improvement']:.0%}",
                "new_probability": None,  # This is historical, not model-computed
                "evidence": f"The combination of all {len(variances)} variances has {combo_analysis['full_combo_rate']:.0%} approval rate ({combo_analysis['full_combo_cases']} cases). Removing {best_sub['removed']} brings it to {best_sub['rate']:.0%} ({best_sub['cases']} cases).",
                "difficulty": "medium",
                "priority": 0,
                "_delta": best_sub['improvement'],
            })

    # ---- 5. Project type change ----
    if project_type and 'new_construction' in project_type.lower().replace(' ', '_'):
        alt_features = build_features_fn(
            parcel_row, proposed_use, variances, 'addition',
            ward, has_attorney, proposed_units, proposed_stories
        )
        alt_prob = _predict_prob(alt_features)
        if alt_prob is not None:
            delta = alt_prob - prediction_prob
            if delta > 0.01:
                # Get real rates
                evidence = f"Model predicts {delta:+.1%} improvement if classified as addition/renovation."
                if state.zba_df is not None:
                    df = state.zba_df[state.zba_df['decision_clean'].notna()]
                    if 'proj_new_construction' in df.columns and 'proj_addition' in df.columns:
                        nc = df[df['proj_new_construction'] == 1]
                        add = df[df['proj_addition'] == 1]
                        if len(nc) >= 10 and len(add) >= 10:
                            nc_rate = float((nc['decision_clean'] == 'APPROVED').mean())
                            add_rate = float((add['decision_clean'] == 'APPROVED').mean())
                            evidence += f" Historically, additions have {add_rate:.0%} approval ({len(add)} cases) vs new construction {nc_rate:.0%} ({len(nc)} cases)."

                recommendations.append({
                    "action": "Consider renovation/addition instead of new construction",
                    "current": "New construction",
                    "suggested": "Reframe as renovation or addition if feasible",
                    "probability_impact": f"+{delta:.2%}",
                    "new_probability": round(alt_prob, 3),
                    "evidence": evidence,
                    "difficulty": "medium",
                    "priority": 0,
                    "_delta": delta,
                })

    # ---- 6. Ward-specific use type advice ----
    ward_use_rates = _get_ward_use_type_rates(ward)
    if ward_use_rates and proposed_use:
        use_lower = proposed_use.lower()
        is_res = bool(pd.Series([use_lower]).str.contains(r'residential|dwelling|family|condo|apartment|housing').iloc[0])
        is_com = bool(pd.Series([use_lower]).str.contains(r'commercial|retail|office|restaurant|store|shop').iloc[0])

        if is_com and 'residential' in ward_use_rates and 'commercial' in ward_use_rates:
            res_rate = ward_use_rates['residential']['rate']
            com_rate = ward_use_rates['commercial']['rate']
            if res_rate > com_rate + 0.08:
                recommendations.append({
                    "action": "Consider mixed-use with residential component",
                    "current": f"Commercial use ({com_rate:.0%} approval in this ward)",
                    "suggested": f"Add residential component ({res_rate:.0%} approval in this ward)",
                    "probability_impact": "varies",
                    "new_probability": None,
                    "evidence": f"Ward {ward}: residential projects have {res_rate:.0%} approval ({ward_use_rates['residential']['cases']} cases) vs commercial {com_rate:.0%} ({ward_use_rates['commercial']['cases']} cases).",
                    "difficulty": "high",
                    "priority": 0,
                    "_delta": res_rate - com_rate,
                })

    # ---- 7. Timing advice ----
    timing = _get_timing_advice(ward)
    if timing and timing['spread'] >= 0.03:
        recommendations.append({
            "action": f"File during {timing['best_quarter']}",
            "current": "Filing timing not optimized",
            "suggested": f"Target {timing['best_quarter']} for filing",
            "probability_impact": f"+{timing['spread']:.0%} vs {timing['worst_quarter']}",
            "new_probability": None,
            "evidence": f"{timing['best_quarter']}: {timing['best_rate']:.0%} approval ({timing['best_cases']} cases). {timing['worst_quarter']}: {timing['worst_rate']:.0%} ({timing['worst_cases']} cases).",
            "difficulty": "low",
            "priority": 0,
            "_delta": timing['spread'],
        })

    # ---- Sort by impact and assign priorities ----
    recommendations.sort(key=lambda r: -r.get('_delta', 0))
    for i, rec in enumerate(recommendations):
        rec['priority'] = i + 1
        rec.pop('_delta', None)  # Remove internal sorting key

    # ---- Similar cases (approved and denied) ----
    similar_approved = _find_similar_approved(
        ward, variances, project_type, has_attorney, proposed_units, limit=5
    )
    similar_denied = _find_similar_denied(
        ward, variances, project_type, has_attorney, proposed_units, limit=3
    )

    # ---- Compute optimized probability (apply top 2 non-conflicting recs) ----
    optimized_prob = prediction_prob
    applied_recs = []
    for rec in recommendations:
        if rec.get('new_probability') is not None and len(applied_recs) < 2:
            optimized_prob = rec['new_probability']
            applied_recs.append(rec['priority'])

    # Better: compute the combined effect by applying the top changes together
    if len(recommendations) >= 2:
        # Build the "best scenario" with all easy/medium recommendations applied
        best_variances = list(variances)
        best_attorney = has_attorney
        best_units = proposed_units
        best_stories = proposed_stories
        best_project_type = project_type

        for rec in recommendations[:3]:  # Top 3 by impact
            action = rec['action'].lower()
            if 'remove' in action and 'variance' in action:
                # Extract variance name
                for v in variances:
                    if v.lower() in action:
                        best_variances = [x for x in best_variances if x != v]
                        break
            elif 'attorney' in action:
                best_attorney = True
            elif 'units' in action and 'reduce' in action:
                # Extract suggested units from the rec
                suggested = rec.get('suggested', '')
                import re
                nums = re.findall(r'(\d+)\s*units', suggested)
                if nums:
                    best_units = int(nums[0])
            elif 'stories' in action and 'reduce' in action:
                suggested = rec.get('suggested', '')
                nums = re.findall(r'(\d+)\s*stories', suggested)
                if nums:
                    best_stories = int(nums[0])

        if (best_variances != variances or best_attorney != has_attorney
                or best_units != proposed_units or best_stories != proposed_stories):
            opt_features = build_features_fn(
                parcel_row, proposed_use, best_variances, best_project_type,
                ward, best_attorney, best_units, best_stories
            )
            opt_prob = _predict_prob(opt_features)
            if opt_prob is not None:
                optimized_prob = round(opt_prob, 3)

    # ---- Build optimization summary ----
    if optimized_prob > prediction_prob + 0.005 and applied_recs:
        summary = (
            f"By implementing the top recommendations, your probability could increase "
            f"from {prediction_prob:.0%} to {optimized_prob:.0%}."
        )
    elif optimized_prob > prediction_prob + 0.005:
        summary = (
            f"Combining recommended changes could increase your probability "
            f"from {prediction_prob:.0%} to {optimized_prob:.0%}."
        )
    else:
        summary = "Your proposal is already well-positioned. Minor optimizations may be available."

    return {
        "current_probability": round(prediction_prob, 3),
        "recommendations": recommendations,
        "similar_approved": similar_approved,
        "similar_denied": similar_denied,
        "optimized_probability": round(optimized_prob, 3),
        "optimization_summary": summary,
    }
