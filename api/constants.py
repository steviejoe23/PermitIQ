"""
Static constants for PermitIQ API.

Single source of truth for variance types, project types, feature labels,
and disclaimer text. No dependencies.
"""

VARIANCE_TYPES = [
    'height', 'far', 'lot_area', 'lot_frontage',
    'front_setback', 'rear_setback', 'side_setback',
    'parking', 'conditional_use', 'open_space', 'density', 'nonconforming'
]

PROJECT_TYPES = [
    'demolition', 'new_construction', 'addition', 'conversion',
    'renovation', 'subdivision', 'adu', 'roof_deck', 'parking',
    'single_family', 'multi_family', 'mixed_use'
]

DISCLAIMER = (
    "IMPORTANT: This is a statistical risk assessment based on historical ZBA decisions, "
    "not a prediction or guarantee of outcome. Actual results depend on factors not captured "
    "in this model including board composition, public testimony, and site-specific conditions. "
    "This does not constitute legal, financial, or professional advice. Always consult a "
    "qualified zoning attorney before making financial commitments based on this analysis."
)

# Human-readable labels for model features (used by SHAP explainability)
FEATURE_LABELS = {
    # Variance features
    'num_variances': 'Number of variances requested',
    'var_height': 'Height variance requested',
    'var_far': 'Floor Area Ratio (FAR) variance',
    'var_lot_area': 'Lot area variance',
    'var_lot_frontage': 'Lot frontage variance',
    'var_front_setback': 'Front setback variance',
    'var_rear_setback': 'Rear setback variance',
    'var_side_setback': 'Side setback variance',
    'var_parking': 'Parking variance requested',
    'var_conditional_use': 'Conditional use permit',
    'var_open_space': 'Open space variance',
    'var_density': 'Density variance',
    'var_nonconforming': 'Nonconforming use extension',
    # Violation types
    'excessive_far': 'Exceeds allowed floor area ratio',
    'insufficient_lot': 'Lot smaller than required',
    'insufficient_frontage': 'Lot frontage below minimum',
    'insufficient_yard': 'Yard/setback below minimum',
    'insufficient_parking': 'Fewer parking spaces than required',
    # Use type
    'is_residential': 'Residential use',
    'is_commercial': 'Commercial use',
    # Representation
    'has_attorney': 'Attorney representation',
    'bpda_involved': 'BPDA review involved',
    'is_building_appeal': 'Building code appeal (vs. zoning)',
    'is_refusal_appeal': 'Appeal of a building permit refusal',
    # Project types
    'proj_demolition': 'Involves demolition',
    'proj_new_construction': 'New construction project',
    'proj_addition': 'Addition to existing building',
    'proj_conversion': 'Building conversion',
    'proj_renovation': 'Renovation project',
    'proj_subdivision': 'Land subdivision',
    'proj_adu': 'Accessory dwelling unit (ADU)',
    'proj_roof_deck': 'Roof deck addition',
    'proj_parking': 'Parking-related project',
    'proj_single_family': 'Single-family project',
    'proj_multi_family': 'Multi-family project',
    'proj_mixed_use': 'Mixed-use project',
    # Legal
    'article_80': 'Large project (Article 80 review)',
    'is_conditional_use': 'Conditional use application',
    'is_variance': 'Variance application',
    'num_articles': 'Number of zoning articles involved',
    # Scale
    'proposed_units': 'Number of proposed units',
    'proposed_stories': 'Number of proposed stories',
    # Location rates
    'ward_approval_rate': 'Historical approval rate in this ward',
    'zoning_approval_rate': 'Historical approval rate for this zoning district',
    'attorney_win_rate': 'Attorney track record at ZBA',
    'contact_win_rate': 'Applicant track record at ZBA',
    'ward_zoning_rate': 'Approval trend for this ward + zoning combination',
    'year_ward_rate': 'Recent approval trend in this ward',
    # Recency
    'year_recency': 'How recent the case is',
    # Property
    'lot_size_sf': 'Lot size (sq ft)',
    'total_value': 'Property assessed value',
    'property_age': 'Building age (years)',
    'living_area': 'Living area (sq ft)',
    'is_high_value': 'High-value property (>$1M)',
    'value_per_sqft': 'Property value per sq ft',
    # Permits
    'prior_permits': 'Number of prior building permits',
    'has_prior_permits': 'Has prior building permit history',
    # Interactions
    'interact_height_stories': 'Height variance x stories (scale of height issue)',
    'interact_attorney_variances': 'Attorney x variance count (complex case with representation)',
    'interact_highvalue_permits': 'High-value property with permit history',
    # Log transforms
    'lot_size_log': 'Lot size (log scale)',
    'total_value_log': 'Property value (log scale)',
    'prior_permits_log': 'Prior permits (log scale)',
    # Additional
    'contact_x_appeal': 'Applicant track record x appeal type',
    'attorney_x_building': 'Attorney x building appeal',
    'many_variances': 'Requesting 3+ variances',
    'has_property_data': 'Property assessment data available',
    # Meta
    'project_complexity': 'Overall project complexity score',
    'total_violations': 'Total number of zoning violations',
    'num_features_active': 'Number of active risk factors',
}
