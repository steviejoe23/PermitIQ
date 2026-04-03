"""
Shared Feature Builder — used by both API prediction and model training.

CRITICAL: Every feature here must be knowable BEFORE the ZBA hearing.
If it describes what happened at/after the hearing, it's data leakage.

PRE-HEARING (SAFE):
  - Variance types and counts (from application)
  - Violation types (from zoning code analysis)
  - Use type (from application)
  - Attorney representation (known at filing)
  - BPDA involvement (pre-hearing review)
  - Project type (from application)
  - Legal articles (from zoning code)
  - Building scale (from application)
  - Ward/zoning approval rates (historical)
  - Attorney/contact win rates (historical)
  - Ward x Zoning, Year x Ward interaction rates (historical)
  - Property data (from assessor database)
  - Permit history (from building permits DB)

POST-HEARING (REMOVED — these leak):
  - has_opposition, no_opposition_noted (hearing testimony)
  - planning_support, planning_proviso (post-review)
  - community_process (describes what happened)
  - support_letters, opposition_letters, non_opposition_letter (from decision text)
  - councilor_involved, mayors_office_involved (hearing testimony)
  - hardship_mentioned (board's determination)
  - has_deferrals, num_deferrals (process events)
  - text_length_log (correlates with outcome)
"""

import re
import numpy as np


# Feature list — must match training pipeline exactly
FEATURE_COLS = [
    # Variance features (13)
    'num_variances',
    'var_height', 'var_far', 'var_lot_area', 'var_lot_frontage',
    'var_front_setback', 'var_rear_setback', 'var_side_setback',
    'var_parking', 'var_conditional_use', 'var_open_space',
    'var_density', 'var_nonconforming',

    # Violation types (5)
    'excessive_far', 'insufficient_lot', 'insufficient_frontage',
    'insufficient_yard', 'insufficient_parking',

    # Use type (2)
    'is_residential', 'is_commercial',

    # Representation (4)
    'has_attorney', 'bpda_involved', 'is_building_appeal', 'is_refusal_appeal',

    # Project types (12)
    'proj_demolition', 'proj_new_construction', 'proj_addition',
    'proj_conversion', 'proj_renovation', 'proj_subdivision',
    'proj_adu', 'proj_roof_deck', 'proj_parking',
    'proj_single_family', 'proj_multi_family', 'proj_mixed_use',

    # Legal framework (4)
    'article_80', 'is_conditional_use', 'is_variance', 'num_articles',

    # Building scale (2)
    'proposed_units', 'proposed_stories',

    # Location-based (6)
    'ward_approval_rate', 'zoning_approval_rate',
    'attorney_win_rate', 'contact_win_rate',
    'ward_zoning_rate', 'year_ward_rate',

    # Recency (1)
    'year_recency',

    # Property features (6)
    'lot_size_sf', 'total_value', 'property_age',
    'living_area', 'is_high_value', 'value_per_sqft',

    # Permit history (2)
    'prior_permits', 'has_prior_permits',

    # Interactions (3)
    'interact_height_stories', 'interact_attorney_variances', 'interact_highvalue_permits',

    # Log transforms (3)
    'lot_size_log', 'total_value_log', 'prior_permits_log',

    # Additional interactions (4)
    'contact_x_appeal', 'attorney_x_building',
    'many_variances', 'has_property_data',

    # Meta-features (3)
    'project_complexity', 'total_violations', 'num_features_active',

    # Variance interactions (3) — from application
    'var_height_and_far', 'var_parking_and_units', 'num_variances_sq',

    # Existing conditions (2) — from assessor DB
    'has_existing_parking', 'existing_parking_count',

    # Scale & density (4) — from application + assessor
    'units_log', 'large_project', 'units_per_lot_area', 'value_per_unit_log',

    # Application type (2) — from tracker description
    'is_change_occupancy', 'is_maintain_use',

    # Complexity signals (4) — engineered
    'multiple_setbacks', 'num_setback_variances',
    'interact_stories_far', 'complex_case_score',
]

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

# Post-hearing features that were removed (documented for transparency)
REMOVED_FEATURES = [
    'has_opposition', 'no_opposition_noted', 'planning_support',
    'planning_proviso', 'community_process', 'support_letters',
    'opposition_letters', 'non_opposition_letter', 'councilor_involved',
    'mayors_office_involved', 'hardship_mentioned', 'has_deferrals',
    'num_deferrals', 'text_length_log',
]
