"""
Unit tests for PermitIQ pure-function modules.
These tests do NOT require a running API server.
Run with: python -m pytest tests/test_unit.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ===========================
# api/utils.py tests
# ===========================

from api.utils import normalize_address, safe_float, safe_int, safe_str


class TestNormalizeAddress:
    def test_basic_normalization(self):
        result = normalize_address("123 Main Street")
        assert "123" in result
        assert "main" in result.lower() or "MAIN" in result

    def test_avenue_normalization(self):
        """Ave and Avenue should normalize to the same thing."""
        r1 = normalize_address("10 Commonwealth Avenue")
        r2 = normalize_address("10 Commonwealth Ave")
        assert r1 == r2

    def test_directional_normalization(self):
        """East/West/North/South should normalize to E/W/N/S."""
        r1 = normalize_address("100 East Broadway")
        r2 = normalize_address("100 E Broadway")
        assert r1 == r2

    def test_zip_removal(self):
        """Zip codes should be stripped."""
        r1 = normalize_address("123 Main St")
        r2 = normalize_address("123 Main St 02101")
        assert r1 == r2

    def test_boston_neighborhood_removal(self):
        """Boston neighborhood names should be stripped."""
        r1 = normalize_address("123 Main St")
        r2 = normalize_address("123 Main St Dorchester")
        assert r1 == r2

    def test_suffix_letter(self):
        """Suffix letters like 18R should be preserved."""
        result = normalize_address("18R Centre St")
        assert "18" in result

    def test_range_address(self):
        """Range addresses like 55-57 should be handled."""
        result = normalize_address("55-57 Centre St")
        assert result  # Should not crash

    def test_empty_input(self):
        result = normalize_address("")
        assert result == ""

    def test_none_input(self):
        result = normalize_address(None)
        assert result == ""

    def test_mount_normalization(self):
        """Mount and MT should normalize to the same thing."""
        assert normalize_address("9 Mount Everett St") == normalize_address("9 MT EVERETT ST")


class TestSafeHelpers:
    def test_safe_float_valid(self):
        assert safe_float("3.14") == 3.14

    def test_safe_float_nan(self):
        import math
        assert safe_float(float('nan')) == 0.0

    def test_safe_float_none(self):
        assert safe_float(None) == 0.0

    def test_safe_float_default(self):
        assert safe_float("bad", default=-1.0) == -1.0

    def test_safe_int_valid(self):
        assert safe_int("42") == 42

    def test_safe_int_float_string(self):
        assert safe_int("3.7") == 3

    def test_safe_int_none(self):
        assert safe_int(None) == 0

    def test_safe_str_valid(self):
        assert safe_str("hello") == "hello"

    def test_safe_str_nan(self):
        assert safe_str(float('nan')) == ""

    def test_safe_str_none(self):
        assert safe_str(None) == ""


# ===========================
# api/constants.py tests
# ===========================

from api.constants import VARIANCE_TYPES, PROJECT_TYPES, FEATURE_LABELS


class TestConstants:
    def test_variance_types_count(self):
        assert len(VARIANCE_TYPES) == 12

    def test_variance_types_contents(self):
        assert "height" in VARIANCE_TYPES
        assert "far" in VARIANCE_TYPES
        assert "parking" in VARIANCE_TYPES
        assert "conditional_use" in VARIANCE_TYPES

    def test_project_types_count(self):
        assert len(PROJECT_TYPES) == 12

    def test_project_types_contents(self):
        assert "new_construction" in PROJECT_TYPES
        assert "demolition" in PROJECT_TYPES
        assert "adu" in PROJECT_TYPES

    def test_feature_labels_covers_key_features(self):
        """Every important feature should have a human-readable label."""
        key_features = [
            'has_attorney', 'ward_approval_rate', 'is_building_appeal',
            'proposed_units', 'num_variances', 'var_height', 'var_parking',
        ]
        for f in key_features:
            assert f in FEATURE_LABELS, f"Missing label for {f}"


# ===========================
# api/services/feature_builder.py tests
# ===========================

from api.services.feature_builder import FEATURE_COLS, REMOVED_FEATURES


class TestFeatureBuilder:
    def test_feature_count(self):
        """Model should have 85 features."""
        assert len(FEATURE_COLS) == 85, f"Expected 85 features, got {len(FEATURE_COLS)}"

    def test_no_leaked_features(self):
        """No post-hearing features should be in FEATURE_COLS."""
        for f in REMOVED_FEATURES:
            assert f not in FEATURE_COLS, f"Leaked feature found: {f}"

    def test_all_variance_features_present(self):
        """All 12 var_* features should be in the feature list."""
        for vt in VARIANCE_TYPES:
            assert f"var_{vt}" in FEATURE_COLS, f"Missing var_{vt}"

    def test_removed_features_documented(self):
        """Should have 14 removed features documented."""
        assert len(REMOVED_FEATURES) == 14

    def test_no_duplicate_features(self):
        """No duplicate features allowed."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplicate features found"


# ===========================
# api/routes/prediction.py tests
# ===========================

from api.routes.prediction import build_features, _auto_detect_ward


class TestBuildFeatures:
    def test_basic_output_keys(self):
        """build_features should return a dict with all expected feature columns."""
        result = build_features(
            parcel_row=None,
            proposed_use="residential",
            variances=["height", "parking"],
            project_type="new_construction",
            ward="7",
            has_attorney=True,
            proposed_units=4,
            proposed_stories=3,
        )
        assert isinstance(result, dict)
        assert 'num_variances' in result
        assert result['num_variances'] == 2
        assert result['var_height'] == 1
        assert result['var_parking'] == 1
        assert result['has_attorney'] == 1
        assert result['proposed_units'] == 4
        assert result['proposed_stories'] == 3

    def test_zero_variances(self):
        """No variances should produce zero counts."""
        result = build_features(
            parcel_row=None, proposed_use="commercial", variances=[],
        )
        assert result['num_variances'] == 0
        assert result['var_height'] == 0
        assert result['var_far'] == 0
        assert result['is_variance'] == 0

    def test_residential_detection(self):
        """Residential use should be detected."""
        result = build_features(parcel_row=None, proposed_use="two-family dwelling", variances=[])
        assert result['is_residential'] == 1
        assert result['is_commercial'] == 0

    def test_commercial_detection(self):
        """Commercial use should be detected."""
        result = build_features(parcel_row=None, proposed_use="retail store", variances=[])
        assert result['is_residential'] == 0
        assert result['is_commercial'] == 1

    def test_article_80_threshold(self):
        """Projects over 15 units or 4 stories should flag article_80."""
        result = build_features(parcel_row=None, proposed_use="residential", variances=[], proposed_units=20, proposed_stories=6)
        assert result['article_80'] == 1

    def test_below_article_80(self):
        """Small projects should not flag article_80."""
        result = build_features(parcel_row=None, proposed_use="residential", variances=[], proposed_units=3, proposed_stories=2)
        assert result['article_80'] == 0

    def test_interaction_features(self):
        """Interaction features should compute correctly."""
        result = build_features(
            parcel_row=None, proposed_use="residential",
            variances=["height"], has_attorney=True,
            proposed_stories=5,
        )
        assert result['interact_height_stories'] == 1 * 5
        assert result['interact_attorney_variances'] == 1 * 1

    def test_large_project_flag(self):
        """Large project should be flagged."""
        result = build_features(parcel_row=None, proposed_use="residential", variances=[], proposed_units=12, proposed_stories=6)
        assert result['large_project'] == 1

    def test_many_variances_flag(self):
        """3+ variances should set many_variances."""
        result = build_features(parcel_row=None, proposed_use="residential", variances=["height", "far", "parking"])
        assert result['many_variances'] == 1


class TestAutoDetectWard:
    def test_returns_none_with_no_data(self):
        """Should return None when no ZBA data available."""
        # state.zba_df is loaded but we pass empty district
        result = _auto_detect_ward("")
        assert result is None

    def test_returns_none_for_nonexistent_district(self):
        """Should return None for a district that doesn't exist."""
        result = _auto_detect_ward("ZZZZZZZ-FAKE-999")
        assert result is None
