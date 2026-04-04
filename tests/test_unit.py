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
