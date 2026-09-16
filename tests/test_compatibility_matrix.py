"""Tests for product-substrate compatibility matrix.

Safety-critical: retrieval gating, explicit incompatibility detection,
and caveat association.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from assistant.compatibility import CompatibilityMatrix, CompatibilityRule


@pytest.fixture
def sample_matrix_data():
    """Sample compatibility matrix data."""
    return {
        "matrix": [
            {
                "product": "Ultra",
                "substrate": "solid_brick",
                "interior": True,
                "exterior": True,
                "recommended": True,
                "caveats": ["requires substrate preparation"],
            },
            {
                "product": "Ultra",
                "substrate": "mgo_board",
                "interior": False,
                "exterior": False,
                "recommended": False,
                "caveats": ["contact us in writing"],
            },
            {
                "product": "Forte",
                "substrate": "solid_brick",
                "interior": True,
                "exterior": False,
                "recommended": True,
                "caveats": [],
            },
        ]
    }


@pytest.fixture
def matrix(sample_matrix_data):
    """CompatibilityMatrix with sample data."""
    rules = [CompatibilityRule(**rule_data) for rule_data in sample_matrix_data["matrix"]]
    return CompatibilityMatrix(rules)


def test_load_matrix_from_file(sample_matrix_data):
    """Load compatibility matrix from JSON file."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "compatibility_matrix.json"
        path.write_text(json.dumps(sample_matrix_data))

        matrix = CompatibilityMatrix.load(path)
        assert len(matrix.rules) == 3


def test_load_matrix_file_not_found():
    """Loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CompatibilityMatrix.load(Path("/nonexistent/path.json"))


def test_load_matrix_invalid_json():
    """Invalid JSON raises JSONDecodeError."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.json"
        path.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            CompatibilityMatrix.load(path)


def test_load_matrix_missing_matrix_key():
    """Missing 'matrix' key raises ValueError."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "no_matrix.json"
        path.write_text(json.dumps({"data": []}))

        with pytest.raises(ValueError, match="matrix"):
            CompatibilityMatrix.load(path)


def test_load_matrix_missing_required_field():
    """Missing required field in rule raises ValueError."""
    with TemporaryDirectory() as tmpdir:
        data = {
            "matrix": [
                {
                    "product": "Ultra",
                    # Missing substrate
                    "recommended": True,
                }
            ]
        }
        path = Path(tmpdir) / "incomplete.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="substrate"):
            CompatibilityMatrix.load(path)


def test_is_compatible_recommended_product(matrix):
    """Recommended compatibility returns True."""
    assert matrix.is_compatible("Ultra", "solid_brick", "interior") is True
    assert matrix.is_compatible("Ultra", "solid_brick", "exterior") is True
    assert matrix.is_compatible("Forte", "solid_brick", "interior") is True


def test_is_compatible_incompatible_product(matrix):
    """Incompatible product returns False."""
    assert matrix.is_compatible("Ultra", "mgo_board", "interior") is False
    assert matrix.is_compatible("Ultra", "mgo_board", "exterior") is False


def test_is_compatible_interior_only(matrix):
    """Interior-only product respects location."""
    assert matrix.is_compatible("Forte", "solid_brick", "interior") is True
    assert matrix.is_compatible("Forte", "solid_brick", "exterior") is False


def test_is_compatible_unknown_product(matrix):
    """Unknown product returns None."""
    assert matrix.is_compatible("UnknownProduct", "solid_brick", "interior") is None


def test_is_compatible_unknown_substrate(matrix):
    """Unknown substrate returns None."""
    assert matrix.is_compatible("Ultra", "unknown_substrate", "interior") is None


def test_is_compatible_unknown_location(matrix):
    """Unknown location uses both interior and exterior."""
    # Ultra is available both interior and exterior on solid_brick
    assert matrix.is_compatible("Ultra", "solid_brick", None) is True
    # Forte is only interior on solid_brick, so None should still match
    assert matrix.is_compatible("Forte", "solid_brick", None) is True


def test_is_compatible_case_insensitive(matrix):
    """Compatibility check is case-insensitive."""
    assert matrix.is_compatible("ULTRA", "SOLID_BRICK", "INTERIOR") is True
    assert matrix.is_compatible("ultra", "solid_brick", "interior") is True
    assert matrix.is_compatible("UlTrA", "SOLID_BRICK", "EXTERIOR") is True


def test_get_caveats_returns_list(matrix):
    """Get caveats returns associated caveats."""
    caveats = matrix.get_caveats("Ultra", "solid_brick", "interior")
    assert isinstance(caveats, list)
    assert "substrate preparation" in caveats[0]


def test_get_caveats_incompatible(matrix):
    """Caveats for incompatible product are returned."""
    caveats = matrix.get_caveats("Ultra", "mgo_board", "interior")
    assert isinstance(caveats, list)
    assert "contact us" in caveats[0]


def test_get_caveats_unknown_returns_empty(matrix):
    """Unknown product returns empty caveats list."""
    caveats = matrix.get_caveats("UnknownProduct", "solid_brick", "interior")
    assert caveats == []


def test_exclude_incompatible_returns_true_for_bad_combo(matrix):
    """Exclude incompatible returns True for bad product-substrate combo."""
    assert matrix.exclude_incompatible("Ultra", "mgo_board", "interior") is True


def test_exclude_incompatible_returns_false_for_good_combo(matrix):
    """Exclude incompatible returns False for good product-substrate combo."""
    assert matrix.exclude_incompatible("Ultra", "solid_brick", "interior") is False


def test_exclude_incompatible_no_substrate_returns_false(matrix):
    """Without substrate, cannot determine exclusion (returns False)."""
    assert matrix.exclude_incompatible("Ultra", None, "interior") is False


def test_exclude_incompatible_unknown_substrate_returns_false(matrix):
    """Unknown substrate is not excluded."""
    assert matrix.exclude_incompatible("Ultra", "unknown_substrate", "interior") is False


def test_exclude_incompatible_unknown_product_returns_false(matrix):
    """Unknown product is not excluded."""
    assert matrix.exclude_incompatible("UnknownProduct", "solid_brick", "interior") is False


def test_compatibility_rule_match_exact(sample_matrix_data):
    """Compatibility rule matches exact product-substrate."""
    rule = CompatibilityRule(**sample_matrix_data["matrix"][0])
    assert rule.matches("Ultra", "solid_brick", "interior") is True


def test_compatibility_rule_no_match_different_product(sample_matrix_data):
    """Compatibility rule doesn't match different product."""
    rule = CompatibilityRule(**sample_matrix_data["matrix"][0])
    assert rule.matches("Forte", "solid_brick", "interior") is False


def test_compatibility_rule_no_match_different_substrate(sample_matrix_data):
    """Compatibility rule doesn't match different substrate."""
    rule = CompatibilityRule(**sample_matrix_data["matrix"][0])
    assert rule.matches("Ultra", "mgo_board", "interior") is False


def test_compatibility_rule_location_interior(sample_matrix_data):
    """Compatibility rule respects interior location."""
    rule = CompatibilityRule(**sample_matrix_data["matrix"][2])  # Forte interior-only
    assert rule.matches("Forte", "solid_brick", "interior") is True
    assert rule.matches("Forte", "solid_brick", "exterior") is False


def test_compatibility_rule_location_synonyms(sample_matrix_data):
    """Compatibility rule recognizes location synonyms."""
    rule = CompatibilityRule(**sample_matrix_data["matrix"][0])  # Ultra both locations
    assert rule.matches("Ultra", "solid_brick", "internal") is True
    assert rule.matches("Ultra", "solid_brick", "inside") is True
    assert rule.matches("Ultra", "solid_brick", "external") is True
    assert rule.matches("Ultra", "solid_brick", "outside") is True


def test_empty_matrix_returns_none(matrix):
    """Empty matrix returns None for any query."""
    empty = CompatibilityMatrix([])
    assert empty.is_compatible("Any", "Any", "Any") is None


def test_matrix_with_defaults(sample_matrix_data):
    """Matrix handles optional fields with defaults."""
    # Rule with no explicit interior/exterior/recommended should use defaults (True)
    rule_data = {
        "product": "Product",
        "substrate": "Substrate",
        # No interior, exterior, recommended, or caveats
    }
    rule = CompatibilityRule(**rule_data)
    assert rule.interior is True
    assert rule.exterior is True
    assert rule.recommended is True
    assert rule.caveats == []
