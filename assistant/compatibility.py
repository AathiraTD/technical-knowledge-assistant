"""Product-substrate compatibility: structure eligibility rules and gate retrieval.

This module loads a structured compatibility matrix and enables the retrieval
layer to exclude incompatible products before ranking, preventing false
positive matches.

DECISIONS 3 and 9 establish that compatibility is enforced by citation when
the matrix is unavailable, but when the matrix exists, explicit incompatibility
produces a specific refusal rather than a generic "not stated". This gates
retrieval deterministically before the model sees anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from . import observability as obs


@dataclass
class CompatibilityRule:
    """One product-substrate-location compatibility rule."""

    product: str
    substrate: str
    interior: bool = True
    exterior: bool = True
    recommended: bool = True
    caveats: list[str] | None = None

    def __post_init__(self):
        """Post-init processing."""
        if self.caveats is None:
            self.caveats = []

    def matches(
        self,
        product: str,
        substrate: str,
        location: Optional[str],
    ) -> bool:
        """Check if this rule matches the given product/substrate/location.

        Returns True only if product and substrate match AND location is supported.
        """
        if product.lower() != self.product.lower():
            return False
        if substrate.lower() != self.substrate.lower():
            return False

        # If no location specified, match if rule applies to at least one location
        if location is None:
            return self.interior or self.exterior

        # Location specified: must be explicitly supported by this rule
        location_lower = location.lower()
        if location_lower in ["interior", "internal", "inside"]:
            return self.interior
        elif location_lower in ["exterior", "external", "outside"]:
            return self.exterior
        else:
            # Unknown location: match if either interior or exterior is supported
            return self.interior or self.exterior


class CompatibilityMatrix:
    """Load and query product-substrate compatibility rules."""

    def __init__(self, rules: list[CompatibilityRule] | None = None):
        """Initialize with compatibility rules.

        Args:
            rules: List of CompatibilityRule objects
        """
        self.rules = rules or []

    @classmethod
    def load(cls, path: Path) -> CompatibilityMatrix:
        """Load compatibility matrix from JSON file.

        Args:
            path: Path to compatibility_matrix.json

        Returns:
            CompatibilityMatrix instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
            ValueError: If matrix format is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Compatibility matrix not found: {path}")

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {path}: {e}", e.doc, e.pos) from e

        if "matrix" not in data or not isinstance(data["matrix"], list):
            raise ValueError("compatibility matrix must have 'matrix' list at top level")

        rules = []
        for i, rule_data in enumerate(data["matrix"]):
            try:
                rule = CompatibilityRule(
                    product=rule_data["product"],
                    substrate=rule_data["substrate"],
                    interior=rule_data.get("interior", True),
                    exterior=rule_data.get("exterior", True),
                    recommended=rule_data.get("recommended", True),
                    caveats=rule_data.get("caveats", []),
                )
                rules.append(rule)
            except KeyError as e:
                raise ValueError(
                    f"Rule {i} missing required field: {e}"
                ) from e

        obs.event("compatibility_matrix_loaded", num_rules=len(rules))
        return cls(rules)

    def is_compatible(
        self,
        product: str,
        substrate: str,
        location: Optional[str] = None,
    ) -> Optional[bool]:
        """Check if a product is compatible with a substrate and location.

        Args:
            product: Product name
            substrate: Substrate type
            location: "interior" or "exterior" (None = unknown)

        Returns:
            True if compatible, False if incompatible (either not suitable or
            location not supported), None if unknown (no rule found)
        """
        for rule in self.rules:
            # Check if product and substrate match
            if product.lower() == rule.product.lower() and substrate.lower() == rule.substrate.lower():
                # Rule exists for this product+substrate combination
                if location is None:
                    # No location specified: return recommendation if rule applies to any location
                    if rule.interior or rule.exterior:
                        return rule.recommended
                else:
                    # Location specified: check if supported by this rule
                    location_lower = location.lower()
                    if location_lower in ["interior", "internal", "inside"]:
                        if rule.interior:
                            return rule.recommended
                        else:
                            return False  # Location not supported
                    elif location_lower in ["exterior", "external", "outside"]:
                        if rule.exterior:
                            return rule.recommended
                        else:
                            return False  # Location not supported
                    else:
                        # Unknown location: return recommendation if rule applies to any location
                        if rule.interior or rule.exterior:
                            return rule.recommended

        # No rule found for this product+substrate combination
        return None

    def get_caveats(
        self,
        product: str,
        substrate: str,
        location: Optional[str] = None,
    ) -> list[str]:
        """Get caveats for a product-substrate-location combination.

        Args:
            product: Product name
            substrate: Substrate type
            location: "interior" or "exterior" (None = unknown)

        Returns:
            List of caveat strings, empty if no rule matches. Caveats are returned
            even if location is not supported, because they explain the incompatibility.
        """
        for rule in self.rules:
            # Check if product and substrate match
            if product.lower() == rule.product.lower() and substrate.lower() == rule.substrate.lower():
                # Found matching rule for product+substrate: return caveats
                # (caveats explain the recommendation or incompatibility)
                return rule.caveats

        return []

    def exclude_incompatible(
        self,
        product: str,
        substrate: Optional[str],
        location: Optional[str],
    ) -> bool:
        """Determine if a product should be excluded from retrieval.

        Args:
            product: Product to check
            substrate: Substrate (None if unknown)
            location: Location (None if unknown)

        Returns:
            True if product is explicitly incompatible and should be excluded
        """
        if substrate is None:
            return False

        compatibility = self.is_compatible(product, substrate, location)
        return compatibility is False
