"""Validation utilities for configuration and payloads."""

from typing import Iterable


def validate_required_env(keys: Iterable[str], values: dict) -> list[str]:
    """Return list of missing keys based on provided values mapping."""
    missing = []
    for key in keys:
        if not values.get(key):
            missing.append(key)
    return missing
