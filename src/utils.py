"""Shared helpers for filesystem and string utilities."""

import os
from typing import Iterable


def ensure_dir(path: str) -> str:
    """Ensure a directory exists and return its absolute path."""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
