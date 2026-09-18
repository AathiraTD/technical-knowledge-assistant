"""Filesystem anchors for the repository, resolved once.

Modules used to recompute the repository root from their own depth below
`assistant/`, so moving one into a subpackage silently repointed it at
`assistant/` instead. Anchoring the root here keeps that depth in a single
place, and a module that moves keeps reading the same directory.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "db"
