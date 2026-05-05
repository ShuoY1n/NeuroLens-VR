"""Paths and tuning constants for the OASIS-TRT subject pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "Dataset"

WINDOW_LOW_PERCENTILE = 1.0
WINDOW_HIGH_PERCENTILE = 99.0
SLICE_INDEX_DECIMAL_PLACES = 4

MESH_BASE_COLOR_RGBA = (0.52, 0.49, 0.47, 1.0)
MESH_METALLIC = 0.0
MESH_ROUGHNESS = 0.82
