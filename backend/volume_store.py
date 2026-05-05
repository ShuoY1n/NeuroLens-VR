"""Lazy, mtime-aware cache for windowed volume.npy + manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np

from backend.volume_data import VolumeData


class VolumeStore:
    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = Lock()
        self._cached: VolumeData | None = None
        self._cache_signature: tuple[float, float] | None = None

    def get(self) -> VolumeData:
        npy_path = self._outputs_dir / "volume.npy"
        manifest_path = self._outputs_dir / "manifest.json"
        if not npy_path.is_file():
            raise FileNotFoundError(
                f"Missing {npy_path}. Run pipeline/process_subject.py first."
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run pipeline/process_subject.py first."
            )

        signature = (
            npy_path.stat().st_mtime,
            manifest_path.stat().st_mtime,
        )
        with self._lock:
            if self._cached is not None and self._cache_signature == signature:
                return self._cached

            voxels = np.load(npy_path, allow_pickle=False)
            if voxels.dtype != np.uint8 or voxels.ndim != 3:
                raise ValueError(
                    "Expected uint8 3D array in volume.npy "
                    f"(got dtype={voxels.dtype}, shape={voxels.shape})."
                )
            with manifest_path.open("r", encoding="utf-8") as file_handle:
                manifest = json.load(file_handle)
            spacing = tuple(float(v) for v in manifest["volume"]["voxelSpacingMm"])
            if len(spacing) != 3:
                raise ValueError(
                    f"manifest.volume.voxelSpacingMm must have 3 entries (got {spacing})."
                )
            dataset_id = str(manifest.get("datasetId", "unknown"))

            self._cached = VolumeData(
                voxels_u8=voxels,
                spacing_mm=spacing,  # type: ignore[arg-type]
                dataset_id=dataset_id,
            )
            self._cache_signature = signature
            return self._cached
