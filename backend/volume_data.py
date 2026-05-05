"""Typed windowed volume loaded from outputs/ for oblique sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolumeData:
    voxels_u8: np.ndarray
    spacing_mm: tuple[float, float, float]
    dataset_id: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self.voxels_u8.shape[0]),
            int(self.voxels_u8.shape[1]),
            int(self.voxels_u8.shape[2]),
        )

    @property
    def center_mm(self) -> tuple[float, float, float]:
        ni, nj, nk = self.shape
        sx, sy, sz = self.spacing_mm
        return ((ni - 1) * sx / 2.0, (nj - 1) * sy / 2.0, (nk - 1) * sz / 2.0)

    @property
    def diagonal_mm(self) -> float:
        ni, nj, nk = self.shape
        sx, sy, sz = self.spacing_mm
        return float(np.sqrt((ni * sx) ** 2 + (nj * sy) ** 2 + (nk * sz) ** 2))
