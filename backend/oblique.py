"""
Dynamic oblique slice sampling.

Resamples the windowed T1 volume on an arbitrary plane and returns a
deterministically oriented 2D image. Used by the /api/slice/oblique endpoint.

Conventions (volume-local, "centered mm" frame):
- Voxel (i, j, k) in `volume[i, j, k]` lives at mm
  `(i*sx - cx, j*sy - cy, k*sz - cz)`, where (cx, cy, cz) is the volume center.
- The plane is described by a unit normal `n` and a scalar `offset` along `n`
  from the volume center; the plane center is `offset * n`.
- The 2D image is sampled on a square grid in the plane's `(right, up)` basis:
    * column 0 -> -h * right    (left of image)
    * column W-1 -> +h * right  (right of image)
    * row 0 -> -h * up          (top of image)
    * row H-1 -> +h * up        (bottom of image)
  where `h = halfExtentMm` covers the volume diagonal so any plane orientation
  shows the full bounding cross-section.
- `(right, up)`: project **e_x**, then **e_y**, then **e_z** onto the plane
  perpendicular to `n` (first non-degenerate wins). At pitch 0, **e_x** gives
  stable `right = e_x` (avoids the pure-axial degeneracy of projecting e_z).

The resulting PNG is meant to be displayed with `texture.flipY = false` so that
plane local +X (right) maps to image right and plane local +Y (up) maps to
image top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.ndimage import map_coordinates


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


class VolumeStore:
    """Lazy, mtime-aware cache for the on-disk windowed volume + manifest."""

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


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, n_unit) orthonormal basis for the plane.

    `right` is the first non-degenerate Gram–Schmidt projection of **e_x**, **e_y**,
    **e_z** onto the plane perpendicular to `n`. Order matters: at pure axial
    (``n ≈ ±e_z``) projecting ``e_z`` is degenerate; **e_x** yields a stable
    ``right = e_x`` and removes pitch≈0 glitches. For general ``n`` this stays
    continuous (no |n·e_z| threshold).
    """
    n = normal.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("Plane normal must be a non-zero finite vector.")
    n_unit = n / norm

    candidates = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    right: np.ndarray | None = None
    for axis in candidates:
        t = axis - float(np.dot(axis, n_unit)) * n_unit
        tn = float(np.linalg.norm(t))
        if tn >= 1e-10:
            right = t / tn
            break
    if right is None:
        raise ValueError("Could not build in-plane basis (degenerate normal).")

    up = np.cross(n_unit, right)
    up /= float(np.linalg.norm(up))
    return right, up, n_unit


def sample_oblique_slice(
    volume: VolumeData,
    normal: np.ndarray,
    offset_mm: float,
    size_px: int,
    half_extent_mm: float | None = None,
) -> np.ndarray:
    """Sample a square oblique slice from `volume` and return it as uint8 (H, W).

    The image follows the orientation contract documented at the module top.
    """
    if size_px < 8 or size_px > 2048:
        raise ValueError(f"size_px out of range (8..2048): {size_px}")

    right, up, n_unit = plane_basis(normal)
    sx, sy, sz = volume.spacing_mm
    cx, cy, cz = volume.center_mm

    h = float(half_extent_mm) if half_extent_mm is not None else volume.diagonal_mm / 2.0

    # Image axis -> mm. Top row = -up, right col = +right.
    u_axis = np.linspace(-h, h, size_px, dtype=np.float64)
    v_axis = np.linspace(-h, h, size_px, dtype=np.float64)
    cols, rows = np.meshgrid(u_axis, v_axis, indexing="xy")

    plane_center_mm = offset_mm * n_unit
    world_x = plane_center_mm[0] + cols * right[0] + rows * up[0]
    world_y = plane_center_mm[1] + cols * right[1] + rows * up[1]
    world_z = plane_center_mm[2] + cols * right[2] + rows * up[2]

    vox_i = (world_x + cx) / sx
    vox_j = (world_y + cy) / sy
    vox_k = (world_z + cz) / sz

    sampled = map_coordinates(
        volume.voxels_u8,
        [vox_i, vox_j, vox_k],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.clip(sampled, 0, 255).astype(np.uint8, copy=False)
