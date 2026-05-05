"""Window T1 to uint8 and export cardinal slice PNG stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import SLICE_INDEX_DECIMAL_PLACES, WINDOW_HIGH_PERCENTILE, WINDOW_LOW_PERCENTILE


def intensity_volume_to_display_u8(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    sample = foreground if foreground.size > 0 else volume.ravel()
    low, high = np.percentile(sample, (WINDOW_LOW_PERCENTILE, WINDOW_HIGH_PERCENTILE))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(volume, low, high)
    scaled = (clipped - low) / (high - low) * 255.0
    return scaled.astype(np.uint8)


def save_grayscale_png(image_u8: np.ndarray, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u8, mode="L").save(file_path)


def _slice_plane_metadata(count: int, index_pad: int) -> dict:
    return {
        "count": count,
        "indexPad": index_pad,
        "defaultIndex": count // 2,
    }


def export_slice_png_stacks(
    volume_u8: np.ndarray,
    output_root: Path,
    index_pad: int = SLICE_INDEX_DECIMAL_PLACES,
) -> dict[str, dict]:
    # NIfTI from nibabel: shape (i, j, k). Coronal/sagittal: rot90 for display.
    size_i, size_j, size_k = volume_u8.shape
    pad = index_pad
    meta: dict[str, dict] = {}

    axial_dir = output_root / "slices" / "axial"
    for k in range(size_k):
        plate_ij = volume_u8[:, :, k]
        save_grayscale_png(plate_ij, axial_dir / f"{k:0{pad}d}.png")
    meta["axial"] = _slice_plane_metadata(size_k, index_pad)

    coronal_dir = output_root / "slices" / "coronal"
    for j in range(size_j):
        plate_ik = np.rot90(volume_u8[:, j, :], k=1)
        save_grayscale_png(plate_ik, coronal_dir / f"{j:0{pad}d}.png")
    meta["coronal"] = _slice_plane_metadata(size_j, index_pad)

    sagittal_dir = output_root / "slices" / "sagittal"
    for i in range(size_i):
        plate_jk = np.rot90(volume_u8[i, :, :], k=1)
        save_grayscale_png(plate_jk, sagittal_dir / f"{i:0{pad}d}.png")
    meta["sagittal"] = _slice_plane_metadata(size_i, index_pad)

    return meta
