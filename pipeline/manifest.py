"""manifest.json shape for static viewers and oblique API hints."""

from __future__ import annotations

import json
from pathlib import Path


def build_manifest(
    dataset_folder_name: str,
    t1_voxels: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    slice_meta: dict[str, dict],
    structures: list[dict] | None = None,
) -> dict:
    # Shape matches NeuroLens Context/05_Asset_Manifest_and_Contract.md
    nx, ny, nz = (int(t1_voxels.shape[0]), int(t1_voxels.shape[1]), int(t1_voxels.shape[2]))

    def slice_block(plane: str) -> dict:
        block = slice_meta[plane]
        return {
            "count": block["count"],
            "urlPattern": f"/outputs/slices/{plane}/{{index}}.png",
            "indexPad": block["indexPad"],
            "defaultIndex": block["defaultIndex"],
        }

    manifest: dict = {
        "schemaVersion": 1,
        "datasetId": dataset_folder_name,
        "orientationNotes": (
            f"MNI152 paired T1 + DKT31_CMA labels; shape ({nx}, {ny}, {nz}); "
            f"spacing_mm {list(voxel_spacing_mm)}; mesh from labels>0 shell."
        ),
        "volume": {
            "dimensions": [nx, ny, nz],
            "voxelSpacingMm": list(voxel_spacing_mm),
            "dataUrl": "/outputs/volume.npy",
        },
        "mesh": {
            "wholeBrainUrl": "/outputs/mesh/brain.glb",
            "units": "millimeters",
        },
        "slices": {
            "axial": slice_block("axial"),
            "coronal": slice_block("coronal"),
            "sagittal": slice_block("sagittal"),
        },
        "obliqueSlice": {
            "endpoint": "/api/slice/oblique",
            "params": {
                "nx": "float (volume-local mm; unit normal x)",
                "ny": "float (volume-local mm; unit normal y)",
                "nz": "float (volume-local mm; unit normal z)",
                "offset": "float (mm along normal from volume center)",
                "size": "int (output PNG side length, default 192)",
            },
        },
    }

    if structures:
        manifest["structures"] = structures

    return manifest


def write_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
