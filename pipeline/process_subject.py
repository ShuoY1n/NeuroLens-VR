"""
One OASIS-TRT subject -> outputs/: PNG slice stacks, brain.glb, manifest.json.
Dataset paths: NeuroLens Context/10_Reference_Dataset_OASIS_TRT_Mindboggle.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from PIL import Image
from skimage.measure import marching_cubes

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "Dataset"

# Window T1 intensities by percentile before 8-bit PNG export.
WINDOW_LOW_PERCENTILE = 1.0
WINDOW_HIGH_PERCENTILE = 99.0

SLICE_INDEX_DECIMAL_PLACES = 4

# GLB looked blown-out white with viewer defaults; gray + matte helps.
MESH_BASE_COLOR_RGBA = (0.52, 0.49, 0.47, 1.0)
MESH_METALLIC = 0.0
MESH_ROUGHNESS = 0.82


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


def mesh_from_foreground_mask(
    foreground_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
) -> trimesh.Trimesh:
    mask = foreground_mask.astype(bool)
    if not np.any(mask):
        raise ValueError("Empty mask — cannot build mesh.")

    vertices, faces, normals, _ = marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=voxel_spacing_mm,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )

    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=list(MESH_BASE_COLOR_RGBA),
        metallicFactor=MESH_METALLIC,
        roughnessFactor=MESH_ROUGHNESS,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def build_manifest(
    dataset_folder_name: str,
    t1_voxels: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    slice_meta: dict[str, dict],
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

    return {
        "schemaVersion": 1,
        "datasetId": dataset_folder_name,
        "orientationNotes": (
            f"MNI152 paired T1 + DKT31_CMA labels; shape ({nx}, {ny}, {nz}); "
            f"spacing_mm {list(voxel_spacing_mm)}; mesh from labels>0 shell."
        ),
        "volume": {
            "dimensions": [nx, ny, nz],
            "voxelSpacingMm": list(voxel_spacing_mm),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process one OASIS-TRT subject into outputs/")
    parser.add_argument(
        "--subject",
        type=str,
        default="1",
        help="Subject id (1 -> OASIS-TRT-20-1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Output directory",
    )
    args = parser.parse_args()

    subject_id = args.subject.strip()
    subject_folder = f"OASIS-TRT-20-{subject_id}"

    t1_path = DATASET_ROOT / "OASIS-TRT-20_volumes" / subject_folder / "t1weighted_brain.MNI152.nii.gz"
    labels_path = (
        DATASET_ROOT
        / "OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2"
        / f"OASIS-TRT-20-{subject_id}_DKT31_CMA_labels_in_MNI152.nii.gz"
    )

    if not t1_path.is_file():
        raise SystemExit(f"Missing T1 volume: {t1_path}")
    if not labels_path.is_file():
        raise SystemExit(f"Missing label volume: {labels_path}")

    output_dir: Path = args.out
    (output_dir / "mesh").mkdir(parents=True, exist_ok=True)

    t1_image = nib.load(str(t1_path))
    labels_image = nib.load(str(labels_path))

    if t1_image.shape != labels_image.shape:
        raise SystemExit("T1 and label volume shapes differ — cannot pair slices with labels.")
    if not np.allclose(t1_image.affine, labels_image.affine):
        raise SystemExit("T1 and label affines differ — voxels are not aligned.")

    t1_voxels = np.asanyarray(t1_image.dataobj, dtype=np.float32)
    label_ids = np.asanyarray(labels_image.dataobj, dtype=np.int32)
    header_zooms = t1_image.header.get_zooms()[:3]
    voxel_spacing_mm = tuple(float(z) for z in header_zooms)

    volume_u8 = intensity_volume_to_display_u8(t1_voxels)
    slice_meta = export_slice_png_stacks(volume_u8, output_dir)

    brain_foreground = label_ids > 0
    brain_mesh = mesh_from_foreground_mask(brain_foreground, voxel_spacing_mm)
    brain_mesh.merge_vertices()
    brain_mesh.export(output_dir / "mesh" / "brain.glb")

    manifest = build_manifest(subject_folder, t1_voxels, voxel_spacing_mm, slice_meta)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)

    print(f"Wrote {manifest_path}, mesh, and slice stacks under {output_dir}.")


if __name__ == "__main__":
    main()
