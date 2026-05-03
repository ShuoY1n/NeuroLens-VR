"""
Milestone 1 helper: OASIS-TRT subject → outputs/ (PNG stacks, brain.glb, manifest.json).

Paths match NeuroLens Context/10_Reference_Dataset_OASIS_TRT_Mindboggle.md
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

# --- defaults for this repo layout ---
REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "Dataset"


def _window_to_u8(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(vol[vol > 0] if np.any(vol > 0) else vol, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(vol, lo, hi)
    return ((x - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _save_slice_png(gray2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray2d, mode="L").save(path)


def _export_slices(
    vol_u8: np.ndarray,
    out_dir: Path,
    index_pad: int = 4,
) -> dict:
    """vol_u8 shape (i, j, k) as NiBabel loaded (axis 0,1,2)."""
    d0, d1, d2 = vol_u8.shape
    planes = {
        "axial": (d2, lambda k: vol_u8[:, :, k]),
        "coronal": (d1, lambda j: vol_u8[:, j, :]),
        "sagittal": (d0, lambda i: vol_u8[i, :, :]),
    }
    meta = {}
    for name, (count, getter) in planes.items():
        sub = out_dir / "slices" / name
        for idx in range(count):
            sl = getter(idx)
            if name == "coronal":
                sl = np.rot90(sl, k=1)
            elif name == "sagittal":
                sl = np.rot90(sl, k=1)
            _save_slice_png(sl, sub / f"{idx:0{index_pad}d}.png")
        meta[name] = {"count": count, "indexPad": index_pad, "defaultIndex": count // 2}
    return meta


def _mesh_from_binary(mask: np.ndarray, spacing: tuple[float, float, float]) -> trimesh.Trimesh:
    mask = mask.astype(bool)
    if not np.any(mask):
        raise ValueError("Empty mask — cannot build mesh.")
    verts, faces, normals, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(description="Process one OASIS-TRT subject into outputs/")
    p.add_argument("--subject", type=str, default="1", help="Subject id, e.g. 1 → OASIS-TRT-20-1")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs",
        help="Output directory (default: repo outputs/)",
    )
    args = p.parse_args()
    sid = args.subject.strip()
    subdir = f"OASIS-TRT-20-{sid}"

    t1_path = DATASET / "OASIS-TRT-20_volumes" / subdir / "t1weighted_brain.MNI152.nii.gz"
    lab_path = (
        DATASET
        / "OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2"
        / f"OASIS-TRT-20-{sid}_DKT31_CMA_labels_in_MNI152.nii.gz"
    )

    if not t1_path.is_file():
        raise SystemExit(f"Missing T1: {t1_path}")
    if not lab_path.is_file():
        raise SystemExit(f"Missing labels: {lab_path}")

    out: Path = args.out
    (out / "mesh").mkdir(parents=True, exist_ok=True)

    t1_img = nib.load(str(t1_path))
    lab_img = nib.load(str(lab_path))
    if t1_img.shape != lab_img.shape or not np.allclose(t1_img.affine, lab_img.affine):
        raise SystemExit("T1 and label shape/affine differ — fix inputs before continuing.")

    t1 = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    labels = np.asanyarray(lab_img.dataobj).astype(np.int32)
    zooms = t1_img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)

    vol_u8 = _window_to_u8(t1)
    slice_meta = _export_slices(vol_u8, out)

    mask = labels > 0
    mesh = _mesh_from_binary(mask, spacing=spacing)
    mesh.merge_vertices()
    mesh.export(out / "mesh" / "brain.glb")

    manifest = {
        "schemaVersion": 1,
        "datasetId": subdir,
        "orientationNotes": (
            f"MNI152 paired T1 + DKT31_CMA labels; shape {tuple(int(x) for x in t1.shape)}; "
            f"spacing_mm {list(spacing)}; mesh from labels>0 shell."
        ),
        "volume": {
            "dimensions": [int(t1.shape[0]), int(t1.shape[1]), int(t1.shape[2])],
            "voxelSpacingMm": list(spacing),
        },
        "mesh": {"wholeBrainUrl": "/outputs/mesh/brain.glb", "units": "millimeters"},
        "slices": {
            "axial": {
                "count": slice_meta["axial"]["count"],
                "urlPattern": "/outputs/slices/axial/{index}.png",
                "indexPad": slice_meta["axial"]["indexPad"],
                "defaultIndex": slice_meta["axial"]["defaultIndex"],
            },
            "coronal": {
                "count": slice_meta["coronal"]["count"],
                "urlPattern": "/outputs/slices/coronal/{index}.png",
                "indexPad": slice_meta["coronal"]["indexPad"],
                "defaultIndex": slice_meta["coronal"]["defaultIndex"],
            },
            "sagittal": {
                "count": slice_meta["sagittal"]["count"],
                "urlPattern": "/outputs/slices/sagittal/{index}.png",
                "indexPad": slice_meta["sagittal"]["indexPad"],
                "defaultIndex": slice_meta["sagittal"]["defaultIndex"],
            },
        },
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {out / 'manifest.json'} and mesh + slices.")


if __name__ == "__main__":
    main()
