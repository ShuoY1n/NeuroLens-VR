"""
One OASIS-TRT subject -> outputs/: PNG slice stacks, brain.glb, manifest.json.
Dataset paths: NeuroLens Context/10_Reference_Dataset_OASIS_TRT_Mindboggle.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from config import REPO_ROOT
from manifest import build_manifest, write_manifest
from mesh_export import export_brain_glb
from oasis_subject import require_existing_paths
from slice_export import export_slice_png_stacks, intensity_volume_to_display_u8


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

    t1_path, labels_path, subject_folder = require_existing_paths(args.subject)

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

    # Backend resamples this for oblique slices; .npy keeps shape/dtype faithfully.
    volume_npy_path = output_dir / "volume.npy"
    np.save(volume_npy_path, volume_u8, allow_pickle=False)

    export_brain_glb(label_ids > 0, voxel_spacing_mm, output_dir / "mesh" / "brain.glb")

    manifest = build_manifest(subject_folder, t1_voxels, voxel_spacing_mm, slice_meta)
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    print(
        f"Wrote {manifest_path}, mesh, slice stacks, and {volume_npy_path.name} "
        f"under {output_dir}."
    )


if __name__ == "__main__":
    main()
