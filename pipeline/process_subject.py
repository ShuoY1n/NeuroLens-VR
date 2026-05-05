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
from mesh_export import export_brain_glb, export_structure_glbs, export_structure_glbs_from_specs
from oasis_subject import require_existing_paths
from slice_export import export_slice_png_stacks, intensity_volume_to_display_u8
from label_mappings import load_label_mappings
from structure_presets import important_structures


def _parse_int_list(csv: str) -> list[int]:
    items: list[int] = []
    for part in csv.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid int in list: {p!r}") from exc
    return items


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
    parser.add_argument(
        "--structures",
        type=str,
        default="",
        help="Comma-separated label ids to export as individual GLBs (e.g. '4,10,23').",
    )
    parser.add_argument(
        "--structures-top",
        type=int,
        default=0,
        help="If >0, export the top-N non-zero labels by voxel count.",
    )
    parser.add_argument(
        "--structures-min-voxels",
        type=int,
        default=1500,
        help="Minimum voxel count for auto-selected structures (used with --structures-top).",
    )
    parser.add_argument(
        "--structures-important",
        action="store_true",
        help="Export a curated set of important subcortical structures (default if no other --structures* flags are provided).",
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

    structures: list[dict] = []
    # Optional label id -> name mapping for friendlier structure labels.
    label_name_by_id: dict[int, str] = {}
    try:
        mappings = load_label_mappings(REPO_ROOT)
        label_name_by_id = {k: v.name for k, v in mappings.items()}
    except FileNotFoundError:
        label_name_by_id = {}
    requested = _parse_int_list(args.structures) if args.structures else []
    if args.structures_top and args.structures_top > 0:
        vals, counts = np.unique(label_ids, return_counts=True)
        pairs = [
            (int(v), int(c))
            for v, c in zip(vals.tolist(), counts.tolist(), strict=False)
            if int(v) != 0 and int(c) >= int(args.structures_min_voxels)
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        requested.extend([v for v, _ in pairs[: int(args.structures_top)]])
    # De-dupe while preserving order.
    seen: set[int] = set()
    requested_unique: list[int] = []
    for lid in requested:
        if lid in seen:
            continue
        seen.add(lid)
        requested_unique.append(lid)

    if requested_unique:
        structures = export_structure_glbs(
            label_ids,
            voxel_spacing_mm,
            output_dir,
            structure_label_ids=requested_unique,
            label_name_by_id=label_name_by_id,
        )
    else:
        # Default behaviour: export curated important structures unless explicitly disabled by providing other selectors.
        if args.structures_important or (not args.structures_top and not args.structures):
            structures = export_structure_glbs_from_specs(
                label_ids,
                voxel_spacing_mm,
                output_dir,
                specs=important_structures(),
                label_name_by_id=label_name_by_id,
            )

    manifest = build_manifest(
        subject_folder, t1_voxels, voxel_spacing_mm, slice_meta, structures=structures
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    print(
        f"Wrote {manifest_path}, mesh, slice stacks, and {volume_npy_path.name} "
        f"under {output_dir}."
    )


if __name__ == "__main__":
    main()
