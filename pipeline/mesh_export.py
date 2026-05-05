"""Label-shell mesh export as PBR GLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from config import MESH_BASE_COLOR_RGBA, MESH_METALLIC, MESH_ROUGHNESS


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


def export_brain_glb(
    brain_foreground: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    glb_path: Path,
) -> None:
    brain_mesh = mesh_from_foreground_mask(brain_foreground, voxel_spacing_mm)
    brain_mesh.merge_vertices()
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    brain_mesh.export(glb_path)


def export_structure_glbs(
    label_ids: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    output_dir: Path,
    *,
    structure_label_ids: list[int],
    label_name_by_id: dict[int, str] | None = None,
) -> list[dict]:
    """Export one GLB per structure label id.

    Each structure is extracted as `label_ids == structure_label_id` and meshed
    with marching cubes in millimeter space.

    Returns a list of manifest-ready dicts:
    `{ id, label, meshUrl, sourceLabelId }`.
    """
    if label_ids.ndim != 3:
        raise ValueError(f"label_ids must be 3D (got shape={label_ids.shape}).")

    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    structures: list[dict] = []
    for lid in structure_label_ids:
        mask = label_ids == int(lid)
        if not np.any(mask):
            continue

        mesh = mesh_from_foreground_mask(mask, voxel_spacing_mm)
        mesh.merge_vertices()
        glb_name = f"label_{int(lid)}.glb"
        glb_path = structures_dir / glb_name
        mesh.export(glb_path)

        lid_int = int(lid)
        mapped_name = label_name_by_id.get(lid_int) if label_name_by_id else None
        structures.append(
            {
                "id": f"label_{lid_int}",
                "label": mapped_name or f"Label {lid_int}",
                "meshUrl": f"/outputs/structures/{glb_name}",
                "sourceLabelId": lid_int,
            }
        )

    return structures


def export_structure_glbs_from_specs(
    label_ids: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    output_dir: Path,
    *,
    specs: list[dict],
    label_name_by_id: dict[int, str] | None = None,
) -> list[dict]:
    """Export structure GLBs using explicit manifest specs.

    Each `spec` must contain:
    - `sourceLabelId` (int): label value in the label NIfTI
    - `label` (str): human-friendly name
    Optional:
    - `id` (str): stable identifier used by the viewer; defaults to `label_<id>`

    Output GLB name is always `label_<sourceLabelId>.glb` for stability across runs.
    """
    if label_ids.ndim != 3:
        raise ValueError(f"label_ids must be 3D (got shape={label_ids.shape}).")

    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    structures: list[dict] = []
    for spec in specs:
        if not isinstance(spec, dict) or "sourceLabelId" not in spec:
            continue
        lid = int(spec["sourceLabelId"])
        mask = label_ids == lid
        if not np.any(mask):
            continue

        mesh = mesh_from_foreground_mask(mask, voxel_spacing_mm)
        mesh.merge_vertices()
        glb_name = f"label_{lid}.glb"
        mesh.export(structures_dir / glb_name)

        mapped_name = label_name_by_id.get(lid) if label_name_by_id else None
        structures.append(
            {
                "id": str(spec.get("id", f"label_{lid}")),
                "label": str(spec.get("label") or mapped_name or f"Label {lid}"),
                "meshUrl": f"/outputs/structures/{glb_name}",
                "sourceLabelId": lid,
            }
        )

    return structures
