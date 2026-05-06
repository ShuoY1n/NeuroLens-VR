"""Label-shell mesh export as PBR GLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label
from skimage.measure import marching_cubes

from config import MESH_BASE_COLOR_RGBA, MESH_METALLIC, MESH_ROUGHNESS

try:
    import pymeshlab  # type: ignore
except Exception:  # pragma: no cover
    pymeshlab = None


def _drop_huge_edge_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove faces with abnormally large edges.

    Some decimation runs can introduce long, skinny triangles that visually appear
    as 'spikes/lines' shooting away from the surface. This removes those outliers.
    """
    if mesh.faces is None or len(mesh.faces) == 0:
        return mesh
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    tri = v[f]
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    maxe = np.maximum(e01, np.maximum(e12, e20))

    # Threshold: 2x the 99.9th percentile, clamped to a reasonable minimum.
    thr = float(np.percentile(maxe, 99.9) * 2.0)
    thr = max(thr, 12.0)
    keep = maxe <= thr
    if bool(np.all(keep)):
        return mesh

    mesh2 = trimesh.Trimesh(vertices=v, faces=f[keep], process=False)
    mesh2.merge_vertices()
    try:
        mesh2.remove_degenerate_faces()
        mesh2.remove_duplicate_faces()
        mesh2.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh2


def mesh_from_foreground_mask(
    foreground_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    *,
    smooth_sigma_vox: float = 0.0,
    target_faces: int | None = None,
    keep_largest_component: bool = False,
) -> trimesh.Trimesh:
    mask = foreground_mask.astype(bool)
    if not np.any(mask):
        raise ValueError("Empty mask — cannot build mesh.")

    if smooth_sigma_vox and smooth_sigma_vox > 0:
        # Pre-smooth the binary mask for better silhouettes (reduces voxel stair-stepping).
        blurred = gaussian_filter(mask.astype(np.float32), sigma=float(smooth_sigma_vox))
        mask = blurred >= 0.5

    if keep_largest_component:
        # Remove tiny stray components that can turn into "spikes" after meshing/decimation.
        # 26-connectivity for 3D.
        structure = np.ones((3, 3, 3), dtype=np.int8)
        labeled, n = cc_label(mask, structure=structure)
        if n > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # background
            keep = int(np.argmax(counts))
            mask = labeled == keep

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

    # Basic cleanup before decimation.
    mesh.merge_vertices()
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        # trimesh API varies slightly across versions; keep best-effort.
        pass

    if target_faces is not None and int(target_faces) > 0:
        tf = int(target_faces)
        if len(mesh.faces) > tf:
            if pymeshlab is not None:
                try:
                    ms = pymeshlab.MeshSet()
                    m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
                    ms.add_mesh(m, "mesh")
                    # Quadric edge collapse with target face count.
                    ms.meshing_decimation_quadric_edge_collapse(
                        targetfacenum=tf,
                        preservenormal=True,
                        preservetopology=True,
                        optimalplacement=True,
                    )
                    m2 = ms.current_mesh()
                    mesh = trimesh.Trimesh(
                        vertices=np.asarray(m2.vertex_matrix(), dtype=np.float64),
                        faces=np.asarray(m2.face_matrix(), dtype=np.int64),
                        process=False,
                    )
                    mesh.merge_vertices()
                except Exception:
                    # Decimation is optional; proceed with the original mesh if it fails.
                    pass

    if keep_largest_component:
        # Decimation can introduce tiny detached fragments; drop everything but the main component.
        try:
            parts = mesh.split(only_watertight=False)
            if len(parts) > 1:
                mesh = max(parts, key=lambda m: len(m.faces))
                mesh.merge_vertices()
        except Exception:
            pass
        # Also drop rare long skinny triangles that render as "spikes".
        mesh = _drop_huge_edge_faces(mesh)

    # Recompute/fix normals after simplification.
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

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
    *,
    smooth_sigma_vox: float = 0.0,
    target_faces: int | None = None,
) -> None:
    brain_mesh = mesh_from_foreground_mask(
        brain_foreground,
        voxel_spacing_mm,
        smooth_sigma_vox=smooth_sigma_vox,
        target_faces=target_faces,
        keep_largest_component=True,
    )
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    brain_mesh.export(glb_path)


def export_structure_glbs(
    label_ids: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    output_dir: Path,
    *,
    structure_label_ids: list[int],
    label_name_by_id: dict[int, str] | None = None,
    smooth_sigma_vox: float = 0.0,
    target_faces: int | None = None,
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

        mesh = mesh_from_foreground_mask(
            mask,
            voxel_spacing_mm,
            smooth_sigma_vox=smooth_sigma_vox,
            target_faces=target_faces,
        )
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
    smooth_sigma_vox: float = 0.0,
    target_faces: int | None = None,
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

        mesh = mesh_from_foreground_mask(
            mask,
            voxel_spacing_mm,
            smooth_sigma_vox=smooth_sigma_vox,
            target_faces=target_faces,
        )
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
