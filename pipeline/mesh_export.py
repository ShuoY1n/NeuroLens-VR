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
