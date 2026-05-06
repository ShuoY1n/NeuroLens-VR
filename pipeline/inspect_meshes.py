"""Inspect triangle counts / sizes of exported GLBs.

Usage:
  .\\.venv\\Scripts\\python.exe pipeline/inspect_meshes.py
"""

from __future__ import annotations

from pathlib import Path

import trimesh


def scene_face_vert_counts(path: Path) -> tuple[int, int, int]:
    scene = trimesh.load(path, force="scene")
    faces = 0
    verts = 0
    geoms = 0
    for g in scene.geometry.values():
        geoms += 1
        if getattr(g, "faces", None) is not None:
            faces += len(g.faces)
        if getattr(g, "vertices", None) is not None:
            verts += len(g.vertices)
    return geoms, faces, verts


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    outputs = repo_root / "outputs"
    glbs = sorted(outputs.rglob("*.glb"))
    if not glbs:
        raise SystemExit("No GLBs found under outputs/. Run pipeline/process_subject.py first.")

    for p in glbs:
        geoms, faces, verts = scene_face_vert_counts(p)
        size_mb = p.stat().st_size / (1024 * 1024)
        rel = p.relative_to(repo_root).as_posix()
        print(f"{rel:45s} geoms={geoms:2d} faces={faces:9d} verts={verts:9d} size={size_mb:6.2f} MB")


if __name__ == "__main__":
    main()

