"""
Dev server: static/ + outputs/ + dynamic oblique slice API.
Use --host 0.0.0.0 for Quest on LAN.
See NeuroLens Context/03_System_Architecture.md.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.oblique import VolumeStore, sample_oblique_slice

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDEX_HTML = STATIC_DIR / "index.html"
PREVIEW_HTML = STATIC_DIR / "preview.html"
SLICES_HTML = STATIC_DIR / "slices.html"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroLens VR dev server")
volume_store = VolumeStore(OUTPUTS_DIR)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/", response_model=None)
def root() -> FileResponse | RedirectResponse:
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML)
    if PREVIEW_HTML.is_file():
        return FileResponse(PREVIEW_HTML)
    return RedirectResponse(url="/docs")


@app.get("/preview.html", response_model=None)
def preview_page() -> FileResponse | RedirectResponse:
    if PREVIEW_HTML.is_file():
        return FileResponse(PREVIEW_HTML)
    return RedirectResponse(url="/docs")


@app.get("/slices.html", response_model=None)
def slices_page() -> FileResponse | RedirectResponse:
    if SLICES_HTML.is_file():
        return FileResponse(SLICES_HTML)
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, bool]:
    return {
        "ok": True,
        "outputs_dir_exists": OUTPUTS_DIR.is_dir(),
    }


@app.get(
    "/api/slice/oblique",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
def oblique_slice(
    nx: float = Query(..., description="Plane normal x (volume-local mm)."),
    ny: float = Query(..., description="Plane normal y (volume-local mm)."),
    nz: float = Query(..., description="Plane normal z (volume-local mm)."),
    offset: float = Query(0.0, description="Signed mm along normal from volume center."),
    size: int = Query(192, ge=64, le=1024, description="Output PNG side length in pixels."),
) -> Response:
    """Return a deterministically oriented oblique slice as PNG.

    See backend/oblique.py for the exact orientation contract.
    """
    try:
        volume = volume_store.get()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    normal = np.array([nx, ny, nz], dtype=np.float64)
    if not np.isfinite(normal).all() or float(np.linalg.norm(normal)) < 1e-6:
        raise HTTPException(status_code=400, detail="Invalid plane normal.")

    try:
        image_u8 = sample_oblique_slice(volume, normal, offset_mm=offset, size_px=size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = BytesIO()
    Image.fromarray(image_u8, mode="L").save(buffer, format="PNG", optimize=False)
    buffer.seek(0)
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
