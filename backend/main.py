"""
Dev server: static/ + outputs/ + dynamic oblique slice API.
Use --host 0.0.0.0 for Quest on LAN.
See NeuroLens Context/03_System_Architecture.md.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.routers.slices import create_oblique_slice_router
from backend.volume_store import VolumeStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDEX_HTML = STATIC_DIR / "index.html"
PREVIEW_HTML = STATIC_DIR / "preview.html"
SLICES_HTML = STATIC_DIR / "slices.html"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroLens VR dev server")
volume_store = VolumeStore(OUTPUTS_DIR)
app.include_router(create_oblique_slice_router(volume_store))

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
