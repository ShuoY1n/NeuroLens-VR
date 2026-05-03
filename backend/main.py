"""
Dev server: static/ + outputs/. Use --host 0.0.0.0 for Quest on LAN.
See NeuroLens Context/03_System_Architecture.md.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREVIEW_HTML = STATIC_DIR / "preview.html"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroLens VR dev server")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
def root() -> FileResponse | RedirectResponse:
    if PREVIEW_HTML.is_file():
        return FileResponse(PREVIEW_HTML)
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, bool]:
    return {
        "ok": True,
        "outputs_dir_exists": OUTPUTS_DIR.is_dir(),
    }
