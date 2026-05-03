"""
Dev server: serves static preview + generated outputs/ (manifest, GLB, PNGs).
Listen on 0.0.0.0 for Quest on LAN — see NeuroLens Context/03_System_Architecture.md
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroLens VR dev server")

if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


@app.get("/")
def root():
    preview = STATIC / "preview.html"
    if preview.is_file():
        return FileResponse(preview)
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"ok": True, "outputs_exists": OUTPUTS.is_dir()}
