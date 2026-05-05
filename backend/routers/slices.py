"""Oblique slice PNG API."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from backend.oblique_sampling import sample_oblique_slice
from backend.volume_store import VolumeStore


def create_oblique_slice_router(volume_store: VolumeStore) -> APIRouter:
    router = APIRouter(tags=["slices"])

    @router.get(
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

        See backend/oblique_sampling.py for the exact orientation contract.
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

    return router
