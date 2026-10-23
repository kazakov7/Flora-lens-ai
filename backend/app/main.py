from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .inference import MODEL_ID, PlantClassifier

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 40_000_000


class Prediction(BaseModel):
    rank: int
    scientific_name: str
    species_id: str
    score: float


class PredictionResponse(BaseModel):
    model: str
    predictions: list[Prediction]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    model_ready: bool
    device: str


async def read_limited_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_image(contents: bytes) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format. Use JPG, PNG, or WebP",
                )
            source.verify()
        with Image.open(io.BytesIO(contents)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image") from exc


def create_app(classifier: PlantClassifier | None = None) -> FastAPI:
    service = classifier or PlantClassifier()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not os.getenv("FLORA_SKIP_MODEL_LOAD"):
            await run_in_threadpool(service.load)
        yield

    app = FastAPI(
        title="Flora Lens AI API",
        version="1.0.0",
        description="Plant species recognition powered by PlantCLEF 2024.",
        lifespan=lifespan,
    )
    app.state.classifier = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.status_code, "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": 422, "message": "An image file is required"}},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok" if service.ready else "starting",
            "model": service.model_id if hasattr(service, "model_id") else MODEL_ID,
            "model_ready": service.ready,
            "device": str(getattr(service, "device", "cpu")),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
        if not service.ready:
            raise HTTPException(status_code=503, detail="The model is not ready yet")
        contents = await read_limited_upload(file)
        image = await run_in_threadpool(decode_image, contents)
        try:
            predictions, inference_time_ms = await run_in_threadpool(service.predict, image, 5)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail="Plant recognition failed") from exc
        return {
            "model": service.model_id if hasattr(service, "model_id") else MODEL_ID,
            "predictions": predictions,
            "inference_time_ms": inference_time_ms,
        }

    return app


app = create_app()

