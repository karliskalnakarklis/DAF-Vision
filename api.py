"""Local HTTP service exposing the puck detector.

Run:
    .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
Endpoints:
    GET  /health   — liveness + model info
    POST /detect   — multipart upload, returns puck count + coordinates
"""

import time
import logging
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from main import build_panel_mask, detect_stickers, detect_pucks


MODEL_INFO = {
    "type": "opencv-classical",
    "version": "1.0.0",
    "pipeline": ["panel_mask", "sticker_detection", "puck_detection"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daf-vision")

app = FastAPI(title="DAF Vision — Puck Detector", version=MODEL_INFO["version"])


class Puck(BaseModel):
    id: int
    x: int
    y: int
    radius: int


class Sticker(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelInfo(BaseModel):
    type: str
    version: str
    pipeline: List[str]


class DetectResponse(BaseModel):
    puck_count: int
    pucks: List[Puck]
    stickers: List[Sticker]
    image: ImageInfo
    model: ModelInfo
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": MODEL_INFO}


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...)):
    payload = await image.read()
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    started = time.perf_counter()
    panel_mask = build_panel_mask(frame)
    stickers = detect_stickers(frame)
    pucks = detect_pucks(frame, stickers, panel_mask)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    height, width = frame.shape[:2]
    response = {
        "puck_count": len(pucks),
        "pucks": [
            {"id": i, "x": int(x), "y": int(y), "radius": int(r)}
            for i, (x, y, r) in enumerate(pucks, start=1)
        ],
        "stickers": [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for (x, y, w, h) in stickers
        ],
        "image": {"width": int(width), "height": int(height)},
        "model": MODEL_INFO,
        "processing_time_ms": elapsed_ms,
    }
    log.info(
        "detect filename=%s bytes=%d pucks=%d stickers=%d elapsed_ms=%d",
        image.filename, len(payload), len(pucks), len(stickers), elapsed_ms,
    )
    return response
