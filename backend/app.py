"""
Surgery Phase Detection — FastAPI Backend
==========================================

REST API that serves the surgery phase detection model.
Accepts image/video uploads and returns phase predictions with
confidence scores and timeline data.
"""

import gc
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# ---- Memory optimisations for constrained environments ----
torch.set_num_threads(1)
torch.set_grad_enabled(False)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery_phase_detection.data.transforms import get_val_transforms
from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHASE_NAMES = [
    "Idle / Empty",
    "Preparation",
    "Procedure Active",
    "Closure / Cleanup",
]

PHASE_COLORS = ["#94a3b8", "#3b82f6", "#ef4444", "#22c55e"]

PHASE_DESCRIPTIONS = [
    "No activity detected — the operating room appears empty or idle.",
    "The team is preparing — patient positioning, equipment setup, clinicians arriving.",
    "Active procedure — multiple clinicians engaged in the surgical intervention.",
    "Winding down — procedure complete, cleanup and clinician departure in progress.",
]

NUM_CLASSES = len(PHASE_NAMES)
DEVICE = torch.device("cpu")  # Always CPU on free tier
MODEL_LOADED = False
model = None
transform = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Surgery Phase Detection API",
    description="Detects surgical phases from operating room images and video.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PhaseResult(BaseModel):
    phase_id: int
    phase_name: str
    confidence: float
    color: str
    description: str


class PredictionResponse(BaseModel):
    phase: PhaseResult
    all_phases: List[PhaseResult]
    inference_time_ms: float


class TimelineEntry(BaseModel):
    frame_index: int
    time_seconds: float
    phase_id: int
    phase_name: str
    confidence: float
    color: str


class VideoResponse(BaseModel):
    timeline: List[TimelineEntry]
    summary: dict
    total_frames: int
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model():
    """Load the model. Uses a lightweight ResNet-18 for fast CPU inference."""
    global model, transform, MODEL_LOADED

    print(f"Loading model on {DEVICE}...")
    model = SurgeryPhaseNet(
        num_classes=NUM_CLASSES,
        backbone="resnet18",
        pretrained=True,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.3,
        freeze_backbone=False,
    )
    model.eval()
    model.to(DEVICE)

    # Check for a trained checkpoint
    ckpt_path = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt
    else:
        print("No checkpoint found — using pretrained backbone (demo mode).")

    transform = get_val_transforms(image_size=224)
    MODEL_LOADED = True
    gc.collect()
    print("Model ready.")


@app.on_event("startup")
async def startup():
    load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def predict_image(image: Image.Image) -> dict:
    """Run inference on a single PIL image."""
    img_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,C,H,W)

    with torch.inference_mode():
        logits, _ = model(img_tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    del img_tensor, logits
    pred_id = int(np.argmax(probs))

    all_phases = []
    for i in range(NUM_CLASSES):
        all_phases.append(PhaseResult(
            phase_id=i,
            phase_name=PHASE_NAMES[i],
            confidence=round(float(probs[i]), 4),
            color=PHASE_COLORS[i],
            description=PHASE_DESCRIPTIONS[i],
        ))

    return {
        "phase": all_phases[pred_id],
        "all_phases": sorted(all_phases, key=lambda x: -x.confidence),
    }


def predict_video_frames(frames: List[Image.Image], fps: float) -> dict:
    """Run inference on a list of video frames with temporal context."""
    seq_len = 5
    timeline = []

    # Transform all frames up-front, then free PIL images
    tensors = [transform(f) for f in frames]
    del frames
    gc.collect()

    for i in range(len(tensors)):
        start = max(0, i - seq_len + 1)
        seq = tensors[start:i + 1]
        while len(seq) < seq_len:
            seq.insert(0, seq[0])

        batch = torch.stack(seq).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

        with torch.inference_mode():
            logits, _ = model(batch)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        del batch, logits
        pred_id = int(np.argmax(probs))
        timeline.append(TimelineEntry(
            frame_index=i,
            time_seconds=round(i / max(fps, 1), 2),
            phase_id=pred_id,
            phase_name=PHASE_NAMES[pred_id],
            confidence=round(float(probs[pred_id]), 4),
            color=PHASE_COLORS[pred_id],
        ))

    del tensors
    gc.collect()

    # Temporal smoothing
    if len(timeline) > 5:
        labels = [t.phase_id for t in timeline]
        smoothed = _majority_smooth(labels, window=5)
        for i, t in enumerate(timeline):
            t.phase_id = smoothed[i]
            t.phase_name = PHASE_NAMES[smoothed[i]]
            t.color = PHASE_COLORS[smoothed[i]]

    # Summary
    summary = {}
    for i, name in enumerate(PHASE_NAMES):
        count = sum(1 for t in timeline if t.phase_id == i)
        summary[name] = {
            "frame_count": count,
            "percentage": round(count / max(len(timeline), 1) * 100, 1),
            "color": PHASE_COLORS[i],
        }

    return {"timeline": timeline, "summary": summary}


def _majority_smooth(labels, window=5):
    smoothed = labels.copy()
    half = window // 2
    for i in range(len(labels)):
        s, e = max(0, i - half), min(len(labels), i + half + 1)
        w = labels[s:e]
        vals, counts = np.unique(w, return_counts=True)
        smoothed[i] = int(vals[np.argmax(counts)])
    return smoothed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": MODEL_LOADED, "device": str(DEVICE)}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}


@app.get("/api/phases")
async def get_phases():
    """Return all phase definitions."""
    return [
        {
            "id": i,
            "name": PHASE_NAMES[i],
            "color": PHASE_COLORS[i],
            "description": PHASE_DESCRIPTIONS[i],
        }
        for i in range(NUM_CLASSES)
    ]


@app.post("/api/predict/image", response_model=PredictionResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    """Predict surgical phase from a single image."""
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded yet.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        del contents
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    t0 = time.time()
    result = predict_image(image)
    elapsed = (time.time() - t0) * 1000

    del image
    gc.collect()

    return PredictionResponse(
        phase=result["phase"],
        all_phases=result["all_phases"],
        inference_time_ms=round(elapsed, 1),
    )


@app.post("/api/predict/video", response_model=VideoResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    sample_rate: int = 25,
    max_frames: int = 100,
):
    """Predict surgical phases from a video file."""
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded yet.")

    # Save uploaded video to temp file
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Cannot open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Downscale large frames immediately to save memory
                pil_img = Image.fromarray(rgb)
                if pil_img.width > 320 or pil_img.height > 320:
                    pil_img.thumbnail((320, 320), Image.LANCZOS)
                frames.append(pil_img)
                del rgb
            del frame
            idx += 1

        cap.release()
        del cap
        gc.collect()

        if not frames:
            raise HTTPException(400, "No frames extracted from video.")

        t0 = time.time()
        result = predict_video_frames(frames, fps / sample_rate)
        elapsed = (time.time() - t0) * 1000

        gc.collect()

        return VideoResponse(
            timeline=result["timeline"],
            summary=result["summary"],
            total_frames=len(result["timeline"]),
            inference_time_ms=round(elapsed, 1),
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/predict/batch")
async def predict_batch_endpoint(files: List[UploadFile] = File(...)):
    """Predict phases for a batch of images (treated as a sequence)."""
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded yet.")

    images = []
    for f in files:
        try:
            contents = await f.read()
            images.append(Image.open(io.BytesIO(contents)).convert("RGB"))
            del contents
        except Exception:
            raise HTTPException(400, f"Invalid image: {f.filename}")

    t0 = time.time()
    result = predict_video_frames(images, fps=1.0)
    elapsed = (time.time() - t0) * 1000

    gc.collect()

    return {
        "timeline": [t.dict() for t in result["timeline"]],
        "summary": result["summary"],
        "total_frames": len(images),
        "inference_time_ms": round(elapsed, 1),
    }
