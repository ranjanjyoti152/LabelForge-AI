from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, List, Set
import hashlib
import uuid

import cv2
import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.core.config import Settings, get_settings
from app.models.sam3_detector import Detection as SamDetection
from app.models.sam3_detector import Sam3Detector
from app.schemas.detection import (
    BoundingBox,
    DetectRequest,
    DetectResponse,
    DetectionResult,
    LabelStudioPredictRequest,
    LabelStudioTask,
)
from app.services.rtsp import RtspFrameSampler, RtspStreamError
from app.services.labelstudio import get_labelstudio_concepts, clear_labels_cache

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="SAM3 Object Detector", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    """Initialize detector and create required directories on startup."""
    print("[STARTUP] Initializing SAM3 Auto Labeler...")
    settings = get_settings()
    
    # Create required directories
    required_dirs = [Path("./weights"), Path("./logs")]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[STARTUP] Directory ensured: {d}")
    
    # Pre-load the detector (downloads model if not available)
    print("[STARTUP] Loading SAM3 model (this may take a while on first run)...")
    try:
        _ = _get_detector()
        print("[STARTUP] SAM3 model loaded successfully!")
    except Exception as e:
        print(f"[STARTUP] Warning: Failed to pre-load model: {e}")
    
    print("[STARTUP] Server ready to accept requests!")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "max_concepts": settings.max_concepts_per_request,
            "score_threshold": settings.score_threshold,
            "max_frames": settings.max_frames,
            "frame_skip": settings.frame_skip,
        },
    )



def get_detector() -> Sam3Detector:
    return _get_detector()


@lru_cache()
def _get_detector() -> Sam3Detector:
    return Sam3Detector(get_settings())


@app.get("/healthz")
@app.get("/health")
@app.get("/predict/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect_objects(
    payload: DetectRequest,
    settings: Settings = Depends(get_settings),
    detector: Sam3Detector = Depends(get_detector),
) -> DetectResponse:
    concepts = payload.concepts
    if concepts and len(concepts) > settings.max_concepts_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many concepts requested. Max allowed is {settings.max_concepts_per_request}.",
        )

    sampler = RtspFrameSampler(
        payload.rtsp_url,
        timeout_seconds=settings.rtsp_timeout,
    )

    max_frames = payload.max_frames or settings.max_frames
    frame_skip = payload.frame_skip or settings.frame_skip

    try:
        sampled_frames = await run_in_threadpool(
            sampler.sample,
            max_frames,
            frame_skip,
        )
    except RtspStreamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    all_detections: List[SamDetection] = []
    for sampled in sampled_frames:
        detections = await run_in_threadpool(
            detector.detect,
            sampled.image,
            sampled.index,
            concepts,
        )
        all_detections.extend(detections)

    response_detections = [
        DetectionResult(
            frame_index=det.frame_index,
            label=det.label,
            bbox=BoundingBox(**det.bbox),
            area=det.area,
            score=det.score,
        )
        for det in all_detections
    ]

    return DetectResponse(
        frames_analyzed=len(sampled_frames),
        detections=response_detections,
    )


@app.get("/live-stream")
def live_stream(
    rtsp_url: str,
    frame_skip: int = Query(default=1, ge=1, le=600),
    concepts: List[str] | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    detector: Sam3Detector = Depends(get_detector),
) -> StreamingResponse:
    if not rtsp_url.startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="Only RTSP urls starting with rtsp:// are supported")

    concept_list = _normalize_concepts(concepts)
    if concept_list and len(concept_list) > settings.max_concepts_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many concepts requested. Max allowed is {settings.max_concepts_per_request}.",
        )

    capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise HTTPException(status_code=400, detail=f"Could not open RTSP stream {rtsp_url}")

    def frame_generator():
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue

                try:
                    detections = detector.detect(frame, frame_index, concept_list)
                except Exception:
                    detections = []

                annotated = _draw_detections(frame.copy(), detections)
                success, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if not success:
                    frame_index += 1
                    continue

                yield _build_mjpeg_frame(buffer.tobytes())
                frame_index += 1
        finally:
            capture.release()

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/predict")
@app.post("/label-studio/predict")
async def label_studio_predict(
    request: Request,
    settings: Settings = Depends(get_settings),
    detector: Sam3Detector = Depends(get_detector),
) -> dict[str, Any]:
    # Log the raw request body
    raw_body = await request.body()
    print(f"[DEBUG] Raw request body: {raw_body.decode('utf-8', errors='replace')}")
    
    import json
    try:
        body_json = json.loads(raw_body)
    except Exception as e:
        print(f"[DEBUG] Failed to parse JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Parse into our schema
    payload = LabelStudioPredictRequest(**body_json)
    
    predictions: List[dict[str, Any]] = []
    for task in payload.tasks:
        predictions.append(await _predict_for_task(task, settings, detector))
    
    # Label Studio expects {"results": [...]} format
    return {"results": predictions}


@app.post("/setup")
@app.post("/predict/setup")
async def label_studio_setup(payload: dict | None = None) -> dict[str, str]:
    # Clear the labels cache on setup to fetch fresh labels
    clear_labels_cache()
    return {"status": "ready"}


@app.post("/refresh-labels")
async def refresh_labels(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Refresh the cached Label Studio labels."""
    clear_labels_cache()
    labels = get_labelstudio_concepts(settings)
    return {"status": "ok", "labels": labels, "count": len(labels)}


@app.get("/labels")
async def get_labels(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Get the current Label Studio labels being used for detection."""
    labels = get_labelstudio_concepts(settings)
    return {"labels": labels, "count": len(labels)}


async def _predict_for_task(
    task: LabelStudioTask,
    settings: Settings,
    detector: Sam3Detector,
) -> dict[str, Any]:
    data = task.data or {}
    
    # Debug logging
    print(f"[DEBUG] Task ID: {task.id}")
    print(f"[DEBUG] Task data keys: {list(data.keys())}")

    # Try to get image from various Label Studio field names
    image_url = data.get("image") or data.get("img") or data.get("video") or data.get("rtsp_url")
    print(f"[DEBUG] Resolved image_url: {str(image_url)[:100] if image_url else None}")
    if not image_url:
        # Return empty prediction if no image source found
        return {
            "task": task.id,
            "result": [],
            "score": 0.0,
            "model_version": settings.labelstudio_model_version,
        }

    # Get concepts: first from task data, then from Label Studio project config
    concepts = _task_concepts(data)
    if not concepts:
        # Fetch concepts from Label Studio project labels - use EXACT labels only
        ls_concepts = get_labelstudio_concepts(settings)
        if ls_concepts:
            # Use exact labels only - no expansion to avoid false positives
            concepts = ls_concepts
            print(f"[DEBUG] Using {len(concepts)} exact labels from Label Studio")
    
    if concepts and len(concepts) > settings.max_concepts_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many concepts requested. Max allowed is {settings.max_concepts_per_request}.",
        )

    # Handle RTSP streams
    if isinstance(image_url, str) and image_url.startswith("rtsp://"):
        frame_skip = _task_frame_skip(data, settings.frame_skip)
        sampler = RtspFrameSampler(image_url, timeout_seconds=settings.rtsp_timeout)
        try:
            sampled_frames = await run_in_threadpool(sampler.sample, 1, frame_skip)
        except RtspStreamError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not sampled_frames:
            return {
                "task": task.id,
                "result": [],
                "score": 0.0,
                "model_version": settings.labelstudio_model_version,
            }
        frame = sampled_frames[0].image
    else:
        # Handle HTTP image URLs
        frame = await run_in_threadpool(_fetch_image_as_cv2, image_url, settings)
        if frame is None:
            return {
                "task": task.id,
                "result": [],
                "score": 0.0,
                "model_version": settings.labelstudio_model_version,
            }

    # Store original dimensions for result scaling
    orig_height, orig_width = frame.shape[:2]
    
    # Resize large images for faster inference
    frame, scale = _resize_image_if_needed(frame, settings.max_image_size)
    
    # Get valid labels from Label Studio for filtering results
    ls_labels = set(get_labelstudio_concepts(settings))
    
    detections = await run_in_threadpool(
        detector.detect,
        frame,
        0,
        concepts,
    )
    
    # Scale bounding boxes and area back to original image size if resized
    if scale != 1.0:
        for det in detections:
            det.bbox["x"] = int(det.bbox["x"] / scale)
            det.bbox["y"] = int(det.bbox["y"] / scale)
            det.bbox["width"] = int(det.bbox["width"] / scale)
            det.bbox["height"] = int(det.bbox["height"] / scale)
            det.area = int(det.area / (scale * scale))

    result_items = _format_label_studio_results(
        detections,
        orig_width,
        orig_height,
        settings.labelstudio_from_name,
        settings.labelstudio_to_name,
        ls_labels,
    )
    top_score = max((det.score for det in detections), default=0.0)

    return {
        "task": task.id,
        "result": result_items,
        "score": top_score,
        "model_version": settings.labelstudio_model_version,
    }


def _task_frame_skip(data: dict[str, Any], default_value: int) -> int:
    override = data.get("frame_skip")
    try:
        parsed = int(override)
    except (TypeError, ValueError):
        return default_value
    return max(1, parsed)


import base64


def _resize_image_if_needed(frame: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Resize image if larger than max_size, return (resized_frame, scale_factor)."""
    # Skip resizing if max_size is 0 or negative (use original size)
    if max_size <= 0:
        return frame, 1.0
    
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        return frame, 1.0
    
    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for downscaling (better quality)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _decode_base64_image(data_uri: str) -> np.ndarray | None:
    """Decode a base64 image (data URI or raw base64) to OpenCV BGR format."""
    try:
        # Handle data URI format: data:image/png;base64,xxxxx
        if data_uri.startswith("data:"):
            # Extract the base64 part after the comma
            header, encoded = data_uri.split(",", 1)
        else:
            # Assume raw base64 string
            encoded = data_uri
        
        # Decode base64
        img_bytes = base64.b64decode(encoded)
        
        # Convert to PIL Image, then to numpy BGR
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None


def _fetch_image_as_cv2(image_url: str, settings: Settings) -> np.ndarray | None:
    """Fetch an image from URL or decode base64, and convert to OpenCV BGR format."""
    try:
        # Handle base64 encoded images
        if image_url.startswith("data:") or _is_base64(image_url):
            return _decode_base64_image(image_url)
        
        # Handle Label Studio local file URLs
        if image_url.startswith("/data/") and settings.labelstudio_api_base:
            api_base = settings.labelstudio_api_base.rstrip("/")
            image_url = f"{api_base}{image_url}"

        headers = {}
        # Add auth for Label Studio hosted images
        if settings.labelstudio_api_token and settings.labelstudio_api_base:
            if image_url.startswith(settings.labelstudio_api_base):
                headers["Authorization"] = f"Token {settings.labelstudio_api_token}"

        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()

        # Convert to PIL Image, then to numpy BGR
        pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        print(f"Failed to fetch image from {image_url[:100]}...: {e}")
        return None


def _expand_concepts(labels: List[str]) -> List[str]:
    """Expand label list with synonyms and variations for better detection."""
    # Map each label to additional search terms
    expansions = {
        "car": ["car", "vehicle", "automobile", "sedan", "suv", "hatchback"],
        "truck": ["truck", "lorry", "pickup truck", "cargo truck", "goods vehicle"],
        "bus": ["bus", "minibus", "coach", "passenger bus", "shuttle"],
        "motorcycle": ["motorcycle", "motorbike", "scooter", "two wheeler", "bike"],
        "bicycle": ["bicycle", "cycle", "bike", "pushbike"],
        "auto rickshaw": ["auto rickshaw", "rickshaw", "auto", "three wheeler", "tuk tuk"],
        "person": ["person", "human", "man", "woman", "pedestrian", "people", "worker"],
        "helmet": ["helmet", "hard hat", "safety helmet", "head protection"],
        "License_Plate": ["license plate", "number plate", "vehicle plate", "registration plate"],
        "traffic light": ["traffic light", "traffic signal", "stoplight", "signal"],
        "fire": ["fire", "flames", "burning", "blaze"],
        "smoke": ["smoke", "fumes", "haze"],
        "ambulance": ["ambulance", "emergency vehicle", "medical vehicle"],
        "dog": ["dog", "puppy", "canine", "stray dog"],
        "chair": ["chair", "seat", "stool"],
    }
    
    expanded = []
    seen = set()
    for label in labels:
        label_lower = label.lower()
        # Add original
        if label not in seen:
            expanded.append(label)
            seen.add(label)
        # Add expansions
        for key, variants in expansions.items():
            if key.lower() == label_lower or label_lower in [v.lower() for v in variants]:
                for variant in variants:
                    if variant not in seen:
                        expanded.append(variant)
                        seen.add(variant)
    
    return expanded


def _is_base64(s: str) -> bool:
    """Check if a string looks like base64 encoded data."""
    # Quick heuristic: base64 strings are long, contain only valid chars, and no slashes like URLs
    if len(s) < 100:
        return False
    if s.startswith(("http://", "https://", "rtsp://", "/")):
        return False
    # Check if it contains only valid base64 characters
    import re
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
    # Check first 100 chars to avoid processing huge strings
    return bool(base64_pattern.match(s[:100]))


def _task_concepts(data: dict[str, Any]) -> List[str] | None:
    concepts = data.get("concepts")
    if concepts is None:
        return None
    if isinstance(concepts, str):
        raw = [concepts]
    elif isinstance(concepts, list):
        raw = [str(item) for item in concepts]
    else:
        return None
    return _normalize_concepts(raw)


def _format_label_studio_results(
    detections: List[SamDetection],
    width: int,
    height: int,
    from_name: str,
    to_name: str,
    valid_labels: Set[str] | None = None,
) -> List[dict[str, Any]]:
    if width <= 0 or height <= 0:
        return []

    # Map SAM3 labels to Label Studio labels (comprehensive synonyms)
    label_mapping = {
        # Fire related
        "flames": "fire",
        "flame": "fire",
        "spark": "fire",
        "sparks": "fire",
        "explosion": "fire",
        "blaze": "fire",
        "burning": "fire",
        "inferno": "fire",
        # Smoke related
        "fumes": "smoke",
        "steam": "smoke",
        "vapor": "smoke",
        "haze": "smoke",
        "smog": "smoke",
        # Vehicle mappings - Car
        "vehicle": "car",
        "automobile": "car",
        "sedan": "car",
        "suv": "car",
        "hatchback": "car",
        "coupe": "car",
        "jeep": "car",
        "taxi": "car",
        "cab": "car",
        # Auto rickshaw
        "auto": "auto rickshaw",
        "rickshaw": "auto rickshaw",
        "auto rickshaw": "auto rickshaw",
        "three wheeler": "auto rickshaw",
        "tuk tuk": "auto rickshaw",
        "tuktuk": "auto rickshaw",
        "autorickshaw": "auto rickshaw",
        "three-wheeler": "auto rickshaw",
        # Motorcycle
        "motorbike": "motorcycle",
        "scooter": "motorcycle",
        "moped": "motorcycle",
        "two wheeler": "motorcycle",
        "two-wheeler": "motorcycle",
        "motor bike": "motorcycle",
        "motor cycle": "motorcycle",
        "scooty": "motorcycle",
        "vespa": "motorcycle",
        # Bicycle
        "bike": "bicycle",
        "cycle": "bicycle",
        "pushbike": "bicycle",
        "push bike": "bicycle",
        "pedal bike": "bicycle",
        # Truck
        "lorry": "truck",
        "dump truck": "truck",
        "pickup truck": "truck",
        "pickup": "truck",
        "semi truck": "truck",
        "trailer": "truck",
        "tanker": "truck",
        "goods vehicle": "truck",
        "cargo truck": "truck",
        # Bus
        "van": "bus",
        "minibus": "bus",
        "coach": "bus",
        "shuttle": "bus",
        "school bus": "bus",
        "city bus": "bus",
        "passenger bus": "bus",
        "tempo": "bus",
        # Person mappings
        "human": "person",
        "people": "person",
        "man": "person",
        "woman": "person",
        "child": "person",
        "kid": "person",
        "boy": "person",
        "girl": "person",
        "pedestrian": "person",
        "worker": "person",
        "adult": "person",
        "individual": "person",
        "cyclist": "person",
        "rider": "person",
        "driver": "person",
        # Helmet mappings
        "hard hat": "helmet",
        "safety helmet": "helmet",
        "construction helmet": "helmet",
        "protective helmet": "helmet",
        "bike helmet": "helmet",
        "motorcycle helmet": "helmet",
        "crash helmet": "helmet",
        "head gear": "helmet",
        "headgear": "helmet",
        # License plate
        "license plate": "License_Plate",
        "number plate": "License_Plate",
        "license_plate": "License_Plate",
        "numberplate": "License_Plate",
        "registration plate": "License_Plate",
        "vehicle plate": "License_Plate",
        "car plate": "License_Plate",
        "plate": "License_Plate",
        "reg plate": "License_Plate",
        # Traffic light
        "traffic signal": "traffic light",
        "stoplight": "traffic light",
        "signal light": "traffic light",
        "traffic lamp": "traffic light",
        "signal": "traffic light",
        "red light": "traffic light",
        "green light": "traffic light",
        # Ambulance
        "emergency vehicle": "ambulance",
        "medical van": "ambulance",
        "paramedic vehicle": "ambulance",
        # Dog
        "puppy": "dog",
        "canine": "dog",
        "stray dog": "dog",
        "street dog": "dog",
        # Chair
        "seat": "chair",
        "stool": "chair",
        "bench": "chair",
        # Smoke
        "steam": "smoke",
        "fumes": "smoke",
    }
    
    # Use provided valid_labels or fall back to default set
    if not valid_labels:
        valid_labels = {
            "auto rickshaw", "car", "truck", "bus", "motorcycle", "bicycle",
            "ambulance", "person", "helmet", "chair", "dog", "fire", "smoke",
            "traffic light", "License_Plate"
        }

    results: List[dict[str, Any]] = []
    for det in detections:
        # Map label if needed
        label = det.label.lower().strip()
        mapped_label = label_mapping.get(label, det.label)
        
        # Skip if label not in valid labels (case-insensitive check)
        label_match = None
        for valid in valid_labels:
            if mapped_label.lower() == valid.lower():
                label_match = valid
                break
        
        # If no exact match, try the original label
        if not label_match:
            for valid in valid_labels:
                if det.label.lower() == valid.lower():
                    label_match = valid
                    break
        
        if not label_match:
            print(f"[DEBUG] Skipping unrecognized label: {det.label} -> {mapped_label}")
            continue
            
        bbox = det.bbox
        x0 = float(bbox["x"])
        y0 = float(bbox["y"])
        bw = float(bbox["width"])
        bh = float(bbox["height"])

        x0 = min(max(x0, 0.0), float(width))
        y0 = min(max(y0, 0.0), float(height))
        bw = min(max(bw, 1.0), max(1.0, float(width) - x0))
        bh = min(max(bh, 1.0), max(1.0, float(height) - y0))

        x_pct = (x0 / width) * 100.0
        y_pct = (y0 / height) * 100.0
        w_pct = (bw / width) * 100.0
        h_pct = (bh / height) * 100.0

        results.append(
            {
                "id": str(uuid.uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [label_match],
                },
                "score": det.score,
                "original_width": width,
                "original_height": height,
            }
        )

    return results


def _normalize_concepts(concepts: List[str] | None) -> List[str] | None:
    if not concepts:
        return None
    cleaned = [concept.strip() for concept in concepts if concept.strip()]
    return cleaned or None


def _build_mjpeg_frame(payload: bytes) -> bytes:
    return b"--frame\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def _draw_detections(frame, detections: List[SamDetection]):
    for det in detections:
        bbox = det.bbox
        x0 = max(0, int(bbox["x"]))
        y0 = max(0, int(bbox["y"]))
        x1 = x0 + int(bbox["width"])
        y1 = y0 + int(bbox["height"])
        color = _color_for_label(det.label)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        caption = f"{det.label} {det.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_y = max(y0, text_h + baseline)
        cv2.rectangle(
            frame,
            (x0, text_y - text_h - baseline - 2),
            (x0 + text_w + 6, text_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            caption,
            (x0 + 3, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return tuple(80 + (digest[i] % 160) for i in range(3))
