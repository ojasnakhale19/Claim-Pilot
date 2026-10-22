"""FastAPI backend for ClaimPilot AI - OpenAI GPT-4o-mini vision only."""

from __future__ import annotations

import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


def _dhash(image: Image.Image) -> int:
    """Return a 64-bit difference hash for near-duplicate screening."""
    gray = ImageOps.fit(image.convert("L"), (9, 8)).resize((9, 8))
    pixels = np.asarray(gray, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(first: int, second: int) -> int:
    return bin(first ^ second).count("1")


# ─── Constants ──────────────────────────────────────────────────────────────
REQUIRED_VIEWS = ("top", "bottom", "left", "right")
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MIN_IMAGE_SIDE = int(os.getenv("AUTOSETTLE_MIN_IMAGE_SIDE", "240"))
MIN_LAPLACIAN_VARIANCE = float(os.getenv("AUTOSETTLE_MIN_LAPLACIAN_VARIANCE", "18"))
NEAR_DUPLICATE_DISTANCE = int(os.getenv("AUTOSETTLE_NEAR_DUPLICATE_DISTANCE", "3"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("AUTOSETTLE_LOW_CONFIDENCE_THRESHOLD", "0.40"))

VIDEO_FRAME_TARGET = 10
MIN_VIDEO_DURATION = 8.0
MAX_VIDEO_DURATION = 10.0
MAX_VIDEO_SIZE_MB = 100
MAX_PHOTO_SIZE_MB = 15
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}

# ─── Image Utilities ────────────────────────────────────────────────────────
def image_metrics(image: Image.Image) -> dict[str, Any]:
    """Return simple image quality measures."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    width, height = image.size
    return {
        "width": int(width),
        "height": int(height),
        "pixel_count": int(width * height),
        "laplacian_variance": round(blur_score, 2),
        "sha256_normalized_pixels": __import__("hashlib").sha256(rgb.tobytes()).hexdigest(),
    }


# ─── OpenAI Configuration ───────────────────────────────────────────────────
AI_PROVIDER = "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─── Cost Estimation ────────────────────────────────────────────────────────
DAMAGE_COST_MAP = {
    "dent": {"minor": (2000, 4000), "moderate": (4000, 8000), "severe": (8000, 15000)},
    "scratch": {"minor": (1500, 3000), "moderate": (3000, 6000), "severe": (6000, 12000)},
    "crack": {"minor": (3000, 5000), "moderate": (5000, 10000), "severe": (10000, 20000)},
    "glass_shatter": {"minor": (8000, 12000), "moderate": (12000, 18000), "severe": (18000, 30000)},
    "lamp_broken": {"minor": (2000, 4000), "moderate": (4000, 8000), "severe": (8000, 15000)},
    "tire_flat": {"minor": (200, 500), "moderate": (500, 1200), "severe": (1200, 3000)},
    "damage": {"minor": (2000, 4000), "moderate": (4000, 8000), "severe": (8000, 15000)},
}

VALID_DAMAGE_CLASSES = {
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"
}

def normalize_damage_type(name: str) -> str:
    s = (name or "").strip().lower()
    if "scratch" in name.lower(): return "scratch"
    if "dent" in name.lower(): return "dent"
    if "crack" in name.lower(): return "crack"
    if "glass" in name.lower() and "shatter" in name.lower(): return "glass_shatter"
    if "lamp" in name.lower() and "broken" in name.lower(): return "lamp_broken"
    if "tire" in name.lower() and "flat" in name.lower(): return "tire_flat"
    return name.lower().replace(" ", "_").replace("-", "_")

def is_valid_damage_class(name: str) -> bool:
    normalized = normalize_damage_type(name)
    return normalized in VALID_DAMAGE_CLASSES

def estimate_costs(detections_by_view: Dict[str, Any]) -> tuple[int, int, str, str, str]:
    all_detections = []
    for view_data in detections_by_view.values():
        all_detections.extend(view_data.get("detections", []))

    if not all_detections:
        return 0, 0, "uncertain", "Unable to determine", "No supported damage classes detected."

    for d in all_detections:
        d["damage_type"] = normalize_damage_type(d["damage_type"])

    valid_detections = [d for d in all_detections if is_valid_damage_class(d["damage_type"])]
    
    if not valid_detections:
        return 0, 0, "uncertain", "Unable to determine", "No valid damage classes detected."

    max_conf = max(d["confidence"] for d in valid_detections)
    total_valid_count = len(valid_detections)

    from collections import Counter
    damage_types = [d["damage_type"] for d in valid_detections]
    primary_damage = Counter(damage_types).most_common(1)[0][0]

    if max_conf >= 0.7 and total_valid_count >= 3:
        severity = "severe"
    elif max_conf >= 0.5 and total_valid_count >= 2:
        severity = "moderate"
    elif max_conf >= 0.3:
        severity = "minor"
    else:
        severity = "uncertain"

    cost_map = DAMAGE_COST_MAP.get(primary_damage, DAMAGE_COST_MAP["damage"])
    min_cost, max_cost = cost_map.get(severity, cost_map["minor"])

    unique_damage_types = set(damage_types)
    unique_count = len(unique_damage_types)
    
    if unique_count > 1:
        min_cost = int(min_cost * (1 + 0.3 * (unique_count - 1)))
        max_cost = int(max_cost * (1 + 0.3 * (unique_count - 1)))
    elif total_valid_count > 1:
        min_cost = int(min_cost * (1 + 0.1 * (total_valid_count - 1)))
        max_cost = int(max_cost * (1 + 0.1 * (total_valid_count - 1)))

    component_map = {
        "dent": "Body Panel", "scratch": "Body Panel", "crack": "Bumper/Fender",
        "glass_shatter": "Windshield/Window", "lamp_broken": "Headlight/Taillight",
        "tire_flat": "Tire/Wheel", "damage": "Vehicle Component",
    }
    primary_component = component_map.get(primary_damage, "Vehicle Component")

    unique_types = list(set(damage_types))
    summary = f"Detected {total_valid_count} valid damage instance(s): {', '.join(unique_types)}. Primary: {primary_damage} ({severity})."

    return min_cost, max_cost, severity, primary_component, summary

# ─── Image Utilities ────────────────────────────────────────────────────────
def decode_image(data: bytes | bytearray | Image.Image) -> Image.Image:
    if isinstance(data, Image.Image):
        image = data.copy()
    else:
        try:
            image = Image.open(io.BytesIO(bytes(data)))
            if image.width * image.height > 40_000_000:
                raise ValueError("Image too large (over 40 megapixels).")
            image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("Could not decode image or exceeds size limit.") from exc
    if image.width * image.height > 40_000_000:
        raise ValueError("Image too large (over 40 megapixels).")
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def validate_four_views(images: Dict[str, Image.Image | bytes | bytearray | None]) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    prepared: dict[str, Image.Image] = {}
    metrics: dict[str, dict[str, Any]] = {}
    hashes: dict[str, int] = {}

    def _dhash(img: Image.Image) -> int:
        gray = ImageOps.fit(img.convert("L"), (9, 8)).resize((9, 8))
        pixels = np.asarray(gray, dtype=np.uint8)
        bits = pixels[:, 1:] > pixels[:, :-1]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return value

    def _hamming(first: int, second: int) -> int:
        return bin(first ^ second).count("1")

    for view in REQUIRED_VIEWS:
        source = images.get(view)
        if source is None:
            blockers.append({"code": "missing_view", "view": view, "message": f"Upload the {view} view."})
            continue
        try:
            image = decode_image(source)
        except ValueError as exc:
            blockers.append({"code": "invalid_image", "view": view, "message": f"{view.title()}: {exc}"})
            continue

        prepared[view] = image
        metrics[view] = {"width": image.width, "height": image.height, "laplacian_variance": float(cv2.Laplacian(np.asarray(image.convert("L")), cv2.CV_64F).var())}
        hashes[view] = _dhash(image)

        width, height = image.size
        if min(width, height) < MIN_IMAGE_SIDE:
            blockers.append({"code": "low_resolution", "view": view, "message": f"{view.title()} image is {width}×{height}; use a clearer photo."})
        blur_score = metrics[view]["laplacian_variance"]
        if blur_score < MIN_LAPLACIAN_VARIANCE:
            blockers.append({"code": "possibly_blurry", "view": view, "message": f"{view.title()} image may be too blurry (sharpness {blur_score})."})

    available_views = list(prepared)
    for index, first in enumerate(available_views):
        for second in available_views[index + 1:]:
            distance = _hamming_distance(hashes[first], hashes[second])
            same_pixels = metrics[first].get("sha256_normalized_pixels") == metrics[second].get("sha256_normalized_pixels")
            if same_pixels:
                blockers.append({"code": "exact_duplicate", "view": f"{first},{second}", "message": f"The {first} and {second} photos are pixel-identical. Upload distinct photos."})
            elif distance <= NEAR_DUPLICATE_DISTANCE:
                warnings.append({"code": "possible_near_duplicate", "view": f"{first},{second}", "message": f"The {first} and {second} photos look very similar."})

    return {
        "valid_for_inference": len(blockers) == 0 and len(prepared) == len(REQUIRED_VIEWS),
        "required_views": list(REQUIRED_VIEWS),
        "received_views": sorted(prepared),
        "blockers": blockers,
        "warnings": warnings,
        "metrics": metrics,
        "images": prepared,
        "viewpoint_verification": "not_available: view labels are supplied by the uploader",
        "vehicle_identity_verification": "not_available: same-vehicle identity is not independently verified",
    }

# ─── OpenAI Vision API ──────────────────────────────────────────────────────
async def call_openai_vision(prompt: str, images: Dict[str, Image.Image], detail: str = "high") -> dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")
    
    url = "https://api.openai.com/v1/chat/completions"
    
    content = [{"type": "text", "text": prompt}]
    for view, img in images.items():
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}
        })
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert automotive damage assessor for insurance claims. Return only valid JSON."},
                {"role": "user", "content": content}
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
        })
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)

# ─── Video Frame Extraction ─────────────────────────────────────────────────
async def extract_video_frames(video_bytes: bytes, filename: str, target_frames: int = 10) -> List[Image.Image]:
    import tempfile
    import cv2
    
    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if duration < MIN_VIDEO_DURATION:
            raise ValueError(f"Video too short: {duration:.1f}s (minimum 8 seconds)")
        if duration > MAX_VIDEO_DURATION:
            raise ValueError(f"Video too long: {duration:.1f}s (maximum 10 seconds)")
        
        if total_frames <= VIDEO_FRAME_TARGET:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, VIDEO_FRAME_TARGET, dtype=int).tolist()
        
        frames = []
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        cap.release()
        return frames
    finally:
        cap.release()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ─── OpenAI Vision Assessment ───────────────────────────────────────────────
async def assess_images_openai(
    vehicle_info: dict,
    images: Dict[str, Image.Image],
    detail: str = "high"
) -> dict[str, Any]:
    prompt = f"""You are an expert automotive damage assessor for insurance claims. Analyze the provided vehicle photos and produce a structured damage assessment.

VEHICLE: {vehicle_info.get('make', 'Unknown')} {vehicle_info.get('model', 'Unknown')}{f" ({vehicle_info.get('year')})" if vehicle_info.get('year') else ''}
CAUSE OF DAMAGE: {vehicle_info.get('causeOfDamage', 'Not specified')}

ANALYSIS REQUIREMENTS:
1. For each image, identify all visible damage and its location
2. Classify each damage: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat, or other
3. Assess severity: minor, moderate, severe, or uncertain
4. Provide confidence score (0.0-1.0) for each finding
5. Identify the vehicle component/panel for each damage
5. Distinguish actual damage from shadows, reflections, dirt, panel gaps, artifacts
6. Detect potential duplicate reports of the same damage across different views
7. Recommend repair vs replace based on visible damage, component material
8. Identify when physical inspection by a mechanic/body shop is needed
9. Estimate repair cost range using reasonable assumptions - state assumptions
10. Explain which conclusions are supported by visual evidence vs estimates

CRITICAL LIMITATIONS - YOU MUST NEVER CLAIM:
- Hidden structural damage has been verified
- Internal mechanical damage has been assessed
- Airbag deployment status is known
- Safety-critical functionality has been verified
- Any conclusion is definitive without physical inspection

OUTPUT FORMAT: Return ONLY valid JSON with this exact structure:
{{
  "damageItems": [
    {{
      "component": "string",
      "view": "top|bottom|left|right|video",
      "damageType": "string",
      "severity": "minor|moderate|severe|uncertain",
      "evidence": ["string"],
      "confidence": 0.0-1.0,
      "recommendation": "repair|replace|inspect|undetermined",
      "reasoning": "string",
      "assumptions": ["string"]
    }}
  ],
  "aiObservations": [
    {{
      "component": "string",
      "view": "top|bottom|left|right|video",
      "description": "string",
      "severity": "minor|moderate|severe|uncertain",
      "confidence": 0.0-1.0,
      "reasoning": "string",
      "boundingBoxHint": {{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}}
    }}
  ],
  "duplicateWarnings": [
    {{
      "detectionIds": ["string"],
      "views": ["top|bottom|left|right|video"],
      "message": "string"
    }}
  ],
  "overallReasoning": "string",
  "confidence": 0.0-1.0,
  "limitations": ["string"]
}}"""

    images_b64 = {}
    for view, img in images.items():
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images_b64[view] = b64
    
    return await call_openai_vision(prompt, images, detail="high")

# ─── Pydantic Models ────────────────────────────────────────────────────────
class DamageDetection(BaseModel):
    damage_type: str
    confidence: float
    bounding_box_xyxy: List[float] = []

class ViewResult(BaseModel):
    detection_count: int
    detections: List[DamageDetection]
    annotated_image_base64: Optional[str] = None

class AssessmentResponse(BaseModel):
    assessment_id: str
    created_at: str
    workflow: str
    model_name: str
    confidence_threshold: float
    image_size: int
    views: Dict[str, ViewResult]
    detection_counts_by_class: Dict[str, int]
    total_detections: int
    review_required: bool
    review_flags: List[Dict[str, str]]
    validation: Dict[str, Any]
    limitations: List[str]
    estimated_repair_cost_min: int
    estimated_repair_cost_max: int
    settlement_offer: int
    settlement_min: int
    settlement_max: int
    severity: str
    primary_component: str
    damage_summary: str

class VideoFourViewAssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: str
    processing_time_ms: int
    video_processed: bool
    video_frames_extracted: int
    video_duration_seconds: float
    photos_processed: int
    video_frames: List[dict] = Field(default_factory=list)
    photos: Dict[str, dict] = Field(default_factory=dict)
    all_detections: List[dict] = Field(default_factory=list)
    total_detections: int
    severity: str
    primary_component: str
    damage_summary: str
    estimated_repair_cost_min: int
    estimated_repair_cost_max: int
    settlement_offer: int
    settlement_min: int
    settlement_max: int
    model_version: str = "gpt-4o-mini"
    cv_backend: str = "openai-gpt4o"
    limitations: List[str] = Field(default_factory=list)

class HybridAssessmentRequest(BaseModel):
    vehicle: dict
    views: Dict[str, str]  # base64 encoded images
    detections: Dict[str, List[DamageDetection]]
    confidence_threshold: float = Field(default=0.25, ge=0.1, le=0.8)
    image_size: int = Field(default=640, ge=320, le=1280)

class HybridAssessmentResponse(BaseModel):
    assessment_id: str
    created_at: str
    workflow: str
    model_name: str
    ai_provider: str
    ai_model: str
    damage_items: List[dict]
    ai_observations: List[dict]
    duplicate_warnings: List[dict]
    overall_reasoning: str
    confidence: float
    limitations: List[str]
    estimated_repair_cost_min: int
    estimated_repair_cost_max: int
    settlement_offer: int
    settlement_min: int
    settlement_max: int
    severity: str
    primary_component: str
    damage_summary: str
    review_required: bool

# ─── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClaimPilot AI Backend",
    description="Vehicle damage assessment using OpenAI GPT-4o-mini vision only",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health & Model Info ────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ClaimPilot AI Backend (OpenAI Vision Only)"}

@app.get("/ai/provider/status")
async def ai_provider_status():
    return {
        "provider": "openai",
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "hybrid_available": bool(OPENAI_API_KEY),
    }

# ─── 4-View Assessment (OpenAI Vision Only) ─────────────────────────────────
@app.post("/assess/four-view", response_model=AssessmentResponse)
async def assess_four_view(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.25),
    image_size: int = Form(default=640),
):
    """Assess vehicle damage from four required views using OpenAI GPT-4o-mini vision only."""
    # Read and validate files
    uploaded_files = {"top": top, "bottom": bottom, "left": left, "right": right}
    image_bytes = {}

    for view, upload_file in uploaded_files.items():
        if not upload_file.filename:
            raise HTTPException(status_code=400, detail=f"{view}: No file provided")
        ext = upload_file.filename.split(".")[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{view}: Unsupported format. Use JPG, PNG, WebP.")
        content = await upload_file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{view}: File too large (max 15MB)")
        image_bytes[view] = content

    # Validate four views
    validation = validate_four_views(image_bytes)
    report = {k: v for k, v in validation.items() if k != "images"}

    if not validation["valid_for_inference"]:
        return JSONResponse(status_code=400, content={
            "error": "Validation failed", "validation": report,
            "message": "Please fix validation issues and re-upload."
        })

    # Prepare images for OpenAI
    images = validation["images"]
    
    # Run OpenAI vision assessment
    try:
        vehicle_info = {"make": "", "model": "", "year": "", "variant": "", "causeOfDamage": ""}
        ai_result = await assess_images_openai(vehicle_info, images)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI assessment failed: {type(exc).__name__}: {exc}")

    # Estimate costs
    all_detections = []
    for item in ai_result.get("damageItems", []):
        all_ai_detections.append({
            "damage_type": normalize_damage_type(item["damageType"]),
            "confidence": item.get("confidence", 0.5),
        })

    detections_by_view = {"top": {"detections": all_ai_detections}}
    min_cost, max_cost, severity, primary_component, summary = estimate_costs(detections_by_view)

    settlement_offer = int((min_cost + max_cost) / 2 * 0.7) if min_cost > 0 else 0
    settlement_min = int(min_cost * 0.55) if min_cost > 0 else 0
    settlement_max = int(max_cost * 0.85) if max_cost > 0 else 0

    review_required = ai_result.get("confidence", 1.0) < 0.7 or len(ai_result.get("duplicateWarnings", [])) > 0

    # Convert AI results to frontend format
    damage_items = ai_result.get("damageItems", [])
    detections_by_view_response = {}
    for item in ai_result.get("damageItems", []):
        detections_by_view_response.setdefault(item["view"], []).append({
            "damage_type": item["damageType"],
            "confidence": item.get("confidence", 0.5),
        })

    response = AssessmentResponse(
        assessment_id=f"ASM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        created_at=datetime.now(timezone.utc).isoformat(),
        workflow="openai_vision_four_view",
        model_name=f"OpenAI {OPENAI_MODEL}",
        confidence_threshold=0.25,
        image_size=640,
        views={
            view: ViewResult(
                detection_count=len(dets),
                detections=[DamageDetection(**d) for d in dets],
                annotated_image_base64=None,  # OpenAI doesn't return annotated images
            )
            for view, dets in detections_by_view_response.items()
        },
        detection_counts_by_class={},
        total_detections=len(all_ai_detections),
        review_required=review_required,
        review_flags=[],
        validation={},
        limitations=ai_result.get("limitations", []),
        estimated_repair_cost_min=min_cost,
        estimated_repair_cost_max=max_cost,
        settlement_offer=settlement_offer,
        settlement_min=settlement_min,
        settlement_max=settlement_max,
        severity=severity,
        primary_component=primary_component,
        damage_summary=summary,
    )
    return response

# ─── Hybrid AI Assessment (OpenAI Vision Only) ───────────────────────────────
@app.post("/assess/hybrid", response_model=HybridAssessmentResponse)
async def assess_hybrid(request: HybridAssessmentRequest):
    """Hybrid AI assessment using OpenAI GPT-4o-mini vision only."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")
    
    # Validate images
    views_images = {}
    for view in REQUIRED_VIEWS:
        b64 = request.views.get(view)
        if not b64:
            raise HTTPException(status_code=400, detail=f"Missing {view} view image")
        try:
            img_data = base64.b64decode(b64)
            img = decode_image(img_data)
            views_images[view] = img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid {view} image: {e}")

    # Run OpenAI vision assessment
    try:
        ai_result = await assess_images_openai(request.vehicle, views_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI assessment failed: {type(e).__name__}: {e}")

    # Use estimate_costs for consistent pricing
    all_ai_detections = []
    for item in ai_result.get("damageItems", []):
        all_ai_detections.append({
            "damage_type": normalize_damage_type(item["damageType"]),
            "confidence": item.get("confidence", 0.5),
        })
    
    detections_by_view = {"top": {"detections": all_ai_detections}}
    min_cost, max_cost, severity, primary_component, summary = estimate_costs(detections_by_view)
    
    settlement_offer = int((min_cost + max_cost) / 2 * 0.7) if min_cost > 0 else 0
    settlement_min = int(min_cost * 0.55) if min_cost > 0 else 0
    settlement_max = int(max_cost * 0.85) if max_cost > 0 else 0
    
    review_required = ai_result.get("confidence", 1.0) < 0.7 or len(ai_result.get("duplicateWarnings", [])) > 0
    
    return HybridAssessmentResponse(
        assessment_id=f"HASM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        created_at=datetime.now(timezone.utc).isoformat(),
        workflow="openai_vision_hybrid",
        model_name=f"OpenAI {OPENAI_MODEL}",
        ai_provider="openai",
        ai_model=OPENAI_MODEL,
        damage_items=ai_result.get("damageItems", []),
        ai_observations=ai_result.get("aiObservations", []),
        duplicate_warnings=ai_result.get("duplicateWarnings", []),
        overall_reasoning=ai_result.get("overallReasoning", ""),
        confidence=ai_result.get("confidence", 0.5),
        limitations=ai_result.get("limitations", []),
        estimated_repair_cost_min=min_cost,
        estimated_repair_cost_max=max_cost,
        settlement_offer=settlement_offer,
        settlement_min=settlement_min,
        settlement_max=settlement_max,
        severity=severity,
        primary_component=primary_component,
        damage_summary=summary,
        review_required=review_required,
    )

# ─── Video + Four-View Assessment (OpenAI Vision Only) ───────────────────────
@app.post("/assess/video-four-view", response_model=VideoFourViewAssessmentResponse)
async def assess_video_four_view(
    video: UploadFile = File(..., description="Video walkaround (8-10 seconds)"),
    top: UploadFile = File(..., description="Top view photo"),
    bottom: UploadFile = File(..., description="Bottom view photo"),
    left: UploadFile = File(..., description="Left side photo"),
    right: UploadFile = File(..., description="Right side photo"),
    confidence_threshold: float = Form(default=0.25),
    image_size: int = Form(default=640),
):
    """
    Video + Four-View Damage Assessment using OpenAI GPT-4o-mini vision only.
    
    Process:
    1. Validate video (8-10s) and 4 photos
    2. Extract 10 frames from video
    3. Run OpenAI GPT-4o-mini on all 14 images (10 video frames + 4 photos)
    4. Aggregate and return assessment
    """
    import time
    start_time = time.time()
    assessment_id = f"VF4V-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    # Validate video
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video: No file provided")
    if video.content_type not in SUPPORTED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Video: Unsupported format. Use MP4, MOV, AVI, or WebM.")
    
    video_content = await video.read()
    if len(video_content) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Video too large (max {MAX_VIDEO_SIZE_MB}MB)")
    if len(video_content) < 1000:
        raise HTTPException(status_code=400, detail="Video file too small or empty")
    
    # Validate photos
    photo_files = {"top": top, "bottom": bottom, "left": left, "right": right}
    photo_contents = {}
    for view, upload_file in photo_files.items():
        if not upload_file.filename:
            raise HTTPException(status_code=400, detail=f"{view}: No file provided")
        ext = upload_file.filename.split(".")[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{view}: Unsupported format. Use JPG, PNG, or WebP.")
        content = await upload_file.read()
        if len(content) > MAX_PHOTO_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{view}: File too large (max {MAX_PHOTO_SIZE_MB}MB)")
        photo_contents[view] = content
    
    # Initialize components
    try:
        video_processor = create_video_processor(max_frames=VIDEO_FRAME_TARGET)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {type(exc).__name__}: {exc}")
    
    # Process video: extract frames
    video_frames = []
    video_detections = []
    video_processed = False
    video_duration = 0.0
    video_warnings = []
    
    try:
        extraction_result = video_processor.extract_frames(
            video_content, video.filename, strategy="uniform", target_frames=VIDEO_FRAME_TARGET
        )
        video_duration = extraction_result.duration_seconds
        video_processed = True
        video_warnings = extraction_result.warnings
        
        # Convert frames to PIL images for OpenAI
        video_frames_images = [frame.image for frame in extraction_result.extracted_frames]
        
    except Exception as e:
        video_warnings = [f"Video processing failed: {str(e)}"]
        video_frames_images = []
    
    # Process photos
    photo_images = {}
    photo_results = {}
    photo_detections = []
    
    for view, content in photo_contents.items():
        try:
            image = decode_image(content)
            # We'll send to OpenAI with video frames
            photo_images[view] = image
        except Exception as e:
            photo_results[view] = {"detection_count": 0, "detections": [], "error": str(e)}
            continue
    
    # Combine all images for OpenAI assessment
    all_images = {}
    if video_frames_images:
        all_images["video"] = video_frames_images[0]  # Use first frame as representative
    all_images.update(photo_images)
    
    if not all_images:
        raise HTTPException(status_code=400, detail="No valid images to process")
    
    # Run OpenAI assessment on combined images
    try:
        vehicle_info = {"make": "", "model": "", "year": "", "variant": "", "causeOfDamage": ""}
        ai_result = await assess_images_openai(vehicle_info, all_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI assessment failed: {type(e).__name__}: {e}")
    
    # Process results
    all_detections = []
    for item in ai_result.get("damageItems", []):
        all_detections.append({
            "damage_type": item["damageType"],
            "confidence": item.get("confidence", 0.5),
            "severity": item.get("severity", "minor"),
            "source": "openai",
            "source_index": 0,
        })
    
    # Estimate costs
    min_cost, max_cost, severity, primary_component, summary = estimate_collective_costs(all_detections)
    settlement_offer = int((min_cost + max_cost) / 2 * 0.7) if min_cost > 0 else 0
    
    # Build response
    processing_time = int((time.time() - start_time) * 1000)
    
    return VideoFourViewAssessmentResponse(
        assessment_id=assessment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=processing_time,
        video_processed=video_processed,
        video_frames_extracted=len(video_frames_images),
        video_duration_seconds=video_duration if video_processed else 0.0,
        photos_processed=len(photo_images),
        video_frames=[{"frame_index": i, "timestamp_ms": 0, "detections_count": 0, "top_detection": None} for i in range(len(video_frames_images))],
        photos={view: {"photo_index": i, "filename": f"{view}.jpg", "detections_count": 0, "top_detection": None} for i, view in enumerate(REQUIRED_VIEWS)},
        all_detections=all_detections,
        total_detections=len(all_detections),
        severity=severity,
        primary_component=primary_component,
        damage_summary=summary,
        estimated_repair_cost_min=min_cost,
        estimated_repair_cost_max=max_cost,
        settlement_offer=settlement_offer,
        settlement_min=int(min_cost * 0.55) if min_cost > 0 else 0,
        settlement_max=int(max_cost * 0.85) if max_cost > 0 else 0,
        limitations=[
            "Video must be 8-10 seconds for optimal 10-frame extraction",
            "Same physical damage may appear in multiple video frames and photos",
            "Damage class and detector confidence are not validated measures of severity or repair cost",
            "Viewpoint labels are user-supplied, not independently verified",
            "This output is an evidence summary only and does not authorize an insurance settlement",
        ],
    )

# ─── Provider Status ────────────────────────────────────────────────────────
@app.get("/ai/provider/status")
async def ai_provider_status():
    return {
        "provider": "openai",
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "hybrid_available": bool(OPENAI_API_KEY),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)