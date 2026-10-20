"""FastAPI backend for ClaimPilot AI - wraps Auto-Settle AI for vehicle damage assessment."""

from __future__ import annotations

import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ─── New: Policy-driven Damage Assessment Router ─────────────────────────────
from damage_assessment import router as damage_assessment_router

# ─── New: Multi-Modal Assessment Router (3+ photos + 1 video) ──────────────────
from multimodal_assessment import router as multimodal_assessment_router

# ─── New: Negotiation Engine Router ───────────────────────────────────────────
from negotiation_engine import create_negotiation_router

# ─── Constants ──────────────────────────────────────────────────────────────
REQUIRED_VIEWS = ("top", "bottom", "left", "right")
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MIN_IMAGE_SIDE = int(os.getenv("AUTOSETTLE_MIN_IMAGE_SIDE", "240"))
MIN_LAPLACIAN_VARIANCE = float(os.getenv("AUTOSETTLE_MIN_LAPLACIAN_VARIANCE", "18"))
NEAR_DUPLICATE_DISTANCE = int(os.getenv("AUTOSETTLE_NEAR_DUPLICATE_DISTANCE", "3"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("AUTOSETTLE_LOW_CONFIDENCE_THRESHOLD", "0.40"))

# ─── Model Loading ──────────────────────────────────────────────────────────
_model: Optional[YOLO] = None
_model_label: str = ""


def load_damage_model() -> tuple[YOLO, str]:
    """Load the YOLO damage detection model."""
    global _model, _model_label
    if _model is not None:
        return _model, _model_label

    from huggingface_hub import hf_hub_download

    local_path = os.getenv("AUTOSETTLE_MODEL_PATH", "").strip()
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"AUTOSETTLE_MODEL_PATH does not exist: {local_path}")
        weights_path = local_path
    else:
        repo_id = os.getenv("AUTOSETTLE_HF_REPO", "shanexf/car-damage-yolo11m-ft")
        filename = os.getenv("AUTOSETTLE_HF_FILENAME", "damage_best.pt")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

    _model = YOLO(weights_path)
    _model_label = os.path.basename(weights_path)
    return _model, _model_label


# ─── Image Utilities ────────────────────────────────────────────────────────
def decode_image(data: bytes | bytearray | Image.Image) -> Image.Image:
    """Decode image bytes safely and normalize orientation/color."""
    if isinstance(data, Image.Image):
        image = data.copy()
    else:
        try:
            image = Image.open(io.BytesIO(bytes(data)))
            if image.width * image.height > 40_000_000:
                raise ValueError("The image is too large to process safely (over 40 megapixels).")
            image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("This file could not be decoded as a valid image, or it exceeds the allowed size.") from exc
    if image.width * image.height > 40_000_000:
        raise ValueError("The image is too large to process safely (over 40 megapixels).")
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


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


# ─── Validation ─────────────────────────────────────────────────────────────
def validate_four_views(images: Dict[str, Image.Image | bytes | bytearray | None]) -> dict[str, Any]:
    """Validate required view presence, basic quality and duplicate evidence."""
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    prepared: dict[str, Image.Image] = {}
    metrics: dict[str, dict[str, Any]] = {}
    hashes: dict[str, int] = {}

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
        metrics[view] = image_metrics(image)
        hashes[view] = _dhash(image)

        width, height = image.size
        if min(width, height) < MIN_IMAGE_SIDE:
            blockers.append({
                "code": "low_resolution",
                "view": view,
                "message": f"{view.title()} image is {width}×{height}; use a clearer photo with both sides at least {MIN_IMAGE_SIDE} px.",
            })
        blur_score = metrics[view]["laplacian_variance"]
        if blur_score < MIN_LAPLACIAN_VARIANCE:
            blockers.append({
                "code": "possibly_blurry",
                "view": view,
                "message": f"{view.title()} image may be too blurry (sharpness score {blur_score}). Retake it in good light and hold the camera steady.",
            })

    # Exact normalized-pixel duplicates are strong signals and block processing.
    available_views = list(prepared)
    for index, first in enumerate(available_views):
        for second in available_views[index + 1 :]:
            distance = _hamming_distance(hashes[first], hashes[second])
            same_pixels = metrics[first]["sha256_normalized_pixels"] == metrics[second]["sha256_normalized_pixels"]
            if same_pixels:
                blockers.append({
                    "code": "exact_duplicate",
                    "view": f"{first},{second}",
                    "message": f"The {first} and {second} photos are pixel-identical. Upload distinct photos for each requested angle.",
                })
            elif distance <= NEAR_DUPLICATE_DISTANCE:
                warnings.append({
                    "code": "possible_near_duplicate",
                    "view": f"{first},{second}",
                    "message": f"The {first} and {second} photos look very similar. This may be a reused image, but different views of the same damage can also look similar; request human verification.",
                })
            elif distance <= NEAR_DUPLICATE_DISTANCE + 2:
                warnings.append({
                    "code": "similar_images",
                    "view": f"{first},{second}",
                    "message": f"The {first} and {second} images look similar; a reviewer should verify that they are distinct angles.",
                })

    return {
        "valid_for_inference": len(blockers) == 0 and len(prepared) == len(REQUIRED_VIEWS),
        "required_views": list(REQUIRED_VIEWS),
        "received_views": sorted(prepared),
        "blockers": blockers,
        "warnings": warnings,
        "metrics": metrics,
        "images": prepared,
        "viewpoint_verification": "not_available: view labels are supplied by the uploader, not verified by a trained viewpoint classifier",
        "vehicle_identity_verification": "not_available: same-vehicle identity is not independently verified",
    }


# ─── Inference ──────────────────────────────────────────────────────────────
def run_four_view_inference(
    model: YOLO,
    images_by_view: Dict[str, Image.Image],
    *,
    confidence: float = 0.25,
    image_size: int = 640,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one YOLO prediction per view and return JSON data plus RGB overlays."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    missing = [view for view in REQUIRED_VIEWS if view not in images_by_view]
    if missing:
        raise ValueError("Missing required views: " + ", ".join(missing))

    all_results: dict[str, Any] = {}
    annotated_images: dict[str, np.ndarray] = {}
    combined_counts: dict[str, int] = {}
    review_flags: list[dict[str, str]] = []
    total_detections = 0

    for view in REQUIRED_VIEWS:
        image = decode_image(images_by_view[view])
        result = model.predict(source=image, conf=confidence, imgsz=image_size, verbose=False)[0]
        detections: list[dict[str, Any]] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                names = result.names
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                else:
                    class_name = str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "damage_type": class_name,
                    "confidence": round(score, 4),
                    "bounding_box_xyxy": xyxy,
                })
                combined_counts[class_name] = combined_counts.get(class_name, 0) + 1
                total_detections += 1
                if score < LOW_CONFIDENCE_THRESHOLD:
                    review_flags.append({
                        "code": "low_confidence_detection",
                        "view": view,
                        "message": f"A {class_name} detection in the {view} view has confidence {score:.2f}; verify manually.",
                    })

        if not detections:
            review_flags.append({
                "code": "no_damage_detected_in_view",
                "view": view,
                "message": f"YOLO detected no supported damage in the {view} view. This does not prove that the area is undamaged.",
            })

        plotted = result.plot()
        # Ultralytics returns BGR overlays; convert to RGB for frontend
        annotated_images[view] = np.ascontiguousarray(plotted[:, :, ::-1])
        all_results[view] = {
            "detection_count": len(detections),
            "detections": detections,
        }

    if total_detections == 0:
        review_flags.append({
            "code": "no_supported_damage_detected",
            "view": "all",
            "message": "No supported damage classes were detected in any view; manual review is recommended before any claim decision.",
        })

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "four_view_visual_damage_screening",
        "model_name": getattr(model, "model_name", None) or model.__class__.__name__,
        "confidence_threshold": confidence,
        "image_size": image_size,
        "views": all_results,
        "detection_counts_by_class_across_views": combined_counts,
        "total_view_detections": total_detections,
        "review_required": bool(review_flags),
        "review_flags": review_flags,
        "important_limitations": [
            "Detection counts are across submitted views; the same physical damage may appear in multiple views and is not automatically deduplicated.",
            "The model detects only its trained classes and may miss damage outside them.",
            "Damage class and detector confidence are not validated measures of severity or repair cost.",
            "The selected top/bottom/left/right viewpoint labels are not independently verified by this model.",
            "This output is an evidence summary only and does not authorize an insurance settlement.",
        ],
    }
    return audit, annotated_images


def public_validation_report(validation: Dict[str, Any]) -> dict[str, Any]:
    """Return validation data without image objects, suitable for JSON audit."""
    return {
        key: value
        for key, value in validation.items()
        if key not in {"images"}
    }


# ─── Pydantic Models ────────────────────────────────────────────────────────
class AssessmentRequest(BaseModel):
    confidence_threshold: float = Field(default=0.25, ge=0.1, le=0.8)
    image_size: int = Field(default=640, ge=320, le=1280)


class DamageDetection(BaseModel):
    damage_type: str
    confidence: float
    bounding_box_xyxy: List[float]


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
    # Settlement estimation (demo)
    estimated_repair_cost_min: int
    estimated_repair_cost_max: int
    settlement_offer: int
    settlement_min: int
    settlement_max: int
    severity: str
    primary_component: str
    damage_summary: str


# ─── Cost Estimation (Demo Logic) ───────────────────────────────────────────
DAMAGE_COST_MAP = {
    "scratch": {"minor": (1500, 3000), "moderate": (3000, 6000), "severe": (6000, 12000)},
    "dent": {"minor": (2000, 4000), "moderate": (4000, 8000), "severe": (8000, 15000)},
    "crack": {"minor": (3000, 5000), "moderate": (5000, 10000), "severe": (10000, 20000)},
    "glass_breakage": {"minor": (8000, 12000), "moderate": (12000, 18000), "severe": (18000, 30000)},
    "paint_damage": {"minor": (1000, 2500), "moderate": (2500, 5000), "severe": (5000, 10000)},
    "deformation": {"minor": (3000, 6000), "moderate": (6000, 12000), "severe": (12000, 25000)},
    "damage": {"minor": (2000, 4000), "moderate": (4000, 8000), "severe": (8000, 15000)},
}


def estimate_costs(detections_by_view: Dict[str, Any]) -> tuple[int, int, str, str, str]:
    """Estimate repair costs based on detections. Returns (min_cost, max_cost, severity, primary_component, summary)."""
    all_detections = []
    for view_data in detections_by_view.values():
        all_detections.extend(view_data.get("detections", []))

    if not all_detections:
        return 0, 0, "uncertain", "Unable to determine", "No supported damage classes detected in the submitted images."

    # Determine severity based on highest confidence detection and count
    max_conf = max(d["confidence"] for d in all_detections)
    total_count = len(all_detections)

    # Get most common damage type
    from collections import Counter
    damage_types = [d["damage_type"].lower() for d in all_detections]
    primary_damage = Counter(damage_types).most_common(1)[0][0]

    # Map to severity
    if max_conf >= 0.7 and total_count >= 3:
        severity = "severe"
    elif max_conf >= 0.5 and total_count >= 2:
        severity = "moderate"
    elif max_conf >= 0.3:
        severity = "minor"
    else:
        severity = "uncertain"

    # Estimate cost
    cost_map = DAMAGE_COST_MAP.get(primary_damage, DAMAGE_COST_MAP["damage"])
    min_cost, max_cost = cost_map.get(severity, cost_map["minor"])

    # Adjust for multiple detections
    if total_count > 1:
        min_cost = int(min_cost * (1 + 0.3 * (total_count - 1)))
        max_cost = int(max_cost * (1 + 0.3 * (total_count - 1)))

    # Component mapping
    component_map = {
        "scratch": "Body Panel",
        "dent": "Body Panel",
        "crack": "Bumper/Fender",
        "glass_breakage": "Windshield/Window",
        "paint_damage": "Paint Surface",
        "deformation": "Structural Panel",
        "damage": "Vehicle Component",
    }
    primary_component = component_map.get(primary_damage, "Vehicle Component")

    # Summary
    unique_types = list(set(damage_types))
    summary = f"Detected {total_count} damage instance(s): {', '.join(unique_types)}. Primary: {primary_damage} ({severity})."

    return min_cost, max_cost, severity, primary_component, summary


# ─── Hybrid AI Configuration ──────────────────────────────────────────────
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─── Hybrid AI Assessment ──────────────────────────────────────────────────
async def run_hybrid_assessment(
    vehicle_info: dict,
    views: Dict[str, Image.Image],
    detections_by_view: Dict[str, Any],
    confidence_threshold: float = 0.25,
    image_size: int = 640,
) -> dict[str, Any]:
    """
    Run multimodal AI assessment using Gemini or OpenAI.
    This sends images and YOLO detections to the vision model for reasoning.
    """
    import httpx
    
    # Prepare images as base64
    view_images = {}
    for view, img in views.items():
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        view_images[view] = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Build prompt with YOLO detections
    detections_summary = []
    for view in REQUIRED_VIEWS:
        view_dets = detections_by_view.get(view, {}).get("detections", [])
        if view_dets:
            for det in view_dets:
                detections_summary.append({
                    "view": view,
                    "class": det["damage_type"],
                    "confidence": det["confidence"],
                    "bbox": det["bounding_box_xyxy"],
                })
    
    prompt = f"""You are an expert automotive damage assessor for insurance claims. Analyze the provided vehicle photos and YOLO detections to produce a structured damage assessment.

VEHICLE: {vehicle_info.get('make', 'Unknown')} {vehicle_info.get('model', 'Unknown')}{f" ({vehicle_info.get('year')})" if vehicle_info.get('year') else ''}

YOLO DETECTIONS BY VIEW:
{json.dumps(detections_summary, indent=2)}

ANALYSIS REQUIREMENTS:
1. For each YOLO detection, verify if the damage visually corresponds to the detected class and location
2. Describe visible damage and severity with uncertainty clearly stated
3. Distinguish actual damage from shadows, reflections, dirt, panel gaps, and artifacts
4. Identify potentially damaged components NOT detected by YOLO (label as AI observations)
5. Detect potential duplicate reports of the same damage across different views
6. Recommend repair vs replace based on visible damage, component material, and assumptions
7. Identify when physical inspection by a mechanic/body shop is needed
8. Estimate repair cost range using reasonable assumptions - explicitly state all assumptions
9. Explain which conclusions are supported by visual evidence vs estimates

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
      "view": "top|bottom|left|right",
      "detectionIds": ["string"],
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
      "view": "top|bottom|left|right",
      "description": "string",
      "severity": "minor|moderate|severe|uncertain",
      "confidence": 0.0-1.0,
      "reasoning": "string",
      "boundingBoxHint": {{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}}
    }}
  ],
  "duplicateDetectionWarnings": [
    {{
      "detectionIds": ["string"],
      "views": ["top|bottom|left|right"],
      "message": "string"
    }}
  ],
  "overallReasoning": "string",
  "confidence": 0.0-1.0,
  "limitations": ["string"]
}}"""

    if AI_PROVIDER == "gemini" and GEMINI_API_KEY:
        return await _call_gemini(prompt, view_images)
    elif AI_PROVIDER == "openai" and OPENAI_API_KEY:
        return await _call_openai(prompt, view_images)
    else:
        raise ValueError("No AI provider configured. Set AI_PROVIDER and corresponding API key.")


async def _call_gemini(prompt: str, images: Dict[str, str]) -> dict[str, Any]:
    """Call Gemini API with images and prompt."""
    import httpx
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    parts = [{"text": prompt}]
    for view, b64 in images.items():
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64
            }
        })
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json={
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ],
        })
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
        
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)


async def _call_openai(prompt: str, images: Dict[str, str]) -> dict[str, Any]:
    """Call OpenAI API with images and prompt."""
    import httpx
    
    url = "https://api.openai.com/v1/chat/completions"
    
    content = [{"type": "text", "text": prompt}]
    for view, b64 in images.items():
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
        })
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert automotive damage assessor. Return only valid JSON."},
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


# ─── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClaimPilot AI Backend",
    description="Vehicle damage assessment API using Auto-Settle AI (YOLO-based) with Hybrid AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Damage Assessment Router ────────────────────────────────────────
app.include_router(damage_assessment_router)

# ─── Include Multi-Modal Assessment Router ────────────────────────────────────
app.include_router(multimodal_assessment_router)

# ─── Include Negotiation Router ───────────────────────────────────────────────
app.include_router(create_negotiation_router())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ClaimPilot AI Backend"}


@app.get("/model/info")
async def model_info():
    """Get model information."""
    model, label = load_damage_model()
    return {
        "model_name": label,
        "model_type": "YOLO",
        "task": "damage_detection",
        "classes": list(model.names.values()) if hasattr(model, "names") else [],
    }


@app.post("/assess/four-view", response_model=AssessmentResponse)
async def assess_four_view(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.25),
    image_size: int = Form(default=640),
):
    """
    Assess vehicle damage from four required views.
    
    Requires exactly 4 images: top, bottom, left, right views of the same damaged area.
    """
    # Read and validate files
    uploaded_files = {"top": top, "bottom": bottom, "left": left, "right": right}
    image_bytes = {}

    for view, upload_file in uploaded_files.items():
        if not upload_file.filename:
            raise HTTPException(status_code=400, detail=f"{view}: No file provided")
        
        ext = upload_file.filename.split(".")[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{view}: Unsupported file format. Use JPG, PNG, or WebP.")

        content = await upload_file.read()
        if len(content) > 15 * 1024 * 1024:  # 15MB
            raise HTTPException(status_code=400, detail=f"{view}: File too large (max 15MB)")
        
        image_bytes[view] = content

    # Validate four views
    validation = validate_four_views(image_bytes)
    report = public_validation_report(validation)

    if not validation["valid_for_inference"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation failed",
                "validation": report,
                "message": "Please fix the validation issues and re-upload.",
            },
        )

    # Run inference
    try:
        model, model_label = load_damage_model()
        audit, annotated = run_four_view_inference(
            model,
            validation["images"],
            confidence=confidence_threshold,
            image_size=image_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}")

    # Add validation warnings to review flags
    for warning in validation["warnings"]:
        audit["review_flags"].append({
            "code": warning["code"],
            "view": warning["view"],
            "message": warning["message"],
        })
    if validation["warnings"]:
        audit["review_required"] = True

    # Estimate costs (demo logic)
    min_cost, max_cost, severity, primary_component, summary = estimate_costs(audit["views"])

    # Calculate settlement offer (demo: 70% of estimated repair cost, with bounds)
    settlement_offer = int((min_cost + max_cost) / 2 * 0.7)
    settlement_min = int(min_cost * 0.55)
    settlement_max = int(max_cost * 0.85)

    # Convert annotated images to base64
    annotated_b64 = {}
    for view, img_array in annotated.items():
        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        annotated_b64[view] = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Build response
    response = AssessmentResponse(
        assessment_id=f"ASM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        created_at=datetime.now(timezone.utc).isoformat(),
        workflow=audit["workflow"],
        model_name=model_label,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        views={
            view: ViewResult(
                detection_count=audit["views"][view]["detection_count"],
                detections=[DamageDetection(**d) for d in audit["views"][view]["detections"]],
                annotated_image_base64=annotated_b64.get(view),
            )
            for view in REQUIRED_VIEWS
        },
        detection_counts_by_class=audit["detection_counts_by_class_across_views"],
        total_detections=audit["total_view_detections"],
        review_required=audit["review_required"],
        review_flags=audit["review_flags"],
        validation=report,
        limitations=audit["important_limitations"],
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


@app.post("/assess/general")
async def assess_general(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(default=0.25),
    image_size: int = Form(default=640),
):
    """
    General damage assessment for any number of images.
    
    This endpoint accepts any number of images and runs detection on each.
    Useful for the NewAssessmentPage which doesn't enforce 4-view structure.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    # Process each file
    results = []
    all_detections = []
    total_cost_min = 0
    total_cost_max = 0
    all_damage_types = []

    try:
        model, model_label = load_damage_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {type(exc).__name__}: {exc}")

    for idx, upload_file in enumerate(files):
        if not upload_file.filename:
            continue
            
        ext = upload_file.filename.split(".")[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        content = await upload_file.read()
        if len(content) > 15 * 1024 * 1024:
            continue

        try:
            image = decode_image(content)
            result = model.predict(source=image, conf=confidence_threshold, imgsz=image_size, verbose=False)[0]
            
            detections = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())
                    names = result.names
                    if isinstance(names, dict):
                        class_name = str(names.get(class_id, class_id))
                    else:
                        class_name = str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
                    xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "damage_type": class_name,
                        "confidence": round(score, 4),
                        "bounding_box_xyxy": xyxy,
                    })
                    all_detections.append({"damage_type": class_name, "confidence": score})
                    all_damage_types.append(class_name.lower())

            # Annotated image
            plotted = result.plot()
            annotated_rgb = np.ascontiguousarray(plotted[:, :, ::-1])
            pil_img = Image.fromarray(annotated_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            results.append({
                "file_index": idx,
                "filename": upload_file.filename,
                "detection_count": len(detections),
                "detections": detections,
                "annotated_image_base64": annotated_b64,
            })
        except Exception as e:
            results.append({
                "file_index": idx,
                "filename": upload_file.filename,
                "error": str(e),
            })

    # Estimate overall costs
    if all_detections:
        from collections import Counter
        damage_counter = Counter(d["damage_type"].lower() for d in all_detections)
        primary_damage = damage_counter.most_common(1)[0][0]
        max_conf = max(d["confidence"] for d in all_detections)
        total_count = len(all_detections)

        if max_conf >= 0.7 and total_count >= 3:
            severity = "severe"
        elif max_conf >= 0.5 and total_count >= 2:
            severity = "moderate"
        elif max_conf >= 0.3:
            severity = "minor"
        else:
            severity = "uncertain"

        cost_map = DAMAGE_COST_MAP.get(primary_damage, DAMAGE_COST_MAP["damage"])
        min_cost, max_cost = cost_map.get(severity, cost_map["minor"])
        if total_count > 1:
            min_cost = int(min_cost * (1 + 0.3 * (total_count - 1)))
            max_cost = int(max_cost * (1 + 0.3 * (total_count - 1)))

        total_cost_min = min_cost
        total_cost_max = max_cost

        component_map = {
            "scratch": "Body Panel",
            "dent": "Body Panel",
            "crack": "Bumper/Fender",
            "glass_breakage": "Windshield/Window",
            "paint_damage": "Paint Surface",
            "deformation": "Structural Panel",
            "damage": "Vehicle Component",
        }
        primary_component = component_map.get(primary_damage, "Vehicle Component")
        unique_types = list(set(all_damage_types))
        summary = f"Analyzed {len(files)} image(s). Detected {total_count} damage instance(s): {', '.join(unique_types)}. Primary: {primary_damage} ({severity})."
    else:
        severity = "uncertain"
        primary_component = "Unable to determine"
        summary = f"Analyzed {len(files)} image(s). No supported damage classes detected."
        total_cost_min = 0
        total_cost_max = 0

    settlement_offer = int((total_cost_min + total_cost_max) / 2 * 0.7) if total_cost_min > 0 else 0
    settlement_min = int(total_cost_min * 0.55) if total_cost_min > 0 else 0
    settlement_max = int(total_cost_max * 0.85) if total_cost_max > 0 else 0

    # Determine if review is required (any low confidence detections or no detections)
    review_required = False
    for result in results:
        if result.get("error"):
            continue
        for det in result.get("detections", []):
            if det.get("confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD:
                review_required = True
                break
        if not result.get("detections"):
            review_required = True
        if review_required:
            break

    return {
        "assessment_id": f"ASM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_label,
        "confidence_threshold": confidence_threshold,
        "image_size": image_size,
        "images_analyzed": len(results),
        "results": results,
        "total_detections": len(all_detections),
        "estimated_repair_cost_min": total_cost_min,
        "estimated_repair_cost_max": total_cost_max,
        "settlement_offer": settlement_offer,
        "settlement_min": settlement_min,
        "settlement_max": settlement_max,
        "severity": severity,
        "primary_component": primary_component,
        "damage_summary": summary,
        "review_required": review_required,
    }


# ─── Hybrid AI Assessment Endpoint ───────────────────────────────────────────
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
    # Combined with local pricing
    estimated_repair_cost_min: int
    estimated_repair_cost_max: int
    settlement_offer: int
    settlement_min: int
    settlement_max: int
    severity: str
    primary_component: str
    damage_summary: str
    review_required: bool


@app.post("/assess/hybrid", response_model=HybridAssessmentResponse)
async def assess_hybrid(request: HybridAssessmentRequest):
    """
    Hybrid AI assessment combining YOLO detections with multimodal vision/LLM reasoning.
    
    Requires:
    - Vehicle info (make, model, year, variant)
    - Base64 encoded images for each view (top, bottom, left, right)
    - YOLO detections per view from the four-view assessment
    """
    # Validate AI provider is configured
    if AI_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key not configured")
    if AI_PROVIDER == "openai" and not OPENAI_API_KEY:
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
    
    # Run YOLO inference first (if not already provided, but we use provided detections)
    # Convert detections to our format
    detections_by_view = {}
    for view in REQUIRED_VIEWS:
        view_dets = request.detections.get(view, [])
        detections_by_view[view] = {
            "detection_count": len(view_dets),
            "detections": [d.model_dump() for d in view_dets],
        }
    
    # Run hybrid AI assessment
    try:
        ai_result = await run_hybrid_assessment(
            request.vehicle,
            views_images,
            detections_by_view,
            request.confidence_threshold,
            request.image_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid AI assessment failed: {type(e).__name__}: {e}")
    
    # Estimate costs using local pricing logic (combine with AI reasoning)
    all_ai_detections = []
    for item in ai_result.get("damageItems", []):
        all_ai_detections.append({
            "damage_type": item["damageType"],
            "confidence": item.get("confidence", 0.5),
        })
    
    # Use existing cost estimation but with AI-determined severity
    if all_ai_detections:
        max_conf = max(d["confidence"] for d in all_ai_detections)
        total_count = len(all_ai_detections)
        damage_types = [d["damage_type"].lower() for d in all_ai_detections]
        from collections import Counter
        primary_damage = Counter(damage_types).most_common(1)[0][0] if damage_types else "damage"
        
        if max_conf >= 0.7 and total_count >= 3:
            severity = "severe"
        elif max_conf >= 0.5 and total_count >= 2:
            severity = "moderate"
        elif max_conf >= 0.3:
            severity = "minor"
        else:
            severity = "uncertain"
        
        cost_map = DAMAGE_COST_MAP.get(primary_damage, DAMAGE_COST_MAP["damage"])
        min_cost, max_cost = cost_map.get(severity, cost_map["minor"])
        if total_count > 1:
            min_cost = int(min_cost * (1 + 0.3 * (total_count - 1)))
            max_cost = int(max_cost * (1 + 0.3 * (total_count - 1)))
    else:
        severity = "uncertain"
        min_cost = 0
        max_cost = 0
        primary_damage = "damage"
    
    component_map = {
        "scratch": "Body Panel",
        "dent": "Body Panel",
        "crack": "Bumper/Fender",
        "glass_breakage": "Windshield/Window",
        "paint_damage": "Paint Surface",
        "deformation": "Structural Panel",
        "damage": "Vehicle Component",
    }
    primary_component = component_map.get(primary_damage, "Vehicle Component")
    unique_types = list(set([d["damage_type"].lower() for d in all_ai_detections]))
    summary = f"Hybrid AI: {len(all_ai_detections)} damage instance(s): {', '.join(unique_types)}. Primary: {primary_damage} ({severity})."
    
    settlement_offer = int((min_cost + max_cost) / 2 * 0.7) if min_cost > 0 else 0
    settlement_min = int(min_cost * 0.55) if min_cost > 0 else 0
    settlement_max = int(max_cost * 0.85) if max_cost > 0 else 0
    
    review_required = ai_result.get("confidence", 1.0) < 0.7 or len(ai_result.get("duplicate_warnings", [])) > 0
    
    return HybridAssessmentResponse(
        assessment_id=f"HASM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        created_at=datetime.now(timezone.utc).isoformat(),
        workflow="hybrid_visual_damage_assessment",
        model_name=f"YOLO + {AI_PROVIDER.capitalize()}",
        ai_provider=AI_PROVIDER,
        ai_model=GEMINI_MODEL if AI_PROVIDER == "gemini" else OPENAI_MODEL,
        damage_items=ai_result.get("damageItems", []),
        ai_observations=ai_result.get("aiObservations", []),
        duplicate_warnings=ai_result.get("duplicateDetectionWarnings", []),
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


# ─── Provider Status Endpoint ───────────────────────────────────────────────
@app.get("/ai/provider/status")
async def ai_provider_status():
    """Get the status of configured AI providers."""
    return {
        "provider": AI_PROVIDER,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "hybrid_available": (AI_PROVIDER == "gemini" and bool(GEMINI_API_KEY)) or (AI_PROVIDER == "openai" and bool(OPENAI_API_KEY)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)