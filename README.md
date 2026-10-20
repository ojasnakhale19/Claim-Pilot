# ClaimPilot AI — Vehicle Damage Assessment & Settlement Platform

A complete, production-ready platform for AI-powered vehicle damage assessment and insurance claim settlement. Combines a cinematic React frontend with a dual-model FastAPI backend featuring policy-driven decision governance.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLAIMPILOT AI                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript + Vite)                          │
│  ├── Cinematic automotive UI (dark theme, 3D animations)       │
│  ├── 5-stage claim flow: Upload → Scan → Report → Negotiate    │
│  └── REST API client (lib/utils.ts)                            │
├─────────────────────────────────────────────────────────────────┤
│  Backend (FastAPI + Python) — Dual Model Architecture          │
│  ├── /assess/general    → HuggingFace YOLOv11 (damage_best.pt) │
│  │   Multi-view, settlement offers, hybrid AI (Gemini/OpenAI)  │
│  ├── /assess/four-view  → Structured 4-view inspection         │
│  ├── /assess/hybrid     → LLM-enhanced reasoning               │
│  └── /assess/           → CarDD YOLOv8n + Policy Engine        │
│      CarDD-trained (10 classes, mAP@50: 0.581)                 │
│      Deterministic decisions: AUTO_APPROVE / HUMAN_REVIEW / ESCALATE │
│      Full decision trace + SOP evidence + human override log   │
├─────────────────────────────────────────────────────────────────┤
│  Governance Layer                                               │
│  ├── policies/rules.yaml     — Explicit decision rules          │
│  ├── policies/damage_triage.md — SOP evidence & next steps     │
│  ├── knowledge/              — RAG knowledge base               │
│  └── Human override audit trail (immutable, timestamped)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- **Node.js 18+** (frontend)
- **Python 3.11+** (backend)
- **Git**

### 1. Clone & Install Frontend
```bash
git clone <repository-url>
cd claimpilot-ai
npm install
```

### 2. Setup Backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Backend
cd backend
cp .env.example .env
# Edit .env with your API keys (optional):
# GEMINI_API_KEY=xxx      # For hybrid AI assessment
# OPENAI_API_KEY=xxx      # Alternative LLM provider
# AI_PROVIDER=gemini      # or openai
```

### 4. Run Development Servers

**Terminal 1 - Backend (port 8000):**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend (port 5173):**
```bash
npm run dev
```

Open **http://localhost:5173** — the frontend proxies API calls to `http://localhost:8000`.

---

## Backend API Endpoints

### Policy-Driven Assessment (New — CarDD + Decision Engine)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/assess/health` | Model + policy engine status |
| `POST` | `/assess/` | **Main endpoint** — single image → detections + policy decision + trace |
| `GET` | `/assess/policy` | Current rules.yaml + SOP configuration |

**Example Response (`/assess/`):**
```json
{
  "assessment_id": "a1b2c3d4e5f6",
  "timestamp": "2026-10-20T12:30:00+00:00",
  "processing_time_ms": 120,
  "damages_detected": [
    {
      "damage_type": "scratch",
      "confidence": 0.87,
      "severity": "minor",
      "location": "rear_bumper",
      "bbox": [100, 200, 300, 250],
      "area_percentage": 2.5,
      "estimated_cost": 150
    }
  ],
  "total_damages": 1,
  "decision": "AUTO_APPROVE",
  "decision_confidence": 0.8,
  "decision_trace": [
    {
      "rule_applied": "auto_approve_minor_scratch",
      "threshold": "damage_type_in=['scratch'] & severity_in=['minor'] & confidence_gte=0.85",
      "evidence": "SOP: Minor scratch with high confidence can be auto-approved..."
    }
  ],
  "model_version": "yolov8n-cardd-v1",
  "policy_version": "v1.0",
  "cv_backend": "yolov8",
  "human_review_required": false
}
```

### Original ClaimPilot Endpoints (Preserved)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/assess/general` | Multi-image, settlement offers, HF YOLOv11 |
| `POST` | `/assess/four-view` | Structured 4-view (top/bottom/left/right) |
| `POST` | `/assess/hybrid` | LLM-enhanced (Gemini/OpenAI) |
| `GET` | `/model/info` | Model metadata |
| `GET` | `/ai/provider/status` | LLM provider config |

---

## Decision Governance (Policy Engine)

The `/assess/` endpoint uses a **deterministic policy engine** — no LLM involved in core decisions.

### Policy Files
```
policies/
├── rules.yaml           # Decision rules (versioned)
└── damage_triage.md     # SOP evidence & next steps

knowledge/
├── repair_playbook.md   # Repair guidance
└── sla_notes.md         # Service level agreements
```

### Decision Rules (`rules.yaml`)
```yaml
version: 1
thresholds:
  auto_approve_conf: 0.85
  human_review_conf: 0.55

rules:
  - name: auto_approve_minor_scratch
    if:
      damage_type_in: ["scratch"]
      severity_in: ["minor"]
      confidence_gte: 0.85
    then:
      action: "AUTO_APPROVE"
      sop_ref: "damage_triage.md#minor-scratch"
      reason: "High-confidence minor scratch matches auto-approval policy."

  - name: human_review_moderate_dent
    if:
      damage_type_in: ["dent"]
      severity_in: ["moderate"]
      confidence_gte: 0.55
    then:
      action: "HUMAN_REVIEW"
      sop_ref: "damage_triage.md#moderate-damage"
      reason: "Moderate dent should be reviewed by an operator before approval."

  - name: escalate_severe_damage
    if:
      severity_in: ["severe"]
      confidence_gte: 0.55
    then:
      action: "ESCALATE"
      sop_ref: "damage_triage.md#severe-damage"
      reason: "Severe damage should be escalated for specialist assessment."

  - name: low_confidence_human_review
    if:
      confidence_lt: 0.55
    then:
      action: "HUMAN_REVIEW"
      sop_ref: "damage_triage.md#low-confidence"
      reason: "Low confidence: route to human review."
```

### Human Override (Governance)
Every decision can be overridden by a human operator:
- Override action: `AUTO_APPROVE` | `HUMAN_REVIEW` | `ESCALATE`
- Override reason: `False severity` | `Poor image quality` | `Hidden damage suspected` | `Vehicle type/context mismatch` | `Other`
- Mandatory comment field
- Immutable audit log with timestamp, original decision, override details

---

## CarDD Model Details

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8n (nano) |
| **Dataset** | CarDD (4,000+ images, 9,000+ annotations) |
| **Classes** | 10: crack, crash, dent, dislocated_part, glass_shatter, lamp_broken, no_part, rub, scratch, tire_flat |
| **mAP@50** | 0.581 |
| **mAP@50-95** | 0.409 |
| **Precision** | 0.675 |
| **Recall** | 0.534 |
| **Image Size** | 640×640 |
| **Epochs** | 50 |
| **Model Size** | 5.9 MB |
| **Inference** | CPU real-time (~50ms) |

Training notebook: `notebooks/train_yolov8_cardd.ipynb` (reproducible on Google Colab T4 GPU, ~20 min)

---

## Docker Deployment

### Build & Run (CPU)
```bash
docker build -t claimpilot-ai:cpu .
docker run --rm -p 8000:8000 -p 5173:5173 claimpilot-ai:cpu
```

### Docker Compose (Production)
```bash
docker-compose up --build
```

**Services:**
- `backend` — FastAPI on port 8000
- `frontend` — Nginx + Vite preview on port 5173

---

## Project Structure

```
claimpilot-ai/
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── backend/
│   ├── main.py                    # FastAPI app (dual-model endpoints)
│   ├── damage_assessment.py       # Policy-driven /assess/ router
│   ├── car_damage_detector.py     # CarDD YOLOv8 wrapper
│   ├── utils.py                   # Shared utilities
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # Environment template
│   ├── agentic/                   # Decision engine
│   │   ├── decision_agent.py      # Policy evaluation
│   │   ├── adapters.py            # CV → signal conversion
│   │   ├── schemas.py             # Pydantic dataclasses
│   │   ├── trace.py               # Decision trace builder
│   │   ├── explainer.py           # Customer explanations
│   │   ├── policy_loader.py       # YAML policy parser
│   │   ├── sop_retriever.py       # SOP section extraction
│   │   ├── rag/simple_retriever.py # Knowledge RAG
│   │   ├── strategies.py          # Repair strategy simulator
│   │   └── vision/                # Inpainting/after-preview
│   ├── policies/                  # Decision policies
│   │   ├── rules.yaml             # Explicit rules
│   │   └── damage_triage.md       # SOP evidence
│   ├── knowledge/                 # RAG knowledge base
│   │   ├── repair_playbook.md
│   │   └── sla_notes.md
│   ├── models/
│   │   └── best.pt                # CarDD-trained YOLOv8n (5.9 MB)
│   ├── tests/
│   │   └── test_api.py            # 22 endpoint tests
│   └── pytest.ini
├── src/                           # React frontend
│   ├── components/
│   │   ├── cinematic/             # 3D wheel, light sweeps, upload bay
│   │   └── ui/                    # shadcn/ui components
│   ├── pages/ClaimFlowPage.tsx    # 5-stage workflow
│   ├── lib/utils.ts               # API client + helpers
│   └── ...
├── notebooks/
│   └── train_yolov8_cardd.ipynb   # Model training (Colab)
├── tests/                         # Backend integration tests
├── docs/screenshots/              # UI documentation
├── data/sample_images/            # Test images
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── LICENSE
├── pytest.ini
└── README.md
```

---

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
# 22 tests covering:
# - Health checks
# - Image validation
# - Assessment endpoints (general, four-view, hybrid)
# - Policy endpoint
# - Error handling
```

### Frontend Linting
```bash
npm run lint
```

---

## Environment Variables

### Backend (`.env`)
```env
# Model Configuration
AUTOSETTLE_MODEL_PATH=              # Local path to custom model (optional)
AUTOSETTLE_HF_REPO=shanexf/car-damage-yolo11m-ft
AUTOSETTLE_HF_FILENAME=damage_best.pt

# Image Validation Thresholds
AUTOSETTLE_MIN_IMAGE_SIDE=240
AUTOSETTLE_MIN_LAPLACIAN_VARIANCE=18
AUTOSETTLE_NEAR_DUPLICATE_DISTANCE=3
AUTOSETTLE_LOW_CONFIDENCE_THRESHOLD=0.40

# Hybrid AI (Optional)
AI_PROVIDER=gemini                  # or openai
GEMINI_API_KEY=                     # Required if AI_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash-latest
OPENAI_API_KEY=                     # Required if AI_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

### Frontend (`.env` at root)
```env
VITE_API_URL=http://localhost:8000
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **CarDD Dataset**: USTC Car Damage Detection dataset
- **Ultralytics**: YOLOv8 framework
- **FastAPI**: Modern Python web framework
- **React/Vite/Tailwind**: Frontend stack
- **shadcn/ui**: Accessible component library