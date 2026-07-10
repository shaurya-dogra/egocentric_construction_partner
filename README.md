# 🏗️ Job Site Safety Copilot

> A real-time, fully offline AI safety assistant for construction sites. Combines high-speed edge computer vision with asynchronous multimodal VLM reasoning to detect hazards, track workers, estimate distances, and deliver spoken safety warnings — all running locally on Apple Silicon.

Built for the **Kaya Hackathon** by Team Antigravity.

---

## Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [AI Models](#ai-models)
- [Pipeline Workflow](#pipeline-workflow)
- [User Flow](#user-flow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration Reference](#configuration-reference)
- [Database & Logging](#database--logging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How It Works

The Safety Copilot processes a live camera feed (webcam, video file, or smart-glasses stream) and performs real-time safety analysis:

1. **Detects** workers, vehicles, machinery, tools, and PPE using multiple YOLO models
2. **Tracks** every object across frames with persistent IDs (BoT-SORT + ReID)
3. **Estimates depth** using monocular depth estimation (Depth Anything V2) to calculate real-world distances in meters
4. **Analyzes poses** to detect falls and estimate worker gaze direction
5. **Evaluates hazards** — vehicle proximity, danger zone intrusion, PPE violations, falls
6. **Tracks attention** — determines if a worker has *noticed* a nearby hazard by checking their gaze direction. If unnoticed for 4+ seconds, escalates the alert
7. **Speaks warnings** via text-to-speech with directional cues ("Vehicle approaching on your left, 3.2 meters away")
8. **Reasons asynchronously** — a background VLM (Gemma 4 / Gemini / Llama Vision) performs semantic scene understanding, task identification, and predictive hazard projection every ~4 seconds

The system supports two perspective modes:

| Mode | Use Case | How It Works |
|------|----------|--------------|
| **Egocentric** | Smart glasses / wearable camera | Camera = worker's eyes. Alerts say "on your left/right/ahead". Gaze = screen center. |
| **Third Person** | Fixed surveillance camera | Overhead view. Tracks multiple workers. Gaze estimated from head yaw + pose keypoints. |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Camera / Video Stream                          │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────┐
│  ⚡ TIER 1: EDGE     │         │  🧠 TIER 2: VLM          │
│  ~15-30 FPS          │         │  ~0.25 FPS (async)       │
│                     │         │                         │
│  Detection+Tracking │         │  Scene Graph (NetworkX) │
│  ├─ YOLO26 (COCO)   │         │  Spatial Memory (3×3)   │
│  ├─ PPE Model (HF)  │  ────►  │  Session Memory         │
│  ├─ YOLO-World      │ scene   │  Document RAG           │
│  │  (tools)         │ state   │  Blueprint Context      │
│  └─ BoT-SORT+ReID   │         │  ─────────────────────  │
│                     │         │  Gemma4 / Gemini /      │
│  Depth Estimation   │         │  Llama 3.2 Vision       │
│  └─ Depth Anything  │         │  ─────────────────────  │
│     V2 Small        │         │  → Task Identification  │
│                     │         │  → Hazard Prediction    │
│  Pose Estimation    │         │  → SOP Compliance Q&A   │
│  └─ YOLO26-Pose     │         │  → Checklist Tracking   │
│     (17 keypoints)  │         └────────────┬────────────┘
│                     │                      │
│  Safety Analysis    │◄─────────────────────┘
│  ├─ Hazard Analyzer │        predictions
│  ├─ Fall Detector   │
│  ├─ Zone Manager    │
│  ├─ PPE Checker     │
│  └─ Attention       │
│     Tracker (novel) │
│                     │
│  Alert Dispatch     │
│  ├─ TTS Engine      │
│  └─ HUD Overlay     │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  📊 SQLite Logger    │
│  (events.db)        │
└─────────────────────┘
```

### Tier 1 — Real-Time Edge Processing

Runs at full frame rate on Apple Silicon (MPS GPU). Handles all latency-critical safety checks with zero network dependency.

| Component | Module | What It Does |
|-----------|--------|--------------|
| Object Detection | `core/detector.py` | Dual/triple YOLO model pipeline: general COCO objects + PPE items + open-vocab tools |
| Multi-Object Tracking | `core/detector.py` | BoT-SORT with ReID appearance embeddings for persistent IDs across occlusions |
| Depth Estimation | `core/depth_estimator.py` | Monocular metric depth from Depth Anything V2 — estimates real-world distance to every object |
| Pose Estimation | `core/pose_estimator.py` | 17-keypoint COCO skeleton + head yaw angle + body torso angle (runs every Nth frame) |
| Hazard Analysis | `safety/hazard_analyzer.py` | Fuses PPE, fall, zone, and proximity signals into scored `HazardAssessment` objects |
| Fall Detection | `safety/fall_detector.py` | Body angle + vertical velocity with multi-frame confirmation state machine |
| PPE Checking | `safety/ppe_checker.py` | Spatial IoU matching between worker bboxes and PPE detections |
| Zone Management | `safety/zones.py` | Polygon-based danger zones with point-in-polygon intersection tests |
| Attention Tracking | `safety/attention_tracker.py` | **Novel feature** — gaze-aware hazard escalation based on worker head direction |
| Alert Manager | `alerts/alert_manager.py` | Priority-based alert dispatch with cooldowns, dedup, and severity-aware speech |
| TTS Engine | `alerts/tts_engine.py` | macOS native speech synthesis with per-severity speech rate control |
| HUD Overlay | `display/overlay.py` | OpenCV renderer for bboxes, skeletons, zones, gaze lines, banners, distance labels |

### Tier 2 — Asynchronous VLM Reasoning

Runs in a background thread at ~0.25 FPS. Provides semantic understanding that rule-based systems cannot.

| Component | Module | What It Does |
|-----------|--------|--------------|
| VLM Hook | `integration/vlm_hook.py` | Frame optimization (448px, JPEG-35, base64), multi-endpoint routing, JSON response parsing |
| Scene Graph | `reasoning/scene_graph.py` | NetworkX graph of workers, vehicles, tools, structures, zones with spatial relationship edges |
| Spatial Memory | `reasoning/spatial_memory.py` | 2D grid heat map tracking worker dwell history |
| Session Memory | `reasoning/session_memory.py` | Rolling buffer of previous VLM outputs for temporal context |
| Document RAG | `reasoning/document_store.py` | SOP document ingestion, chunking, and retrieval (direct or embedding-based) |
| Blueprint Store | `reasoning/blueprint_store.py` | PDF/CAD blueprint text extraction and page rendering for VLM vision input |
| Reasoning Coordinator | `reasoning/pipeline.py` | Orchestrates all context sources into a compact VLM prompt with jitter-filtered state hashing |

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core application |
| **Deep Learning** | PyTorch 2.3+ (MPS backend) | GPU inference on Apple Silicon |
| **Object Detection** | Ultralytics YOLO26 | Workers, vehicles, PPE, machinery detection |
| **Open-Vocab Detection** | YOLOv8s-World v2 | Zero-shot tool detection (hammers, drills, saws) |
| **Pose Estimation** | YOLO26-Pose | 17-keypoint skeleton + head yaw estimation |
| **Multi-Object Tracking** | BoT-SORT + ReID (OSNet) | Persistent identity tracking across occlusions |
| **Depth Estimation** | Depth Anything V2 Small (HuggingFace) | Monocular metric distance estimation |
| **VLM Reasoning** | Gemma 4 / Gemini 1.5 Flash / Llama 3.2 Vision | Semantic scene understanding + predictive hazards |
| **Scene Graphs** | NetworkX | Dynamic spatial relationship modeling |
| **Document RAG** | Ollama `nomic-embed-text` | SOP document embedding and retrieval |
| **Speech** | macOS `say` / pyttsx3 | Real-time text-to-speech alerts |
| **Video Processing** | OpenCV 4.9+ | Capture, rendering, overlays |
| **Database** | SQLite | Persistent event and alert logging |
| **Configuration** | PyYAML | YAML-based configuration |
| **PDF Processing** | PyPDF + PyMuPDF (fitz) | Blueprint text extraction and page rendering |

---

## AI Models

The system loads up to **5 AI models** simultaneously, all running locally:

| Model | File | Size | Device | Frequency | Purpose |
|-------|------|------|--------|-----------|---------|
| **YOLO26 Nano** | `yolo26n.pt` | 5.5 MB | MPS | Every frame | General object detection (COCO 80 classes) |
| **PPE Detector** | HuggingFace / Roboflow | ~5.5 MB | MPS | Every frame | Hardhat, vest, mask, goggles, boots, gloves, safety cone, machinery, vehicle |
| **YOLO26 Nano Pose** | `yolo26n-pose.pt` | 7.9 MB | MPS | Every 3rd frame | 17-keypoint skeleton, head yaw, body angle |
| **YOLOv8s-World v2** | `yolov8s-worldv2.pt` | 25.9 MB | MPS | Every frame | Open-vocabulary tool detection: hammer, drill, saw, measuring tape |
| **Depth Anything V2 Small** | HuggingFace (cached) | ~100 MB | MPS | Every 3rd frame | Monocular depth map → metric distance in meters |

### VLM (Tier 2 — Background)

| Model | Endpoint | Use Case |
|-------|----------|----------|
| **Gemma 4** (`gemma4:cloud`) | Ollama local (`:11434`) | Default — auto-routes to Gemini or runs locally |
| **Gemini 1.5 Flash** | `generativelanguage.googleapis.com` | Cloud fallback for `gemma4:cloud` |
| **Llama 3.2 11B Vision** | `api.groq.com` | Alternative cloud VLM via Groq |
| **nomic-embed-text** | Ollama local | Document embedding for RAG retrieval |

### Tracking

| Tracker | Config | Features |
|---------|--------|----------|
| **BoT-SORT + ReID** | `models/botsort_reid.yaml` | Appearance embeddings (OSNet / CLIP ViT-B/32), `track_buffer=60`, `match_thresh=0.8`, camera motion compensation |

---

## Pipeline Workflow

Each frame passes through the following pipeline (see `main.py` → `_process_frame()`):

```
Frame N arrives
    │
    ├─ 1. DETECT + TRACK ──────────────────────────────────────────────────
    │      YOLO26 + PPE Model + YOLO-World → Detections
    │      BoT-SORT+ReID → Persistent Track IDs
    │      Update TrackedObject dict (position history, active/inactive)
    │
    ├─ 2. DEPTH ESTIMATION (every Nth frame) ──────────────────────────────
    │      Depth Anything V2 → Full-frame depth map
    │      For each active object: median(depth_crop) → distance_meters
    │
    ├─ 3. POSE ESTIMATION (every Nth frame) ───────────────────────────────
    │      YOLO26-Pose → 17 keypoints per person
    │      Calculate head_yaw (gaze direction) + body_angle (fall indicator)
    │
    ├─ 4. HAZARD ANALYSIS ─────────────────────────────────────────────────
    │      ├─ PPE Compliance: IoU match workers ↔ PPE detections
    │      ├─ Fall Detection: body_angle + velocity → state machine
    │      ├─ Zone Proximity: worker bbox ∩ danger zone polygons
    │      ├─ Vehicle Proximity:
    │      │    ├─ Third Person: worker-vehicle pixel distance + machine exclusion zones
    │      │    └─ Egocentric: distance_meters thresholds (≤3m CRITICAL, ≤6m DANGER, ≤12m WARNING)
    │      └─ PPE Severity Modulation: missing PPE escalates hazard severity +1 tier
    │
    ├─ 5. ATTENTION TRACKING (novel) ──────────────────────────────────────
    │      For each hazard:
    │        PASSIVE ──(no gaze for 4s)──→ UNNOTICED ──(cooldown)──→ ESCALATED
    │        PASSIVE ──(worker looks at hazard for 0.5s)──→ ACKNOWLEDGED (silences alert)
    │      State persists across frames. Escalation bumps severity +1 tier (capped).
    │
    ├─ 6. VLM ESCALATION (if available) ───────────────────────────────────
    │      Any ESCALATED hazard → send scene to VLM for semantic analysis
    │
    ├─ 7. ALERT GENERATION ────────────────────────────────────────────────
    │      Filter by state (skip PASSIVE unless DANGER+, skip ACKNOWLEDGED)
    │      Sort by severity (CRITICAL > DANGER > WARNING > INFO)
    │      Speak highest-priority alert (with cooldown + interrupt logic)
    │      Build directional message: "Vehicle approaching on your left, 3.2 meters away"
    │
    ├─ 8. EVENT LOGGING ───────────────────────────────────────────────────
    │      Log UNNOTICED/ESCALATED hazards + all alerts → SQLite
    │
    └─ 9. BACKGROUND VLM REASONING (async, ~every 4s) ────────────────────
           Build scene graph → spatial memory → session memory → document RAG
           Compress frame (448px, JPEG Q=35) → base64
           Send to VLM → parse structured JSON response
           Extract: tasks, predictions, checklists, guidance
           High-confidence predictions → create predicted_hazard alerts
```

---

## User Flow

### Starting the Application

```bash
# Live webcam
python main.py --source webcam

# Video file
python main.py --source path/to/video.mp4

# Headless mode (no window, for testing)
python main.py --source video.mp4 --no-display --no-voice

# Force CPU (disable Apple Silicon GPU)
python main.py --source webcam --device cpu

# Load custom danger zones
python main.py --source webcam --zones data/zones/site_a.json
```

### What You See (HUD Overlay)

| Element | Description |
|---------|-------------|
| **Green bounding boxes** | Tracked workers with `person 87% #3 [4.2m]` labels |
| **Orange bounding boxes** | Vehicles / machinery with distance labels |
| **Teal / Red PPE boxes** | PPE items detected (teal = worn, red = missing) |
| **Violet bounding boxes** | Detected tools (hammer, drill, saw) |
| **Skeleton overlay** | 17-keypoint pose with colored limbs (blue=left, red=right, green=center) |
| **Yellow arrow** | Gaze direction line from nose keypoint |
| **Semi-transparent polygons** | Danger zones (orange) and machine exclusion zones |
| **Hazard border** | Color-coded by state: yellow=PASSIVE, orange=UNNOTICED, red=ESCALATED, green=ACKNOWLEDGED |
| **Dwell timer** | Shows how long a hazard has been active (e.g., "4.1s") |
| **Violet "carrying" lines** | Lines from worker wrists to nearby tools |
| **Top banner** | Most severe active alert message (color-coded) |
| **FPS counter** | Top-right corner |

### What You Hear (TTS Alerts)

Alerts are spoken through macOS text-to-speech with severity-scaled speech rate:

| Severity | Speech Rate | Example Message |
|----------|-------------|-----------------|
| **INFO** | 160 wpm | "Worker near restricted area." |
| **WARNING** | 180 wpm | "Vehicle approaching on your left." |
| **DANGER** | 220 wpm | "Warning! Vehicle approaching closely ahead! 4.2 meters away" |
| **CRITICAL** | 240 wpm | "CRITICAL! Vehicle collision risk on your right! Move away NOW! 2.1 meters away" |

Alert behavior:
- **Cooldown**: Same hazard won't re-alert for 5 seconds
- **Priority**: Only the highest-severity alert is spoken; lower ones are visual-only
- **Interruption**: A higher-severity alert interrupts an active lower-severity speech
- **Acknowledgment**: If the worker looks at the hazard (gaze check), the alert speaks once then silences
- **Escalation**: If unnoticed for 4+ seconds, severity bumps up one tier and re-alerts

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit the application |
| `s` | Silence all active TTS alerts |
| `v` | **Interactive VLM Q&A** — pauses the video, prompts for a question, sends it to the VLM with the current scene context, prints and speaks the answer |

### Interactive VLM Q&A (press `v`)

```
==================================================
🎤 Ask the Safety Assistant a question: What is the worker doing?
Thinking...

🤖 VLM: Worker-1 is operating the backhoe loader, positioned in the cab
   near a freshly dug trench. The machine appears to be performing
   excavation work based on the dirt pile positioning.
==================================================
```

---

## Project Structure

```
kaya-hackathon/
│
├── main.py                          # Application entry point & pipeline orchestrator
├── config.yaml                      # Central configuration (all thresholds, model paths, VLM settings)
├── requirements.txt                 # Python dependencies
├── download_model.py                # Utility to download Roboflow PPE model weights
│
├── core/                            # ── Tier 1: Perception Engine ──
│   ├── capture.py                   #    Video stream ingestion (webcam / file / image)
│   ├── detector.py                  #    Multi-model YOLO detection + BoT-SORT tracking
│   ├── pose_estimator.py            #    YOLO26-Pose keypoint extraction + head yaw
│   ├── depth_estimator.py           #    Depth Anything V2 monocular depth estimation
│   ├── device.py                    #    Apple Silicon MPS device detection + smoke test
│   └── models.py                    #    All shared dataclasses (Detection, TrackedObject,
│                                    #    HazardAssessment, Alert, Severity, etc.)
│
├── safety/                          # ── Safety Analysis ──
│   ├── hazard_analyzer.py           #    Master analyzer: fuses all safety signals
│   ├── attention_tracker.py         #    Novel gaze-aware hazard escalation engine
│   ├── fall_detector.py             #    Pose-based fall detection state machine
│   ├── ppe_checker.py               #    Spatial PPE compliance matching
│   └── zones.py                     #    Danger zone polygon management
│
├── alerts/                          # ── Alert Delivery ──
│   ├── alert_manager.py             #    Priority-based alert orchestration + dedup
│   └── tts_engine.py                #    macOS native text-to-speech engine
│
├── display/                         # ── Visual Overlay ──
│   └── overlay.py                   #    OpenCV HUD renderer (bboxes, skeletons, banners, zones)
│
├── reasoning/                       # ── Tier 2: VLM Reasoning ──
│   ├── pipeline.py                  #    Main reasoning coordinator + VLM prompt assembly
│   ├── scene_graph.py               #    NetworkX scene graph builder (nodes & edges)
│   ├── spatial_memory.py            #    2D grid heat map of worker positions
│   ├── session_memory.py            #    Rolling buffer of previous VLM outputs
│   ├── document_store.py            #    RAG document ingestion + embedding retrieval
│   ├── blueprint_store.py           #    PDF/CAD blueprint loader + renderer
│   └── models.py                    #    Tier 2 dataclasses (TaskEstimate, HazardPrediction, etc.)
│
├── integration/                     # ── VLM Bridge ──
│   └── vlm_hook.py                  #    Background-threaded VLM dispatcher, multi-endpoint
│                                    #    routing, frame optimization, JSON response parsing
│
├── logging_/                        # ── Persistent Logging ──
│   ├── event_logger.py              #    SQLite event logger (thread-safe)
│   └── schemas.py                   #    Database schema definitions (4 tables)
│
├── models/                          # ── Tracker Configs ──
│   ├── botsort_reid.yaml            #    BoT-SORT + ReID tracker (production)
│   └── botsort_reid_test.yaml       #    Deterministic tracker config (testing)
│
├── weights/                         # ── Pre-downloaded Weights ──
│   └── clip/ViT-B-32.pt            #    CLIP ViT-B/32 for ReID appearance embeddings (354 MB)
│
├── data/                            # ── Runtime Data ──
│   ├── events.db                    #    SQLite database (hazards, alerts, reasoning logs)
│   ├── documents/                   #    SOP PDFs/text files for RAG queries
│   ├── blueprints/                  #    PDF/image blueprints for VLM context
│   │   └── rendered/                #    Rendered blueprint page images
│   └── indexes/                     #    Document embedding indexes
│
├── tests/                           # ── Unit Tests ──
│   ├── test_depth.py                #    Depth estimation distance calculation tests
│   ├── test_ollama_fallbacks.py     #    VLM endpoint fallback URL tests
│   ├── test_reasoning_document_store.py  #  Document RAG retrieval tests
│   └── test_reasoning_scene_graph.py     #  Scene graph serialization tests
│
├── test_videos/                     # ── Sample Videos ──
│   └── *.mp4                        #    Construction site footage for development/demo
│
├── yolo26n.pt                       # YOLO26 nano detection weights (COCO)
├── yolo26n-pose.pt                  # YOLO26 nano pose estimation weights
├── yolo26n-cls.pt                   # YOLO26 nano classification weights
└── yolov8s-worldv2.pt               # YOLOv8s-World open-vocab tool detection weights
```

---

## Getting Started

### Prerequisites

- **macOS** (Apple Silicon recommended for MPS GPU acceleration)
- **Python 3.10+**
- **Ollama** (optional, for local VLM reasoning)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd kaya-hackathon

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for depth estimation (not in requirements.txt)
pip install transformers pillow
```

### VLM Setup (Optional — Tier 2 Reasoning)

The system works fully without a VLM (Tier 1 only). To enable Tier 2:

**Option A — Local via Ollama:**
```bash
# Install Ollama: https://ollama.ai
ollama pull gemma4:cloud
ollama pull nomic-embed-text    # For document RAG
```

**Option B — Google Gemini (Cloud):**
```bash
# Set environment variable
export GOOGLE_API_KEY="your-api-key"
```
The system auto-routes `gemma4:cloud` to `generativelanguage.googleapis.com` when Ollama is unavailable.

**Option C — Groq (Cloud):**
```yaml
# In config.yaml:
vlm:
  api_url: "https://api.groq.com/openai/v1/chat/completions"
  model: "llama-3.2-11b-vision-preview"
  api_key: "your-groq-api-key"
```

### Run

```bash
# Webcam (live)
python main.py --source webcam

# Video file
python main.py --source test_videos/13771068_1920_1080_60fps.mp4

# Headless + silent (for testing/benchmarking)
python main.py --source video.mp4 --no-display --no-voice
```

---

## Configuration Reference

All configuration lives in `config.yaml`. Key parameters:

### Perspective Mode

```yaml
perspective: "egocentric"     # "egocentric" (smart glasses) | "third_person" (surveillance)
```

### Model Paths & Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models.general.path` | `null` | COCO model (null = disabled, rely on PPE model) |
| `models.ppe.path` | HuggingFace model | PPE detection model (Roboflow ID or HF path) |
| `models.pose.run_every_n_frames` | `3` | Pose runs every Nth frame to save compute |
| `models.tool.enabled` | `true` | Enable open-vocab tool detection |
| `models.tool.classes` | `[hammer, drill, saw, measuring tape]` | Tool categories to detect |
| `models.depth.enabled` | `true` | Enable monocular depth estimation |
| `models.depth.scale_factor` | `15.0` | Calibration: `distance_meters = scale_factor / relative_depth` |
| `models.depth.run_every_n_frames` | `3` | Depth runs every Nth frame |

### Safety Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `escalation.dwell_threshold_seconds` | `4.0` | Time before unnoticed hazard escalates |
| `escalation.gaze_angle_tolerance_degrees` | `45.0` | ±degrees to count as "looking at" hazard |
| `escalation.acknowledgment_gaze_duration_seconds` | `0.5` | Sustained gaze time to acknowledge |
| `escalation.escalation_cooldown_seconds` | `10.0` | Cooldown between re-escalations |
| `fall_detection.body_angle_threshold_degrees` | `30.0` | Torso angle from vertical to trigger |
| `fall_detection.confirmation_frames` | `2` | Consecutive suspicious frames required |
| `alerts.cooldown_seconds` | `5.0` | Min gap between alerts for same hazard |

### Vehicle Proximity Thresholds (Egocentric Mode)

| Distance | Severity | Alert Example |
|----------|----------|---------------|
| **≤ 3.0 m** | 🔴 CRITICAL | "CRITICAL! Vehicle collision risk ahead!" |
| **≤ 6.0 m** | 🟠 DANGER | "Warning! Vehicle approaching closely on your left!" |
| **≤ 12.0 m** | 🟡 WARNING | "Vehicle approaching on your right." |
| **> 12.0 m** | — | No alert |

### VLM Reasoning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm.enabled` | `true` | Enable Tier 2 VLM reasoning |
| `vlm.model` | `gemma4:cloud` | VLM model identifier |
| `vlm.fps` | `0.25` | VLM polling rate (~every 4 seconds) |
| `vlm.prediction_alert_threshold` | `0.65` | Min confidence for VLM predictions to become alerts |
| `vlm.reasoning.reason_every_seconds` | `4.0` | Min interval between VLM calls |
| `vlm.reasoning.documents.directory` | `data/documents` | Folder for SOP documents (PDF/text) |

---

## Database & Logging

All events are logged to `data/events.db` (SQLite) with 4 tables:

| Table | Contents |
|-------|----------|
| `hazards` | Every detected hazard with type, severity, state, worker ID, timestamp |
| `alerts` | Every spoken/visual alert with message text, severity, escalation flag |
| `events` | Generic pipeline events (detections, acknowledgments, resolutions) |
| `reasoning_events` | VLM reasoning outputs (tasks, predictions, checklists, guidance) |

### Query Examples

```sql
-- Recent critical alerts
SELECT timestamp, message, severity FROM alerts
WHERE severity = 'critical' ORDER BY timestamp DESC LIMIT 20;

-- Hazard escalation history for a specific worker
SELECT * FROM hazards
WHERE worker_track_id = 3 AND state IN ('unnoticed', 'escalated')
ORDER BY timestamp;

-- VLM predictions
SELECT timestamp, output FROM reasoning_events
ORDER BY timestamp DESC LIMIT 10;
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **MPS device error** | The system auto-detects and falls back to CPU. Force with `--device cpu` |
| **VLM not responding** | Check Ollama is running (`ollama serve`). The system works without it (Tier 1 only) |
| **NumPy matmul warnings** | Already suppressed in `main.py`. Caused by BoT-SORT ReID internals — harmless |
| **Depth model slow to load** | First run downloads ~100MB from HuggingFace. Subsequent runs use cache (`~/.cache/huggingface/`) |
| **No PPE detections** | Verify `models.ppe.path` in `config.yaml`. Use `download_model.py` to fetch Roboflow weights |
| **TTS not working** | macOS only. Ensure `say` command works in terminal. Check `alerts.voice` matches an installed voice (`say -v ?`) |
| **Low FPS** | Increase `pose.run_every_n_frames` and `depth.run_every_n_frames`. Disable tool model (`tool.enabled: false`) |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_depth.py -v
```

---

## License

MIT License
