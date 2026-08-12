# 🏗️ Job Site Safety Copilot

> A real-time, high-speed, fully offline AI safety assistant for construction sites. Combines high-frame-rate edge computer vision with asynchronous multimodal VLM reasoning to detect hazards, track workers, estimate depth/distances, and deliver spoken safety warnings — running locally on Apple Silicon paired with Raspberry Pi live camera streams.

Built for the **Kaya Hackathon** by Team Antigravity.

---

## 📋 Table of Contents

- [Overview & How It Works](#overview--how-it-works)
- [Raspberry Pi 5 Live Camera Streaming](#raspberry-pi-5-live-camera-streaming)
- [System Architecture](#system-architecture)
- [Performance & High-Speed Optimizations (28.7 FPS)](#performance--high-speed-optimizations-287-fps)
- [Technology Stack](#technology-stack)
- [AI Models](#ai-models)
- [Pipeline Workflow](#pipeline-workflow)
- [User Flow & Interface](#user-flow--interface)
- [Project Structure](#project-structure)
- [Getting Started & Installation](#getting-started--installation)
- [Configuration Reference](#configuration-reference)
- [Database & Logging](#database--logging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview & How It Works

The **Safety Copilot** ingests a live camera feed (webcam, video file, or an IP/Raspberry Pi camera stream over Wi-Fi/Ethernet) and performs real-time edge safety analysis:

1. **Detects** workers, vehicles, machinery, tools, and PPE using multi-model YOLO object detection.
2. **Tracks** every worker and vehicle across frames with persistent IDs using high-speed ByteTrack filtering.
3. **Estimates Monocular Depth** using Depth Anything V2 in a background thread to calculate real-world metric distances (in meters) to every tracked object.
4. **Analyzes Worker Poses** with a 17-keypoint skeleton to detect fall events and estimate head yaw / gaze direction.
5. **Evaluates Hazards** — vehicle proximity, danger zone intrusion, PPE non-compliance, fall risks.
6. **Tracks Worker Attention (Novel)** — determines if a worker has *noticed* a nearby hazard based on head yaw bearing. If unnoticed for 4+ seconds, automatically escalates the alert.
7. **Delivers Spoken Warnings** via text-to-speech with spatial/directional cues (*"Warning! Vehicle approaching on your left, 3.2 meters away"*).
8. **Asynchronous VLM Reasoning** — an on-device VLM (FastVLM-0.5B / Moondream2 / Gemma 4 / Gemini) performs semantic scene understanding, SOP compliance Q&A, and predictive hazard projection.

### Dual Perspective Modes

| Mode | Target Hardware | Description |
|------|-----------------|-------------|
| **Egocentric** | Smart glasses / worker-mounted camera | Camera = worker's eyes. Distance thresholds trigger spatial directional alerts (*"ahead / on your left / on your right"*). |
| **Third Person** | Fixed job-site surveillance camera | Overhead view. Tracks multiple workers simultaneously, estimating gaze direction via pose head yaw. |

---

## 📹 Raspberry Pi 5 Live Camera Streaming

The repository includes a dedicated lightweight streaming server (`pi_stream/stream_server.py`) designed specifically for Raspberry Pi 5 with IMX219 / libcamera modules.

```
┌────────────────────────────────────────┐                ┌────────────────────────────────────────┐
│           Raspberry Pi 5               │                │               Apple Mac                │
│  ┌──────────────────────────────────┐  │  MJPEG / HTTP  │  ┌──────────────────────────────────┐  │
│  │ Camera (IMX219 via picamera2)    │  ├───────────────►│  │ FrameSource (OpenCV FFMPEG)     │  │
│  │ stream_server.py (Port 8554)     │  │  (10.236.6.195)│  │ Safety Copilot Pipeline         │  │
│  └──────────────────────────────────┘  │                │  └──────────────────────────────────┘  │
└────────────────────────────────────────┘                └────────────────────────────────────────┘
```

### Pi Server Features:
- **Thread-safe MJPEG server** built using Python's `http.server` and `Picamera2`.
- **Zero double-encoding overhead**: Captures `BGR888` arrays directly and encodes JPEG frames in a background thread.
- **Auto-reconnect & HTTP Endpoints**:
  - `http://<pi-ip>:8554/stream` — MJPEG live stream endpoint.
  - `http://<pi-ip>:8554/snapshot` — High-resolution single frame snapshot.
  - `http://<pi-ip>:8554/` — Server status JSON endpoint.

---

## ⚡ Performance & High-Speed Optimizations (28.7 FPS)

To achieve **28.7 FPS sustained** real-time annotated output on Apple Silicon (up from 7–8 FPS), four critical architectural optimizations were applied:

```
BEFORE:   Detection ──► BoT-SORT+ReID ──► Depth (blocking) ──► Pose ──► Tool ──► VLM (GPU stall) = ~130ms (7-8 FPS)
AFTER:    Detection ──► ByteTrack ─────┬─ Pose (thread) ───────────────────────────────────────┐ = ~34ms (28.7 FPS)
                                       ├─ Async Depth Worker (background thread) ──────────────┤
                                       ├─ Async Tool Detection (downscaled, thread) ───────────┤
                                       └─ On-Demand VLM (prevents PyTorch MPS GPU lockouts) ───┘
```

1. **ByteTrack Object Tracking (`bytetrack.yaml`)**: Replaced BoT-SORT + ReID with ByteTrack Kalman filtering, eliminating deep neural network feature extraction per box and reducing tracking latency from **35ms to <1ms**.
2. **Asynchronous Non-Blocking Depth Worker (`AsyncDepthWorker`)**: Depth-Anything-V2-Small runs in a dedicated background worker thread (`submit_async`). The main frame loop fetches the latest depth map with **0ms wait time**. Input images are downscaled to 384px prior to depth processing.
3. **Non-Blocking Frame Queue**: `FrameSource` in `core/capture.py` uses non-blocking `put_nowait()` with drop-oldest frame replacement to prevent network stream lag and 500ms queue stalls.
4. **MPS GPU Contention Prevention**: Background VLM polling is set to run on-demand (`vlm.background_polling: false`). PyTorch Metal MPS GPU locks on Apple Silicon during long LLM/VLM text generation are eliminated, leaving 100% GPU bandwidth for Tier 1 real-time vision.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Camera Feed / Network Stream (HTTP)                  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────┐
│  ⚡ TIER 1: EDGE     │         │  🧠 TIER 2: VLM          │
│  ~28.7 FPS          │         │  On-Demand / Escalated  │
│                     │         │                         │
│  Detection+Tracking │         │  Scene Graph (NetworkX) │
│  ├─ PPE Model (HF)  │  ────►  │  Spatial Memory (3×3)   │
│  ├─ YOLO-World      │ scene   │  Session Memory         │
│  │  (tools)         │ state   │  Document RAG           │
│  └─ ByteTrack       │         │  Blueprint Context      │
│                     │         │  ─────────────────────  │
│  Depth Estimation   │         │  FastVLM / Moondream2 / │
│  └─ Async Depth     │         │  Gemma4 / Gemini        │
│     (384px)         │         │  ─────────────────────  │
│                     │         │  → Task Identification  │
│  Pose Estimation    │         │  → Hazard Prediction    │
│  └─ YOLO26-Pose     │         │  → SOP Compliance Q&A   │
│     (17 keypoints)  │         └────────────┬────────────┘
│                     │                      │
│  Safety Analysis    │◄─────────────────────┘
│  ├─ Hazard Analyzer │        predictions
│  ├─ Fall Detector   │
│  ├─ Zone Manager    │
│  ├─ PPE Checker     │
│  └─ Attention       │
│     Tracker (gaze)  │
│                     │
│  Alert Dispatch     │
│  ├─ TTS Engine      │
│  └─ HUD Overlay     │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  📊 SQLite Logger    │
│  (data/events.db)   │
└─────────────────────┘
```

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Application logic |
| **Deep Learning** | PyTorch 2.3+ (MPS backend) | Hardware-accelerated inference on Apple Silicon |
| **Object Detection** | Ultralytics YOLO26 | Worker, vehicle, machinery, and PPE detection |
| **Open-Vocab Detection** | YOLOv8s-World v2 | Zero-shot tool detection (hammer, drill, saw, measuring tape) |
| **Pose Estimation** | YOLO26-Pose | 17-keypoint skeleton & head yaw estimation |
| **Multi-Object Tracking** | ByteTrack (`bytetrack.yaml`) | Zero-overhead persistent identity tracking |
| **Depth Estimation** | Depth Anything V2 Small (HuggingFace) | Asynchronous monocular metric depth estimation |
| **VLM Reasoning** | FastVLM-0.5B / Moondream2 / Gemma 4 / Gemini | Semantic scene understanding & interactive Q&A |
| **Scene Graphs** | NetworkX | Dynamic spatial relationship graph modeling |
| **Document RAG** | Ollama `nomic-embed-text` | SOP document embedding & retrieval |
| **Speech** | macOS native `say` / pyttsx3 | Real-time text-to-speech audio warnings |
| **Video Processing** | OpenCV 4.9+ | Ingestion, rendering, and HUD overlays |
| **Database** | SQLite | Persistent event, alert, and reasoning logs |

---

## 🤖 AI Models

| Model | Weight Source | Size | Device | Frequency | Purpose |
|-------|--------------|------|--------|-----------|---------|
| **PPE Detector** | HuggingFace (`yihong1120/Construction-Hazard-Detection`) | ~5.5 MB | MPS | Every frame | Hardhat, Mask, Safety Vest, Person, Machinery, Vehicle, Safety Cone |
| **YOLO26-Pose** | `yolo26n-pose.pt` | 7.9 MB | MPS | Every 3rd frame | 17-keypoint skeleton, head yaw, torso body angle |
| **YOLOv8s-World v2** | `yolov8s-worldv2.pt` | 25.9 MB | MPS | Every 4th frame (Async) | Open-vocabulary tool detection (hammer, drill, saw, measuring tape) |
| **Depth Anything V2 Small** | HuggingFace (`depth-anything/Depth-Anything-V2-Small-hf`) | ~100 MB | MPS | Every 10th frame (Async Thread) | Monocular depth map → real-world metric distance in meters |
| **FastVLM-0.5B** | HuggingFace (`apple/FastVLM-0.5B`) | ~1.0 GB | MPS | On-demand / Escalated | Fast on-device multimodal reasoning & Q&A |

---

## 🔄 Pipeline Workflow

Each frame is processed through the following stages (`main.py` → `_process_frame()`):

```
Frame N Ingested
    │
    ├─ 1. DETECTION & TRACKING ─────────────────────────────────────────────
    │      PPE Model + YOLO-World → Bounding Boxes & Classes
    │      ByteTrack → Update Persistent Track IDs & Position Histories
    │
    ├─ 2. ASYNCHRONOUS PARALLEL TASKS ──────────────────────────────────────
    │      Thread 1: YOLO26-Pose (17 keypoints + head yaw angle + body angle)
    │      Thread 2: Async Tool Detection (downscaled 640x360)
    │      Worker Thread: Async Depth Anything V2 (downscaled 384px)
    │
    ├─ 3. HAZARD & ATTENTION ANALYSIS ──────────────────────────────────────
    │      PPE Compliance: IoU matching between workers and PPE items
    │      Fall Detection: Torso angle + vertical velocity state machine
    │      Zone Proximity: Worker bbox ∩ Danger Zone polygon
    │      Vehicle Proximity: Metric distance thresholds (≤3m CRITICAL, ≤6m DANGER, ≤12m WARNING)
    │      Attention Tracking: Unnoticed hazard > 4.0s → ESCALATED
    │                          Worker gaze towards hazard > 0.5s → ACKNOWLEDGED
    │
    ├─ 4. ALERT GENERATION & SPEECH ────────────────────────────────────────
    │      Sort hazards by severity (CRITICAL > DANGER > WARNING > INFO)
    │      Dispatch spoken warning via TTS engine with spatial direction
    │
    └─ 5. EVENT LOGGING & OVERLAY RENDER ──────────────────────────────────
           Write hazard & alert records to SQLite (data/events.db)
           Render OpenCV HUD (skeletons, gaze vectors, zones, banners)
```

---

## 🖥️ User Flow & Interface

### Starting the Application

```bash
# 1. Live Raspberry Pi Camera Feed (Recommended)
python main.py --source http://10.236.6.195:8554/stream

# 2. Local Mac Webcam
python main.py --source webcam

# 3. Local Video File
python main.py --source test_videos/site_sample.mp4

# 4. Headless Mode (no GUI window or voice, for server/testing)
python main.py --source http://10.236.6.195:8554/stream --no-display --no-voice
```

### HUD Visual Overlay

- **Green Bounding Boxes**: Tracked workers with IDs and distances (e.g., `Person #2 [3.5m]`).
- **Yellow / Orange Bounding Boxes**: Vehicles and machinery.
- **Teal / Red PPE Indicators**: Teal = PPE detected; Red = missing required PPE.
- **Violet Bounding Boxes**: Detected tools (hammer, drill, saw, measuring tape).
- **Skeleton & Gaze Vectors**: 17 keypoints with head directional gaze rays (yellow).
- **Semi-Transparent Polygons**: Active Danger Zones (orange) and Machine Exclusion Zones.
- **Top Banner**: High-priority alert message displayed with color coding.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Exit the application safely. |
| `s` | Silence all active speech alerts immediately. |
| `v` | **Trigger Interactive VLM Assistant** — pauses video feed, prompts for a natural language question in the terminal, runs VLM visual reasoning, and speaks the answer. |

---

## 📁 Project Structure

```
kaya-hackathon/
│
├── main.py                          # Entry point & main pipeline orchestrator
├── config.yaml                      # Central configuration (thresholds, models, devices)
├── requirements.txt                 # Python dependencies
│
├── pi_stream/                       # ── Raspberry Pi 5 Streaming Subsystem ──
│   ├── stream_server.py             #    Thread-safe Picamera2 MJPEG server (Port 8554)
│   └── deploy.sh                    #    Remote SSH deployment script
│
├── core/                            # ── Tier 1 Perception Core ──
│   ├── capture.py                   #    FrameSource iterator (webcam, video, HTTP stream)
│   ├── detector.py                  #    Multi-model YOLO detector + ByteTrack
│   ├── pose_estimator.py            #    YOLO26-Pose keypoint estimator & head yaw
│   ├── depth_estimator.py           #    Async Depth Anything V2 monocular depth engine
│   ├── device.py                    #    Apple Silicon MPS device detection & smoke test
│   └── models.py                    #    Dataclasses (Detection, TrackedObject, Hazard, Alert)
│
├── safety/                          # ── Safety Logic & Analysis ──
│   ├── hazard_analyzer.py           #    Master hazard fusion engine
│   ├── attention_tracker.py         #    Gaze-aware hazard escalation system
│   ├── fall_detector.py             #    Pose-based fall detection state machine
│   ├── ppe_checker.py               #    Spatial PPE compliance matcher
│   └── zones.py                     #    Polygon danger zone manager
│
├── alerts/                          # ── Alert Dispatch ──
│   ├── alert_manager.py             #    Priority-based alert dispatcher
│   └── tts_engine.py                #    macOS native text-to-speech engine
│
├── display/                         # ── Visual Rendering ──
│   └── overlay.py                   #    OpenCV HUD overlay renderer
│
├── reasoning/                       # ── Tier 2 VLM Reasoning ──
│   ├── pipeline.py                  #    Reasoning coordinator & prompt builder
│   ├── scene_graph.py               #    NetworkX spatial scene graph builder
│   ├── spatial_memory.py            #    2D grid heat map for worker positioning
│   ├── session_memory.py            #    Rolling buffer of reasoning history
│   ├── document_store.py            #    SOP document RAG loader & embedder
│   └── blueprint_store.py           #    PDF/CAD blueprint loader & page renderer
│
├── integration/                     # ── VLM Integration ──
│   ├── vlm_hook.py                  #    VLM dispatcher & JSON response parser
│   ├── fastvlm_engine.py            #    On-device FastVLM-0.5B engine
│   └── moondream_engine.py          #    On-device Moondream2 engine
│
└── logging_/                        # ── Database & Event Logs ──
    ├── event_logger.py              #    SQLite event logger
    └── schemas.py                   #    Database table schemas (hazards, alerts, reasoning)
```

---

## 🚀 Getting Started & Installation

### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4 recommended for MPS GPU acceleration).
- **Python 3.10+** (Python 3.10 or 3.13 supported).
- **Raspberry Pi 5** (optional, for remote camera feed).

### Installation

```bash
# Clone the repository
git clone https://github.com/shaurya-dogra/egocentric_construction_partner.git
cd egocentric_construction_partner

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Starting the Pi Camera Stream (Optional)

On your Raspberry Pi 5 with camera connected:

```bash
# Transfer pi_stream folder or run directly on Pi:
cd pi_stream
python3 stream_server.py
```
The server will output: `Stream: http://0.0.0.0:8554/stream`.

---

## ⚙️ Configuration Reference

All settings are managed in `config.yaml`:

```yaml
perspective: "egocentric"       # "egocentric" (smart glasses) | "third_person" (surveillance)

models:
  ppe:
    path: "hf:yihong1120/Construction-Hazard-Detection:models/yolo26/pt/yolo26n.pt"
    confidence: 0.35
  pose:
    path: "yolo26n-pose.pt"
    run_every_n_frames: 3
  tool:
    enabled: true
    path: "yolov8s-worldv2.pt"
    classes: ["hammer", "drill", "saw", "measuring tape"]
  depth:
    enabled: true
    path: "depth-anything/Depth-Anything-V2-Small-hf"
    scale_factor: 15.0

tracking:
  tracker: "bytetrack.yaml"     # High-speed ByteTrack filtering

escalation:
  dwell_threshold_seconds: 4.0           # Seconds before unnoticed hazard escalates
  gaze_angle_tolerance_degrees: 45.0     # ±degrees to count as "looking at" hazard

vlm:
  enabled: true
  backend: "gemini"             # "gemini" (Google AI Studio Free Tier) | "groq" | "ollama" | "fastvlm" | "moondream"
  model: "gemini-2.0-flash"     # Free-tier fast multimodal model (or "gemini-1.5-flash")
  background_polling: true      # Uses 0% local GPU over cloud REST, leaving 28.7 FPS for local vision
```

### VLM Setup (Tier 2 Multimodal Reasoning)

To use **Google Gemini 2.0 Flash** (Free Tier):

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory (automatically git-ignored):
   ```bash
   GEMINI_API_KEY=your_google_ai_studio_api_key
   ```

---

## 📊 Database & Logging

All pipeline activities are saved to an SQLite database at `data/events.db`.

### Database Schema:

- `hazards`: Hazard type, severity, state (`PASSIVE`, `UNNOTICED`, `ESCALATED`, `ACKNOWLEDGED`), worker track ID, timestamps.
- `alerts`: Spoken and visual alerts dispatched, severity level, message text.
- `events`: System events, acknowledgments, and resolutions.
- `reasoning_events`: Output from Tier 2 VLM reasoning passes (tasks, predicted risks, SOP compliance).

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Pi Camera returns red/blue tint** | Color channel mismatch | Ensure `stream_server.py` uses `BGR888` format (already configured). |
| **Camera times out on Pi** | Unstable power supply | Ensure Pi 5 is powered using an official 27W USB-C supply (5V/5A). |
| **Low FPS on Mac** | MPS backend fallback | Verify PyTorch MPS availability. Ensure `bytetrack.yaml` is set in `config.yaml`. |
| **TTS audio disabled** | System voice muted | Check macOS volume or verify `alerts.enabled: true` in `config.yaml`. |

---

## 📜 License

MIT License. Developed for the Kaya Hackathon by Team Antigravity.
