# Kaya — Real-Time Job Site Safety Copilot

> **AI-powered construction site monitoring.** Real-time multi-model computer vision + multimodal voice assistant running on Apple Silicon.

---

## What It Does

Kaya is a full-stack AI safety system that runs on an Apple Silicon Mac (M-series, MPS GPU) and streams a live web dashboard. It:

- Detects **people, heavy machinery, tools** and **PPE compliance** in real-time
- Estimates **metric depth** per-frame using Depth Anything V2
- Tracks **3D human pose** and **eye gaze** of every worker
- Listens to voice queries → reasons over live video context → speaks back (Push-to-Talk)
- Maintains a **temporal ring buffer** so the AI can answer questions about *what just happened*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py  (entry point)                      │
│                                                                     │
│   SafetyCopilot loop          Kaya Web App (FastAPI + Uvicorn)      │
│   ┌────────────────┐          ┌──────────────────────────────────┐  │
│   │ FrameSource    │          │  /api/video_feed   (MJPEG)       │  │
│   │ (webcam/Pi)    │──frame──▶│  /api/ask          (voice)       │  │
│   └───────┬────────┘          │  /api/ask-text     (text)        │  │
│           │                   │  /api/pose         (3D viewer)   │  │
│           ▼                   │  /api/status       (telemetry)   │  │
│   ┌───────────────────────┐   └──────────────────────────────────┘  │
│   │   Detection Pipeline  │                  │                      │
│   │  ┌─────────────────┐  │                  ▼                      │
│   │  │ YOLO11n         │  │         CopilotBridge (singleton)       │
│   │  │ (80-class COCO) │  │         ┌────────────────────────────┐  │
│   │  ├─────────────────┤  │         │ latest_raw_frame           │  │
│   │  │ YOLO-World v2   │  │         │ latest_result (FrameResult)│  │
│   │  │ (125 tool/PPE   │  │         │ latest_depth_map           │  │
│   │  │  custom classes)│  │◀─result─│ overlay_renderer           │  │
│   │  ├─────────────────┤  │         │ temporal_ring_buffer       │  │
│   │  │ PPE YOLO (HF)   │  │         │   (1 FPS, 8-frame window)  │  │
│   │  │ yolo26n.pt      │  │         └────────────────────────────┘  │
│   │  │ 11 PPE classes  │  │                                         │
│   │  └─────────────────┘  │                                         │
│   │  ┌─────────────────┐  │                                         │
│   │  │ YOLO-Pose       │  │                                         │
│   │  │ yolo26n-pose.pt │  │                                         │
│   │  │ 17-kp COCO      │  │                                         │
│   │  └─────────────────┘  │                                         │
│   │  ┌─────────────────┐  │                                         │
│   │  │ Depth Anything  │  │                                         │
│   │  │ V2 Metric Small │  │                                         │
│   │  │ (async thread)  │  │                                         │
│   │  └─────────────────┘  │                                         │
│   └───────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘

Browser (http://127.0.0.1:8000)
┌─────────────────────────────────────────────────────────────────────┐
│  View Mode Tabs                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Combined  │ │Normal    │ │3D Pose   │ │ Depth    │ │PPE / Obj │  │
│  │(server)  │ │(server)  │ │(Three.js)│ │(WebGPU!) │ │(server)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                     │
│  🌊 Depth Tab — Runs entirely in browser (mirrors v.fast depth):    │
│    onnx-community/depth-anything-v2-small (WebGPU FP16)             │
│    LUT ImageNet preprocessing → persistent Float32 buffer           │
│    64-sample adaptive range + EMA smoothing (85%/15%)               │
│    Analytical TURBO colormap (5th-order polynomial)                 │
│                                                                     │
│  👁️ 3D Pose Tab — Three.js real-time 3D skeleton:                  │
│    Poll /api/pose (15 FPS) → keypoints depth-lifted to 3D           │
│    Colored bones per COCO limb, joint spheres, gaze ray             │
│    Auto-orbit PerspectiveCamera, GridHelper floor                   │
│                                                                     │
│  🎙️ Voice Chat (Push-to-Talk):                                      │
│    STT: Gemini Flash → VLM: Gemini Flash (temporal frames)          │
│    TTS: Sarvam Bulbul v3 → audio/wav autoplay                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Stack

| Model | Purpose | Device | Notes |
|---|---|---|---|
| `yolo11n.pt` | General object detection (80 COCO classes) | MPS | YOLO11 nano |
| `yolov8s-worldv2.pt` | Open-vocabulary tool + PPE detection (125 custom classes) | MPS | YOLO-World v2 |
| `yihong1120/Construction-Hazard-Detection yolo26n.pt` | PPE compliance (Hardhat, Vest, No-Hardhat, No-Vest…) | MPS | HuggingFace Hub |
| `yolo26n-pose.pt` | Human pose estimation (17 COCO keypoints) | MPS | Run every N=3 frames |
| `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | Metric monocular depth (in metres) | MPS | Async background thread |
| `onnx-community/depth-anything-v2-small` | Browser-side relative depth (Depth tab) | WebGPU FP16 | Runs in Chrome — zero Python |
| `gemini-3.5-flash` | STT + Vision reasoning (VLM) | Google API | Temporal 8-frame window |
| `sarvam/bulbul:v3/shubh` | Text-to-Speech (Hindi-English) | Sarvam API | Returns WAV audio |

---

## YOLO-World Custom Vocabulary (125 classes)

<details>
<summary>Hand Tools</summary>

claw hammer · rubber mallet · screwdriver · adjustable wrench · socket wrench set · pliers · wire cutters · utility knife · hand saw · hacksaw · measuring tape · spirit level · chisel · hand file · allen key set · c-clamp · pipe wrench · putty knife · hand trowel · staple gun · crowbar · bolt cutters

</details>

<details>
<summary>Power Tools & Equipment</summary>

cordless drill · impact driver · angle grinder · circular saw · reciprocating saw · jigsaw · table saw · miter saw · belt sander · orbital sander · nail gun · jackhammer · rotary hammer drill · heat gun · soldering iron · bench grinder · air compressor · pressure washer · chainsaw · welding machine · welding torch · extension cord reel · power strip

</details>

<details>
<summary>PPE</summary>

hard hat · safety vest · safety glasses · safety goggles · face shield · ear muffs · ear plugs · dust mask · respirator mask · work gloves · cut-resistant gloves · safety boots · steel-toe boots · knee pads · safety harness · welding helmet · high-visibility jacket

</details>

<details>
<summary>Site Infrastructure, Vehicles, Hazards & Materials</summary>

step ladder · extension ladder · scaffolding · safety cone · barricade · caution tape · guardrail · warning sign · toolbox · tool cart · storage bin · bucket · wheelbarrow · dumpster · dolly cart · workbench · shipping container · port-a-potty · forklift · excavator · bulldozer · backhoe loader · cement mixer · crane · dump truck · pickup truck · skid steer loader · road roller · scissor lift · boom lift · concrete pump truck · flatbed trailer · exposed wire · cinder block · rebar · gas cylinder · propane tank · fire extinguisher · wooden pallet · spill puddle · sandbag · lumber stack · metal pipe · concrete slab · electrical panel · open manhole · debris pile · sharp metal shard · asbestos warning sign · cell phone · laptop · clipboard · water bottle · backpack · hard hat with attached radio · two-way radio · lunch box · first aid kit · coffee cup · folding chair · trash can · broom · dustpan

</details>

---

## Directory Structure

```
kaya hackathon/
├── main.py                     # Entry point — starts SafetyCopilot + web server
├── config.yaml                 # Runtime configuration (models, thresholds, providers)
├── requirements.txt
│
├── core/
│   ├── capture.py              # FrameSource — webcam or Pi MJPEG stream
│   ├── detector.py             # Multi-model detector (YOLO11 + YOLO-World + PPE)
│   ├── pose_estimator.py       # YOLO-Pose 17-keypoint estimator
│   ├── depth_estimator.py      # Depth Anything V2 — async background thread
│   ├── tracker.py              # ByteTrack object tracker
│   ├── models.py               # Pydantic dataclasses (FrameResult, Detection, PoseData…)
│   └── device.py               # MPS / CUDA / CPU device selection
│
├── display/
│   └── overlay.py              # OverlayRenderer — HUD, skeletons, depth, PPE audit
│                               # All 6 view modes with EMA depth + TURBO colormap
│
├── app/
│   ├── main.py                 # FastAPI app — all HTTP endpoints
│   ├── copilot_bridge.py       # Thread-safe singleton bridge (frame state + temporal buffer)
│   ├── pipeline.py             # STT → VLM → TTS execution pipeline
│   ├── interfaces.py           # Provider ABC definitions
│   ├── factory.py              # Provider factory (selects STT/VLM/TTS from config)
│   └── providers/
│       ├── stt/                # gemini_stt, sarvam_stt, mock_stt
│       ├── tts/                # sarvam_tts, mock_tts
│       └── vision/             # gemini_vision, mock_vision
│
├── integration/
│   └── vlm_hook.py             # VLMHook — injects temporal frames into Gemini calls
│
├── logging_/
│   └── event_logger.py         # SQLite event logger
│
├── static/
│   ├── index.html              # Dashboard HTML (Three.js + HF Transformers via CDN)
│   ├── app.js                  # Frontend logic:
│   │                           #   - Browser-side WebGPU depth inference (v.fast depth port)
│   │                           #   - Three.js 3D pose viewer (/api/pose polling)
│   │                           #   - Push-to-Talk MediaRecorder
│   │                           #   - TTS audio autoplay
│   └── styles.css              # Clean white dashboard CSS
│
├── pi_stream/
│   └── stream_server.py        # Raspberry Pi MJPEG stream server
│
├── tests/                      # pytest test suite (17 tests)
└── data/                       # SQLite event database (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (MPS GPU) — or any CUDA/CPU machine
- Chrome / Edge (required for WebGPU depth tab)

### Install

```bash
git clone https://github.com/shaurya-dogra/egocentric_construction_partner
cd "kaya hackathon"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

### Run

```bash
source .venv/bin/activate
python main.py
```

Open **http://127.0.0.1:8000** in Chrome.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/video_feed` | GET | MJPEG stream (`?mode=all\|raw\|pose\|depth\|ppe\|objects`) |
| `/api/ask` | POST | Voice query (multipart audio) |
| `/api/ask-text` | POST | Text query |
| `/api/pose` | GET | Current pose keypoints + depth JSON for 3D viewer |
| `/api/status` | GET | System telemetry (FPS, providers, temporal buffer) |
| `/api/reset` | POST | Clear conversation history |

---

## View Modes

| Tab | How it works |
|---|---|
| 🎯 **Combined** | All detections overlaid (YOLO + PPE + Pose + Depth tags) |
| 📹 **Normal** | Raw camera feed, no overlays |
| 👁️ **3D Pose** | Three.js 3D scene — skeleton bones + joints + gaze ray, depth-lifted coordinates, orbit camera |
| 🌊 **Depth** | Browser-side WebGPU inference via `onnx-community/depth-anything-v2-small` (FP16), rendered with analytical TURBO colormap + EMA temporal smoothing |
| 🦺 **PPE** | PPE compliance audit — compliance halo per worker, missing gear warnings |
| 🔨 **Objects** | Tool + machinery detections only, wrist-to-tool carrying links |

---

## Temporal Feed

The voice assistant has memory of the last **8 seconds** of video:

1. Every processed frame is sampled at ~1 FPS into a ring buffer in `CopilotBridge`
2. When a voice/text query arrives, `get_latest_temporal_frames(max_frames=8)` returns those JPEG frames
3. All 8 frames + the question are sent to Gemini Flash as a multimodal temporal sequence
4. Gemini can answer questions like *"What happened in the last few seconds?"* or *"Did someone just pick up a tool?"*

---

## Raspberry Pi Integration

Kaya can accept video from a remote Raspberry Pi instead of a local webcam:

```bash
# On the Pi (run pi_stream/stream_server.py)
python pi_stream/stream_server.py

# In config.yaml
capture:
  source: "http://<pi-ip>:8080/stream"
```

---

## Configuration (`config.yaml`)

```yaml
capture:
  source: 0                    # 0 = webcam, or MJPEG URL

models:
  general: yolo11n.pt
  tool: yolov8s-worldv2.pt
  ppe: hf:yihong1120/...
  pose: yolo26n-pose.pt
  depth:
    path: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
    enabled: true
    run_every_n_frames: 3

kaya:
  stt_provider: gemini
  vision_provider: gemini
  tts_provider: sarvam
  model: gemini-3.5-flash
  frame_mode: TEMPORAL_FRAMES   # or SINGLE_FRAME
  temporal_buffer_seconds: 6.0
  max_temporal_frames: 8
```

---

## Tech Stack

**Backend**: Python 3.13 · FastAPI · Uvicorn · PyTorch (MPS) · Ultralytics · HuggingFace Transformers · OpenCV · NumPy  
**Frontend**: Vanilla JS (ES Modules) · Three.js · @huggingface/transformers (WebGPU) · MediaRecorder API  
**AI APIs**: Google Gemini Flash · Sarvam Bulbul TTS  
**Models**: YOLO11 · YOLO-World v2 · Depth Anything V2 (PyTorch + ONNX WebGPU) · YOLO-Pose
