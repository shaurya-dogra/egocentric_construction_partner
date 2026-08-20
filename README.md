# 🏗️ Kaya — Job Site Safety Copilot & Multimodal RAG Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-Detection%20%2B%20Pose-00FFFF?logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![Docling](https://img.shields.io/badge/Docling-Structure--Aware%20RAG-7928CA?logo=ibm&logoColor=white)](https://ds4sd.github.io/docling/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.1%20Flash%20Lite-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Sarvam AI](https://img.shields.io/badge/Sarvam%20AI-Bulbul%20v3%20TTS-FF6F00)](https://sarvam.ai)
[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen?logo=pytest&logoColor=white)](https://pytest.org)

**An intelligent, low-latency construction safety copilot combining real-time edge computer vision (YOLO detection, 3D pose, and Depth Anything V2 monocular depth) with a structure-aware multimodal RAG knowledge assistant.**

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
  - [1. Real-Time Computer Vision Pipeline](#1-real-time-computer-vision-pipeline)
  - [2. Multi-Mode Video Dashboard & 3D Spatial Viewer](#2-multi-mode-video-dashboard--3d-spatial-viewer)
  - [3. Structure-Aware Docling RAG Subsystem](#3-structure-aware-docling-rag-subsystem)
  - [4. Multimodal Conversational Voice & Text Chatbot](#4-multimodal-conversational-voice--text-chatbot)
- [Repository Structure](#-repository-structure)
- [Model Stack & Technologies](#-model-stack--technologies)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Ingesting Construction Documents](#ingesting-construction-documents)
  - [Running the Unified Application](#running-the-unified-application)
- [Verification & Automated Tests](#-verification--automated-tests)
- [Latency Benchmarks](#-latency-benchmarks)
- [License](#-license)

---

## 🌟 Overview

**Kaya** is designed for high-risk industrial, construction, and engineering environments where workers need instant safety assistance, real-time hazard detection, compliance verification, and voice-guided standard operating procedure (SOP) queries.

Kaya runs offline computer vision on Apple Silicon (MPS) or edge hardware while seamlessly connecting to Google Gemini multimodal reasoning and Sarvam AI voice synthesis.

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │      CAMERA / WEBCAM / RTSP   │
                                  └───────────────┬───────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │     REAL-TIME CV SAFETY PIPELINE │             │    TEMPORAL FRAME RING BUFFER    │
        │ ──────────────────────────────── │             │ ──────────────────────────────── │
        │ • YOLO11/YOLO26 Object Detection │             │ • 6.0s Rolling Buffer @ 1.0 FPS  │
        │ • Multi-Object Tracking (ByteTrack)            │ • Keyframe Subsampling (Max 4-8) │
        │ • YOLO-Pose 3D Skeletons & Gaze  │             │ • JPEG Dynamic Downscaling 768px │
        │ • Depth Anything V2 Indoor Depth │             └────────────────┬─────────────────┘
        │ • PPE & Fall Hazard Analyzer     │                              │
        │ • Local macOS Alert TTS Engine   │                              │
        └────────────────┬─────────────────┘                              │
                         │                                                │
                         ▼                                                │
        ┌──────────────────────────────────┐                              │
        │     COPILOT THREAD-SAFE BRIDGE   │                              │
        │ ──────────────────────────────── │                              │
        │ • MJPEG View Mode Multiplexer    │                              │
        │ • /api/video_feed (6 View Modes) │                              │
        │ • /api/pose (3D Depth Keypoints) │                              │
        └────────────────┬─────────────────┘                              │
                         │                                                │
                         ▼                                                │
   ╔══════════════════════════════════════════════╗                       │
   ║          KAYA DUAL-PANEL WEB DASHBOARD       ║                       │
   ╠══════════════════════════════════════════════╣                       │
   ║  LEFT: CV Live Stream & 3D Spatial Viewer    ║                       │
   ║  RIGHT: Voice/Text RAG Assistant Chatbot     ║                       │
   ╚═════════════════════╤════════════════════════╝                       │
                         │                                                │
                         │ (User Query: Microphone Audio / Text)          │
                         ▼                                                │
        ┌──────────────────────────────────┐                              │
        │      STT TRANSCRIPTION ENGINE    │                              │
        │ ──────────────────────────────── │                              │
        │ • Gemini STT / Sarvam Saaras v3  │                              │
        └────────────────┬─────────────────┘                              │
                         │                                                │
                         ▼                                                │
        ┌──────────────────────────────────┐                              │
        │        RAG INTENT ROUTER         │                              │
        │ ──────────────────────────────── │                              │
        │ • Sub-millisecond Regex Router   │                              │
        │ • Detects SOP/Manual/Rule Queries│                              │
        └────────┬─────────────────┬───────┘                              │
                 │                 │                                      │
       [Requires RAG: YES]   [Requires RAG: NO]                           │
                 │                 │                                      │
                 ▼                 │                                      │
   ┌───────────────────────────┐   │                                      │
   │  DOCLING VECTOR RETRIEVER │   │                                      │
   │ ───────────────────────── │   │                                      │
   │ • Docling Structure Chunks│   │                                      │
   │ • Gemini 3072d Embeddings │   │                                      │
   │ • Top-K Cosine Retrieval  │   │                                      │
   └─────────────┬─────────────┘   │                                      │
                 │ (Retrieved Chunks)                                     │
                 └───────────────┬─┴──────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │           MULTIMODAL VISION REASONER             │
        │ ──────────────────────────────────────────────── │
        │ • Google Gemini 3.1 Flash Lite                   │
        │ • Inputs: Camera Keyframes + SOP Chunks + Query  │
        │ • Automatic Function Calling (AFC) Disabled      │
        │ • Concise Construction Safety Rules Prompting    │
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │            HIGH-SPEED TTS SYNTHESIZER            │
        │ ──────────────────────────────────────────────── │
        │ • Sarvam AI Bulbul v3 (1.4x Pace, Markdown-safe) │
        │ • On-device macOS fallback (215 WPM)             │
        └──────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. Real-Time Computer Vision Pipeline
- **Object Detection & Tracking**: Real-time tracking of workers, machinery, vehicles, power tools, and obstacles using YOLO11/YOLO26 and ByteTrack.
- **PPE Compliance Verification**: Live auditing of hardhats, high-visibility vests, gloves, boots, and missing safety equipment.
- **3D Pose & Gaze Estimation**: YOLO-Pose extracts 17 COCO skeletal keypoints per person, calculating `head_yaw`, torso `body_angle`, and attention escalation.
- **Monocular Depth Estimation**: Utilizes `Depth-Anything-V2-Metric-Indoor-Small-hf` with foreground-focused core percentile sampling to produce accurate metric distance measurements (e.g. `0.9m` indoors) without background bleed.

### 2. Multi-Mode Video Dashboard & 3D Spatial Viewer
- **🎯 Combined Mode**: Complete safety overlay with bounding boxes, distance tags, PPE badges, danger zones, and active alert banners.
- **📹 Normal Mode**: Clean, un-annotated raw camera feed.
- **👁️ 3D Pose Mode**: 
  - *Stream Overlay*: Real-time skeleton projections and posture angle telemetry.
  - *3D Spatial Canvas*: Interactive Three.js WebGL scene with orbit controls, 17-bone anatomical connectivity, and head gaze perspective rays.
- **🌊 Depth Map Mode**: Full-resolution TURBO colormap gradient depth map with distance pins and metric scale bar.
- **🦺 PPE Audit Mode**: Focused strictly on worker compliance and safety gear.
- **🔨 Objects & Tools Mode**: Focuses on heavy machinery, power tools, and spatial distances.

### 3. Structure-Aware Docling RAG Subsystem
- **Deep Structure Parsing**: Uses IBM's **Docling** (`DocumentConverter` & `HybridChunker`) to parse construction safety manuals, OSHA guidelines, and SOPs while preserving table hierarchies, headings, and page metadata.
- **High-Dimensional Embeddings**: Chunks are embedded using `gemini-embedding-001` (3072 dimensions) and indexed in a local vector database.
- **RAG Intent Router**: Sub-millisecond classifier prevents unnecessary retrieval latency for purely visual queries while activating retrieval for compliance questions.
- **Grounding Citations**: Chatbot responses include exact source files and page numbers (e.g., `📖 ladder_safety_sop.txt`, `📄 worker_manual_construction_eng.pdf (p.2)`).

### 4. Multimodal Conversational Voice & Text Chatbot
- **Voice & Text Input**: Push-to-talk microphone recording (`Space` key shortcut) or direct text query form.
- **Low-Latency Reasoning**: Optimized Gemini 3.1 Flash Lite inference (~1.8s–2.2s) with frame subsampling and AFC disabled.
- **High-Speed TTS**: Audio synthesized via Sarvam AI Bulbul v3 at **1.4x pace** (~4.5s for spoken paragraphs) with automatic browser playback and `▶ Replay` buttons.
- **Granular Latency Badges**: Real-time telemetry pills for `RAG`, `VLM`, `TTS`, and `Total` execution time.

---

## 📁 Repository Structure

```
kaya hackathon/
├── app/
│   ├── main.py                     # FastAPI web application with endpoints & lifespan
│   ├── config.py                   # Pydantic settings & environment configuration
│   ├── factory.py                  # Provider factory (Vision, STT, TTS, RAG)
│   ├── interfaces.py               # Abstract Base Classes (KnowledgeRetriever, VisionReasoner, etc.)
│   ├── pipeline.py                 # Core Kaya turn pipeline coordinating STT, RAG, VLM, and TTS
│   ├── copilot_bridge.py           # Thread-safe bridge connecting CV pipeline to FastAPI
│   ├── rag/
│   │   └── router.py               # RAG Intent Router with regex pattern matching
│   └── providers/
│       ├── rag/
│       │   ├── docling_rag.py      # Docling LocalVectorStore & DoclingVectorRetriever
│       │   ├── gemini_rag.py       # Gemini File Search RAG provider
│       │   └── mock_rag.py         # Mock RAG retriever for unit tests
│       ├── vision/
│       │   ├── gemini_vision.py    # Gemini 3.1 Flash Lite VLM provider with RAG context injection
│       │   ├── nvidia_vision.py    # NVIDIA NIM vision provider
│       │   ├── ollama_vision.py    # Local Ollama vision provider
│       │   └── mock_vision.py      # Mock vision reasoner for unit tests
│       ├── tts/
│       │   ├── sarvam_tts.py       # Sarvam AI Bulbul v3 TTS provider (1.4x pace)
│       │   ├── mac_tts.py          # Native macOS say TTS provider (215 WPM)
│       │   └── mock_tts.py         # Mock TTS provider for unit tests
│       └── stt/
│           ├── gemini_stt.py       # Gemini audio transcription provider
│           ├── sarvam_stt.py       # Sarvam Saaras v3 STT provider
│           └── mock_stt.py         # Mock STT provider for unit tests
├── core/
│   ├── capture.py                  # FrameSource wrapper for webcam, video, and image inputs
│   ├── detector.py                 # YOLO11 & YOLO-World object & tool detection
│   ├── depth_estimator.py          # Depth Anything V2 Indoor metric depth estimator
│   ├── device.py                   # Compute device resolution (MPS / CPU / CUDA)
│   ├── models.py                   # Dataclasses (Detection, PoseData, FrameResult, HazardAssessment)
│   └── pose_estimator.py           # YOLO-Pose keypoint & head yaw estimator
├── safety/
│   ├── attention_tracker.py        # Gaze tracking and attention escalation logic
│   ├── fall_detector.py            # Fall detection via body angle and velocity thresholds
│   ├── hazard_analyzer.py          # Spatial proximity & danger zone hazard analysis
│   ├── ppe_checker.py              # Worker PPE compliance checker
│   └── zones.py                    # Danger zone manager & polygon intersection
├── alerts/
│   ├── alert_manager.py            # Alert deduplication and escalation manager
│   └── tts_engine.py               # Multiprocessing local macOS TTS audio alert engine
├── display/
│   └── overlay.py                  # OverlayRenderer for HUD, TURBO depth colormap, & 3D pose
├── integration/
│   └── vlm_hook.py                 # On-demand VLM escalation hook
├── knowledge/
│   ├── manifest.json               # Index of ingested documents & chunk counts
│   └── vector_store.json           # Serialized 3072-dim embeddings & structure-aware chunks
├── pdfs_for_rag/                   # Source construction safety PDFs & SOP text files
├── scripts/
│   ├── ingest.py                   # Docling document ingestion & embedding CLI
│   └── test_live_docling_rag.py    # End-to-end live RAG evaluation runner
├── static/
│   ├── index.html                  # Dual-panel web dashboard UI
│   ├── styles.css                  # Responsive CSS design system
│   └── app.js                      # Three.js 3D pose, video stream, & RAG chat controller
├── tests/                          # 34 automated unit & integration test suites
├── config.yaml                     # CV models, danger zones, & alert configurations
├── main.py                         # Single unified entrypoint for CV copilot & web server
├── requirements.txt                # Python package dependencies
└── pytest.ini                      # Pytest configuration
```

---

## 🤖 Model Stack & Technologies

| Domain | Model / Tool | Source | Purpose |
|---|---|---|---|
| **Object Detection** | YOLO11n / YOLO26 | Ultralytics | Real-time person, vehicle, and obstacle detection |
| **Tool Detection** | YOLOv8s-Worldv2 | Ultralytics | Open-vocabulary detection of construction tools |
| **Pose & Posture** | YOLO26n-Pose | Ultralytics | 17 COCO keypoints, head yaw, and fall detection |
| **Monocular Depth** | Depth Anything V2 (Indoor) | HuggingFace | 0.2m–20m metric distance estimation |
| **Document Ingestion** | Docling (HybridChunker) | IBM DS4SD | Structure-aware chunking preserving tables & headings |
| **Embedding Model** | `gemini-embedding-001` | Google GenAI | 3072-dimensional vector embeddings for cosine search |
| **Multimodal VLM** | Gemini 3.1 Flash Lite | Google GenAI | Voice + Vision reasoning grounded in SOP chunks |
| **Speech Synthesizer** | Sarvam Bulbul v3 (`shubh`) | Sarvam AI | Indian-accented high-speed voice synthesis (1.4x pace) |
| **3D Rendering** | Three.js (r128) | CDN | Interactive browser-side 3D skeleton pose canvas |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dual-panel interactive web dashboard |
| `GET` | `/api/status` | System health, provider statuses, copilot telemetry, & RAG metadata |
| `GET` | `/api/video_feed?mode={mode}` | Live MJPEG multipart stream (`all`, `raw`, `pose`, `depth`, `ppe`, `objects`) |
| `GET` | `/api/pose` | Latest 3D keypoints, head yaw, and per-joint metric depths as JSON |
| `GET` | `/api/knowledge/status` | Document count, filenames, and vector store readiness |
| `POST` | `/api/ask` | Multipart voice turn processing (audio payload + temporal frames) |
| `POST` | `/api/ask-text` | Multipart text turn processing (text question + temporal frames) |
| `POST` | `/api/reset` | Resets conversation context and turn history |

---

## 🚀 Getting Started

### Prerequisites
- **macOS** (Apple Silicon M1/M2/M3/M4 recommended) or Linux with Python 3.10+
- **Google Gemini API Key** (for VLM, STT, and Embeddings)
- **Sarvam AI API Key** (for high-speed Indian-accented TTS)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shaurya-dogra/egocentric_construction_partner.git
   cd egocentric_construction_partner
   ```

2. **Create & activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Create or update `.env` in the root directory:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# Sarvam AI API
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SPEAKER=shubh
SARVAM_TTS_PACE=1.4

# Providers
VISION_PROVIDER=gemini
STT_PROVIDER=gemini
TTS_PROVIDER=sarvam

# RAG Knowledge Configuration
RAG_PROVIDER=docling
RAG_ENABLED=true
RAG_ROUTER_MODE=auto
RAG_TOP_K=4
EMBEDDING_MODEL=gemini-embedding-001
KNOWLEDGE_DIR=pdfs_for_rag

# Temporal Buffer & Server
FRAME_MODE=TEMPORAL_FRAMES
TEMPORAL_BUFFER_SECONDS=6.0
TEMPORAL_FPS=1.0
TEMPORAL_MAX_FRAMES=8
HOST=127.0.0.1
PORT=8000
```

### Ingesting Construction Documents

To index new or updated PDFs and SOPs into the local Docling vector store:

```bash
python scripts/ingest.py --dir pdfs_for_rag
```

### Running the Unified Application

Start the unified system (launches the real-time CV pipeline + the web dashboard server in one command):

```bash
python main.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🧪 Verification & Automated Tests

Kaya includes an end-to-end test suite covering CV detectors, depth estimators, overlay rendering modes, FastAPI REST endpoints, Docling RAG retrieval, and the full multimodal pipeline.

Run the test suite:
```bash
pytest tests/ -v
```

Output:
```
============================== test session starts ===============================
collected 34 items

tests/test_depth.py ..                                                    [  5%]
tests/test_detector.py ...                                                [ 14%]
tests/test_endpoints.py .....                                             [ 29%]
tests/test_interfaces_and_factories.py .....                              [ 44%]
tests/test_overlay_modes.py .                                             [ 47%]
tests/test_pipeline.py .                                                  [ 50%]
tests/test_rag.py ......                                                  [ 67%]
tests/test_voice_assistant_endpoints.py .....                             [ 82%]
tests/test_voice_assistant_factories.py .....                             [ 97%]
tests/test_voice_assistant_pipeline.py .                                  [100%]

============================== 34 passed in 8.58s ===============================
```

To run the live multimodal RAG evaluation script:
```bash
python scripts/test_live_docling_rag.py
```

---

## ⚡ Latency Benchmarks

| Stage | Latency | Optimization Technique |
|---|---|---|
| **STT Transcription** | ~0 ms | Direct text bypass / streaming Whisper & Saaras v3 |
| **RAG Intent Router** | **< 1 ms** | Regex pre-filter avoiding redundant vector lookups |
| **Docling Vector Retrieval** | **~500 ms – 650 ms** | Cosine similarity over cached 3072-dim embeddings |
| **Gemini Multimodal VLM** | **~1.8 s – 2.2 s** | Keyframe subsampling, 768px downscaling, AFC disabled |
| **Sarvam AI TTS** | **~4.5 s – 5.0 s** | Markdown stripping, sentence buffering, 1.4x pace |
| **Total Turnaround Time** | **~6.5 s – 7.5 s** | End-to-end question to voice playback |

---

## 📄 License

This project is licensed under the Apache 2.0 License.
