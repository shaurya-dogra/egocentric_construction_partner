/**
 * Kaya — Job Site Safety Copilot
 * app.js (ES Module)
 *
 * Depth tab  : Browser-side WebGPU depth inference via @huggingface/transformers,
 *              mirroring v.fast depth architecture exactly (LUT preprocessing,
 *              analytical TURBO colormap, EMA temporal smoothing).
 * Pose tab   : Three.js real-time 3D skeleton viewer, keypoints depth-lifted
 *              to 3D space using the /api/pose JSON endpoint.
 * Voice/Chat : Push-to-Talk MediaRecorder, text query, chip suggestions,
 *              latency pills, TTS autoplay.
 */

// ── DOM refs ──────────────────────────────────────────────────────

const videoFeed           = document.getElementById('videoFeed');
const depthCanvas         = document.getElementById('depthCanvas');
const poseCanvas          = document.getElementById('poseCanvas');
const depthLoadingOverlay = document.getElementById('depthLoadingOverlay');
const depthLoadMsg        = document.getElementById('depthLoadMsg');
const depthProgressBar    = document.getElementById('depthProgressBar');
const depthStatsTag       = document.getElementById('telDepthStats');
const viewModeNav         = document.getElementById('viewModeNav');
const frameModeToggle     = document.getElementById('frameModeToggle');
const btnReset            = document.getElementById('btnReset');
const btnReconnect        = document.getElementById('btnReconnect');
const engineInfo          = document.getElementById('engineInfo');

const telObjects  = document.getElementById('telObjects');
const telHazards  = document.getElementById('telHazards');
const telFps      = document.getElementById('telFps');
const pillVisionLabel = document.getElementById('pillVisionLabel');
const pillSTTLabel    = document.getElementById('pillSTTLabel');
const pillTTSLabel    = document.getElementById('pillTTSLabel');
const msgCount        = document.getElementById('msgCount');

const hudListening = document.getElementById('hudListening');
const hudThinking  = document.getElementById('hudThinking');
const hudSpeaking  = document.getElementById('hudSpeaking');

const convFeed    = document.getElementById('convFeed');
const welcomeCard = document.getElementById('welcomeCard');
const textForm    = document.getElementById('textForm');
const textInput   = document.getElementById('textInput');
const btnSend     = document.getElementById('btnSend');
const btnPTT      = document.getElementById('btnPTT');
const pttLabel    = document.getElementById('pttLabel');
const toastContainer = document.getElementById('toastContainer');

// ── State ─────────────────────────────────────────────────────────

let currentMode = 'all';
let frameMode   = 'TEMPORAL_FRAMES';
let msgTotal    = 0;
let isRecording = false;
let isProcessing = false;

let mediaRecorder = null;
let audioChunks   = [];
let audioStream   = null;

// ════════════════════════════════════════════════════════════════════
//  DEPTH TAB — Browser-side WebGPU inference (v.fast depth exact port)
// ════════════════════════════════════════════════════════════════════

let depthPipelineReady = false;
let depthPipelineLoading = false;
let depthInferencing = false;

// EMA state (mirrors GPUDepthRenderer smoothedMin/smoothedMax)
let dSmoothedMin = 0;
let dSmoothedMax = 1;
let dHasRange = false;

// Offscreen canvas for frame capture from MJPEG img element
let depthOffscreen = null;
let depthOffCtx = null;

// Depth model reference
let depthModel = null;

// Persistent pre-allocated buffer + LUTs (mirrors DepthPipeline constructor)
const DEPTH_RES = 336;
let persistentPixelBuffer = null;
let lutR = null, lutG = null, lutB = null;

function buildDepthLUTs() {
  lutR = new Float32Array(256);
  lutG = new Float32Array(256);
  lutB = new Float32Array(256);
  const inv255 = 1 / 255;
  const meanR = 0.485, meanG = 0.456, meanB = 0.406;
  const invStdR = 1 / 0.229, invStdG = 1 / 0.224, invStdB = 1 / 0.225;
  for (let i = 0; i < 256; i++) {
    lutR[i] = (i * inv255 - meanR) * invStdR;
    lutG[i] = (i * inv255 - meanG) * invStdG;
    lutB[i] = (i * inv255 - meanB) * invStdB;
  }
  persistentPixelBuffer = new Float32Array(3 * DEPTH_RES * DEPTH_RES);
}

async function initDepthPipeline() {
  if (depthPipelineReady || depthPipelineLoading) return;
  depthPipelineLoading = true;

  depthLoadingOverlay.classList.remove('hidden');
  depthLoadMsg.textContent = 'Initializing WebGPU Depth Pipeline...';
  depthProgressBar.style.width = '0%';

  buildDepthLUTs();

  depthOffscreen = document.createElement('canvas');
  depthOffscreen.width = DEPTH_RES;
  depthOffscreen.height = DEPTH_RES;
  depthOffCtx = depthOffscreen.getContext('2d', { willReadFrequently: true });

  try {
    const { AutoModelForDepthEstimation, env } = await import('@huggingface/transformers');
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const MODEL_NAME = 'onnx-community/depth-anything-v2-small';
    depthLoadMsg.textContent = `Loading ${MODEL_NAME} (WebGPU FP16)…`;

    depthModel = await AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, {
      device: 'webgpu',
      dtype: 'fp16',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.progress != null) {
          const pct = Math.round(p.progress);
          depthProgressBar.style.width = pct + '%';
          depthLoadMsg.textContent = `Downloading model: ${pct}%`;
        } else if (p.status === 'initiate') {
          depthLoadMsg.textContent = `Fetching: ${p.file || ''}…`;
        }
      },
    });

    // Warmup passes (mirrors DepthPipeline.warmup())
    depthLoadMsg.textContent = 'Warming up GPU…';
    persistentPixelBuffer.fill(0);
    const { Tensor } = await import('@huggingface/transformers');
    const dummyTensor = new Tensor('float32', persistentPixelBuffer, [1, 3, DEPTH_RES, DEPTH_RES]);
    for (let i = 0; i < 3; i++) {
      await depthModel({ pixel_values: dummyTensor });
    }

    depthPipelineReady = true;
    depthLoadingOverlay.classList.add('hidden');
    depthProgressBar.style.width = '100%';
    engineInfo.textContent = `YOLO11 · YOLO-World · Depth Anything V2 (WebGPU FP16) · YOLO-Pose`;
    depthStatsTag.classList.remove('hidden');
    toast('WebGPU depth pipeline ready', 'success');
    startDepthLoop();

  } catch (err) {
    console.error('[Depth] Init error:', err);
    depthLoadMsg.textContent = `Error: ${err.message}`;
    toast('WebGPU depth failed — falling back to server stream', 'error');
    // Fall back: show server-side MJPEG depth stream
    depthLoadingOverlay.classList.add('hidden');
    depthCanvas.classList.add('hidden');
    videoFeed.src = `/api/video_feed?mode=depth&t=${Date.now()}`;
    videoFeed.classList.remove('hidden');
    depthPipelineLoading = false;
  }
}

async function runDepthInference() {
  if (!depthPipelineReady || depthInferencing) return;
  if (currentMode !== 'depth') return;
  depthInferencing = true;

  try {
    const { Tensor } = await import('@huggingface/transformers');

    // ── 1. Capture current frame from raw MJPEG img into offscreen canvas ──
    const t0 = performance.now();
    depthOffCtx.drawImage(videoFeed, 0, 0, DEPTH_RES, DEPTH_RES);
    const imgData = depthOffCtx.getImageData(0, 0, DEPTH_RES, DEPTH_RES).data;

    // ── 2. LUT preprocessing — planar RGB, ImageNet norm (mirrors DepthPipeline) ──
    const N = DEPTH_RES * DEPTH_RES;
    for (let i = 0, j = 0; i < N; i++, j += 4) {
      persistentPixelBuffer[i]       = lutR[imgData[j]];
      persistentPixelBuffer[N + i]   = lutG[imgData[j + 1]];
      persistentPixelBuffer[N*2 + i] = lutB[imgData[j + 2]];
    }
    const preprocessMs = performance.now() - t0;

    // ── 3. WebGPU inference ──
    const t2 = performance.now();
    const inputTensor = new Tensor('float32', persistentPixelBuffer, [1, 3, DEPTH_RES, DEPTH_RES]);
    const output = await depthModel({ pixel_values: inputTensor });
    const inferenceMs = performance.now() - t2;

    // ── 4. Extract depth array ──
    const rawTensor = output.predicted_depth || output.depth || output[Object.keys(output)[0]];
    let depthData;
    let outW = DEPTH_RES, outH = DEPTH_RES;
    if (rawTensor && rawTensor.data) {
      depthData = rawTensor.data instanceof Float32Array ? rawTensor.data : new Float32Array(rawTensor.data);
      if (rawTensor.dims && rawTensor.dims.length >= 2) {
        outW = rawTensor.dims[rawTensor.dims.length - 1];
        outH = rawTensor.dims[rawTensor.dims.length - 2];
      }
    } else {
      depthData = new Float32Array(N);
    }

    // ── 5. Render to canvas ──
    const t4 = performance.now();
    renderDepthToCanvas(depthData, outW, outH);
    const renderMs = performance.now() - t4;

    const totalMs = performance.now() - t0;
    depthStatsTag.textContent = `WebGPU ${(1000/totalMs).toFixed(1)} FPS | inf ${inferenceMs.toFixed(0)}ms`;

  } catch (err) {
    console.error('[Depth] Inference error:', err);
  } finally {
    depthInferencing = false;
  }
}

/**
 * Render Float32 depth data to the depth canvas using the analytical TURBO
 * colormap polynomial — exact port of WGSL colormapTurbo() from gpu-renderer.ts.
 * EMA temporal smoothing matches GPUDepthRenderer (85% history, 15% current).
 */
function renderDepthToCanvas(depthData, w, h) {
  const canvas = depthCanvas;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);
  const pixels = imageData.data;
  const n = depthData.length;

  // ── Adaptive range: 64-sample strided scan (mirrors GPU renderer) ──
  const step = Math.max(1, Math.floor(n / 64));
  let curMin = Infinity, curMax = -Infinity;
  for (let i = 0; i < n; i += step) {
    const v = depthData[i];
    if (v < curMin) curMin = v;
    if (v > curMax) curMax = v;
  }
  if (!isFinite(curMin)) curMin = 0;
  if (!isFinite(curMax) || curMax <= curMin) curMax = curMin + 1;

  // ── EMA smoothing: 85% history + 15% current (matches GPUDepthRenderer) ──
  if (!dHasRange) {
    dSmoothedMin = curMin; dSmoothedMax = curMax; dHasRange = true;
  } else {
    dSmoothedMin = dSmoothedMin * 0.85 + curMin * 0.15;
    dSmoothedMax = dSmoothedMax * 0.85 + curMax * 0.15;
  }

  const dRange = dSmoothedMax - dSmoothedMin || 1;

  // ── Analytical TURBO colormap polynomial (exact match to WGSL shader) ──
  for (let i = 0; i < n; i++) {
    const x = Math.min(1, Math.max(0, (depthData[i] - dSmoothedMin) / dRange));
    const x2 = x*x, x3 = x2*x, x4 = x3*x, x5 = x4*x;

    const r = Math.min(1, Math.max(0, 0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.13108234*x3 - 152.94239396*x4 + 59.28637943*x5));
    const g = Math.min(1, Math.max(0, 0.09140261 + 2.19418839*x +  4.84296658*x2 -  14.18503333*x3 +   4.27729857*x4 +  2.82956604*x5));
    const b = Math.min(1, Math.max(0, 0.10667447 +12.64194608*x - 60.58204836*x2 + 110.36276771*x3 -  89.90310912*x4 + 27.34824973*x5));

    const off = i * 4;
    pixels[off]   = r * 255;
    pixels[off+1] = g * 255;
    pixels[off+2] = b * 255;
    pixels[off+3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);

  // ── Scale bar ──
  const barX = w - 20, barH = h - 40;
  for (let y = 0; y < barH; y++) {
    const t = y / barH;
    const t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    const br = Math.min(1,Math.max(0, 0.13572138 + 4.61539260*t - 42.66032258*t2 + 132.13108234*t3 - 152.94239396*t4 + 59.28637943*t5));
    const bg = Math.min(1,Math.max(0, 0.09140261 + 2.19418839*t +  4.84296658*t2 -  14.18503333*t3 +   4.27729857*t4 +  2.82956604*t5));
    const bb = Math.min(1,Math.max(0, 0.10667447 +12.64194608*t - 60.58204836*t2 + 110.36276771*t3 -  89.90310912*t4 + 27.34824973*t5));
    ctx.fillStyle = `rgb(${(br*255)|0},${(bg*255)|0},${(bb*255)|0})`;
    ctx.fillRect(barX, 20 + y, 10, 1);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.strokeRect(barX - 1, 19, 12, barH + 2);
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.font = '9px monospace';
  ctx.fillText(`${dSmoothedMin.toFixed(1)}`, barX - 24, 24);
  ctx.fillText(`${dSmoothedMax.toFixed(1)}`, barX - 24, 20 + barH);
}

function startDepthLoop() {
  const loop = () => {
    if (currentMode === 'depth') runDepthInference();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

// ════════════════════════════════════════════════════════════════════
//  POSE TAB — Three.js real-time 3D skeleton viewer
// ════════════════════════════════════════════════════════════════════

let threeRenderer = null;
let threeScene    = null;
let threeCamera   = null;
let poseActive    = false;
let poseObjects   = [];  // { boneLine, joint }

// COCO skeleton connections [kp_a, kp_b, color_hex]
const SKELETON = [
  [0,1,'#a78bfa'], [0,2,'#a78bfa'],          // nose-eyes
  [1,3,'#7c3aed'], [2,4,'#7c3aed'],          // eyes-ears
  [5,6,'#22d3ee'],                            // shoulders
  [5,7,'#3b82f6'], [7,9,'#3b82f6'],          // left arm
  [6,8,'#f97316'], [8,10,'#f97316'],         // right arm
  [5,11,'#22c55e'], [6,12,'#22c55e'],        // torso sides
  [11,12,'#22d3ee'],                          // hips
  [11,13,'#3b82f6'], [13,15,'#3b82f6'],      // left leg
  [12,14,'#f97316'], [14,16,'#f97316'],      // right leg
];

function initThreePose() {
  if (threeRenderer) return;

  const canvas = poseCanvas;
  const W = canvas.parentElement.clientWidth  || 640;
  const H = canvas.parentElement.clientHeight || 480;
  canvas.width  = W;
  canvas.height = H;

  threeRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeRenderer.setSize(W, H);
  threeRenderer.setClearColor(0x0a0a1a, 1);

  threeScene = new THREE.Scene();

  // Perspective camera
  threeCamera = new THREE.PerspectiveCamera(60, W / H, 0.1, 100);
  threeCamera.position.set(0, 0, 5);
  threeCamera.lookAt(0, 0, 0);

  // Subtle ambient + point lights
  threeScene.add(new THREE.AmbientLight(0x222244, 1));
  const pt = new THREE.PointLight(0x6366f1, 2, 20);
  pt.position.set(2, 3, 4);
  threeScene.add(pt);

  // Grid floor
  const grid = new THREE.GridHelper(10, 20, 0x1e1e3a, 0x1e1e3a);
  grid.position.y = -2;
  threeScene.add(grid);

  // Slow orbit animation
  let angle = 0;
  const animate = () => {
    requestAnimationFrame(animate);
    if (currentMode !== 'pose') return;
    angle += 0.004;
    threeCamera.position.x = Math.sin(angle) * 5;
    threeCamera.position.z = Math.cos(angle) * 5;
    threeCamera.lookAt(0, 0, 0);
    threeRenderer.render(threeScene, threeCamera);
  };
  animate();
}

function clearPoseMeshes() {
  poseObjects.forEach(o => {
    threeScene.remove(o);
    o.geometry && o.geometry.dispose();
    o.material && o.material.dispose();
  });
  poseObjects = [];
}

function updatePose3D(poseData) {
  if (!threeScene) return;
  clearPoseMeshes();

  const FW = poseData.frame_width  || 1280;
  const FH = poseData.frame_height || 720;

  // Convert 2D pixel + depth → 3D coords
  // Normalize: x ∈ [-2.5, 2.5], y ∈ [-1.5, 1.5] (inverted), z from depth
  function toVec3(kp) {
    if (!kp || kp.conf < 0.2) return null;
    const x = ((kp.x / FW) - 0.5) * 5.0;
    const y = -((kp.y / FH) - 0.5) * 3.0;
    const z = kp.depth ? -(kp.depth * 0.5) : 0;  // depth-lifted Z
    return new THREE.Vector3(x, y, z);
  }

  for (const person of poseData.poses) {
    const kps = person.keypoints;
    if (!kps || kps.length < 17) continue;

    // ── Joints (spheres) ──
    for (const kp of kps) {
      if (kp.conf < 0.2) continue;
      const pos = toVec3(kp);
      if (!pos) continue;
      const geo = new THREE.SphereGeometry(0.045, 8, 8);
      const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, emissive: 0x6366f1, emissiveIntensity: 0.8 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pos);
      threeScene.add(mesh);
      poseObjects.push(mesh);
    }

    // ── Bones (lines) ──
    for (const [a, b, colorHex] of SKELETON) {
      const pA = toVec3(kps[a]);
      const pB = toVec3(kps[b]);
      if (!pA || !pB) continue;

      const points = [pA, pB];
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 2 });
      const line = new THREE.Line(geo, mat);
      threeScene.add(line);
      poseObjects.push(line);
    }

    // ── Gaze ray ──
    if (person.head_yaw != null && kps[0] && kps[0].conf >= 0.2) {
      const nose = toVec3(kps[0]);
      if (nose) {
        const yawRad = (person.head_yaw * Math.PI) / 180;
        const dir = new THREE.Vector3(Math.cos(yawRad) * 1.2, 0, Math.sin(yawRad) * 0.5);
        const end = nose.clone().add(dir);
        const geo = new THREE.BufferGeometry().setFromPoints([nose, end]);
        const mat = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
        const line = new THREE.Line(geo, mat);
        threeScene.add(line);
        poseObjects.push(line);
      }
    }
  }

  // Empty state label
  if (poseData.poses.length === 0) {
    // Keep scene empty — grid still shows
  }
}

async function pollPoseData() {
  if (currentMode !== 'pose') return;
  try {
    const res = await fetch('/api/pose');
    if (res.ok) {
      const data = await res.json();
      updatePose3D(data);
    }
  } catch (_) {}
}

// ── View mode switching ───────────────────────────────────────────

function setActiveViewElement(mode) {
  // Hide all
  videoFeed.classList.add('hidden');
  depthCanvas.classList.add('hidden');
  poseCanvas.classList.add('hidden');

  if (mode === 'depth') {
    depthCanvas.classList.remove('hidden');
    // Also pipe raw video to the offscreen for depth inference capture
    videoFeed.src = `/api/video_feed?mode=raw&t=${Date.now()}`;
  } else if (mode === 'pose') {
    poseCanvas.classList.remove('hidden');
  } else {
    videoFeed.classList.remove('hidden');
  }
}

viewModeNav.addEventListener('click', e => {
  const tab = e.target.closest('.view-tab');
  if (!tab) return;
  const mode = tab.dataset.mode;
  if (mode === currentMode) return;

  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  currentMode = mode;

  setActiveViewElement(mode);

  if (mode === 'depth') {
    depthStatsTag.classList.remove('hidden');
    if (!depthPipelineReady && !depthPipelineLoading) {
      initDepthPipeline();
    } else if (depthPipelineReady) {
      depthLoadingOverlay.classList.add('hidden');
    }
  } else {
    depthStatsTag.classList.add('hidden');
    depthLoadingOverlay.classList.add('hidden');
    if (mode !== 'pose') {
      videoFeed.src = `/api/video_feed?mode=${mode}&t=${Date.now()}`;
    }
  }

  if (mode === 'pose') {
    initThreePose();
    pollPoseData();
  }
});

// Poll pose data at 15fps while in pose mode
setInterval(() => { if (currentMode === 'pose') pollPoseData(); }, 66);

btnReconnect.addEventListener('click', () => {
  if (currentMode !== 'depth' && currentMode !== 'pose') {
    videoFeed.src = `/api/video_feed?mode=${currentMode}&t=${Date.now()}`;
  }
  toast('Reconnected');
});

videoFeed.addEventListener('error', () => {
  if (currentMode !== 'depth' && currentMode !== 'pose') {
    setTimeout(() => { videoFeed.src = `/api/video_feed?mode=${currentMode}&t=${Date.now()}`; }, 2000);
  }
});

// ── Frame mode toggle ─────────────────────────────────────────────

frameModeToggle.addEventListener('click', e => {
  const btn = e.target.closest('.mode-tab-btn');
  if (!btn) return;
  document.querySelectorAll('.mode-tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  frameMode = btn.id === 'btnModeTemporal' ? 'TEMPORAL_FRAMES' : 'SINGLE_FRAME';
});

// ── Status polling ────────────────────────────────────────────────

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();
    const c = d.copilot || {};
    const p = d.providers || {};
    telObjects.textContent = `${c.tracked_count ?? 0} objects`;
    telHazards.textContent = `${c.hazard_count  ?? c.hazards_count ?? 0} hazards`;
    telFps.textContent     = `${Number(c.fps ?? 0).toFixed(1)} fps`;
    if (p.vision) pillVisionLabel.textContent = `Vision: ${p.vision.split(':')[0]}`;
    if (p.stt)    pillSTTLabel.textContent    = `STT: ${p.stt}`;
    if (p.tts)    pillTTSLabel.textContent    = `TTS: ${p.tts}`;
    const turns = d.history_turns ?? 0;
    msgCount.textContent = `${turns} message${turns !== 1 ? 's' : ''}`;
  } catch (_) {}
}

pollStatus();
setInterval(pollStatus, 2000);

// ── HUD helpers ───────────────────────────────────────────────────

function showHUD(state) {
  hudListening.classList.add('hidden');
  hudThinking.classList.add('hidden');
  hudSpeaking.classList.add('hidden');
  if (state === 'listening') hudListening.classList.remove('hidden');
  if (state === 'thinking')  hudThinking.classList.remove('hidden');
  if (state === 'speaking')  hudSpeaking.classList.remove('hidden');
}

function hideHUD() {
  hudListening.classList.add('hidden');
  hudThinking.classList.add('hidden');
  hudSpeaking.classList.add('hidden');
}

// ── Push-to-Talk ──────────────────────────────────────────────────

function mimeType() {
  for (const t of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

async function getStream() {
  if (!audioStream || audioStream.getTracks().every(t => t.readyState === 'ended')) {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  }
  return audioStream;
}

async function startRec() {
  if (isRecording || isProcessing) return;
  try {
    const stream = await getStream();
    const mime = mimeType();
    mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
    audioChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data?.size > 0) audioChunks.push(e.data); };
    mediaRecorder.start(100);
    isRecording = true;
    btnPTT.classList.add('recording');
    pttLabel.textContent = 'Recording — Release to send';
    showHUD('listening');
  } catch {
    toast('Microphone access denied', 'error');
  }
}

async function stopRec() {
  if (!isRecording || !mediaRecorder) return;
  await new Promise(res => { mediaRecorder.onstop = res; mediaRecorder.stop(); });
  isRecording = false;
}

async function submitVoice() {
  await stopRec();
  btnPTT.classList.remove('recording');
  pttLabel.textContent = 'Push to Talk';
  if (!audioChunks.length) { hideHUD(); return; }

  const mime = mimeType() || 'audio/webm';
  const blob = new Blob(audioChunks, { type: mime });
  audioChunks = [];
  const userRow = addMsg('user', '🎙️ Voice query...');
  isProcessing = true;
  showHUD('thinking');

  try {
    const fd = new FormData();
    const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'mp4' : 'webm';
    fd.append('audio', blob, `rec.${ext}`);
    fd.append('frame_mode', frameMode);
    const t0 = Date.now();
    const res = await fetch('/api/ask', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Error ${res.status}`);
    const data = await res.json();
    if (data.transcript) userRow.querySelector('.bubble').textContent = `🎙️ "${data.transcript}"`;
    addAssistantMsg(data, Date.now() - t0);
  } catch (err) {
    toast(err.message, 'error');
    userRow.remove();
  } finally {
    isProcessing = false;
    hideHUD();
  }
}

btnPTT.addEventListener('mousedown', e => { e.preventDefault(); startRec(); });
btnPTT.addEventListener('mouseup', () => { if (isRecording) submitVoice(); });
btnPTT.addEventListener('mouseleave', () => { if (isRecording) submitVoice(); });
btnPTT.addEventListener('touchstart', e => { e.preventDefault(); startRec(); });
btnPTT.addEventListener('touchend', e => { e.preventDefault(); if (isRecording) submitVoice(); });

let spaceDown = false;
document.addEventListener('keydown', e => {
  if (e.target === textInput) return;
  if (e.code === 'Space' && !spaceDown && !isProcessing) { e.preventDefault(); spaceDown = true; startRec(); }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space' && spaceDown) { e.preventDefault(); spaceDown = false; if (isRecording) submitVoice(); }
});

// ── Text submission ───────────────────────────────────────────────

textForm.addEventListener('submit', async e => {
  e.preventDefault();
  const q = textInput.value.trim();
  if (!q || isProcessing) return;
  textInput.value = '';
  isProcessing = true;
  btnSend.disabled = true;
  addMsg('user', q);
  showHUD('thinking');
  try {
    const fd = new FormData();
    fd.append('question', q);
    fd.append('frame_mode', frameMode);
    const t0 = Date.now();
    const res = await fetch('/api/ask-text', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Error ${res.status}`);
    const data = await res.json();
    addAssistantMsg(data, Date.now() - t0);
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    isProcessing = false;
    hideHUD();
    btnSend.disabled = false;
  }
});

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    if (isProcessing) return;
    textInput.value = chip.dataset.query;
    textForm.dispatchEvent(new Event('submit', { cancelable: true }));
  });
});

// ── Context reset ─────────────────────────────────────────────────

btnReset.addEventListener('click', async () => {
  try {
    await fetch('/api/reset', { method: 'POST' });
    convFeed.innerHTML = '';
    convFeed.appendChild(welcomeCard);
    welcomeCard.classList.remove('hidden');
    msgTotal = 0;
    msgCount.textContent = '0 messages';
    hideHUD();
    toast('Conversation cleared');
  } catch { toast('Reset failed', 'error'); }
});

// ── Chat rendering ────────────────────────────────────────────────

function addMsg(role, text) {
  welcomeCard.classList.add('hidden');
  const row = document.createElement('div');
  row.className = `msg-row ${role === 'user' ? 'user' : 'kaya'}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  const ts = document.createElement('span');
  ts.className = 'meta-time';
  ts.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  meta.appendChild(ts);
  row.appendChild(bubble);
  row.appendChild(meta);
  convFeed.appendChild(row);
  msgTotal++;
  msgCount.textContent = `${msgTotal} message${msgTotal !== 1 ? 's' : ''}`;
  scrollDown();
  return row;
}

function addAssistantMsg(data, elapsed) {
  const text = data.response || data.answer || data.text || 'No response.';
  const row = addMsg('kaya', text);
  const meta = row.querySelector('.msg-meta');

  const t = data.timings || {};
  [['STT', t.stt_ms], ['VLM', t.vision_ms], ['TTS', t.tts_ms]].forEach(([label, val]) => {
    if (val == null) return;
    const p = document.createElement('span');
    p.className = 'lat-pill';
    p.textContent = `${label}: ${Math.round(val)}ms`;
    meta.appendChild(p);
  });
  if (!t.stt_ms && !t.vision_ms && elapsed) {
    const p = document.createElement('span');
    p.className = 'lat-pill';
    p.textContent = `${Math.round(elapsed)}ms`;
    meta.appendChild(p);
  }

  const audiob64 = data.audio_base64;
  if (audiob64) {
    const getAudio = () => new Audio(`data:audio/wav;base64,${audiob64}`);
    const btn = document.createElement('button');
    btn.className = 'btn-replay';
    btn.textContent = '▶ Replay';
    btn.onclick = () => getAudio().play().catch(() => {});
    meta.appendChild(btn);
    showHUD('speaking');
    const audio = getAudio();
    audio.onended = hideHUD;
    audio.onerror = () => { console.warn('TTS error'); hideHUD(); };
    audio.play().catch(err => { console.warn('TTS blocked:', err); hideHUD(); });
  }

  scrollDown();
  return row;
}

// ── Toasts ────────────────────────────────────────────────────────

function toast(msg, type = 'default', ms = 3000) {
  const el = document.createElement('div');
  el.className = `toast${type === 'success' ? ' success' : type === 'error' ? ' error' : ''}`;
  el.textContent = msg;
  toastContainer.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 200); }, ms);
}

function scrollDown() {
  requestAnimationFrame(() => { convFeed.scrollTop = convFeed.scrollHeight; });
}

// ── Init ──────────────────────────────────────────────────────────

// Ensure initial view element is correct
setActiveViewElement('all');
videoFeed.src = `/api/video_feed?mode=all&t=${Date.now()}`;
