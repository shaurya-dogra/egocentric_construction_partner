/**
 * Kaya — Job Site Safety Copilot
 * Clean, simple client script
 */

// DOM
const videoFeed     = document.getElementById('videoFeed');
const viewModeNav   = document.getElementById('viewModeNav');
const frameModeToggle = document.getElementById('frameModeToggle');
const btnModeTemporal = document.getElementById('btnModeTemporal');
const btnModeSingle   = document.getElementById('btnModeSingle');
const btnReset      = document.getElementById('btnReset');
const btnReconnect  = document.getElementById('btnReconnect');

const telObjects    = document.getElementById('telObjects');
const telHazards    = document.getElementById('telHazards');
const telFps        = document.getElementById('telFps');
const pillVisionLabel = document.getElementById('pillVisionLabel');
const pillSTTLabel  = document.getElementById('pillSTTLabel');
const pillTTSLabel  = document.getElementById('pillTTSLabel');
const msgCount      = document.getElementById('msgCount');

const hudListening  = document.getElementById('hudListening');
const hudThinking   = document.getElementById('hudThinking');
const hudSpeaking   = document.getElementById('hudSpeaking');

const convFeed      = document.getElementById('convFeed');
const welcomeCard   = document.getElementById('welcomeCard');
const textForm      = document.getElementById('textForm');
const textInput     = document.getElementById('textInput');
const btnSend       = document.getElementById('btnSend');
const btnPTT        = document.getElementById('btnPTT');
const pttLabel      = document.getElementById('pttLabel');
const toastContainer = document.getElementById('toastContainer');

// State
let currentMode   = 'all';
let frameMode     = 'TEMPORAL_FRAMES';
let msgTotal      = 0;
let isRecording   = false;
let isProcessing  = false;
let mediaRecorder = null;
let audioChunks   = [];
let audioStream   = null;

// ── View mode switching ───────────────────────────────────────────

viewModeNav.addEventListener('click', e => {
  const tab = e.target.closest('.view-tab');
  if (!tab) return;
  const mode = tab.dataset.mode;
  if (mode === currentMode) return;
  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  currentMode = mode;
  reloadStream();
});

function reloadStream() {
  videoFeed.src = `/api/video_feed?mode=${currentMode}&t=${Date.now()}`;
}

btnReconnect.addEventListener('click', () => { reloadStream(); toast('Stream reconnected'); });
videoFeed.addEventListener('error', () => setTimeout(reloadStream, 3000));

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

    const obj = c.tracked_count ?? c.objects_count ?? 0;
    const haz = c.hazard_count ?? 0;
    const fps = Number(c.fps ?? 0).toFixed(1);

    telObjects.textContent = `${obj} object${obj !== 1 ? 's' : ''}`;
    telHazards.textContent = `${haz} hazard${haz !== 1 ? 's' : ''}`;
    telFps.textContent     = `${fps} fps`;

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

// PTT mouse
btnPTT.addEventListener('mousedown', e => { e.preventDefault(); startRec(); });
btnPTT.addEventListener('mouseup', () => { if (isRecording) submitVoice(); });
btnPTT.addEventListener('mouseleave', () => { if (isRecording) submitVoice(); });

// PTT touch
btnPTT.addEventListener('touchstart', e => { e.preventDefault(); startRec(); });
btnPTT.addEventListener('touchend', e => { e.preventDefault(); if (isRecording) submitVoice(); });

// Space key
let spaceDown = false;
document.addEventListener('keydown', e => {
  if (e.target === textInput) return;
  if (e.code === 'Space' && !spaceDown && !isProcessing) {
    e.preventDefault(); spaceDown = true; startRec();
  }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space' && spaceDown) {
    e.preventDefault(); spaceDown = false;
    if (isRecording) submitVoice();
  }
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

// Quick chips
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
  } catch {
    toast('Reset failed', 'error');
  }
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
  // The API returns: response, audio_base64, timings.{stt_ms, vision_ms, tts_ms}
  const text = data.response || data.answer || data.text || 'No response.';
  const row = addMsg('kaya', text);
  const meta = row.querySelector('.msg-meta');

  // Latency pills — from nested timings object
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

  // Audio replay + autoplay — API returns audio_base64 (WAV from Sarvam TTS)
  const audiob64 = data.audio_base64;
  if (audiob64) {
    const getAudio = () => {
      return new Audio(`data:audio/wav;base64,${audiob64}`);
    };

    const btn = document.createElement('button');
    btn.className = 'btn-replay';
    btn.textContent = '▶ Replay';
    btn.onclick = () => getAudio().play().catch(() => {});
    meta.appendChild(btn);

    // Autoplay TTS
    showHUD('speaking');
    const audio = getAudio();
    audio.onended = hideHUD;
    audio.onerror = () => { console.warn('TTS audio play error'); hideHUD(); };
    audio.play().catch(err => { console.warn('TTS autoplay blocked:', err); hideHUD(); });
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

// ── Utilities ─────────────────────────────────────────────────────

function scrollDown() {
  requestAnimationFrame(() => { convFeed.scrollTop = convFeed.scrollHeight; });
}

// Init
reloadStream();
