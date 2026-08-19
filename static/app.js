/**
 * Kaya — Job Site Safety Copilot
 * Client-Side Application Script
 * ---------------------------------
 * Handles:
 *  - Real-time view mode tab switching (MJPEG stream query param)
 *  - Frame mode toggle (Temporal / Single Frame)
 *  - Push-to-Talk via MediaRecorder + Space key hold/release
 *  - Text query submission via /api/ask-text
 *  - Voice query submission via /api/ask
 *  - Quick suggestion chip auto-submit
 *  - Chat message rendering with latency pills & audio replay
 *  - Status polling (/api/status) every 2 seconds
 *  - Toast notifications
 *  - Context reset via /api/reset
 */

// =============================================================================
// DOM References
// =============================================================================

const videoFeed         = document.getElementById('copilotVideoFeed');
const viewModeNav       = document.getElementById('viewModeNav');
const frameModeGroup    = document.getElementById('frameModeToggleGroup');
const btnModeTemporal   = document.getElementById('btnModeTemporal');
const btnModeSingle     = document.getElementById('btnModeSingle');
const btnResetContext   = document.getElementById('btnResetContext');
const btnRefreshFeed    = document.getElementById('btnRefreshFeed');

// Status telemetry
const trackedCountTag   = document.getElementById('trackedCountTag');
const hazardsCountTag   = document.getElementById('hazardsCountTag');
const bufferStatusTag   = document.getElementById('bufferStatusTag');
const copilotFpsTag     = document.getElementById('copilotFpsTag');
const pillVisionLabel   = document.getElementById('pillVisionLabel');
const pillSTTLabel      = document.getElementById('pillSTTLabel');
const pillTTSLabel      = document.getElementById('pillTTSLabel');
const historyCounter    = document.getElementById('historyCounter');

// HUD overlays
const statusListening   = document.getElementById('statusListening');
const statusThinking    = document.getElementById('statusThinking');
const statusSpeaking    = document.getElementById('statusSpeaking');

// Chat panel
const conversationFeed  = document.getElementById('conversationFeed');
const feedEmptyState    = document.getElementById('feedEmptyState');
const textQueryForm     = document.getElementById('textQueryForm');
const inputQueryText    = document.getElementById('inputQueryText');
const btnSubmitText     = document.getElementById('btnSubmitText');
const btnPushToTalk     = document.getElementById('btnPushToTalk');
const pttMainText       = document.getElementById('pttMainText');
const toastContainer    = document.getElementById('toastContainer');

// Quick chips
const chipBtns          = document.querySelectorAll('.chip-btn');

// =============================================================================
// Application State
// =============================================================================

let currentMode     = 'all';   // Active video stream mode
let frameMode       = 'TEMPORAL_FRAMES'; // API frame_mode param
let messageCount    = 0;
let isRecording     = false;
let isProcessing    = false;

// MediaRecorder state
let mediaRecorder   = null;
let audioChunks     = [];
let audioStream     = null;

// Playback
let lastAudioBlob   = null;
let latestAudioEl   = null;

// Status polling
let statusInterval  = null;

// =============================================================================
// View Mode Tab Switching
// =============================================================================

viewModeNav.addEventListener('click', (e) => {
  const tab = e.target.closest('.view-tab');
  if (!tab) return;

  const newMode = tab.dataset.mode;
  if (newMode === currentMode) return;

  // Update active tab
  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');

  // Switch the MJPEG stream
  currentMode = newMode;
  refreshVideoStream();
});

function refreshVideoStream() {
  videoFeed.src = `/api/video_feed?mode=${currentMode}&t=${Date.now()}`;
}

// Reconnect button
btnRefreshFeed.addEventListener('click', () => {
  refreshVideoStream();
  showToast('🔄 Stream reconnected', 'success');
});

// =============================================================================
// Frame Mode Toggle (Temporal vs. Single Frame)
// =============================================================================

frameModeGroup.addEventListener('click', (e) => {
  const btn = e.target.closest('.mode-tab-btn');
  if (!btn) return;

  document.querySelectorAll('.mode-tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (btn.id === 'btnModeTemporal') {
    frameMode = 'TEMPORAL_FRAMES';
    showToast('🎞️ Temporal 6s mode active', 'success');
  } else {
    frameMode = 'SINGLE_FRAME';
    showToast('🖼️ Single frame mode active');
  }
});

// =============================================================================
// Status Polling
// =============================================================================

function startStatusPolling() {
  pollStatus(); // Immediate first call
  statusInterval = setInterval(pollStatus, 2000);
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();

    const copilot = data.copilot || {};
    const config  = data.config  || {};
    const providers = data.providers || {};

    // Telemetry pills
    const objCount = copilot.tracked_count ?? copilot.objects_count ?? 0;
    const hazCount = copilot.hazard_count  ?? 0;
    const fps      = copilot.fps           ?? 0;
    const bufCurr  = copilot.buffer_count  ?? copilot.frame_count ?? 0;
    const bufMax   = copilot.buffer_max    ?? 8;

    trackedCountTag.textContent = `● ${objCount} Object${objCount !== 1 ? 's' : ''}`;
    hazardsCountTag.textContent = `● ${hazCount} Hazard${hazCount !== 1 ? 's' : ''}`;
    bufferStatusTag.textContent = `● Buffer: ${bufCurr}/${bufMax}`;
    copilotFpsTag.textContent   = `${Number(fps).toFixed(1)} FPS`;

    // Provider badges
    if (providers.vision) {
      const parts = providers.vision.split(':');
      pillVisionLabel.textContent = `Vision: ${parts[0]}`;
    }
    if (providers.stt) {
      pillSTTLabel.textContent = `STT: ${providers.stt}`;
    }
    if (providers.tts) {
      pillTTSLabel.textContent = `TTS: ${providers.tts}`;
    }

    // History counter
    const turns = data.history_turns ?? 0;
    historyCounter.textContent = `${turns} message${turns !== 1 ? 's' : ''}`;

  } catch (_) {
    // Network error — silently ignore
  }
}

// =============================================================================
// HUD State Helpers
// =============================================================================

function showHUD(state) {
  statusListening.classList.add('hidden');
  statusThinking.classList.add('hidden');
  statusSpeaking.classList.add('hidden');

  if (state === 'listening') statusListening.classList.remove('hidden');
  else if (state === 'thinking') statusThinking.classList.remove('hidden');
  else if (state === 'speaking') statusSpeaking.classList.remove('hidden');
}

function hideHUD() {
  statusListening.classList.add('hidden');
  statusThinking.classList.add('hidden');
  statusSpeaking.classList.add('hidden');
}

// =============================================================================
// Push-to-Talk Logic
// =============================================================================

async function ensureAudioStream() {
  if (!audioStream || audioStream.getTracks().every(t => t.readyState === 'ended')) {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  }
  return audioStream;
}

function getMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

async function startRecording() {
  if (isRecording || isProcessing) return;

  try {
    const stream = await ensureAudioStream();
    const mimeType = getMimeType();
    const options = mimeType ? { mimeType } : {};

    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.start(100); // 100ms time slices
    isRecording = true;

    // Update UI
    btnPushToTalk.classList.add('recording');
    pttMainText.textContent = 'Recording...';
    showHUD('listening');

  } catch (err) {
    console.error('Mic access error:', err);
    showToast('🎙️ Microphone access denied', 'error');
  }
}

async function stopRecording() {
  if (!isRecording || !mediaRecorder) return;

  return new Promise((resolve) => {
    mediaRecorder.onstop = () => resolve();
    mediaRecorder.stop();
    isRecording = false;
  });
}

async function submitVoiceQuery() {
  await stopRecording();

  if (audioChunks.length === 0) {
    resetPTTButton();
    hideHUD();
    return;
  }

  const mimeType = getMimeType() || 'audio/webm';
  const audioBlob = new Blob(audioChunks, { type: mimeType });
  audioChunks = [];

  // Append a user "voice" message placeholder
  const userMsgEl = appendMessage('user', '🎙️ Voice query...', null);

  isProcessing = true;
  resetPTTButton();
  showHUD('thinking');

  try {
    const formData = new FormData();
    const ext = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'mp4' : 'webm';
    formData.append('audio', audioBlob, `recording.${ext}`);
    formData.append('frame_mode', frameMode);

    const startTs = Date.now();
    const res = await fetch('/api/ask', { method: 'POST', body: formData });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    const elapsed = Date.now() - startTs;

    // Update user bubble if transcript is available
    if (data.transcript && userMsgEl) {
      userMsgEl.querySelector('.msg-bubble').textContent = `🎙️ "${data.transcript}"`;
    }

    appendAssistantMessage(data, elapsed);

  } catch (err) {
    hideHUD();
    showToast(`❌ ${err.message}`, 'error');
    userMsgEl?.remove();
    console.error('Voice query error:', err);
  } finally {
    isProcessing = false;
    hideHUD();
  }
}

// PTT button — mousedown/mouseup
btnPushToTalk.addEventListener('mousedown', (e) => {
  e.preventDefault();
  startRecording();
});

btnPushToTalk.addEventListener('mouseup', () => {
  if (isRecording) submitVoiceQuery();
});

btnPushToTalk.addEventListener('mouseleave', () => {
  if (isRecording) submitVoiceQuery();
});

// Touch events for mobile
btnPushToTalk.addEventListener('touchstart', (e) => {
  e.preventDefault();
  startRecording();
});

btnPushToTalk.addEventListener('touchend', (e) => {
  e.preventDefault();
  if (isRecording) submitVoiceQuery();
});

// Keyboard Space key hold
let spaceHeld = false;
document.addEventListener('keydown', (e) => {
  // Ignore if typing in text field
  if (document.activeElement === inputQueryText) return;
  if (e.code === 'Space' && !spaceHeld && !isProcessing) {
    e.preventDefault();
    spaceHeld = true;
    startRecording();
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && spaceHeld) {
    e.preventDefault();
    spaceHeld = false;
    if (isRecording) submitVoiceQuery();
  }
});

function resetPTTButton() {
  btnPushToTalk.classList.remove('recording');
  pttMainText.textContent = 'Push to Talk';
}

// =============================================================================
// Text Query Submission
// =============================================================================

textQueryForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const question = inputQueryText.value.trim();
  if (!question || isProcessing) return;

  inputQueryText.value = '';
  isProcessing = true;

  appendMessage('user', question, null);
  showHUD('thinking');
  btnSubmitText.disabled = true;

  try {
    const formData = new FormData();
    formData.append('question', question);
    formData.append('frame_mode', frameMode);

    const startTs = Date.now();
    const res = await fetch('/api/ask-text', { method: 'POST', body: formData });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    const elapsed = Date.now() - startTs;

    appendAssistantMessage(data, elapsed);

  } catch (err) {
    showToast(`❌ ${err.message}`, 'error');
    console.error('Text query error:', err);
  } finally {
    isProcessing = false;
    hideHUD();
    btnSubmitText.disabled = false;
  }
});

// =============================================================================
// Quick Prompt Chips
// =============================================================================

chipBtns.forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.dataset.query;
    if (!query || isProcessing) return;
    inputQueryText.value = query;
    textQueryForm.dispatchEvent(new Event('submit', { cancelable: true }));
  });
});

// =============================================================================
// Chat Rendering
// =============================================================================

/**
 * Append a user or assistant message.
 * @param {'user'|'assistant'} role
 * @param {string} text
 * @param {object|null} meta  — latency metadata from API
 * @returns {HTMLElement}
 */
function appendMessage(role, text, meta = null) {
  // Hide the welcome empty state on first message
  if (feedEmptyState && !feedEmptyState.classList.contains('hidden')) {
    feedEmptyState.classList.add('hidden');
  }

  const row = document.createElement('div');
  row.className = `message-bubble-row ${role === 'user' ? 'user-row' : 'assistant-row'}`;

  const bubble = document.createElement('div');
  bubble.className = `msg-bubble ${role === 'user' ? 'user-bubble' : 'assistant-bubble'}`;
  bubble.textContent = text;

  const metaRow = document.createElement('div');
  metaRow.className = 'msg-meta-row';

  const ts = document.createElement('span');
  ts.className = 'msg-timestamp';
  ts.textContent = formatTime(new Date());
  metaRow.appendChild(ts);

  row.appendChild(bubble);
  row.appendChild(metaRow);
  conversationFeed.appendChild(row);

  messageCount++;
  historyCounter.textContent = `${messageCount} message${messageCount !== 1 ? 's' : ''}`;

  scrollFeedToBottom();
  return row;
}

/**
 * Append a full assistant response with latency pills and audio replay.
 */
function appendAssistantMessage(data, totalElapsed) {
  const answer = data.answer || data.text || data.response || 'No response.';
  const row = appendMessage('assistant', answer, data);

  const metaRow = row.querySelector('.msg-meta-row');

  // Latency pills
  if (data.latency_ms || data.stt_ms || data.vlm_ms || data.tts_ms || totalElapsed) {
    const pillsWrap = document.createElement('div');
    pillsWrap.className = 'latency-pills-wrap';

    const addPill = (label, val) => {
      if (val == null) return;
      const p = document.createElement('span');
      p.className = 'latency-pill';
      p.textContent = `${label}: ${Math.round(val)}ms`;
      pillsWrap.appendChild(p);
    };

    addPill('STT', data.stt_ms);
    addPill('VLM', data.vlm_ms);
    addPill('TTS', data.tts_ms);

    if (!data.stt_ms && !data.vlm_ms && totalElapsed) {
      addPill('Total', totalElapsed);
    }

    metaRow.appendChild(pillsWrap);
  }

  // Audio replay button
  if (data.audio_b64 || data.audio_url) {
    const replayBtn = document.createElement('button');
    replayBtn.type = 'button';
    replayBtn.className = 'btn-replay-audio';
    replayBtn.title = 'Replay TTS audio';
    replayBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
        <polygon points="5,3 19,12 5,21"/>
      </svg>
      Replay
    `;

    replayBtn.addEventListener('click', () => {
      if (data.audio_url) {
        const audio = new Audio(data.audio_url);
        audio.play();
      } else if (data.audio_b64) {
        const mime = data.audio_mime || 'audio/mpeg';
        const src = `data:${mime};base64,${data.audio_b64}`;
        const audio = new Audio(src);
        audio.play().catch(console.error);
      }
    });

    metaRow.appendChild(replayBtn);

    // Auto-play TTS if this is a fresh voice response
    if (data.audio_b64) {
      showHUD('speaking');
      const mime = data.audio_mime || 'audio/mpeg';
      const src = `data:${mime};base64,${data.audio_b64}`;
      const audio = new Audio(src);
      audio.onended = () => hideHUD();
      audio.onerror = () => hideHUD();
      audio.play().catch(() => hideHUD());
    } else if (data.audio_url) {
      showHUD('speaking');
      const audio = new Audio(data.audio_url);
      audio.onended = () => hideHUD();
      audio.onerror = () => hideHUD();
      audio.play().catch(() => hideHUD());
    }
  }

  scrollFeedToBottom();
  return row;
}

// =============================================================================
// Context Reset
// =============================================================================

btnResetContext.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    if (!res.ok) throw new Error('Reset failed');

    // Clear chat
    conversationFeed.innerHTML = '';
    conversationFeed.appendChild(feedEmptyState);
    feedEmptyState.classList.remove('hidden');
    messageCount = 0;
    historyCounter.textContent = '0 messages';
    hideHUD();

    showToast('🔄 Conversation context cleared', 'success');
  } catch (err) {
    showToast(`❌ Reset failed: ${err.message}`, 'error');
  }
});

// =============================================================================
// Toast Notifications
// =============================================================================

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'default'|'success'|'error'} type
 * @param {number} duration ms
 */
function showToast(message, type = 'default', duration = 3000) {
  const toast = document.createElement('div');
  toast.className = `toast${type === 'success' ? ' toast-success' : type === 'error' ? ' toast-error' : ''}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 220);
  }, duration);
}

// =============================================================================
// Utility Helpers
// =============================================================================

function scrollFeedToBottom() {
  requestAnimationFrame(() => {
    conversationFeed.scrollTop = conversationFeed.scrollHeight;
  });
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// =============================================================================
// Video Feed Error Recovery
// =============================================================================

videoFeed.addEventListener('error', () => {
  // Retry connection after 3 seconds
  setTimeout(refreshVideoStream, 3000);
});

// =============================================================================
// Initialization
// =============================================================================

(function init() {
  // Start status polling
  startStatusPolling();

  // Ensure the initial stream is set
  refreshVideoStream();

  console.log(
    '%c Kaya Safety Copilot %c v0.3 Ready ',
    'background:#4f46e5;color:#fff;padding:4px 8px;border-radius:4px 0 0 4px;font-weight:700',
    'background:#10b981;color:#fff;padding:4px 8px;border-radius:0 4px 4px 0;font-weight:600'
  );
})();
