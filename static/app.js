/**
 * Kaya — Job Site Safety Copilot & Multimodal Assistant Logic
 * Connects Live YOLO/Depth/Pose Stream, Push-to-Talk Voice Recording,
 * Temporal Sequence VLM Reasoning, and Audio Speech Playback.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ==========================================
  // DOM Elements
  // ==========================================
  const copilotVideoFeed = document.getElementById('copilotVideoFeed');
  const btnRefreshFeed = document.getElementById('btnRefreshFeed');

  // Mode buttons & Badges
  const btnModeTemporal = document.getElementById('btnModeTemporal');
  const btnModeSingle = document.getElementById('btnModeSingle');
  const pillVisionLabel = document.getElementById('pillVisionLabel');
  const pillSTTLabel = document.getElementById('pillSTTLabel');
  const pillTTSLabel = document.getElementById('pillTTSLabel');
  
  // Live Metrics
  const trackedCountTag = document.getElementById('trackedCountTag');
  const hazardsCountTag = document.getElementById('hazardsCountTag');
  const bufferStatusTag = document.getElementById('bufferStatusTag');
  const copilotFpsTag = document.getElementById('copilotFpsTag');
  const objectsSummaryText = document.getElementById('objectsSummaryText');
  const historyCounter = document.getElementById('historyCounter');

  // HUD & Indicators
  const statusListening = document.getElementById('statusListening');
  const statusThinking = document.getElementById('statusThinking');
  const statusSpeaking = document.getElementById('statusSpeaking');

  // Chat & Controls
  const conversationFeed = document.getElementById('conversationFeed');
  const feedEmptyState = document.getElementById('feedEmptyState');
  const btnPushToTalk = document.getElementById('btnPushToTalk');
  const pttMainText = document.getElementById('pttMainText');
  const textQueryForm = document.getElementById('textQueryForm');
  const inputQueryText = document.getElementById('inputQueryText');
  const btnResetContext = document.getElementById('btnResetContext');
  const toastContainer = document.getElementById('toastContainer');

  // ==========================================
  // Application State
  // ==========================================
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let isSpacePressed = false;
  let currentAudio = null;
  let totalTurnsCount = 0;
  let frameMode = 'TEMPORAL_FRAMES'; // 'TEMPORAL_FRAMES' | 'SINGLE_FRAME'

  let currentViewMode = 'all'; // 'all' | 'raw' | 'pose' | 'depth' | 'ppe' | 'objects'

  // ==========================================
  // Live Stream & Mode Controls
  // ==========================================
  const viewTabBtns = document.querySelectorAll('.view-tab-btn');
  viewTabBtns.forEach((tabBtn) => {
    tabBtn.addEventListener('click', () => {
      const mode = tabBtn.getAttribute('data-mode') || 'all';
      currentViewMode = mode;

      viewTabBtns.forEach((b) => b.classList.remove('active'));
      tabBtn.classList.add('active');

      // Update stream source with cachebuster
      copilotVideoFeed.src = `/api/video_feed?mode=${encodeURIComponent(mode)}&t=${Date.now()}`;
      showToast(`Switched to ${tabBtn.textContent.trim()}`, 'success');
    });
  });

  function setFrameMode(mode) {
    frameMode = mode;
    if (mode === 'TEMPORAL_FRAMES') {
      btnModeTemporal.classList.add('active');
      btnModeSingle.classList.remove('active');
    } else {
      btnModeSingle.classList.add('active');
      btnModeTemporal.classList.remove('active');
    }
  }

  btnModeTemporal.addEventListener('click', () => setFrameMode('TEMPORAL_FRAMES'));
  btnModeSingle.addEventListener('click', () => setFrameMode('SINGLE_FRAME'));

  btnRefreshFeed.addEventListener('click', () => {
    copilotVideoFeed.src = `/api/video_feed?mode=${encodeURIComponent(currentViewMode)}&t=${Date.now()}`;
    showToast('Reconnected to Safety Copilot video stream', 'success');
  });

  // ==========================================
  // Push-to-Talk Microphone Recording
  // ==========================================
  async function startRecording() {
    if (isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      audioChunks = [];
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      }

      mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.start(100);
      isRecording = true;

      btnPushToTalk.classList.add('recording');
      pttMainText.textContent = 'Release to Send';
      statusListening.classList.add('active');

      if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
      }
    } catch (err) {
      console.error('Microphone access error:', err);
      showToast('Microphone access denied', 'error');
    }
  }

  async function stopRecordingAndSubmit() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;

    btnPushToTalk.classList.remove('recording');
    pttMainText.textContent = 'Hold to Speak';
    statusListening.classList.remove('active');

    return new Promise((resolve) => {
      mediaRecorder.onstop = async () => {
        try {
          const mime = mediaRecorder.mimeType || 'audio/webm';
          const audioBlob = new Blob(audioChunks, { type: mime });
          mediaRecorder.stream.getTracks().forEach(t => t.stop());

          if (audioBlob.size < 1000) {
            showToast('Query too short. Please hold to speak.', 'error');
            resolve();
            return;
          }

          await submitVoiceQuery(audioBlob);
        } catch (e) {
          console.error('Error stopping recording:', e);
        }
        resolve();
      };
      mediaRecorder.stop();
    });
  }

  // Pointer & Touch Listeners
  btnPushToTalk.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    startRecording();
  });

  window.addEventListener('pointerup', () => {
    if (isRecording && !isSpacePressed) stopRecordingAndSubmit();
  });

  // Spacebar Keyboard PTT Shortcut
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !isSpacePressed && document.activeElement !== inputQueryText) {
      e.preventDefault();
      isSpacePressed = true;
      startRecording();
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && isSpacePressed) {
      e.preventDefault();
      isSpacePressed = false;
      if (!isRecording) return;
      stopRecordingAndSubmit();
    }
  });

  // ==========================================
  // Pipeline Query Submissions
  // ==========================================
  async function submitVoiceQuery(audioBlob) {
    statusThinking.classList.add('active');
    const formData = new FormData();
    formData.append('audio', audioBlob, 'speech.webm');
    formData.append('frame_mode', frameMode);

    await handleTurnSubmission('/api/ask', formData);
  }

  async function submitTextQuery(questionText) {
    statusThinking.classList.add('active');
    const formData = new FormData();
    formData.append('question', questionText.trim());
    formData.append('frame_mode', frameMode);

    await handleTurnSubmission('/api/ask-text', formData);
  }

  async function handleTurnSubmission(url, formData) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      statusThinking.classList.remove('active');

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error (${response.status})`);
      }

      const result = await response.json();
      totalTurnsCount += 1;
      historyCounter.textContent = `${totalTurnsCount} turn${totalTurnsCount === 1 ? '' : 's'}`;

      if (feedEmptyState) feedEmptyState.classList.add('hidden');

      renderTurn(
        result.transcript,
        result.response,
        result.timings,
        result.providers,
        result.frame_mode || frameMode,
        result.frame_count || 1
      );

      playResponseAudio(result.audio_base64, result.response);

    } catch (err) {
      console.error('Pipeline processing failed:', err);
      statusThinking.classList.remove('active');
      showToast(`Processing error: ${err.message}`, 'error');
    }
  }

  textQueryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = inputQueryText.value.trim();
    if (!query) return;

    inputQueryText.value = '';
    await submitTextQuery(query);
  });

  // ==========================================
  // Audio Playback with Web Speech API Failover
  // ==========================================
  function speakWithWebSpeech(text) {
    if (!('speechSynthesis' in window) || !text) return;
    try {
      window.speechSynthesis.cancel();
      const cleanText = text.replace(/[*#`_]/g, '');
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;

      const voices = window.speechSynthesis.getVoices();
      const naturalVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Samantha') || v.name.includes('Siri') || v.name.includes('Natural') || v.name.includes('Google')));
      if (naturalVoice) utterance.voice = naturalVoice;

      statusSpeaking.classList.add('active');
      utterance.onend = () => statusSpeaking.classList.remove('active');
      utterance.onerror = () => statusSpeaking.classList.remove('active');

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Web Speech API error:', e);
      statusSpeaking.classList.remove('active');
    }
  }

  function playResponseAudio(base64Wav, textFallback = "") {
    if (currentAudio) {
      try {
        currentAudio.pause();
        currentAudio = null;
      } catch (e) {}
    }

    if (!base64Wav || base64Wav.length === 0) {
      if (textFallback) speakWithWebSpeech(textFallback);
      return;
    }

    try {
      const audioUrl = `data:audio/wav;base64,${base64Wav}`;
      currentAudio = new Audio(audioUrl);
      statusSpeaking.classList.add('active');

      currentAudio.onended = () => {
        statusSpeaking.classList.remove('active');
        currentAudio = null;
      };

      currentAudio.onerror = (e) => {
        console.warn('Audio playback error, falling back to Web Speech:', e);
        statusSpeaking.classList.remove('active');
        currentAudio = null;
        if (textFallback) speakWithWebSpeech(textFallback);
      };

      const playPromise = currentAudio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.warn('Autoplay blocked or playback error, falling back to Web Speech:', err);
          statusSpeaking.classList.remove('active');
          if (textFallback) speakWithWebSpeech(textFallback);
        });
      }
    } catch (err) {
      console.warn('Failed to construct audio element, falling back to Web Speech:', err);
      statusSpeaking.classList.remove('active');
      if (textFallback) speakWithWebSpeech(textFallback);
    }
  }

  // ==========================================
  // Safe UI Rendering (DOM Construction)
  // ==========================================
  function renderTurn(userText, kayaText, timings, providers, mode, frameCount) {
    const turnWrap = document.createElement('div');
    turnWrap.className = 'chat-turn';

    // 1. User Message Bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'bubble bubble-user';
    const userHeader = document.createElement('div');
    userHeader.className = 'bubble-header';
    userHeader.textContent = 'You';
    const userContent = document.createElement('div');
    userContent.className = 'bubble-content';
    userContent.textContent = userText || '(Spoken Query)';
    userBubble.appendChild(userHeader);
    userBubble.appendChild(userContent);

    // 2. Kaya Message Bubble
    const kayaBubble = document.createElement('div');
    kayaBubble.className = 'bubble bubble-kaya';
    const kayaHeader = document.createElement('div');
    kayaHeader.className = 'bubble-header';

    const kayaName = document.createElement('span');
    kayaName.textContent = 'Kaya Copilot';

    const modePill = document.createElement('span');
    modePill.className = 'bubble-mode-pill';
    modePill.textContent = mode === 'TEMPORAL_FRAMES' ? `🎞️ ${frameCount} frames` : `🖼️ 1 frame`;

    kayaHeader.appendChild(kayaName);
    kayaHeader.appendChild(modePill);

    const kayaContent = document.createElement('div');
    kayaContent.className = 'bubble-content';
    kayaContent.textContent = kayaText;

    // Latency Pills Footer
    const metaFooter = document.createElement('div');
    metaFooter.className = 'bubble-meta';

    if (timings && timings.formatted) {
      const fmt = timings.formatted;
      if (fmt.stt && fmt.stt !== '0 ms') {
        const sttPill = document.createElement('span');
        sttPill.className = 'latency-pill';
        sttPill.textContent = `STT: ${fmt.stt}`;
        metaFooter.appendChild(sttPill);
      }

      if (fmt.vision) {
        const vlmPill = document.createElement('span');
        vlmPill.className = 'latency-pill';
        vlmPill.textContent = `VLM: ${fmt.vision}`;
        metaFooter.appendChild(vlmPill);
      }

      if (fmt.tts && fmt.tts !== '0 ms') {
        const ttsPill = document.createElement('span');
        ttsPill.className = 'latency-pill';
        ttsPill.textContent = `TTS: ${fmt.tts}`;
        metaFooter.appendChild(ttsPill);
      }

      if (fmt.total) {
        const totalPill = document.createElement('span');
        totalPill.className = 'latency-pill latency-total';
        totalPill.textContent = `Total: ${fmt.total}`;
        metaFooter.appendChild(totalPill);
      }
    }

    kayaBubble.appendChild(kayaHeader);
    kayaBubble.appendChild(kayaContent);
    if (metaFooter.childNodes.length > 0) {
      kayaBubble.appendChild(metaFooter);
    }

    turnWrap.appendChild(userBubble);
    turnWrap.appendChild(kayaBubble);
    conversationFeed.appendChild(turnWrap);

    // Auto-scroll
    conversationFeed.scrollTo({
      top: conversationFeed.scrollHeight,
      behavior: 'smooth'
    });
  }

  // ==========================================
  // Context Reset & Status Polling
  // ==========================================
  btnResetContext.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', { method: 'POST' });
      conversationFeed.innerHTML = '';
      if (feedEmptyState) {
        conversationFeed.appendChild(feedEmptyState);
        feedEmptyState.classList.remove('hidden');
      }
      totalTurnsCount = 0;
      historyCounter.textContent = '0 messages';
      showToast('Conversation context cleared', 'success');
    } catch (e) {
      showToast('Failed to reset history', 'error');
    }
  });

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();

      if (data.providers) {
        pillVisionLabel.textContent = `Vision: ${data.providers.vision}`;
        pillSTTLabel.textContent = `STT: ${data.providers.stt}`;
        pillTTSLabel.textContent = `TTS: ${data.providers.tts}`;
      }

      if (data.copilot) {
        const c = data.copilot;
        trackedCountTag.textContent = `${c.tracked_count} Objects`;
        hazardsCountTag.textContent = `${c.hazards_count} Hazards`;
        copilotFpsTag.textContent = `${c.fps || 30} FPS`;
        if (c.objects_summary && c.objects_summary.length > 0) {
          objectsSummaryText.textContent = c.objects_summary.join(' • ');
        }
        bufferStatusTag.textContent = `● Buffer: ${c.buffer_frames_count || 6}/8 (1 FPS)`;
      }

      if (data.config && data.config.frame_mode) {
        setFrameMode(data.config.frame_mode);
      }
    } catch (e) {
      console.debug('Status fetch error:', e);
    }
  }

  // Toast System
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' : ''}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 3200);
  }

  // Periodic Status Polling
  fetchStatus();
  setInterval(fetchStatus, 1500);
});
