import { transcribeAudio } from "./api.js";

const SILENCE_THRESHOLD = 0.015;
const SILENCE_MS = 1500;
const MIN_RECORDING_MS = 500;

/**
 * @param {{
 *   formEl: HTMLFormElement;
 *   inputEl: HTMLTextAreaElement;
 *   voiceBtn: HTMLButtonElement;
 *   autoSendEl: HTMLInputElement;
 *   getStreaming: () => boolean;
 *   setStatus: (text: string, isError?: boolean) => void;
 * }} options
 */
export function initVoiceInput({
  formEl,
  inputEl,
  voiceBtn,
  autoSendEl,
  getStreaming,
  setStatus,
}) {
  if (!voiceBtn || !navigator.mediaDevices?.getUserMedia) {
    if (voiceBtn) {
      voiceBtn.disabled = true;
      voiceBtn.title = "Entrada por voz no disponible en este navegador";
    }
    return;
  }

  /** @type {MediaRecorder | null} */
  let recorder = null;
  /** @type {MediaStream | null} */
  let stream = null;
  /** @type {AudioContext | null} */
  let audioContext = null;
  /** @type {number | null} */
  let silenceTimer = null;
  /** @type {number | null} */
  let monitorFrame = null;
  /** @type {number} */
  let recordingStartedAt = 0;
  /** @type {Blob[]} */
  let chunks = [];
  let recording = false;
  let transcribing = false;

  function updateVoiceButton() {
    const busy = getStreaming() || transcribing;
    voiceBtn.disabled = busy || (!recording && transcribing);
    voiceBtn.classList.toggle("recording", recording);
    voiceBtn.setAttribute("aria-pressed", recording ? "true" : "false");
    voiceBtn.textContent = recording ? "⏹" : "🎤";
  }

  function cleanupMonitor() {
    if (silenceTimer !== null) {
      window.clearTimeout(silenceTimer);
      silenceTimer = null;
    }
    if (monitorFrame !== null) {
      cancelAnimationFrame(monitorFrame);
      monitorFrame = null;
    }
  }

  function stopTracks() {
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      stream = null;
    }
    if (audioContext) {
      void audioContext.close();
      audioContext = null;
    }
  }

  function appendTranscription(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    const current = inputEl.value.trim();
    inputEl.value = current ? `${current} ${trimmed}` : trimmed;
    inputEl.dispatchEvent(new Event("input", { bubbles: true }));
  }

  async function finishRecording() {
    if (!recorder || recorder.state === "inactive") return;
    recorder.stop();
  }

  function startSilenceMonitor(analyser) {
    const buffer = new Float32Array(analyser.fftSize);

    const tick = () => {
      if (!recording) return;
      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i += 1) {
        sum += buffer[i] * buffer[i];
      }
      const rms = Math.sqrt(sum / buffer.length);

      if (rms < SILENCE_THRESHOLD) {
        if (silenceTimer === null) {
          silenceTimer = window.setTimeout(() => {
            if (recording && Date.now() - recordingStartedAt >= MIN_RECORDING_MS) {
              void finishRecording();
            }
          }, SILENCE_MS);
        }
      } else if (silenceTimer !== null) {
        window.clearTimeout(silenceTimer);
        silenceTimer = null;
      }

      monitorFrame = requestAnimationFrame(tick);
    };

    monitorFrame = requestAnimationFrame(tick);
  }

  async function handleRecordingStop() {
    cleanupMonitor();
    stopTracks();
    recording = false;
    updateVoiceButton();

    const blob = new Blob(chunks, { type: recorder?.mimeType || "audio/webm" });
    chunks = [];
    recorder = null;

    if (!blob.size) {
      setStatus("");
      return;
    }

    transcribing = true;
    updateVoiceButton();
    setStatus("Transcribiendo…");

    try {
      const file = new File([blob], "voice-input.webm", {
        type: blob.type || "audio/webm",
      });
      const result = await transcribeAudio(file);
      const text = result?.text || "";
      appendTranscription(text);
      setStatus("Transcripción lista");
      if (autoSendEl?.checked && text.trim()) {
        formEl.requestSubmit();
      } else {
        inputEl.focus();
      }
    } catch (err) {
      setStatus(err.message || "No se pudo transcribir el audio", true);
    } finally {
      transcribing = false;
      updateVoiceButton();
      if (!getStreaming()) {
        window.setTimeout(() => setStatus(""), 2500);
      }
    }
  }

  async function startRecording() {
    if (recording || transcribing || getStreaming()) return;

    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus("Permiso de micrófono denegado o no disponible", true);
      return;
    }

    chunks = [];
    recordingStartedAt = Date.now();
    recording = true;
    updateVoiceButton();
    setStatus("Escuchando…");

    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    startSilenceMonitor(analyser);

    const preferredMime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    recorder = new MediaRecorder(stream, { mimeType: preferredMime });
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    });
    recorder.addEventListener("stop", () => {
      void handleRecordingStop();
    });
    recorder.start(250);
  }

  voiceBtn.addEventListener("click", () => {
    if (recording) {
      void finishRecording();
      return;
    }
    void startRecording();
  });

  updateVoiceButton();

  return {
    refresh: updateVoiceButton,
  };
}
