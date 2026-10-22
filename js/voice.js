// Voice: Whisper speech-to-text (Transformers.js), RMS-based voice activity detection,
// and the browser's native text-to-speech.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";
import { WHISPER_MODEL } from "./config.js";

const SPEECH_RMS = 0.02; // Louder than this counts as speech
const SILENCE_MS = 1500; // Silence after speech that ends the recording
const NO_SPEECH_MS = 8000; // Give up if nobody speaks
const MAX_RECORDING_MS = 20000;

let transcriber = null;
let recording = null;

export const isListening = () => recording !== null;

async function getTranscriber() {
  transcriber ??= await pipeline("automatic-speech-recognition", WHISPER_MODEL);
  return transcriber;
}

// Whisper expects 16 kHz mono samples, not the compressed bytes MediaRecorder produces
async function decodeTo16kHz(blob) {
  const context = new AudioContext({ sampleRate: 16000 });
  try {
    const audio = await context.decodeAudioData(await blob.arrayBuffer());
    return audio.getChannelData(0);
  } finally {
    context.close();
  }
}

// Records until the speaker stops (VAD), then returns the transcription ("" if nothing was said).
// onState receives "loading", "listening" and "processing".
export async function listen(onState) {
  if (recording) return "";
  onState("loading");
  const stt = await getTranscriber();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  const audioContext = new AudioContext();
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  audioContext.createMediaStreamSource(stream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);

  const recorder = new MediaRecorder(stream);
  const parts = [];
  recorder.ondataavailable = e => parts.push(e.data);

  let heardSpeech = false;
  const blob = await new Promise(resolve => {
    recording = { stop: () => recorder.state === "recording" && recorder.stop() };
    recorder.onstop = () => resolve(new Blob(parts, { type: recorder.mimeType }));
    recorder.start();
    onState("listening");

    // VAD: root-mean-square amplitude every 100 ms
    const startedAt = Date.now();
    let lastSpeechAt = Date.now();
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0) / samples.length);
      const now = Date.now();
      if (rms > SPEECH_RMS) {
        heardSpeech = true;
        lastSpeechAt = now;
      }
      const done = heardSpeech ? now - lastSpeechAt > SILENCE_MS : now - startedAt > NO_SPEECH_MS;
      if (done || now - startedAt > MAX_RECORDING_MS || recorder.state !== "recording") {
        clearInterval(timer);
        recording.stop();
      }
    }, 100);
  });

  recording = null;
  stream.getTracks().forEach(track => track.stop());
  audioContext.close();
  if (!heardSpeech) return "";

  onState("processing");
  const result = await stt(await decodeTo16kHz(blob));
  return (result.text || "").trim();
}

export function stopListening() {
  recording?.stop();
}

export function speak(text, onEnd) {
  if (!("speechSynthesis" in window)) return onEnd?.();
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.onend = () => onEnd?.();
  utterance.onerror = () => onEnd?.();
  speechSynthesis.speak(utterance);
}
