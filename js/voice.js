// Voice: Whisper speech-to-text (Transformers.js), RMS-based voice activity detection,
// and the browser's native text-to-speech.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";
import { WHISPER_MODEL } from "./config.js";
import { detectLanguage } from "./language.js";

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

// --- Text-to-speech ---
// The voice must match the language of the text: without it the browser uses the system
// default voice (French on a French Mac) and reads English with a French accent.

// Natural-sounding voices, best first (macOS, Windows, Chrome). Novelty voices are excluded,
// including their localized names on a French system ("Bonnes nouvelles", "Bouffon"...).
const PREFERRED_VOICES = {
  en: ["Samantha", "Ava", "Allison", "Daniel", "Karen", "Moira", "Tessa", "Serena", "Alex", "Tom", "Microsoft Aria", "Microsoft Jenny", "Microsoft Guy", "Google US English", "Google UK English Female"],
  fr: ["Thomas", "Amélie", "Audrey", "Aurélie", "Marie", "Microsoft Denise", "Microsoft Henri", "Google français"],
};
const NOVELTY_VOICES = /albert|bad news|bahh|bells|boing|bubbles|cellos|good news|jester|organ|superstar|trinoi|whisper|wobble|zarvox|deranged|hysterical|nouvelles|bouffon|bulles|cloches|murmure|orgue|violoncelles|grandma|grandpa|junior|ralph|fred|kathy|rocko/i;

let voices = [];
if ("speechSynthesis" in window) {
  voices = speechSynthesis.getVoices();
  speechSynthesis.addEventListener("voiceschanged", () => { voices = speechSynthesis.getVoices(); });
}

// Prefers voices installed on the device: network voices (e.g. Chrome's "Google US English")
// send the text to a server, which breaks the app's local-only promise
function pickVoice(language) {
  if (voices.length === 0) voices = speechSynthesis.getVoices(); // Chrome loads them lazily
  const region = language === "fr" ? "fr-FR" : "en-US";
  const preferred = PREFERRED_VOICES[language];
  const rank = voice => {
    const index = preferred.findIndex(name => voice.name.startsWith(name));
    return index === -1 ? 0 : preferred.length - index;
  };
  const score = voice =>
    (voice.localService ? 100 : 0) +
    rank(voice) * 5 +
    (/premium|enhanced|natural/i.test(voice.name) ? 3 : 0) +
    (voice.lang.replace("_", "-") === region ? 1 : 0);
  return voices
    .filter(v => v.lang.toLowerCase().startsWith(language) && !NOVELTY_VOICES.test(v.name))
    .sort((a, b) => score(b) - score(a))[0];
}

// Markdown, bullets and emojis are read out loud literally otherwise
const speakable = text => text.replace(/[*#•⟦⟧]/g, " ").replace(/\p{Extended_Pictographic}/gu, "").replace(/\s+/g, " ").trim();

let speechId = 0;

export const isSpeaking = () => "speechSynthesis" in window && (speechSynthesis.speaking || speechSynthesis.pending);

export function stopSpeaking() {
  speechId++;
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

// language: that of the question the text answers. Detecting it from the answer itself fails when a
// French answer quotes English titles and terms. onEnd runs only when the whole text was read.
export function speak(text, { language, onEnd } = {}) {
  if (!("speechSynthesis" in window)) return;
  stopSpeaking();
  const id = speechId;
  const clean = speakable(text);
  language ??= detectLanguage(clean);
  const voice = pickVoice(language);

  // One utterance per sentence: Chrome cuts long utterances after ~15 seconds
  const sentences = clean.match(/[^.!?]+[.!?]*/g) || [clean];
  sentences.forEach((sentence, i) => {
    const utterance = new SpeechSynthesisUtterance(sentence.trim());
    if (voice) utterance.voice = voice;
    utterance.lang = voice?.lang || (language === "fr" ? "fr-FR" : "en-US");
    if (i === sentences.length - 1) utterance.onend = () => id === speechId && onEnd?.();
    speechSynthesis.speak(utterance);
  });
}
