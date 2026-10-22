// AI Paper Assistant — entry point: wires the UI to the RAG engine, the agent, the model and voice.
// Everything runs in the browser: no server, no upload.

import { MODELS, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT } from "./js/config.js";
import { extractPdf } from "./js/pdf.js";
import { store, initEmbedder, isEmbedderReady, addPaper, removePaper } from "./js/rag.js";
import { loadModel, isModelReady, settings } from "./js/llm.js";
import { answerQuestion, literatureReview, isLiteratureReviewRequest } from "./js/agent.js";
import { stripCitations } from "./js/grounding.js";
import { listen, stopListening, isListening, speak, stopSpeaking } from "./js/voice.js";
import * as ui from "./js/ui.js";

const { els } = ui;
const history = []; // { role, content } without citation markers
let busy = false;

// --- Chat ---

function selectedPaperIds() {
  const value = els.paperScope.value;
  return !value || value === "all" ? store.papers.map(p => p.id) : [Number(value)];
}

function showCitation(citation) {
  if (citation) ui.showCitation(citation);
}

// --- Reading answers aloud: one answer at a time, the button doubles as a stop button ---

let speakingButton = null;

function stopReading() {
  stopSpeaking();
  if (speakingButton) speakingButton.textContent = "🔊 Read aloud";
  speakingButton = null;
}

function readAloud(message, button, onDone) {
  const wasReadingThis = speakingButton === button;
  stopReading();
  if (wasReadingThis) return;
  speakingButton = button;
  button.textContent = "⏹ Stop reading";
  speak(stripCitations(message.text), {
    language: message.language,
    onEnd: () => {
      button.textContent = "🔊 Read aloud";
      speakingButton = null;
      onDone?.();
    },
  });
}

async function sendMessage(text = els.chatInput.value.trim()) {
  if (!text || busy) return;
  if (!isModelReady()) {
    ui.notice("The language model is still loading — please wait a moment.", "error");
    return;
  }
  busy = true;
  stopReading();
  els.chatInput.value = "";
  ui.addUserMessage(text);

  const bubble = ui.createAssistantMessage({ onCitation: showCitation, onSpeak: readAloud });
  const options = {
    history: [...history],
    paperIds: selectedPaperIds(),
    onStatus: status => bubble.setStatus(status),
    onToken: partial => bubble.setStreaming(partial),
  };

  try {
    const message = store.papers.length > 0 && isLiteratureReviewRequest(text)
      ? await literatureReview(text, options)
      : await answerQuestion(text, options);

    const speakButton = bubble.finish(message);
    ui.updateCitationsPanel([...message.citations, ...(message.notes || []).flatMap(n => n.citations)], showCitation);
    history.push({ role: "user", content: text }, { role: "assistant", content: stripCitations(message.text) });

    if (els.handsFree.checked) readAloud(message, speakButton, () => els.handsFree.checked && startVoice());
  } catch (err) {
    console.error("Chat error:", err);
    bubble.fail(`⚠️ Generation failed: ${err.message}. Try again, or switch to the 1B model in Settings if memory is low.`);
  } finally {
    busy = false;
  }
}

function runLiteratureReview() {
  if (store.papers.length === 0) {
    ui.notice("Upload at least one PDF first.", "error");
    return;
  }
  sendMessage("Generate a concise literature review of the selected papers.");
}

// --- Papers ---

async function handleFiles(files) {
  const pdfs = [...files].filter(f => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  if (pdfs.length === 0) return;
  if (!isEmbedderReady()) {
    ui.notice("The embedding model is still loading — try again in a few seconds.", "error");
    return;
  }

  for (const file of pdfs) {
    if (store.papers.some(p => p.name === file.name)) {
      ui.notice(`${file.name} is already uploaded.`, "info");
      continue;
    }
    const status = ui.notice(`Reading ${file.name}…`);
    try {
      const extracted = await extractPdf(file);
      const paper = await addPaper(file, extracted, (done, total) => {
        status.textContent = `Indexing ${file.name}: passage ${done}/${total}…`;
      });
      status.remove();
      const authors = paper.authors.length ? ` — ${paper.authors.slice(0, 2).join(", ")}${paper.authors.length > 2 ? " et al." : ""}` : "";
      ui.notice(`✅ ${paper.title}${authors} · ${paper.pageCount} pages, ${paper.chunkCount} passages indexed`, "success");
    } catch (err) {
      console.error("PDF processing failed:", err);
      status.remove();
      ui.notice(`❌ Could not read ${file.name}: ${err.message}`, "error");
    }
  }
  refreshPapers();
}

function refreshPapers() {
  ui.updateMemoryBank();
  ui.renderPaperScope();
  ui.renderPapers(paper => {
    if (!confirm(`Remove "${paper.title}"?`)) return;
    removePaper(paper.id);
    refreshPapers();
  });
}

// --- Voice (hands-free: VAD ends the recording, the question is sent, the answer is read aloud) ---

const VOICE_STATES = {
  loading: ["Loading Whisper…", "pending"],
  listening: ["🎤 Listening…", "active"],
  processing: ["⏳ Processing…", "pending"],
};

async function startVoice() {
  if (isListening()) {
    stopListening();
    return;
  }
  stopReading();
  try {
    const text = await listen(state => ui.setStatus(els.voiceStatus, ...VOICE_STATES[state]));
    ui.setStatus(els.voiceStatus, "", "ready");
    if (!text) return;
    if (els.handsFree.checked) sendMessage(text);
    else els.chatInput.value = text;
  } catch (err) {
    console.error("Voice error:", err);
    ui.setStatus(els.voiceStatus, "Microphone unavailable", "error");
  }
}

// --- Model ---

const modelName = modelId => MODELS[modelId].split(" (")[0];

async function switchModel(modelId) {
  els.modelSelect.value = modelId;
  ui.setStatus(els.modelStatus, `${modelName(modelId)} · loading…`, "pending");
  try {
    await loadModel(modelId, percent => ui.setStatus(els.modelStatus, `${modelName(modelId)} · ${percent}%`, "pending"));
    ui.setStatus(els.modelStatus, `${modelName(modelId)} · ready`, "ready");
  } catch (err) {
    console.error("Model loading failed:", err);
    ui.setStatus(els.modelStatus, "Model failed to load (WebGPU needed)", "error");
  }
}

// --- Wiring ---

function wireEvents() {
  // Keep the browser from opening files dropped outside the drop zone
  ["dragover", "drop"].forEach(name => window.addEventListener(name, e => e.preventDefault()));

  els.sendBtn.addEventListener("click", () => sendMessage());
  els.chatInput.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
  });
  els.reviewBtn.addEventListener("click", runLiteratureReview);
  els.micBtn.addEventListener("click", startVoice);

  // Hands-free starts listening right away; unchecking it stops listening and reading
  els.handsFree.addEventListener("change", () => {
    if (els.handsFree.checked) {
      if (!isListening() && !busy) startVoice();
    } else {
      stopListening();
      stopReading();
    }
  });

  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("dragover", () => els.dropZone.classList.add("ring-2", "ring-indigo-400"));
  els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("ring-2", "ring-indigo-400"));
  els.dropZone.addEventListener("drop", e => {
    els.dropZone.classList.remove("ring-2", "ring-indigo-400");
    handleFiles(e.dataTransfer.files);
  });
  els.fileInput.addEventListener("change", e => {
    handleFiles(e.target.files);
    e.target.value = ""; // Allow picking the same file again
  });

  els.chatTab.addEventListener("click", () => ui.switchView("chat"));
  els.papersTab.addEventListener("click", () => ui.switchView("papers"));

  // Settings: temperature, model and system prompt apply immediately
  els.temperature.addEventListener("input", e => {
    settings.temperature = Number(e.target.value);
    els.temperatureValue.textContent = settings.temperature.toFixed(1);
  });
  els.modelSelect.addEventListener("change", e => switchModel(e.target.value));
  els.systemPrompt.value = settings.systemPrompt;
  els.applyPrompt.addEventListener("click", () => {
    settings.systemPrompt = els.systemPrompt.value.trim() || DEFAULT_SYSTEM_PROMPT;
    ui.notice("System prompt updated.", "success");
  });
  els.resetPrompt.addEventListener("click", () => {
    settings.systemPrompt = els.systemPrompt.value = DEFAULT_SYSTEM_PROMPT;
  });

  // Modals close on backdrop click, close buttons and Escape
  for (const modal of [els.pdfModal, els.citationModal]) {
    modal.addEventListener("click", e => {
      if (e.target === modal || e.target.closest("[data-close]")) ui.closeModal(modal);
    });
  }
  document.addEventListener("keydown", e => {
    if (e.key !== "Escape") return;
    [els.pdfModal, els.citationModal].forEach(ui.closeModal);
    stopReading();
  });
}

function init() {
  els.modelSelect.innerHTML = Object.entries(MODELS).map(([id, label]) => `<option value="${id}">${label}</option>`).join("");
  wireEvents();
  refreshPapers();
  ui.updateCitationsPanel([], showCitation);

  ui.setStatus(els.ragStatus, "Loading embedder…", "pending");
  initEmbedder()
    .then(() => ui.setStatus(els.ragStatus, "Ready", "ready"))
    .catch(err => {
      console.error("Embedder failed:", err);
      ui.setStatus(els.ragStatus, "Embedder error", "error");
    });

  switchModel(DEFAULT_MODEL);
}

init();
