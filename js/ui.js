// DOM rendering: chat messages with citation chips, citations panel, memory bank,
// paper library, paper scope selector, PDF viewer and status indicators.

import { store, paperById } from "./rag.js";
import { CITATION_MARKER } from "./grounding.js";

const $ = id => document.getElementById(id);
export const els = {
  chatHistory: $("chat-history"),
  welcome: $("welcome"),
  chatInput: $("chat-input"),
  sendBtn: $("send-btn"),
  micBtn: $("mic-btn"),
  reviewBtn: $("literature-review-btn"),
  paperScope: $("paper-scope"),
  handsFree: $("hands-free-toggle"),
  voiceStatus: $("voice-status"),
  dropZone: $("drop-zone"),
  fileInput: $("file-input"),
  chunkCount: $("chunk-count"),
  paperCount: $("document-count"),
  ragStatus: $("rag-status"),
  modelStatus: $("model-status"),
  modelSelect: $("model-select"),
  temperature: $("temperature-slider"),
  temperatureValue: $("temperature-value"),
  systemPrompt: $("system-prompt"),
  applyPrompt: $("apply-prompt-btn"),
  resetPrompt: $("reset-prompt-btn"),
  citationsList: $("citations-list"),
  chatTab: $("chat-tab"),
  papersTab: $("papers-tab"),
  chatView: $("chat-view"),
  papersView: $("papers-view"),
  papersGrid: $("papers-grid"),
  papersEmpty: $("papers-empty-state"),
  pdfModal: $("pdf-viewer-modal"),
  pdfTitle: $("pdf-viewer-title"),
  pdfFrame: $("pdf-viewer-frame"),
  citationModal: $("citation-modal"),
  citationBody: $("citation-modal-body"),
};

export function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const shortTitle = title => {
  const words = title.split(/\s+/);
  return words.length > 5 ? `${words.slice(0, 5).join(" ")}…` : title;
};

// When every citation comes from the same paper, the chip shows only the page; the title stays in the tooltip
function citationChip(citation, index, pageOnly) {
  if (!citation) return "";
  const label = pageOnly ? `p. ${citation.page}` : `${shortTitle(citation.title)}, p. ${citation.page}`;
  return `<button type="button" data-cite="${index}" class="cite-chip inline-flex items-center align-baseline bg-indigo-50 border border-indigo-200 text-indigo-700 hover:bg-indigo-100 rounded-md px-1.5 py-0.5 mx-0.5 text-[11px] font-medium transition-colors" title="${escapeHtml(`${citation.title} — p. ${citation.page}\n“${citation.quote}”`)}">${escapeHtml(label)}</button>`;
}

// Model output is untrusted: escape first, then add our own light formatting
export function renderRichText(text, citations = []) {
  const pageOnly = new Set(citations.map(c => c.paperId)).size <= 1;
  return escapeHtml(text)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/^#{1,4}\s*(.+)$/gm, "<strong>$1</strong>")
    .replace(/^([A-Z][A-Z &]{3,}[A-Z])(\s*[—:-])?/gm, "<strong>$1</strong>$2")
    .replace(/^\s*[-*•]\s+/gm, "• ")
    .replace(CITATION_MARKER, (_, i) => citationChip(citations[Number(i)], i, pageOnly))
    .replace(/\n/g, "<br>");
}

export function hideWelcome() {
  els.welcome?.remove();
  els.welcome = null;
}

export function scrollToBottom() {
  els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

export function addUserMessage(text) {
  hideWelcome();
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-start gap-3 justify-end animate-fade-in-up";
  wrapper.innerHTML = `
    <div class="glass-bubble-user p-3 md:p-4 rounded-2xl rounded-tr-none text-white max-w-[80%] shadow-md">
      <p class="text-sm font-medium whitespace-pre-wrap"></p>
    </div>
    <div class="w-8 h-8 rounded-full bg-gray-400 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-md">ME</div>`;
  wrapper.querySelector("p").textContent = text;
  els.chatHistory.appendChild(wrapper);
  scrollToBottom();
}

export function notice(text, kind = "info") {
  hideWelcome();
  const colors = { info: "bg-blue-50 text-blue-800", success: "bg-green-50 text-green-800", error: "bg-red-50 text-red-800" };
  const div = document.createElement("div");
  div.className = `px-3 py-2 rounded-lg text-xs md:text-sm ${colors[kind]}`;
  div.textContent = text;
  els.chatHistory.appendChild(div);
  scrollToBottom();
  return div;
}

function reportLine(report) {
  if (!report) return "";
  const parts = [];
  const details = [];
  if (report.supported > 0) {
    parts.push(`✓ ${report.supported} statement${report.supported > 1 ? "s" : ""} sourced`);
    details.push("Each sourced statement is linked to the passage of the paper that supports it. Citations are computed by semantic matching, not written by the model.");
  } else {
    parts.push("⚠ no statement matches a passage");
    details.push("No sentence of this answer is close to a passage of your papers: treat it with caution.");
  }
  if (report.unsupported > 0) {
    parts.push(`ℹ ${report.unsupported} unsourced`);
    details.push(`${report.unsupported} statement(s) matched no passage: general wording, or not supported by the papers.`);
  }
  if (report.unsupportedNumbers.length > 0) {
    parts.push(`⚠ ${report.unsupportedNumbers.length} figure${report.unsupportedNumbers.length > 1 ? "s" : ""} not found`);
    details.push(`Figures not found in the passages (verify before using): ${report.unsupportedNumbers.join(", ")}.`);
  }
  return `<details class="mt-3 pt-2 border-t border-gray-200 text-xs text-gray-500">
    <summary class="cursor-pointer select-none">${escapeHtml(parts.join(" · "))}</summary>
    <ul class="mt-1 list-disc pl-4 space-y-0.5">${details.map(d => `<li>${escapeHtml(d)}</li>`).join("")}</ul>
  </details>`;
}

function notesBlock(notes, open) {
  if (!notes?.length) return "";
  return `<details class="mt-3 text-sm" ${open ? "open" : ""}>
    <summary class="cursor-pointer select-none text-xs font-medium text-indigo-600">Per-paper notes (${notes.length})</summary>
    ${notes.map((note, n) => `
      <div class="mt-2 p-2 rounded-lg bg-white/60">
        <div class="font-semibold text-slate-700 text-xs mb-1">📄 ${escapeHtml(note.title)}</div>
        <div class="text-slate-700 leading-relaxed" data-note="${n}">${renderRichText(note.text, note.citations)}</div>
      </div>`).join("")}
  </details>`;
}

// An assistant bubble that shows progress, then the streamed text, then the final grounded answer
export function createAssistantMessage({ onCitation, onSpeak }) {
  hideWelcome();
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-start gap-3 animate-fade-in-up";
  wrapper.innerHTML = `
    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-md">AI</div>
    <div class="glass-bubble-ai p-3 md:p-4 rounded-2xl rounded-tl-none max-w-[85%] shadow-sm min-w-0">
      <div class="status text-xs text-indigo-500 flex items-center gap-2"><span class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span><span class="status-text">Thinking…</span></div>
      <div class="body text-sm text-slate-700 leading-relaxed whitespace-pre-wrap hidden"></div>
    </div>`;
  const bubble = wrapper.lastElementChild;
  const status = bubble.querySelector(".status");
  const body = bubble.querySelector(".body");
  els.chatHistory.appendChild(wrapper);
  scrollToBottom();

  return {
    setStatus(text) {
      bubble.querySelector(".status-text").textContent = text;
    },
    setStreaming(text) {
      body.classList.remove("hidden");
      body.textContent = text;
      scrollToBottom();
    },
    finish(message) {
      status.remove();
      body.classList.remove("hidden", "whitespace-pre-wrap");
      body.innerHTML = renderRichText(message.text, message.citations) + notesBlock(message.notes, message.citations.length === 0) + reportLine(message.report);
      const speakBtn = document.createElement("button");
      speakBtn.type = "button";
      speakBtn.className = "mt-2 text-gray-400 hover:text-indigo-500 text-xs";
      speakBtn.title = "Read aloud";
      speakBtn.textContent = "🔊 Read aloud";
      speakBtn.addEventListener("click", () => onSpeak(message));
      body.appendChild(speakBtn);

      // Chips in the answer refer to message.citations, chips in a note to that note's citations
      body.addEventListener("click", e => {
        const chip = e.target.closest(".cite-chip");
        if (!chip) return;
        const note = chip.closest("[data-note]");
        const citations = note ? message.notes[Number(note.dataset.note)].citations : message.citations;
        onCitation(citations[Number(chip.dataset.cite)]);
      });
      scrollToBottom();
    },
    fail(text) {
      status.remove();
      body.classList.remove("hidden");
      body.classList.add("text-red-700");
      body.textContent = text;
    },
  };
}

export function updateCitationsPanel(citations, onCitation) {
  els.citationsList.innerHTML = "";
  if (citations.length === 0) {
    els.citationsList.innerHTML = `<div class="text-center text-gray-400 text-xs py-6">Citations of the latest answer appear here</div>`;
    return;
  }
  citations.forEach(citation => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "w-full text-left bg-white/70 border border-indigo-100 rounded-lg p-2 text-xs hover:bg-indigo-50 transition-colors";
    item.innerHTML = `
      <div class="flex justify-between gap-2 font-semibold text-slate-700"><span class="truncate">📄 ${escapeHtml(citation.title)}</span><span class="text-indigo-600 flex-shrink-0">p. ${citation.page}</span></div>
      <div class="mt-1 italic text-slate-500 line-clamp-3">“${escapeHtml(citation.quote)}”</div>`;
    item.addEventListener("click", () => onCitation(citation));
    els.citationsList.appendChild(item);
  });
}

export function showCitation(citation) {
  els.citationBody.innerHTML = `
    <div class="text-xs text-gray-500 mb-1">${escapeHtml(citation.title)} — page ${citation.page}</div>
    <blockquote class="border-l-4 border-indigo-400 bg-indigo-50/60 px-3 py-2 rounded-r italic text-slate-700 text-sm">“${escapeHtml(citation.quote)}”</blockquote>
    <button type="button" id="open-cited-page" class="mt-4 px-4 py-2 text-sm bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-lg hover:shadow-lg">Open the PDF at page ${citation.page}</button>`;
  els.citationBody.querySelector("#open-cited-page").addEventListener("click", () => {
    closeModal(els.citationModal);
    openPdf(paperById(citation.paperId), citation.page);
  });
  openModal(els.citationModal);
}

export function openModal(modal) {
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

export function closeModal(modal) {
  modal.classList.add("hidden");
  modal.classList.remove("flex");
  if (modal === els.pdfModal) els.pdfFrame.src = "";
}

export function openPdf(paper, page = 1) {
  if (!paper) return;
  els.pdfTitle.textContent = `${paper.title} — page ${page}`;
  els.pdfFrame.src = `${paper.url}#page=${page}`;
  openModal(els.pdfModal);
}

export function updateMemoryBank() {
  els.chunkCount.textContent = store.chunks.length;
  els.paperCount.textContent = store.papers.length;
}

export function renderPaperScope() {
  const previous = els.paperScope.value;
  const options = store.papers.length > 1 ? [`<option value="all">All papers (${store.papers.length})</option>`] : [];
  options.push(...store.papers.map(p => `<option value="${p.id}">${escapeHtml(shortTitle(p.title))}</option>`));
  els.paperScope.innerHTML = options.join("");
  els.paperScope.classList.toggle("hidden", store.papers.length === 0);
  if ([...els.paperScope.options].some(o => o.value === previous)) els.paperScope.value = previous;
}

export function renderPapers(onRemove) {
  const empty = store.papers.length === 0;
  els.papersEmpty.classList.toggle("hidden", !empty);
  els.papersGrid.classList.toggle("hidden", empty);
  els.papersGrid.innerHTML = "";
  for (const paper of store.papers) {
    const card = document.createElement("div");
    card.className = "glass-panel p-4 rounded-2xl";
    card.innerHTML = `
      <h3 class="font-semibold text-gray-800 text-sm">${escapeHtml(paper.title)}</h3>
      <p class="text-xs text-gray-500 mt-1">${escapeHtml(paper.authors.length ? paper.authors.slice(0, 3).join(", ") + (paper.authors.length > 3 ? " et al." : "") : paper.name)}${paper.year ? ` · ${paper.year}` : ""}</p>
      <p class="text-xs text-gray-400 mt-1">${paper.pageCount} pages · ${paper.chunkCount} passages indexed</p>
      <div class="flex gap-2 mt-3">
        <button type="button" class="view px-3 py-1 bg-indigo-100 text-indigo-700 rounded-lg text-xs font-medium hover:bg-indigo-200">View</button>
        <button type="button" class="remove px-3 py-1 bg-red-100 text-red-700 rounded-lg text-xs font-medium hover:bg-red-200">Remove</button>
      </div>`;
    card.querySelector(".view").addEventListener("click", () => openPdf(paper, 1));
    card.querySelector(".remove").addEventListener("click", () => onRemove(paper));
    els.papersGrid.appendChild(card);
  }
}

export function switchView(view) {
  const active = "tab-btn text-sm font-bold text-indigo-600 bg-white/50 px-3 py-1 rounded-full shadow-sm";
  const inactive = "tab-btn text-sm font-medium text-gray-500 hover:text-indigo-600 px-3 py-1 transition-colors";
  els.chatView.classList.toggle("hidden", view !== "chat");
  els.papersView.classList.toggle("hidden", view !== "papers");
  els.chatTab.className = view === "chat" ? active : inactive;
  els.papersTab.className = view === "papers" ? active : inactive;
}

export function setStatus(element, text, tone = "pending") {
  const tones = { pending: "text-orange-500", ready: "text-green-600", error: "text-red-600", active: "text-indigo-600" };
  element.textContent = text;
  element.className = tones[tone];
}
