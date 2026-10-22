# Local LLM Multi-Agent Literature Reviewer 📚

A **privacy-first** AI research assistant that runs **fully in your browser**.  
It reads PDF papers, builds a local **Vector DB**, and answers questions with **RAG**.  
It can also generate a **concise Literature Review** with **citations** (the exact chunks used).

**Local AI:** no cloud calls, no uploads, your documents stay on your device.

---

## Live Demo

- Live URL: https://emiliejng.github.io/Local-LLM-Multi-Agent-Literature-Reviewer/
- Recommended desktop browser: recent **Chrome / Edge** with **WebGPU** enabled

---

## Visuals 

<img width="1510" height="901" alt="Capture d’écran 2026-01-13 à 03 05 06" src="https://github.com/user-attachments/assets/2fcf548e-ce14-4133-8d83-ea360670c8fd" />

<img width="1510" height="901" alt="Capture d’écran 2026-01-13 à 03 05 10" src="https://github.com/user-attachments/assets/a20a07c5-afd2-494c-b525-6d98c928148a" />



---

## Overview

This app implements an end-to-end **RAG pipeline in Vanilla JavaScript**:

1. **Upload PDFs** (drag & drop)
2. Extract text page by page with **PDF.js**: figure labels, running headers and the **reference list are removed**
3. Split each page into chunks of **whole sentences** (≤ 700 chars), so every chunk keeps its **page number**
4. Create embeddings with **Transformers.js** (`Xenova/all-MiniLM-L6-v2`) and store them in a client-side **Vector DB**
5. **Hybrid retrieval**: cosine similarity (pure JS) + BM25 keywords, merged by reciprocal rank fusion. Follow-up and non-English questions are first rewritten into a standalone English query; the paper's first page (title, authors, abstract) is added for questions about the paper as a whole
6. **WebLLM** answers from the retrieved excerpts only (streamed token by token)
7. **Grounding**: each sentence of the answer is matched to the passage that supports it and cited as **[page]**, with the exact quoted sentence; figures absent from the sources are flagged

Everything runs locally in the browser using **WebGPU**.

---

## Features

- 100% client-side: **no server**, **no uploads**
- Multi-PDF ingestion, paper selector ("All papers" or one paper)
- Citations with **page numbers**: click one to read the quoted passage and open the PDF at that page
- Source check under every answer (sourced statements, unverified figures)
- **Literature review agent**: summarizes each paper from its own passages, then writes a structured synthesis (Introduction, Thematic synthesis, Comparison, Gaps & conclusion)
- Memory Bank (passages / papers / status)
- Settings: temperature, model (Llama 3.2 3B or 1B), editable system prompt — applied in real time
- Voice: Whisper speech-to-text, RMS voice-activity detection, hands-free mode with spoken answers. Answers are read with a local voice matching their language (English or French); the read-aloud button doubles as a stop button (Esc also stops)

---

## Tech Stack

- **WebLLM** (LLM inference in the browser, WebGPU)
- **Transformers.js** (embeddings + optional Whisper STT)
- **Tailwind CSS** (CDN UI)
- **Vanilla JavaScript**
- **PDF.js** (PDF extraction)

---

## Getting Started (Local)

You must serve the project over HTTP (ES modules + browser security).  
Do **not** open `index.html` by double-clicking.

### Option A — Python (recommended)

```bash
cd /path/to/your/project
python -m http.server 8000
# Open http://localhost:8000
```

### Option B — VS Code Live Server

1. Install the Live Server extension
2. Right-click `index.html` → Open with Live Server

---

## Usage Guide

### 1. Wait for the model
- The model loads automatically (Llama 3.2 3B, ~1.5 GB the first time, then cached)
- Low on GPU memory? Pick Llama 3.2 1B in **Settings**

### 2. Upload PDFs
- Drag & drop one or more PDF papers
- Wait for indexing: PDF → text → chunks → embeddings

### 3. Ask Questions
- Type a question about your papers (any language), or use the mic
- Each statement of the answer is followed by its page, e.g. `p. 4`; click it to see the exact passage and open the PDF there

### 4. Generate a Literature Review
- Click **📚 Literature review**, or ask "Generate a concise literature review about …"
- The agent summarizes each selected paper, then synthesizes themes across them, with page citations

---

## Project Structure

```
index.html          # UI (Tailwind) + layout
main.js             # Entry point: wires the UI to the modules below
js/config.js        # Models, chunking, retrieval and prompt settings
js/pdf.js           # PDF.js extraction: pages, noise and reference-list removal, title/authors
js/rag.js           # Chunking, embeddings, vector store, cosine similarity, BM25, hybrid search
js/llm.js           # WebLLM loading and streaming completions
js/agent.js         # Question answering and literature review pipelines
js/grounding.js     # Page citations, quoted passages and source check
js/ui.js            # Chat rendering, citations panel, paper library, PDF viewer
js/voice.js         # Whisper STT, voice activity detection, text-to-speech
PROJECT_LLM.md      # Full project specifications
```

---

## Configuration

Edit the constants in `js/config.js`:

```javascript
export const CHUNK_MAX_CHARS = 700;      // whole sentences, never across pages
export const TOP_K = 6;                  // passages retrieved per question
export const SUPPORT_THRESHOLD = 0.3;    // minimum similarity for a citation
export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const DEFAULT_MODEL = "Llama-3.2-3B-Instruct-q4f32_1-MLC";
```

---

## Notes (Privacy & Performance)

- **Privacy:** all processing happens locally in your browser. No data is sent to external servers.
- **Performance:** WebGPU is required for good speed. Desktop browsers work best.
- **After an update:** hard-reload the page (Cmd/Ctrl+Shift+R) so the browser does not keep old JavaScript modules in cache.
