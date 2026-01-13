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

<img width="1503" height="901" alt="Capture d’écran 2026-01-13 à 01 26 26" src="https://github.com/user-attachments/assets/e2d0189d-e48a-4980-b3f7-56d522a8290b" />

---

## Overview

This app implements an end-to-end **RAG pipeline in Vanilla JavaScript**:

1. **Upload PDFs** (drag & drop)
2. Extract text with **PDF.js**
3. Split text into chunks (default: **400 chars**, **80 overlap**)
4. Create embeddings with **Transformers.js** (`Xenova/all-MiniLM-L6-v2`)
5. Store chunks + vectors in a client-side **Vector DB** (`state.vectorStore`)
6. Retrieve top chunks using **cosine similarity**
7. Inject retrieved context into a **WebLLM** prompt
8. Show **citations** for every answer

Everything runs locally in the browser using **WebGPU**.

---

## Features

- 100% client-side: **no server**, **no uploads**
- Multi-PDF ingestion (drag & drop)
- Chunking + embeddings + cosine similarity retrieval (pure JS)
- WebLLM chat with history
- Citations UI (shows which chunks support the answer)
- Memory bank view (chunk count / indexing status)
- Optional voice mode (TTS / STT / VAD)

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

### 1. Initialize Model
- Click "Initialize Model"
- Wait for WebLLM + embedding model to load

### 2. Upload PDFs
- Drag & drop one or more PDF papers
- Wait for indexing: PDF → text → chunks → embeddings

### 3. Ask Questions
- Type a question about your papers
- The app retrieves the most relevant chunks and shows citations

### 4. Generate a Literature Review
- Ask: "Generate a concise literature review about …"
- The agent synthesizes themes across papers with citations

---

## Project Structure

```
index.html          # UI (Tailwind) + layout
main.js             # RAG engine + WebLLM chat + citations (+ optional audio)
PROJECT_LLM.md      # Full project specifications
images/             # screenshot.png, rag-demo.gif
```

---

## Configuration

Edit the constants in `main.js`:

```javascript
const CHUNK_SIZE = 400;
const CHUNK_OVERLAP = 80;
const TOP_K_CHUNKS = 8;

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const SELECTED_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";
```

---

## Notes (Privacy & Performance)

- **Privacy:** all processing happens locally in your browser. No data is sent to external servers.
- **Performance:** WebGPU is required for good speed. Desktop browsers work best.
