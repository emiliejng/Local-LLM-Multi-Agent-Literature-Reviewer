# Local LLM Multi-Agent Literature Reviewer 📚

A privacy-first research assistant that runs **fully in your browser**.

This app reads PDF papers.  
It builds a local vector database.  
It answers questions with RAG.  
It can write a short literature review with citations.

No cloud. No upload.  
Everything stays on your device.

---

## Live Demo

- Live URL: **(add your GitHub Pages link)**
- Recommended browser: recent **Chrome / Edge** with **WebGPU** enabled
- Best on desktop: model loading and embeddings are heavy

---

## Overview

The workflow is simple:

1. Drop PDF papers into the page  
2. Extract text with **PDF.js**  
3. Split text into chunks  
4. Create embeddings with **Transformers.js**  
5. Store chunks + vectors in a local **Vector DB**  
6. Search with **cosine similarity**  
7. Send the best chunks to a local LLM with **WebLLM**

The LLM replies using your documents.  
The UI shows which chunks were used (citations).

---

## Features

### RAG Engine (PDF → Chunks → Vectors → Search)
- Drag & drop PDF upload
- Text extraction with **PDF.js**
- Chunking with overlap (400 chars, 80 overlap)
- Embeddings with **Transformers.js** (`Xenova/all-MiniLM-L6-v2`)
- Simple local Vector Store (`state.vectorStore`)
- Pure JS cosine similarity search
- Multiple PDFs at the same time
- Memory view (chunks count / storage info)

### Agentic Chat (WebLLM + Context Injection)
- Local model loading with **WebLLM**
- Loading status in the UI
- Chat history (the agent remembers)
- System prompt: "Academic Researcher"
- Top-N retrieved chunks injected into the prompt
- Citations UI: shows which document parts are used
- Optional controls panel (temperature, system prompt)

### Bonus: Voice (Optional)
- TTS with browser **SpeechSynthesis**
- STT with Transformers.js (`Xenova/whisper-tiny`)
- VAD logic to auto-send when you stop speaking
- Visual state: Listening / Processing

---

## Screenshot / Video

**Project Title & Visuals:** Add a screenshot of the interface with a generated Literature Review.

Recommended visuals for your repo:
- Screenshot: `images/screenshot.png` - showing the full interface
- Optional GIF: `images/rag-demo.gif` - demonstrating the RAG process  
- Optional video: `demo.mp4` - complete workflow demonstration

Your screenshot should show:
- PDF upload zone with uploaded papers
- Memory bank (chunks count / storage stats)
- Chat interface with citations
- A generated literature review output

---

## Tech Stack

**Core Technologies (as specified in PROJECT_LLM.md):**

- **WebLLM** (LLM inference in the browser, WebGPU)
- **Transformers.js** (embeddings + optional Whisper)
- **PDF.js** (PDF text extraction)
- **Vanilla JavaScript** (RAG, vector store, UI logic)
- **Tailwind CSS** (CDN)

---

## Getting Started (Local)

**Important:** You must run a local server.  
Do not open `index.html` by double-clicking.

**Local Setup:** As required by the project specifications, this application needs a local server (e.g., Python or VS Code Live Server) for development due to ES modules and CORS requirements.

### Prerequisites
- Chrome/Edge with WebGPU
- Python 3 (or any local server)

### Usage Guide - Clear instructions to run locally:

```bash
cd /path/to/your/project
python -m http.server
# Open http://localhost:8000
```

**Alternative local servers:**
```bash
# VS Code Live Server extension
# Right-click index.html → "Open with Live Server"

# Node.js
npx serve .

# PHP
php -S localhost:8000
```

### First Use
1. Click "Initialize Model" to load WebLLM
2. Wait for embedding model to load
3. Drag PDF files into the upload zone
4. Start asking questions about your papers

---

## Project Architecture

```
index.html          # Main page, UI components, Tailwind CSS
main.js             # RAG engine, WebLLM integration, chat logic
tailwind.css        # Custom styles and animations
PROJECT_LLM.md      # Full project specifications
```

### Key Functions
- `processPDF()`: PDF → text → chunks → embeddings → vector store
- `searchVectorStore()`: Query → embeddings → cosine similarity → top chunks
- `generateResponse()`: Context + query → WebLLM → response with citations

---

## Configuration

Edit the constants in `main.js`:

```javascript
const CHUNK_SIZE = 400;              // Characters per chunk
const CHUNK_OVERLAP = 80;            // Overlap between chunks  
const TOP_K_CHUNKS = 8;              // Max chunks per query
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const SELECTED_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";
```

---

## Privacy & Performance

**Privacy**: All processing happens locally. No data is sent to any server.

**Performance Tips**:
- Use Chrome/Edge with WebGPU for best speed
- Start with smaller PDFs (< 10MB each)
- The 1B model is fastest, 3B+ models need more RAM
- Close other browser tabs during model loading

---

## Credits

Built for the **Local LLM Multi-Agent Literature Reviewer** project.

Technologies:
- [WebLLM](https://webllm.mlc.ai/) by MLC Team
- [Transformers.js](https://huggingface.co/docs/transformers.js/) by Hugging Face  
- [PDF.js](https://mozilla.github.io/pdf.js/) by Mozilla
| **Architecture** | Vanilla JavaScript | Zero dependencies, pure client-side |

## Quick Start

### Option 1: Online (Deployed Version)
Visit the live demo: **[https://your-github-username.github.io/projet_LLM_AICG](https://your-github-username.github.io/projet_LLM_AICG)**

### Option 2: Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/projet_LLM_AICG.git
   cd projet_LLM_AICG
   ```

2. **Start a local server** (required for ES modules)
   ```bash
   # Option A: Python
   python -m http.server 8000
   
   # Option B: Node.js
   npx serve .
   
   # Option C: VS Code Live Server Extension
   # Right-click index.html → "Open with Live Server"
   ```

3. **Open your browser**
   - Navigate to `http://localhost:8000`
   - **Chrome/Edge 113+** recommended (WebGPU support)
   - Enable experimental features: `chrome://flags/#enable-unsafe-webgpu`

## Usage Guide

### 1. **Upload Research Papers**
- Drag & drop PDF files to the upload zone
- Watch real-time processing: PDF → Text → Chunks → Embeddings
- Monitor Memory Bank statistics as papers are indexed

### 2. **Smart Questioning**
- Ask natural language questions about your papers
- See which document sections are being used (citation tracking)
- Switch between different AI personas (Academic, Technical, etc.)

### 3. **Automated Analysis**
- Click **Auto Literature Review** for structured academic reviews
- Use **Analyze Methods** to compare research methodologies  
- Try **Compare Papers** for side-by-side analysis

### 4. **Voice Interaction** (Bonus)
- Click 🎤 button to start voice input
- Enable "Hands-Free Mode" for continuous listening
- AI responses can be read aloud automatically

## Advanced Configuration

### Model Selection
The app supports multiple LLM models:
- **Llama 3.2 1B**: Fast, lightweight (~600MB)
- **Llama 3.2 3B**: Balanced performance (~1.5GB)  
- **Llama 3.1 8B**: Highest quality (~4GB)
- **Phi 3.5 Mini**: Microsoft's efficient model (~2GB)
- **Gemma 2 2B**: Google's compact model (~1GB)

### System Prompts
Customize AI behavior with built-in templates:
- **Academic Researcher**: Structured literature reviews
- **Technical Analyst**: Deep methodology focus
- **Methodology Critic**: Research design analysis
- **Comparative Synthesizer**: Cross-paper synthesis

### Performance Tuning
- **Temperature**: Control AI creativity (0.0 = focused, 1.0 = creative)
- **Chunk Size**: Adjust context window (default: 400 chars)
- **Similarity Threshold**: Filter relevance (default: 0.1)

## Privacy & Security

**100% Client-Side Processing**
- All AI models run locally in your browser
- No data sent to external servers
- PDFs processed entirely on your machine
- Vector embeddings stored in browser memory

**Local Storage Only**
- Conversation history: Browser session
- Uploaded papers: Temporary browser memory
- Settings: LocalStorage (optional)

### Key Files
- `index.html`: UI structure with Tailwind CSS
- `main.js`: Core logic (RAG, WebLLM, voice features)
- `tailwind.css`: Custom styles and animations

## Acknowledgments

- [WebLLM Team](https://webllm.mlc.ai/) for browser-based LLM inference
- [Hugging Face](https://huggingface.co/) for Transformers.js and model hosting
- [Mozilla](https://mozilla.github.io/pdf.js/) for PDF.js library
- Academic research community for inspiration and use cases

---
