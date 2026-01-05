// main.js
console.log("✅ main.js loaded");

/**
 * AI Paper Assistant - Core Logic with RAG Engine
 * RAG (embeddings + vector store) should work even if WebLLM fails.
 */

import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";

// --- Configuration ---
let SELECTED_MODEL = "Llama-3.2-1B-Instruct-q4f32_1-MLC";
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 100;
const TOP_K_CHUNKS = 5;
const MAX_MB = 25;
const MIN_SIMILARITY_THRESHOLD = 0.1; // Minimum similarity for relevant chunks

// Available Models Configuration
const AVAILABLE_MODELS = {
  "Llama-3.2-1B-Instruct-q4f32_1-MLC": {
    name: "Llama 3.2 1B",
    description: "Fast, lightweight model for quick responses",
    size: "~600MB"
  },
  "Llama-3.2-3B-Instruct-q4f32_1-MLC": {
    name: "Llama 3.2 3B",
    description: "Balanced performance and quality",
    size: "~1.5GB"
  },
  "Llama-3.1-8B-Instruct-q4f32_1-MLC": {
    name: "Llama 3.1 8B",
    description: "High quality responses, slower",
    size: "~4GB"
  },
  "Phi-3.5-mini-instruct-q4f32_1-MLC": {
    name: "Phi 3.5 Mini",
    description: "Microsoft's efficient model",
    size: "~2GB"
  },
  "gemma-2-2b-it-q4f32_1-MLC": {
    name: "Gemma 2 2B",
    description: "Google's compact model",
    size: "~1GB"
  }
};

// System Control Variables
let currentTemperature = 0.7;
let systemPrompts = {
  default: "You are an expert Academic Researcher and Literature Reviewer. " +
           "Your goal is to synthesize information from uploaded PDF research papers. " +
           "When provided with document context, always cite your sources using the filename. " +
           "Structure your responses professionally with clear arguments and evidence. " +
           "For literature reviews, organize your response with: Introduction, Key Themes, Comparison of Approaches, and Conclusion.",
  
  technical: "You are a Technical Research Analyst specializing in deep technical analysis of research papers. " +
             "Focus on methodologies, algorithms, experimental setups, and technical contributions. " +
             "Analyze technical limitations, implementation details, and reproducibility concerns. " +
             "Always cite sources and provide technical depth in your analysis.",
             
  methodological: "You are a Research Methodology Critic. Your role is to analyze and critique the research methods, " +
                   "experimental designs, statistical approaches, and validity of conclusions in academic papers. " +
                   "Identify methodological strengths and weaknesses. Compare different methodological approaches across papers. " +
                   "Cite sources and maintain academic rigor.",
                   
  comparative: "You are a Comparative Research Synthesizer. Your specialty is identifying patterns, " +
               "contradictions, and complementary findings across multiple research papers. " +
               "Focus on how different papers relate to each other, their agreements and disagreements, " +
               "and the evolution of ideas in the field. Always cite sources and provide comparative analysis.",
               
  custom: ""
};

let currentPromptType = 'default';

// --- State ---
let engine = null;
let embedder = null;
let isModelLoading = false;
let isEmbedderLoading = false;
let isTyping = false;

let vectorStore = [];
let uploadedPapers = [];

let conversationHistory = [
  {
    role: "system",
    content: systemPrompts.default
  },
];

// --- UI Refs ---
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const chatHistoryContainer = document.getElementById("chat-history");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const vectorStoreInfo = document.getElementById("vector-store-info");
const papersList = document.getElementById("papers-list");
const ragStatus = document.getElementById("rag-status");

// Navigation refs
const chatTab = document.getElementById("chat-tab");
const papersTab = document.getElementById("papers-tab");
const dashboardTab = document.getElementById("dashboard-tab");
const chatView = document.getElementById("chat-view");
const papersView = document.getElementById("papers-view");
const papersGrid = document.getElementById("papers-grid");
const papersEmptyState = document.getElementById("papers-empty-state");
const gotoUploadBtn = document.getElementById("goto-upload-btn");

// PDF Viewer refs
const pdfViewerModal = document.getElementById("pdf-viewer-modal");
const pdfViewerTitle = document.getElementById("pdf-viewer-title");
const pdfViewerFrame = document.getElementById("pdf-viewer-frame");
const closePdfViewerBtn = document.getElementById("close-pdf-viewer");

// System Controls UI refs
const temperatureSlider = document.getElementById("temperature-slider");
const temperatureValue = document.getElementById("temperature-value");
const modelSelect = document.getElementById("model-select");
const modelStatus = document.getElementById("model-status");
const systemPromptSelect = document.getElementById("system-prompt-select");
const editPromptBtn = document.getElementById("edit-prompt-btn");
const promptModal = document.getElementById("prompt-modal");
const customPromptTextarea = document.getElementById("custom-prompt-textarea");
const cancelPromptBtn = document.getElementById("cancel-prompt-btn");
const savePromptBtn = document.getElementById("save-prompt-btn");

// --- Status helpers ---
function setRagStatus(text, cls = "text-orange-500") {
  if (!ragStatus) return;
  ragStatus.textContent = text;
  ragStatus.className = cls;
}

function setModelStatus(text, cls = "text-orange-500") {
  if (!modelStatus) return;
  modelStatus.textContent = text;
  modelStatus.className = cls;
}

// --- RAG Engine ---

async function initEmbedder() {
  if (isEmbedderLoading || embedder) return;

  isEmbedderLoading = true;
  setRagStatus("Loading embedder…", "text-orange-500");
  console.log("Loading embedding model...");

  try {
    embedder = await pipeline("feature-extraction", EMBEDDING_MODEL);
    console.log("✅ Embedding model loaded successfully.");
    setRagStatus("Ready", "text-green-600");
    updateVectorStoreUI();
  } catch (error) {
    console.error("❌ Failed to load embedding model:", error);
    setRagStatus("Embedder error", "text-red-600");
  } finally {
    isEmbedderLoading = false;
  }
}

// ✅ pdf.js from CDN => window.pdfjsLib
async function extractTextFromPDF(file) {
  const pdfjsLib = window.pdfjsLib;
  if (!pdfjsLib) throw new Error("pdf.js not loaded (window.pdfjsLib is undefined).");

  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  let fullText = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    fullText += textContent.items.map((item) => item.str).join(" ") + "\n";
  }
  return fullText;
}

function chunkText(text, filename) {
  console.log(`📝 Chunking text from ${filename}: ${text.length} characters`);
  const chunks = [];
  
  // Clean and normalize text
  const cleanText = text.replace(/\s+/g, ' ').trim();
  
  // Sliding window approach with character-based chunking
  for (let i = 0; i < cleanText.length; i += (CHUNK_SIZE - CHUNK_OVERLAP)) {
    const chunk = cleanText.slice(i, i + CHUNK_SIZE);
    
    if (chunk.trim().length > 0) {
      chunks.push({ 
        text: chunk.trim(), 
        source: filename, 
        embedding: null,
        chunkIndex: chunks.length
      });
    }
    
    // Break if we've reached the end
    if (i + CHUNK_SIZE >= cleanText.length) break;
  }
  
  console.log(`🧩 Created ${chunks.length} chunks with ${CHUNK_OVERLAP} character overlap`);
  return chunks;
}

async function generateEmbeddings(chunks) {
  if (!embedder) {
    console.warn("❌ Embedder not loaded yet, skipping embeddings.");
    return chunks;
  }

  console.log(`🔢 Generating embeddings for ${chunks.length} chunks using ${EMBEDDING_MODEL}...`);
  let successCount = 0;
  let errorCount = 0;
  
  for (let i = 0; i < chunks.length; i++) {
    try {
      // Generate embedding with proper options
      const output = await embedder(chunks[i].text, { 
        pooling: "mean", 
        normalize: true 
      });
      
      // Convert tensor to array
      chunks[i].embedding = Array.from(output.data);
      successCount++;
      
      // Progress logging every 10 chunks
      if (i % 10 === 0 || i === chunks.length - 1) {
        console.log(`📊 Embedding progress: ${i + 1}/${chunks.length} (${successCount} success, ${errorCount} errors)`);
      }
      
    } catch (error) {
      console.error(`❌ Failed to generate embedding for chunk ${i}:`, error);
      chunks[i].embedding = null; // Mark as failed
      errorCount++;
    }
  }

  console.log(`✅ Embedding generation complete: ${successCount} successful, ${errorCount} failed`);
  return chunks.filter(chunk => chunk.embedding !== null); // Filter out failed embeddings
}

/**
 * Calculate cosine similarity between two vectors
 * Formula: similarity = (Á·B) / (||Á|| × ||B||)
 * @param {number[]} vecA - First vector
 * @param {number[]} vecB - Second vector
 * @returns {number} Similarity score between 0 and 1
 */
function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB) {
    console.warn("❌ One or both vectors are null/undefined");
    return 0;
  }
  
  if (vecA.length !== vecB.length) {
    console.warn(`❌ Vector length mismatch: ${vecA.length} vs ${vecB.length}`);
    return 0;
  }
  
  if (vecA.length === 0) return 0;

  // Calculate dot product and magnitudes
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magnitudeA += vecA[i] * vecA[i];
    magnitudeB += vecB[i] * vecB[i];
  }
  
  // Calculate magnitudes (L2 norm)
  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);
  
  // Avoid division by zero
  if (magnitudeA === 0 || magnitudeB === 0) {
    console.warn("❌ One or both vectors have zero magnitude");
    return 0;
  }
  
  // Return cosine similarity
  const similarity = dotProduct / (magnitudeA * magnitudeB);
  return Math.max(0, Math.min(1, similarity)); // Clamp between 0 and 1
}

async function searchSimilarChunks(query) {
  if (!embedder) {
    console.warn("❌ Embedder not available for search");
    return [];
  }
  
  if (vectorStore.length === 0) {
    console.warn("❌ Vector store is empty - no documents to search");
    return [];
  }
  
  console.log(`🔍 Searching for: "${query.substring(0, 100)}${query.length > 100 ? '...' : ''}"`);  
  console.log(`📊 Vector store contains ${vectorStore.length} chunks`);

  try {
    // Generate query embedding
    const queryEmbedding = await embedder(query, { 
      pooling: "mean", 
      normalize: true 
    });
    const queryVector = Array.from(queryEmbedding.data);
    
    console.log(`🎯 Query embedding generated: ${queryVector.length} dimensions`);

    // Calculate similarities and sort
    const results = vectorStore
      .map((chunk, index) => {
        if (!chunk.embedding) {
          console.warn(`⚠️ Chunk ${index} has no embedding`);
          return { ...chunk, similarity: 0, index };
        }
        
        const similarity = cosineSimilarity(queryVector, chunk.embedding);
        return { ...chunk, similarity, index };
      })
      .filter(result => result.similarity >= MIN_SIMILARITY_THRESHOLD)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, TOP_K_CHUNKS);

    console.log(`📊 Found ${results.length} relevant chunks above threshold ${MIN_SIMILARITY_THRESHOLD}`);
    
    if (results.length > 0) {
      console.log("🎯 Top similarities:", results.slice(0, 3).map(r => ({
        source: r.source,
        similarity: r.similarity.toFixed(3),
        preview: r.text.substring(0, 50) + "..."
      })));
    }

    return results;
    
  } catch (error) {
    console.error("❌ Search failed:", error);
    return [];
  }
}

async function processPDF(file) {
  console.log("Starting processPDF:", file.name);

  const statusMessage = document.createElement("div");
  statusMessage.className = "p-2 bg-blue-100 text-blue-800 rounded text-sm mb-2";
  statusMessage.textContent = `Processing ${file.name}...`;
  chatHistoryContainer.appendChild(statusMessage);
  scrollToBottom();

  try {
    const text = await extractTextFromPDF(file);
    if (!text || text.length === 0) throw new Error("No text could be extracted from this PDF.");

    const chunks = chunkText(text, file.name);
    const embeddedChunks = await generateEmbeddings(chunks);

    vectorStore.push(...embeddedChunks);
    uploadedPapers.push({ 
      name: file.name, 
      chunks: embeddedChunks.length, 
      uploadTime: new Date(),
      file: file // Store the original file for viewing
    });

    updateVectorStoreUI();
    updatePapersListUI();

    statusMessage.remove();

    const ok = document.createElement("div");
    ok.className = "p-2 bg-green-100 text-green-800 rounded text-sm mb-2";
    ok.textContent = `✅ Successfully processed ${file.name} (${embeddedChunks.length} chunks)`;
    chatHistoryContainer.appendChild(ok);
    scrollToBottom();
  } catch (err) {
    console.error("processPDF error:", err);
    statusMessage.remove();

    const bad = document.createElement("div");
    bad.className = "p-2 bg-red-100 text-red-800 rounded text-sm mb-2";
    bad.textContent = `❌ Failed to process ${file.name}: ${err.message}`;
    chatHistoryContainer.appendChild(bad);
    scrollToBottom();

    alert(`Failed to process ${file.name}: ${err.message}`);
  }
}

// --- WebLLM (LLM part) ---
// IMPORTANT: does NOT block embedder / RAG.
async function initWebLLM(modelId = SELECTED_MODEL) {
  if (isModelLoading) return;
  
  // Dispose previous engine if exists
  if (engine) {
    try {
      await engine.unload();
    } catch (e) {
      console.warn("Failed to unload previous model:", e);
    }
    engine = null;
  }

  isModelLoading = true;
  const modelInfo = AVAILABLE_MODELS[modelId];
  console.log(`Initializing WebLLM with ${modelInfo?.name || modelId}...`);
  
  setModelStatus(`Loading ${modelInfo?.name || modelId}...`, "text-orange-500");

  const originalPlaceholder = chatInput?.placeholder ?? "";
  if (chatInput) {
    chatInput.placeholder = `Loading ${modelInfo?.name || 'AI Model'} (${modelInfo?.size || 'this may take a moment'})...`;
    chatInput.disabled = true;
  }

  try {
    engine = await CreateMLCEngine(modelId, {
      initProgressCallback: (progress) => {
        const percent = Math.ceil(progress.progress * 100);
        console.log(`Model Loading: ${percent}%`);
        setModelStatus(`Loading ${percent}%`, "text-orange-500");
      },
    });

    console.log("✅ WebLLM Loaded Successfully.");
    setModelStatus(`${modelInfo?.name || modelId} Ready`, "text-green-600");
    
    // Add success message to chat
    const successMsg = document.createElement("div");
    successMsg.className = "p-2 bg-green-100 text-green-800 rounded text-sm mb-2";
    successMsg.textContent = `🚀 Model switched to: ${modelInfo?.name || modelId}`;
    chatHistoryContainer.appendChild(successMsg);
    scrollToBottom();
    
  } catch (error) {
    console.error("❌ Failed to load WebLLM model:", error);
    setModelStatus("Failed to load", "text-red-600");
    
    // Add error message to chat
    const errorMsg = document.createElement("div");
    errorMsg.className = "p-2 bg-red-100 text-red-800 rounded text-sm mb-2";
    errorMsg.textContent = `❌ Failed to load ${modelInfo?.name || modelId}: ${error.message}`;
    chatHistoryContainer.appendChild(errorMsg);
    scrollToBottom();
  } finally {
    if (chatInput) {
      chatInput.placeholder = originalPlaceholder;
      chatInput.disabled = false;
    }
    isModelLoading = false;
  }
}

// Function to change models
async function changeModel(newModelId) {
  if (newModelId === SELECTED_MODEL && engine) {
    console.log("Model already loaded:", newModelId);
    return;
  }
  
  SELECTED_MODEL = newModelId;
  await initWebLLM(newModelId);
}

// --- Navigation functions ---
function switchToView(viewName) {
  // Hide all views
  if (chatView) chatView.classList.add('hidden');
  if (papersView) papersView.classList.add('hidden');
  
  // Reset all tab styles
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.className = 'tab-btn text-sm font-medium text-gray-500 hover:text-indigo-600 transition-colors';
  });
  
  // Show selected view and highlight tab
  switch(viewName) {
    case 'chat':
      if (chatView) chatView.classList.remove('hidden');
      if (chatTab) chatTab.className = 'tab-btn text-sm font-bold text-indigo-600 bg-white/50 px-3 py-1 rounded-full shadow-sm';
      break;
    case 'papers':
      if (papersView) papersView.classList.remove('hidden');
      if (papersTab) papersTab.className = 'tab-btn text-sm font-bold text-indigo-600 bg-white/50 px-3 py-1 rounded-full shadow-sm';
      updatePapersGrid();
      break;
    case 'dashboard':
      if (chatView) chatView.classList.remove('hidden'); // Default to chat for now
      if (dashboardTab) dashboardTab.className = 'tab-btn text-sm font-bold text-indigo-600 bg-white/50 px-3 py-1 rounded-full shadow-sm';
      break;
  }
}

function updatePapersGrid() {
  if (!papersGrid || !papersEmptyState) return;
  
  if (uploadedPapers.length === 0) {
    papersGrid.classList.add('hidden');
    papersEmptyState.classList.remove('hidden');
    return;
  }
  
  papersGrid.classList.remove('hidden');
  papersEmptyState.classList.add('hidden');
  
  papersGrid.innerHTML = '';
  
  uploadedPapers.forEach((paper, index) => {
    const paperCard = document.createElement('div');
    paperCard.className = 'glass-panel p-4 rounded-2xl hover:shadow-lg transition-all duration-300 cursor-pointer group hover:scale-[1.02]';
    paperCard.innerHTML = `
      <div class="flex items-start gap-3">
        <div class="w-12 h-12 rounded-xl bg-red-100 text-red-500 flex items-center justify-center text-lg font-bold flex-shrink-0">
          📄
        </div>
        <div class="flex-1 min-w-0">
          <h3 class="font-semibold text-gray-800 truncate group-hover:text-indigo-600 transition-colors">
            ${paper.name}
          </h3>
          <p class="text-sm text-gray-500 mt-1">
            ${paper.chunks} chunks • Uploaded ${paper.uploadTime.toLocaleDateString()}
          </p>
          <div class="flex gap-2 mt-3">
            <button onclick="viewPdf(${index})" class="flex items-center gap-1 px-3 py-1 bg-indigo-100 text-indigo-700 rounded-lg text-xs font-medium hover:bg-indigo-200 transition-colors">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
              </svg>
              View
            </button>
            <button onclick="removePaper(${index})" class="flex items-center gap-1 px-3 py-1 bg-red-100 text-red-700 rounded-lg text-xs font-medium hover:bg-red-200 transition-colors">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
              </svg>
              Remove
            </button>
          </div>
        </div>
      </div>
    `;
    papersGrid.appendChild(paperCard);
  });
}

// PDF Viewer functions
function viewPdf(paperIndex) {
  const paper = uploadedPapers[paperIndex];
  if (!paper || !paper.file) return;
  
  const fileURL = URL.createObjectURL(paper.file);
  
  if (pdfViewerTitle) pdfViewerTitle.textContent = paper.name;
  if (pdfViewerFrame) pdfViewerFrame.src = fileURL;
  if (pdfViewerModal) {
    pdfViewerModal.classList.remove('hidden');
    pdfViewerModal.classList.add('flex');
  }
}

function closePdfViewer() {
  if (pdfViewerModal) {
    pdfViewerModal.classList.add('hidden');
    pdfViewerModal.classList.remove('flex');
  }
  if (pdfViewerFrame) {
    pdfViewerFrame.src = '';
  }
}

function removePaper(paperIndex) {
  if (!confirm('Are you sure you want to remove this paper?')) return;
  
  const paper = uploadedPapers[paperIndex];
  
  // Remove chunks from vector store
  vectorStore = vectorStore.filter(chunk => chunk.source !== paper.name);
  
  // Remove from uploaded papers
  uploadedPapers.splice(paperIndex, 1);
  
  // Update UI
  updateVectorStoreUI();
  updatePapersListUI();
  updatePapersGrid();
  
  // Show success message
  const statusMsg = document.createElement("div");
  statusMsg.className = "p-2 bg-orange-100 text-orange-800 rounded text-sm mb-2";
  statusMsg.textContent = `🗑️ Removed paper: ${paper.name}`;
  chatHistoryContainer.appendChild(statusMsg);
  scrollToBottom();
}

// Make functions available globally for onclick handlers
window.viewPdf = viewPdf;
window.removePaper = removePaper;

// --- Chat ---
async function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text || isTyping) return;

  chatInput.value = "";
  chatHistoryContainer.appendChild(createMessageBubble(text, true));
  scrollToBottom();

  conversationHistory.push({ role: "user", content: text });
  isTyping = true;
  
  // Show typing indicator
  const typingIndicator = createTypingIndicator();
  chatHistoryContainer.appendChild(typingIndicator);
  scrollToBottom();

  try {
    let aiResponseText = "";

    if (engine) {
      const relevantChunks = await searchSimilarChunks(text);

      let contextString = "";
      if (relevantChunks.length > 0) {
        contextString = "\n\n--- DOCUMENT CONTEXT ---\n";
        relevantChunks.forEach((chunk) => {
          contextString += `[Source: ${chunk.source}]\n${chunk.text}\n\n`;
        });
        contextString += "--- END CONTEXT ---\n\n";
      }

      const enhancedPrompt =
        contextString +
        (relevantChunks.length > 0
          ? "Based on the provided document context above, please answer the following question. Always cite your sources using the filenames provided.\n\n"
          : "") +
        text;

      const enhancedHistory = [...conversationHistory];
      enhancedHistory[enhancedHistory.length - 1].content = enhancedPrompt;

      const reply = await engine.chat.completions.create({
        messages: enhancedHistory,
        temperature: currentTemperature,
        max_tokens: 1024,
      });

      aiResponseText = reply.choices[0].message.content;
    } else {
      // LLM not available
      aiResponseText =
        "⚠️ WebLLM is not available (WebGPU/model load failed). " +
        "RAG ingestion still works (PDFs + embeddings).";
    }

    conversationHistory.push({ role: "assistant", content: aiResponseText });
    
    // Remove typing indicator before showing response
    removeTypingIndicator();
    
    chatHistoryContainer.appendChild(createMessageBubble(aiResponseText, false));
    scrollToBottom();
  } catch (err) {
    console.error("Chat Error:", err);
    
    // Remove typing indicator on error
    removeTypingIndicator();
    
    chatHistoryContainer.appendChild(createMessageBubble("Error generating response.", false));
  } finally {
    isTyping = false;
  }
}

// --- UI helpers ---
function createTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-start gap-3 animate-fade-in-up typing-indicator-wrapper";
  wrapper.id = "typing-indicator";

  const avatar = document.createElement("div");
  avatar.className = "w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-md";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "glass-bubble-ai p-4 rounded-2xl rounded-tl-none max-w-[80%] shadow-sm";

  const typingContainer = document.createElement("div");
  typingContainer.className = "typing-indicator";
  
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.className = "typing-dot";
    typingContainer.appendChild(dot);
  }

  bubble.appendChild(typingContainer);
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}

function removeTypingIndicator() {
  const indicator = document.getElementById("typing-indicator");
  if (indicator) {
    indicator.remove();
  }
}

function createMessageBubble(text, isUser = false) {
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-start gap-3 animate-fade-in-up";

  const avatar = document.createElement("div");
  avatar.className = `w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-md ${
    isUser ? "bg-gray-400 order-2" : "bg-gradient-to-br from-indigo-500 to-purple-500"
  }`;
  avatar.textContent = isUser ? "ME" : "AI";

  const bubble = document.createElement("div");
  bubble.className = isUser
    ? "glass-bubble-user p-4 rounded-2xl rounded-tr-none text-white max-w-[80%] shadow-md order-1 ml-auto"
    : "glass-bubble-ai p-4 rounded-2xl rounded-tl-none max-w-[80%] shadow-sm";

  const textP = document.createElement("p");
  textP.className = isUser ? "text-sm font-medium" : "text-sm text-slate-700 leading-relaxed";
  textP.innerText = text;

  bubble.appendChild(textP);
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}

function scrollToBottom() {
  chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

function updateVectorStoreUI() {
  if (!vectorStoreInfo) return;
  vectorStoreInfo.innerHTML = `
    <div class="text-center">
      <div class="text-2xl font-bold text-indigo-600">${vectorStore.length}</div>
      <div class="text-xs text-gray-500">Chunks</div>
    </div>
    <div class="text-center">
      <div class="text-2xl font-bold text-purple-600">${uploadedPapers.length}</div>
      <div class="text-xs text-gray-500">Papers</div>
    </div>
  `;
}

function updatePapersListUI() {
  if (!papersList) return;

  papersList.innerHTML = "";
  if (uploadedPapers.length === 0) {
    papersList.innerHTML = `
      <li class="p-3 bg-white/40 rounded-2xl flex items-center gap-3 hover:bg-white/60 transition-colors cursor-pointer border border-transparent hover:border-white/50">
        <div class="w-8 h-8 rounded-lg bg-red-100 text-red-400 flex items-center justify-center text-xs font-bold">PDF</div>
        <div class="overflow-hidden">
          <h4 class="text-sm font-semibold text-gray-700 truncate">No papers uploaded yet</h4>
          <p class="text-[10px] text-gray-400">Upload PDFs to get started</p>
        </div>
      </li>
    `;
    return;
  }

  uploadedPapers.forEach((paper, index) => {
    const li = document.createElement("li");
    li.className =
      "p-3 bg-white/40 rounded-2xl flex items-center gap-3 hover:bg-white/60 transition-colors cursor-pointer border border-transparent hover:border-white/50";
    li.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-red-100 text-red-400 flex items-center justify-center text-xs font-bold">PDF</div>
      <div class="overflow-hidden flex-1">
        <h4 class="text-sm font-semibold text-gray-700 truncate">${paper.name}</h4>
        <p class="text-[10px] text-gray-400">${paper.chunks} chunks • ${paper.uploadTime.toLocaleTimeString()}</p>
      </div>
    `;
    li.addEventListener('click', () => viewPdf(index));
    papersList.appendChild(li);
  });
}

// --- Drop zone logic ---
function isPdfFile(file) {
  const byMime = file.type === "application/pdf";
  const byExt = file.name?.toLowerCase().endsWith(".pdf");
  return byMime || byExt;
}

function setDropActive(active) {
  if (!dropZone) return;
  dropZone.classList.toggle("ring-2", active);
  dropZone.classList.toggle("ring-indigo-400", active);
  dropZone.classList.toggle("bg-indigo-50/50", active);
}

function handleFileDrop(files) {
  if (!files || files.length === 0) return;

  Array.from(files).forEach((file) => {
    if (!isPdfFile(file)) {
      alert(`Please upload only PDF files. "${file.name}" is not a PDF.`);
      return;
    }

    const tooBig = file.size > MAX_MB * 1024 * 1024;
    if (tooBig) {
      alert(`"${file.name}" is too large (max ${MAX_MB} MB).`);
      return;
    }

    processPDF(file);
  });
}

// --- System Controls Functions ---

function updateSystemPrompt(promptType) {
  currentPromptType = promptType;
  const newPrompt = systemPrompts[promptType];
  
  // Update conversation history
  conversationHistory[0].content = newPrompt;
  
  // Visual feedback
  const statusMsg = document.createElement("div");
  statusMsg.className = "p-2 bg-blue-100 text-blue-800 rounded text-sm mb-2";
  statusMsg.textContent = `🎯 System prompt updated: ${getPromptDisplayName(promptType)}`;
  chatHistoryContainer.appendChild(statusMsg);
  scrollToBottom();
}

function getPromptDisplayName(type) {
  const names = {
    default: "Literature Reviewer",
    technical: "Technical Analyzer", 
    methodological: "Methodology Critic",
    comparative: "Comparative Researcher",
    custom: "Custom Prompt"
  };
  return names[type] || type;
}

function openPromptModal() {
  if (!promptModal) return;
  
  // Load current custom prompt or default
  const currentContent = currentPromptType === 'custom' 
    ? systemPrompts.custom 
    : systemPrompts[currentPromptType];
  
  if (customPromptTextarea) {
    customPromptTextarea.value = currentContent;
  }
  
  promptModal.classList.remove('hidden');
  promptModal.classList.add('flex');
  
  // Focus textarea
  setTimeout(() => customPromptTextarea?.focus(), 100);
}

function closePromptModal() {
  if (!promptModal) return;
  promptModal.classList.add('hidden');
  promptModal.classList.remove('flex');
}

function saveCustomPrompt() {
  if (!customPromptTextarea) return;
  
  const newPrompt = customPromptTextarea.value.trim();
  if (!newPrompt) {
    alert("Please enter a valid system prompt.");
    return;
  }
  
  // Save custom prompt
  systemPrompts.custom = newPrompt;
  
  // Update select to custom
  if (systemPromptSelect) {
    systemPromptSelect.value = 'custom';
  }
  
  // Apply the new prompt
  updateSystemPrompt('custom');
  
  closePromptModal();
}

// --- Event Listeners ---
document.addEventListener("DOMContentLoaded", async () => {
  console.log("DOM loaded");

  // ✅ Prevent browser opening dropped files, WITHOUT breaking drop-zone handlers
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(
      eventName,
      (e) => {
        e.preventDefault();
      },
      { passive: false, capture: true }
    );
  });

  // ✅ Start embedder immediately (RAG status updates even if WebLLM fails)
  await initEmbedder();

  // ✅ Then try WebLLM (optional)
  initWebLLM();

  // Chat events
  sendBtn?.addEventListener("click", sendChatMessage);
  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChatMessage();
  });

  // Drop zone events
  if (!dropZone || !fileInput) {
    console.error("dropZone or fileInput not found:", { dropZone, fileInput });
    return;
  }

  const openPicker = () => fileInput.click();

  dropZone.addEventListener("click", openPicker);
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  });

  dropZone.addEventListener("dragover", () => setDropActive(true));
  dropZone.addEventListener("dragleave", () => setDropActive(false));
  dropZone.addEventListener("drop", (e) => {
    setDropActive(false);
    handleFileDrop(e.dataTransfer.files);
  });

  fileInput.addEventListener("change", (e) => {
    handleFileDrop(e.target.files);
    e.target.value = ""; // ✅ allow picking the same file again
  });

  // --- System Controls Event Listeners ---
  
  // Navigation
  if (chatTab) {
    chatTab.addEventListener('click', () => switchToView('chat'));
  }
  if (papersTab) {
    papersTab.addEventListener('click', () => switchToView('papers'));
  }
  if (dashboardTab) {
    dashboardTab.addEventListener('click', () => switchToView('dashboard'));
  }
  if (gotoUploadBtn) {
    gotoUploadBtn.addEventListener('click', () => switchToView('chat'));
  }
  
  // PDF Viewer
  if (closePdfViewerBtn) {
    closePdfViewerBtn.addEventListener('click', closePdfViewer);
  }
  if (pdfViewerModal) {
    pdfViewerModal.addEventListener('click', (e) => {
      if (e.target === pdfViewerModal) {
        closePdfViewer();
      }
    });
  }
  
  // Close PDF viewer on ESC key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (pdfViewerModal && !pdfViewerModal.classList.contains('hidden')) {
        closePdfViewer();
      } else if (promptModal && !promptModal.classList.contains('hidden')) {
        closePromptModal();
      }
    }
  });
  
  // --- System Controls Event Listeners (original) ---
  
  // Temperature slider
  if (temperatureSlider && temperatureValue) {
    temperatureSlider.addEventListener('input', (e) => {
      currentTemperature = parseFloat(e.target.value);
      temperatureValue.textContent = currentTemperature.toFixed(1);
    });
  }
  
  // Model selection
  if (modelSelect) {
    modelSelect.addEventListener('change', async (e) => {
      const selectedModel = e.target.value;
      if (selectedModel !== SELECTED_MODEL) {
        await changeModel(selectedModel);
      }
    });
  }
  
  // System prompt dropdown
  if (systemPromptSelect) {
    systemPromptSelect.addEventListener('change', (e) => {
      const selectedType = e.target.value;
      if (selectedType === 'custom') {
        openPromptModal();
      } else {
        updateSystemPrompt(selectedType);
      }
    });
  }
  
  // Modal controls
  if (editPromptBtn) {
    editPromptBtn.addEventListener('click', openPromptModal);
  }
  
  if (cancelPromptBtn) {
    cancelPromptBtn.addEventListener('click', closePromptModal);
  }
  
  if (savePromptBtn) {
    savePromptBtn.addEventListener('click', saveCustomPrompt);
  }
  
  // Close modal when clicking backdrop
  if (promptModal) {
    promptModal.addEventListener('click', (e) => {
      if (e.target === promptModal) {
        closePromptModal();
      }
    });
  }
});
