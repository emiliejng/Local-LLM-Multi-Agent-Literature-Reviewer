// RAG engine: sentence-aware chunking with page numbers, MiniLM embeddings,
// a pure-JS vector store, BM25 keyword index, and hybrid search.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";
import { EMBEDDING_MODEL, CHUNK_MAX_CHARS } from "./config.js";

env.allowLocalModels = false; // Models come from the Hugging Face CDN, not from this site

// The vector store: one entry per chunk { id, paperId, source, page, text, embedding, tokens }
export const store = { papers: [], chunks: [] };

let embedder = null;
let nextPaperId = 1;

export async function initEmbedder() {
  embedder = await pipeline("feature-extraction", EMBEDDING_MODEL);
}

export const isEmbedderReady = () => embedder !== null;

export async function embed(text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// similarity = Σ(Aᵢ·Bᵢ) / (√ΣAᵢ² · √ΣBᵢ²)
export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return normA === 0 || normB === 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Splits on sentence ends, but not after common abbreviations ("Fig. 3", "et al.", "e.g.")
export function splitSentences(text) {
  return text
    .replace(/\b(Fig|Figs|Eq|Eqs|Ref|Refs|al|e\.g|i\.e|vs|approx|No)\.\s/g, "$1.\u00A0")
    .split(/(?<=[.!?])(?<!\b[A-Z]\.)\s+(?=[A-Z(“"])/)
    .map(s => s.replace(/\u00A0/g, " ").trim())
    .filter(s => s.length > 0);
}

// Groups whole sentences into chunks of at most CHUNK_MAX_CHARS, one sentence of overlap,
// never across pages. Over-long "sentences" (tables, captions) are cut hard.
function chunkPages(pages) {
  const chunks = [];
  for (const page of pages) {
    const sentences = splitSentences(page.text).flatMap(s =>
      s.length <= CHUNK_MAX_CHARS ? [s] : s.match(new RegExp(`.{1,${CHUNK_MAX_CHARS}}(\\s|$)`, "g"))
    );
    let current = [];
    let length = 0;
    for (const sentence of sentences) {
      if (current.length > 0 && length + sentence.length > CHUNK_MAX_CHARS) {
        chunks.push({ page: page.number, text: current.join(" ") });
        current = current.length > 1 ? [current[current.length - 1]] : [];
        length = current.reduce((sum, s) => sum + s.length + 1, 0);
      }
      current.push(sentence);
      length += sentence.length + 1;
    }
    if (current.length > 0) chunks.push({ page: page.number, text: current.join(" ") });
  }
  return chunks.filter(c => c.text.length >= 40);
}

export async function addPaper(file, extracted, onProgress) {
  const paper = {
    id: nextPaperId++,
    name: file.name,
    title: extracted.title || file.name.replace(/\.pdf$/i, ""),
    authors: extracted.authors,
    year: extracted.year,
    pageCount: extracted.pageCount,
    url: URL.createObjectURL(file),
  };

  const rawChunks = chunkPages(extracted.pages);
  if (rawChunks.length === 0) throw new Error("No text could be extracted (scanned PDF?).");

  // One text per call: batching was slower in the browser (padding to the longest text)
  const embeddings = [];
  for (const raw of rawChunks) {
    embeddings.push(await embed(raw.text));
    onProgress?.(embeddings.length, rawChunks.length);
  }

  const chunks = rawChunks.map((raw, i) => ({
    id: `${paper.id}:${i}`,
    paperId: paper.id,
    index: i,
    source: paper.name,
    page: raw.page,
    text: raw.text,
    embedding: embeddings[i],
    tokens: tokenize(raw.text),
  }));

  paper.chunkCount = chunks.length;
  store.papers.push(paper);
  store.chunks.push(...chunks);
  rebuildKeywordIndex();
  return paper;
}

export function removePaper(paperId) {
  const paper = store.papers.find(p => p.id === paperId);
  if (paper) URL.revokeObjectURL(paper.url);
  store.papers = store.papers.filter(p => p.id !== paperId);
  store.chunks = store.chunks.filter(c => c.paperId !== paperId);
  rebuildKeywordIndex();
}

export const paperById = id => store.papers.find(p => p.id === id);
export const chunksOf = paperId => store.chunks.filter(c => c.paperId === paperId);

// --- BM25 keyword index: finds exact terms, names and numbers that embeddings miss ---
const STOPWORDS = new Set((
  "a an the and or of to in on for with by from at as is are was were be been this that these those it its " +
  "what which who whom how why when where does do did can could would should about into than then there their " +
  "paper study article le la les un une des du de et ou en dans sur pour par avec est sont ce cet cette ces " +
  "qui que quoi quel quelle quels comment pourquoi papier étude il elle ils elles au aux"
).split(" "));

export function tokenize(text) {
  // "Liu1" → "liu": affiliation and footnote markers stick to the word in PDFs
  return (text.toLowerCase().replace(/(\p{L}{2,})\d+\b/gu, "$1").match(/[\p{L}\p{N}][\p{L}\p{N}.-]*/gu) || [])
    .map(t => t.replace(/[.-]+$/, ""))
    .filter(t => t.length >= 2 && !STOPWORDS.has(t));
}

const keywordIndex = { df: new Map(), avgLength: 0 };

function rebuildKeywordIndex() {
  keywordIndex.df = new Map();
  let total = 0;
  for (const chunk of store.chunks) {
    total += chunk.tokens.length;
    for (const term of new Set(chunk.tokens)) keywordIndex.df.set(term, (keywordIndex.df.get(term) || 0) + 1);
  }
  keywordIndex.avgLength = store.chunks.length ? total / store.chunks.length : 0;
}

function bm25(queryTerms, chunk, k1 = 1.2, b = 0.75) {
  const n = store.chunks.length;
  let score = 0;
  for (const term of queryTerms) {
    const tf = chunk.tokens.filter(t => t === term).length;
    if (tf === 0) continue;
    const df = keywordIndex.df.get(term) || 0;
    const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * chunk.tokens.length / keywordIndex.avgLength));
  }
  return score;
}

// Hybrid search: dense (cosine) and keyword (BM25) rankings merged with Reciprocal Rank Fusion.
// A chunk is kept only if it is semantically close or shares a query term.
export async function search(query, { paperIds, k, keywordQuery = query }) {
  const pool = store.chunks.filter(c => !paperIds || paperIds.includes(c.paperId));
  if (pool.length === 0) return [];

  const queryVector = await embed(query);
  const queryTerms = [...new Set(tokenize(keywordQuery))];
  const scored = pool.map(chunk => ({
    chunk,
    dense: cosineSimilarity(queryVector, chunk.embedding),
    keyword: bm25(queryTerms, chunk),
  }));

  const rank = key => new Map([...scored].sort((a, b) => b[key] - a[key]).map((s, i) => [s.chunk.id, i]));
  const denseRank = rank("dense");
  const keywordRank = rank("keyword");

  return scored
    .filter(s => s.dense >= 0.2 || s.keyword > 0)
    .map(s => ({ ...s, score: 1 / (60 + denseRank.get(s.chunk.id)) + 1 / (60 + keywordRank.get(s.chunk.id)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}
