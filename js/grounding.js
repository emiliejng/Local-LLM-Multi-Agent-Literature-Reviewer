// Grounding (anti-hallucination).
// The model never writes citations itself: small models invent or misplace them.
// After generation, each sentence is matched to the closest passage the model actually received
// (cosine similarity on MiniLM embeddings). The citation points to that passage's page and
// quotes its most similar sentence. Figures are checked against the passages and invented
// reference lists are removed.

import { embed, cosineSimilarity, splitSentences, tokenize, paperById } from "./rag.js";
import { SUPPORT_THRESHOLD, SECOND_SOURCE_MARGIN } from "./config.js";

const NUMBER_REGEX = /\d+(?:[.,]\d+)*/g;
export const CITATION_MARKER = /⟦(\d+)⟧/g;

// "1,200" and "1200" are the same figure; "87,5" and "87.5" too
function normalizeNumber(raw) {
  return /^\d{1,3}(,\d{3})+$/.test(raw) ? raw.replace(/,/g, "") : raw.replace(",", ".");
}

export function cleanModelOutput(response) {
  return response
    .replace(/\[(source|excerpt)[^\]]*\]/gi, "") // Model-written tags, real or placeholder
    .replace(/\s*\((excerpt|extrait)s?\s*\d+[^)]*\)/gi, "") // "(Excerpt 7, page 3)"
    .replace(/\n+[*#\s]*(references|bibliography|works cited)[*:\s]*\n[\s\S]*$/i, "") // Invented reference lists
    .replace(/^\s*\(?note:[^\n]*$/gim, "") // Meta comments about how the instructions were followed
    .replace(/^\s*\(?(excerpt|extrait)\s*\d+[^\n]{0,40}$/gim, "") // Echoed excerpt labels
    .replace(/\b(excerpts?|extraits?) \d+(?:\s*(?:,|and|et|&)\s*\d+)*/gi, "the paper") // "According to Excerpt 1"
    .replace(/(^|[.!?]\s+)the paper/gm, "$1The paper")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

// Share of the sentence's words found in the passage: catches lists of names or terms
// that embeddings do not see as "similar in meaning"
function lexicalSupport(sentenceTokens, chunk) {
  if (sentenceTokens.length < 4) return 0;
  chunk.tokenSet ??= new Set(chunk.tokens);
  const overlap = sentenceTokens.filter(t => chunk.tokenSet.has(t)).length / sentenceTokens.length;
  return overlap >= 0.6 ? overlap * 0.6 : 0;
}

const shortTitle = title => {
  const words = title.split(/\s+/);
  return words.length > 6 ? `${words.slice(0, 6).join(" ")}…` : title;
};

export const stripCitations = text => text.replace(CITATION_MARKER, "").replace(/[ \t]{2,}/g, " ").trim();

// The sentence of the chunk closest to the claim: shown to the user as the quoted passage
async function bestQuote(chunk, claimVector) {
  if (!chunk.sentences) {
    chunk.sentences = splitSentences(chunk.text);
    chunk.sentenceVectors = await Promise.all(chunk.sentences.map(s => embed(s)));
  }
  let best = 0;
  chunk.sentenceVectors.forEach((vector, i) => {
    if (cosineSimilarity(claimVector, vector) > cosineSimilarity(claimVector, chunk.sentenceVectors[best])) best = i;
  });
  return chunk.sentences[best];
}

// Returns the text with ⟦n⟧ markers, the citations they point to, and a verification report.
// paperChunks maps a label such as "P1" to that paper's chunks, for sentences the model tagged [P1];
// paperTitles maps it to the title, for when the model uses the label as a noun ("While [P1] finds...").
export async function groundAnswer(response, providedChunks, paperChunks = {}, paperTitles = {}) {
  const citations = [];
  const citationIndex = new Map();
  const seen = new Set();
  let supported = 0;
  let unsupported = 0;
  const lines = [];

  for (const line of cleanModelOutput(response).split("\n")) {
    const out = [];
    // Sentence ends, but not after an initial ("George G. Malliaras")
    for (let sentence of line.split(/(?<=[.!?])(?<!\b[A-Z]\.)\s+(?=[A-Z"“(*•\[-])/)) {
      const labels = [...sentence.matchAll(/\[(P\d+)\]/g)].map(m => m[1]);
      sentence = sentence
        .replace(/\s*(?:\b(?:in|by|from|see)\s+)?\[P\d+\](?=\s*(?:\[P\d+\]|[.!?;:)]|$))/g, "")
        .replace(/\[(P\d+)\]/g, (_, label) => paperTitles[label] ? `“${shortTitle(paperTitles[label])}”` : "");

      // Small models repeat themselves
      const key = sentence.trim().toLowerCase();
      if (key.length > 20 && seen.has(key)) continue;
      seen.add(key);

      // Headings ("COMPARISON OF METHODS") and fragments carry no claim
      const plain = sentence.replace(/[*#•]/g, "").trim();
      if (plain.length < 25 || plain === plain.toUpperCase()) {
        out.push(sentence);
        continue;
      }

      const labelled = labels.flatMap(label => paperChunks[label] || []);
      const pool = labelled.length > 0 ? labelled : providedChunks;
      const vector = await embed(plain);
      const sentenceTokens = [...new Set(tokenize(plain))];
      const ranked = pool
        .map(chunk => ({ chunk, score: Math.max(cosineSimilarity(vector, chunk.embedding), lexicalSupport(sentenceTokens, chunk)) }))
        .filter(r => r.score >= SUPPORT_THRESHOLD)
        .sort((a, b) => b.score - a.score);

      // Best passage, plus a close runner-up from another page or paper:
      // a sentence that combines two facts or compares two papers cites both
      const cited = ranked.slice(0, 1);
      const runnerUp = ranked.find(r =>
        r.score >= ranked[0].score - SECOND_SOURCE_MARGIN &&
        (r.chunk.paperId !== ranked[0].chunk.paperId || r.chunk.page !== ranked[0].chunk.page));
      if (runnerUp) cited.push(runnerUp);

      if (cited.length === 0) {
        out.push(sentence);
        unsupported++;
        continue;
      }

      const markers = [];
      for (const { chunk } of cited) {
        const quote = await bestQuote(chunk, vector);
        const key = `${chunk.id}|${quote}`;
        if (!citationIndex.has(key)) {
          citationIndex.set(key, citations.length);
          const paper = paperById(chunk.paperId);
          citations.push({ chunkId: chunk.id, paperId: chunk.paperId, title: paper?.title || chunk.source, page: chunk.page, quote });
        }
        markers.push(`⟦${citationIndex.get(key)}⟧`);
      }
      out.push(`${sentence.trimEnd()} ${markers.join("")}`);
      supported++;
    }
    lines.push(out.join(" "));
  }
  const text = lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();

  // Figures must appear in the passages the model received
  const sourceNumbers = new Set((providedChunks.map(c => c.text).join(" ").match(NUMBER_REGEX) || []).map(normalizeNumber));
  const unsupportedNumbers = new Set();
  for (const raw of stripCitations(text).match(NUMBER_REGEX) || []) {
    // Small integers are list numbering or counts, not reported findings
    if (!/[.,]/.test(raw) && Number(raw) < 10) continue;
    if (!sourceNumbers.has(normalizeNumber(raw))) unsupportedNumbers.add(raw);
  }

  return { text, citations, report: { supported, unsupported, unsupportedNumbers: [...unsupportedNumbers] } };
}
