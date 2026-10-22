// The research agent: answers questions from the papers and writes literature reviews.
// Each answer goes through: query rewriting → hybrid retrieval → generation → grounding.

import { store, search, chunksOf, paperById } from "./rag.js";
import { complete, ask, settings } from "./llm.js";
import { groundAnswer, cleanModelOutput, stripCitations } from "./grounding.js";
import { detectLanguage, LANGUAGE_NAMES } from "./language.js";
import {
  TOP_K, CONTEXT_CHAR_BUDGET, MAX_HISTORY_MESSAGES, HISTORY_MESSAGE_MAX_CHARS, ANSWER_MAX_TOKENS,
  REVIEW_CHUNKS_PER_PAPER, REVIEW_SUMMARIES_MAX_CHARS,
} from "./config.js";

export function isLiteratureReviewRequest(text) {
  return /\b(literature|lit\.?)\s+review\b|revue de (la )?litt[ée]rature|state of the art|[ée]tat de l'art/i.test(text);
}

// "Who wrote this paper?" is answered from the detected metadata: faster and more reliable than the model
// (letter lookarounds instead of \b, which does not treat "é" as a letter)
const AUTHORS_QUESTION = /(?<!\p{L})(authors?|wrote|written by|auteurs?|[ée]crit|r[ée]dig[ée])(?!\p{L})/iu;

// Questions about the paper as a whole are answered by its first page (title, authors, abstract)
const PAPER_LEVEL_QUESTION = /\b(authors?|wrote|written|title|about|idea|summar\w*|main|contribution|abstract|overview|goal|objective|purpose|this (paper|article|study|work)|auteurs?|titre|r[ée]sum\w*|id[ée]e|sujet|objectif|ce papier|cet article|cette [ée]tude)\b/i;

export function paperLabel(paper) {
  const authors = paper.authors?.length
    ? ` by ${paper.authors.slice(0, 3).join(", ")}${paper.authors.length > 3 ? " et al." : ""}`
    : "";
  return `"${paper.title}"${authors}${paper.year ? ` (${paper.year})` : ""}`;
}

// Full metadata for the prompt: the model reads authors here more reliably than in the
// first-page text, where affiliation markers clutter the names ("Liu1 †, Jing Bai 1 †")
const paperDetails = paper =>
  `- Title: "${paper.title}"${paper.authors?.length ? ` — Authors: ${paper.authors.join(", ")}` : ""}${paper.year ? ` — Year: ${paper.year}` : ""}`;

// System prompt + recent turns, shortened so the window stays free for the excerpts
function recentHistory(history) {
  const recent = history.slice(-MAX_HISTORY_MESSAGES).map(m => ({
    role: m.role,
    content: m.content.length > HISTORY_MESSAGE_MAX_CHARS ? m.content.slice(0, HISTORY_MESSAGE_MAX_CHARS) + "…" : m.content,
  }));
  while (recent.length > 0 && recent[0].role !== "user") recent.shift();
  return [{ role: "system", content: settings.systemPrompt }, ...recent];
}

// Turns follow-ups ("and its limits?") and non-English questions into a standalone English
// search query, since the embedding model is English-only.
async function rewriteQuery(question, history, papers) {
  const context = history.slice(-4).map(m => `${m.role}: ${stripCitations(m.content).slice(0, 200)}`).join("\n");
  try {
    const rewritten = await ask(
      "You rewrite questions into standalone search queries for a search engine over scientific papers.",
      `The user is reading: ${papers.map(p => `"${p.title}"`).join("; ")}
${context ? `\nConversation so far:\n${context}\n` : ""}
Last question: ${question}

Rewrite the last question as ONE standalone search query in English. Use the conversation only to resolve references such as "it" or "this paper"; if the question is already self-contained, just translate it. Keep its meaning and do not add topics it does not ask about. Output only the query.`,
      { maxTokens: 48, temperature: 0 }
    );
    const query = rewritten.split("\n")[0].replace(/^(search )?query:\s*/i, "").replace(/^["']|["']$/g, "").trim();
    return query.length >= 3 ? query : question;
  } catch (err) {
    console.warn("Query rewriting failed:", err);
    return question;
  }
}

// Interleaves two rankings without duplicates: 1st of each, then 2nd of each...
function mergeResults(a, b) {
  const merged = [];
  const seen = new Set();
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    for (const r of [a[i], b[i]]) {
      if (r && !seen.has(r.chunk.id)) {
        seen.add(r.chunk.id);
        merged.push(r);
      }
    }
  }
  return merged;
}

function buildContext(chunks) {
  const used = [];
  let context = "";
  for (const chunk of chunks) {
    const paper = paperById(chunk.paperId);
    const entry = `[Excerpt ${used.length + 1} — "${paper.title}", page ${chunk.page}]\n"${chunk.text}"\n\n`;
    if (used.length > 0 && context.length + entry.length > CONTEXT_CHAR_BUDGET) break;
    context += entry;
    used.push(chunk);
  }
  return { context, used };
}

function groundedPrompt(question, papers, context, language) {
  // The question appears before and after the excerpts: small models otherwise drift to the last excerpt
  return `Question: ${question}

PAPERS:
${papers.map(paperDetails).join("\n")}

DOCUMENT EXCERPTS (the only allowed sources):
${context}
Instructions:
1. Answer the question directly in your own words, in 2 to 6 sentences, using ONLY the excerpts. Do not copy the excerpt list.
2. Copy numbers exactly as they appear in the excerpts. Never compute new numbers.
3. Ignore excerpts that are unrelated to the question. If none answers it, say so plainly instead of guessing.
4. Never mention papers, authors or results that are not in the PAPERS list or the excerpts. For questions about authors, title or year, use the PAPERS list.
5. Do not add citations, excerpt numbers or a reference list: sources are attached automatically. Refer to "the paper(s)", never to "the excerpts".
6. Write the whole answer in ${LANGUAGE_NAMES[language]}, even though the excerpts are in another language.

Question: ${question}
Answer (in ${LANGUAGE_NAMES[language]}):`;
}

export async function answerQuestion(question, { history, paperIds, onStatus, onToken }) {
  const language = detectLanguage(question);
  if (store.papers.length === 0) {
    const messages = recentHistory(history);
    messages.push({
      role: "user",
      content: "No documents are uploaded yet. Answer briefly from general knowledge and say that it is general knowledge. " +
               `Do not cite papers and do not invent references. Answer in ${LANGUAGE_NAMES[language]}.\n\nQuestion: ` + question,
    });
    const answer = await complete(messages, { maxTokens: ANSWER_MAX_TOKENS, onToken });
    return { text: `💬 General knowledge (no paper uploaded)\n\n${cleanModelOutput(answer)}`, citations: [], report: null, language };
  }

  const papers = paperIds.map(paperById);
  if (papers.length === 1 && papers[0].authors.length >= 2 && AUTHORS_QUESTION.test(question)) {
    const names = papers[0].authors.join(", ");
    const text = language === "fr"
      ? `« ${papers[0].title} » a été écrit par ${names}.`
      : `"${papers[0].title}" was written by ${names}.`;
    return { ...(await groundAnswer(text, chunksOf(papers[0].id).slice(0, 1))), language };
  }

  onStatus?.("Understanding the question…");
  const query = await rewriteQuery(question, history, papers);

  onStatus?.("Searching the papers…");
  // Search with both the rewritten query and the original question: a poor rewrite
  // (e.g. pulled off-topic by a long previous answer) must not hide the right passage
  const [rewrittenResults, originalResults] = await Promise.all([
    search(query, { paperIds, k: TOP_K, keywordQuery: `${question} ${query}` }),
    search(question, { paperIds, k: TOP_K }),
  ]);
  const results = mergeResults(rewrittenResults, originalResults).slice(0, TOP_K);

  // First page of the paper(s): needed for "who are the authors", "what is the main idea"...
  const wantsOverview = paperIds.length === 1 || PAPER_LEVEL_QUESTION.test(`${question} ${query}`);
  const headers = wantsOverview ? paperIds.slice(0, 4).map(id => chunksOf(id)[0]).filter(Boolean) : [];
  const chunks = [...headers, ...results.map(r => r.chunk).filter(c => !headers.includes(c))];

  if (chunks.length === 0) {
    const text = language === "fr"
      ? "Je n'ai trouvé aucun passage lié à cette question dans le(s) papier(s) sélectionné(s), donc je ne réponds pas de mémoire. Essayez de reformuler avec les termes du papier."
      : "I could not find any passage related to this question in the selected paper(s), so I will not answer from memory. Try rephrasing with terms used in the paper.";
    return { text, citations: [], report: null, language };
  }

  const { context, used } = buildContext(chunks);
  const messages = recentHistory(history);
  messages.push({ role: "user", content: groundedPrompt(question, papers, context, language) });

  onStatus?.("Writing the answer…");
  const answer = await complete(messages, { maxTokens: ANSWER_MAX_TOKENS, onToken });
  onStatus?.("Checking sources…");
  return { ...(await groundAnswer(answer, used)), language };
}

// --- Literature review: summarize each paper from its own chunks, then synthesize ---

const REFUSAL = /^\s*(i can['’]?t|i cannot|i'm unable|i am unable|i'm sorry|sorry)/i;
const REVIEW_QUERY = "research objective, methodology, data, key results with numbers, conclusions and limitations";

async function reviewChunks(paperId, topic) {
  const best = await search(`${REVIEW_QUERY}. ${topic}`, { paperIds: [paperId], k: REVIEW_CHUNKS_PER_PAPER });
  const header = chunksOf(paperId)[0];
  const selected = [header, ...best.map(r => r.chunk).filter(c => c !== header)].slice(0, REVIEW_CHUNKS_PER_PAPER);
  return selected.sort((a, b) => a.index - b.index);
}

async function summarizePaper(paper, chunks, language) {
  const excerpts = chunks.map((c, i) => `[Excerpt ${i + 1}, page ${c.page}]\n"${c.text}"`).join("\n\n");
  const summary = await ask(
    "You are an academic researcher. You summarize ONE paper using ONLY the excerpts provided. Never invent facts, numbers, authors or references.",
    `Paper: ${paperLabel(paper)}

Excerpts:
${excerpts}

Summarize this paper in exactly 5 bullet points, one sentence each:
- Objective:
- Method / data:
- Key results (include every number from the excerpts, copied exactly):
- Conclusion:
- Limitations (only if stated):
If an element is not in the excerpts, write "not stated". Output only the 5 bullets, with no introduction and no citations. Write the bullets in ${LANGUAGE_NAMES[language]}.`,
    { maxTokens: 350, temperature: Math.min(settings.temperature, 0.3) }
  );
  // Keep only the bullets (drops "Here is a summary..." preambles)
  const bullets = summary.split("\n").filter(line => /^\s*([-•*]|\d+\.)\s/.test(line));
  return cleanModelOutput(bullets.length > 0 ? bullets.join("\n") : summary);
}

export async function literatureReview(request, { history, paperIds, onStatus }) {
  const language = detectLanguage(request);
  const previousQuestions = history
    .filter(m => m.role === "user" && !isLiteratureReviewRequest(m.content))
    .slice(-3)
    .map(m => m.content.slice(0, 200));
  const topic = [...previousQuestions, request].join(" | ");

  // Map: one summary per paper, grounded in its own excerpts
  const summaries = [];
  for (let i = 0; i < paperIds.length; i++) {
    const paper = paperById(paperIds[i]);
    onStatus?.(`Summarizing paper ${i + 1}/${paperIds.length}: ${paper.title}`);
    const chunks = await reviewChunks(paper.id, topic);
    try {
      const summary = await summarizePaper(paper, chunks, language);
      summaries.push({ paper, chunks, summary, grounded: await groundAnswer(summary, chunks) });
    } catch (err) {
      console.warn(`Summary failed for ${paper.name}:`, err);
    }
  }
  if (summaries.length === 0) throw new Error("No paper could be summarized.");

  const notes = summaries.map(s => ({ title: paperLabel(s.paper), ...s.grounded }));
  if (summaries.length === 1) {
    const text = language === "fr" ? "Un seul papier est sélectionné : voici son résumé structuré." : "Only one paper is selected, so here is its structured summary.";
    return { text, citations: [], report: null, notes, language };
  }

  // Reduce: synthesis across papers. Short labels [P1] are easier for a small model than titles,
  // and they scope the citation matching to the right paper.
  onStatus?.(`Writing the synthesis of ${summaries.length} papers…`);
  const paperChunks = Object.fromEntries(summaries.map((s, i) => [`P${i + 1}`, s.chunks]));
  const perSummaryMax = Math.floor(REVIEW_SUMMARIES_MAX_CHARS / summaries.length);
  const summaryBlock = summaries
    .map((s, i) => `=== [P${i + 1}] ${paperLabel(s.paper)} ===\n${s.summary.slice(0, perSummaryMax)}`)
    .join("\n\n");

  const prompt = `User request / topic of interest: ${topic}

Summaries of the ${summaries.length} papers:

${summaryBlock}

Write a concise, structured literature review with these sections:
INTRODUCTION — the research area and the papers covered (2-3 sentences).
THEMATIC SYNTHESIS — group the papers by shared themes and contrast them (e.g. "While <title 1> finds X, <title 2> argues Y").
COMPARISON OF METHODS AND RESULTS — compare approaches and cite specific numbers from the summaries.
GAPS AND CONCLUSION — open questions and an overall conclusion.

Rules:
- Refer to papers by their title, and end each sentence about a paper with its label, e.g. [P1] or [P1][P2].
- Copy numbers exactly from the summaries. Never compute new numbers.
- Discuss ONLY these ${summaries.length} papers; do not simply list them one after another.
- Do not write a references list or citations: sources are attached automatically.
- Write the whole review in ${LANGUAGE_NAMES[language]}.`;

  // Small Llama models sometimes refuse harmless long prompts ("I can't fulfill your request"):
  // retry once with a plainer system prompt, then fall back to the per-paper notes.
  let synthesis = await ask(
    "You are an expert academic researcher. Use ONLY the paper summaries provided. Never mention papers, authors, numbers or results that are not in them.",
    prompt,
    { maxTokens: 900, temperature: Math.min(settings.temperature, 0.3) }
  );
  if (REFUSAL.test(synthesis)) {
    synthesis = await ask("You are a helpful assistant who writes literature reviews from paper summaries.", prompt, { maxTokens: 900, temperature: 0.5 });
  }
  if (REFUSAL.test(synthesis)) {
    return { text: "⚠️ The model did not manage to write the synthesis. Here are the per-paper notes; try again or switch model in Settings.", citations: [], report: null, notes, language: "en" };
  }

  onStatus?.("Checking sources…");
  const paperTitles = Object.fromEntries(summaries.map((s, i) => [`P${i + 1}`, s.paper.title]));
  const grounded = await groundAnswer(synthesis, summaries.flatMap(s => s.chunks), paperChunks, paperTitles);
  return { ...grounded, notes, language };
}

