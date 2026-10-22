// Central configuration: models, chunking, retrieval and prompt settings.

export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const WHISPER_MODEL = "Xenova/whisper-tiny";

export const MODELS = {
  "Llama-3.2-3B-Instruct-q4f32_1-MLC": "Llama 3.2 3B (recommended, ~1.5 GB)",
  "Llama-3.2-1B-Instruct-q4f32_1-MLC": "Llama 3.2 1B (fast, ~600 MB)",
};
export const DEFAULT_MODEL = "Llama-3.2-3B-Instruct-q4f32_1-MLC";

// Chunks are built from whole sentences and never cross a page, so every chunk has one page number
export const CHUNK_MAX_CHARS = 700;

// Retrieval
export const TOP_K = 6;
export const CONTEXT_CHAR_BUDGET = 5000; // Excerpts sent to the model; keeps prompt + answer inside a 4k-token window
export const MAX_HISTORY_MESSAGES = 6;
export const HISTORY_MESSAGE_MAX_CHARS = 500;
export const ANSWER_MAX_TOKENS = 700;

// Grounding: a sentence is cited only if a passage is at least this similar to it
export const SUPPORT_THRESHOLD = 0.3;
export const SECOND_SOURCE_MARGIN = 0.08; // Another paper scoring this close to the best one is cited too

// Literature review
export const REVIEW_CHUNKS_PER_PAPER = 5;
export const REVIEW_SUMMARIES_MAX_CHARS = 6000;

export const DEFAULT_TEMPERATURE = 0.3;
export const DEFAULT_SYSTEM_PROMPT =
  "You are an academic research assistant helping a user read and review scientific papers. " +
  "When DOCUMENT EXCERPTS are provided, answer ONLY from them, copy numbers exactly, " +
  "and say clearly when the excerpts do not contain the answer. " +
  "Never invent papers, authors, results, numbers or citations. " +
  "Sources are attached automatically: do not write citations or reference lists. " +
  "Write clearly and concisely, in the language of the user's question.";
