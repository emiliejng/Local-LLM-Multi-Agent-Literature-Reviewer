// PDF extraction with pdf.js: text per page, figure-label noise removed,
// reference list cut off, and conservative title/author/year detection.

const pdfjsLib = window.pdfjsLib;
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const REFERENCES_HEADING = /^\s*(\d+\.?\s*)?(references( and notes)?|bibliography|literature cited|works cited|références( bibliographiques)?)\s*$/i;

const rawLines = items => items.map(item => item.str + (item.hasEOL ? "\n" : "")).join("").split("\n");

function pageLines(items) {
  const lines = rawLines(items)
    .map(line => line.replace(/\s+/g, " ").trim())
    // Figure labels and axis ticks come out as tiny lines ("G", "On", "1 µm")
    // (the "References" heading is short too, but removeReferences needs it)
    .filter(line => line.length >= 12 || /[.!?:]$/.test(line) || /\p{L}{3,}\s+\S*\p{L}{3,}/u.test(line) || REFERENCES_HEADING.test(line));

  // Re-join words split across lines (soft hyphens, then "hy-\nphen")
  return lines
    .join("\n")
    .replace(/\u00AD\s*\n\s*/g, "")
    .replace(/\u00AD/g, "")
    .replace(/([a-z])-\n([a-z])/g, "$1$2")
    .split("\n");
}

// Removes the reference list (and what follows it: acknowledgments, funding...).
// Indexing it made the model quote cited papers' authors as the paper's own.
function removeReferences(pages) {
  const searchFrom = Math.floor(pages.length * 0.4);
  for (let p = searchFrom; p < pages.length; p++) {
    const index = pages[p].lines.findIndex(line => REFERENCES_HEADING.test(line));
    if (index !== -1) {
      pages[p].lines = pages[p].lines.slice(0, index);
      return pages.slice(0, p + 1);
    }
  }
  return pages;
}

// The title is usually the largest text of page 1. Rotated text (arXiv's side stamp) is ignored.
function largestText(items) {
  const sized = items
    .filter(item => item.str.trim() && item.transform[1] === 0 && item.transform[2] === 0)
    .map(item => ({ str: item.str, size: Math.abs(item.transform[3]) }));
  const max = Math.max(0, ...sized.map(item => item.size));
  return sized.filter(item => item.size >= max * 0.95).map(item => item.str).join(" ").replace(/\s+/g, " ").trim();
}

function detectTitle(pdfTitle, fontTitle, lines) {
  const title = (pdfTitle || "").trim();
  if (title.length >= 10 && !/untitled|microsoft|\.docx?$|\.pdf$|\.tex$|^doi|^paper$/i.test(title)) return title;
  if (fontTitle.length >= 10 && fontTitle.length <= 250 && fontTitle.split(" ").length >= 2) return fontTitle;

  for (let i = 0; i < Math.min(15, lines.length); i++) {
    const line = lines[i];
    if (line.length < 15 || line.length > 200 || line.split(" ").length < 3) continue;
    if (/^(page|www\.|http|doi|©|abstract|introduction|\d)/i.test(line) || line === line.toUpperCase()) continue;
    // Titles often wrap onto a second line that starts in lowercase
    const next = lines[i + 1] || "";
    return /^[a-z]/.test(next) && !/^by\s/.test(next) && next.length < 120 ? `${line} ${next}` : line;
  }
  return null;
}

// Authors: the longest run of "Firstname Lastname" items near the top of page 1.
// Uses raw lines because affiliation markers (¹, †, *) come out as separate fragments
// that break the author list into pieces. Empty when unsure.
const ORGANIZATION = /\b(google|brain|research|universit\w*|institut\w*|labs?|inc|ltd|department|school|college|facebook|microsoft|meta|deepmind|openai|corporation|center|centre)\b/i;

function detectAuthors(lines, title = "") {
  // \p{M} and \p{Sk}: accents that PDFs store as separate marks ("O˘guz")
  const namePattern = /^[A-Z][\p{L}\p{M}\p{Sk}'’.-]+(?:\s+[A-Z][\p{L}\p{M}\p{Sk}'’.-]*){1,3}$/u;
  const items = lines
    .slice(0, 60)
    .join("\n")
    .replace(/[\d†‡*∗⋆★§¶♠♣♦♥◦✝⁰¹²³⁴⁵⁶⁷⁸⁹?]+/g, "") // "?" is how PDF.js renders unknown marker glyphs
    .split(/,|\n|\band\b|&/)
    .map(item => item.trim())
    .filter(Boolean);

  let best = [];
  let run = [];
  for (const item of items) {
    // Title lines can look like names ("Knowledge-Intensive NLP Tasks")
    if (namePattern.test(item) && item.split(" ").length <= 4 && !ORGANIZATION.test(item) && !title.includes(item)) {
      run.push(item);
      if (run.length > best.length) best = [...run];
    } else {
      run = [];
    }
  }
  return best.length >= 2 ? best.slice(0, 30) : [];
}

// Running headers/footers ("Science 20 November 2025 824") repeat on most pages: drop them.
// Digits are ignored when comparing, since page numbers change.
function removeRepeatedLines(pages) {
  if (pages.length < 3) return;
  const key = line => line.replace(/\d+/g, "#");
  const counts = new Map();
  for (const page of pages) {
    for (const k of new Set(page.lines.map(key))) counts.set(k, (counts.get(k) || 0) + 1);
  }
  for (const page of pages) {
    page.lines = page.lines.filter(line => counts.get(key(line)) < pages.length / 2);
  }
}

// arXiv identifiers encode the first submission date ("arXiv:1706.03762" → 2017); the date on
// the side stamp is the latest revision. Otherwise: the first standalone year, ignoring identifiers.
function detectYear(text) {
  const arxiv = text.match(/arXiv:(\d{2})(\d{2})\.\d{4,5}/);
  if (arxiv && Number(arxiv[2]) >= 1 && Number(arxiv[2]) <= 12) return `20${arxiv[1]}`;
  const match = text.replace(/arXiv:\S+/g, "").match(/(?<![\d.])(19|20)\d{2}(?![\d.])/);
  return match && Number(match[0]) <= new Date().getFullYear() ? match[0] : null;
}

export async function extractPdf(file) {
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;

  let pdfTitle = "";
  try {
    pdfTitle = (await pdf.getMetadata()).info?.Title || "";
  } catch {
    // Metadata is optional
  }

  let pages = [];
  let firstPageRaw = [];
  let fontTitle = "";
  for (let n = 1; n <= pdf.numPages; n++) {
    const content = await (await pdf.getPage(n)).getTextContent();
    if (n === 1) {
      firstPageRaw = rawLines(content.items);
      fontTitle = largestText(content.items);
    }
    pages.push({ number: n, lines: pageLines(content.items) });
  }
  const pageCount = pages.length;

  // Metadata is read before running headers are removed: they often hold the journal and year
  const firstLines = pages[0]?.lines || [];
  const title = detectTitle(pdfTitle, fontTitle, firstLines);
  const metadata = {
    title,
    authors: detectAuthors(firstPageRaw, title || ""),
    year: detectYear(firstPageRaw.join(" ")),
  };

  removeRepeatedLines(pages);
  pages = removeReferences(pages);
  return { pageCount, pages: pages.map(p => ({ number: p.number, text: p.lines.join(" ") })), ...metadata };
}
