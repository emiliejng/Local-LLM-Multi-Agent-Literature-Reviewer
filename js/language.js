// Language of a question or answer (French or English): the answer is written in the language of the
// question, and read aloud with a voice of that language. Small models do not follow
// "answer in the language of the question" once the excerpts are in English, so it is set explicitly.

export const LANGUAGE_NAMES = { fr: "French", en: "English" };

export function detectLanguage(text) {
  const lower = text.toLowerCase();
  const count = pattern => (lower.match(pattern) || []).length;
  const french =
    count(/\b(le|la|les|des|est|une|un|et|que|qui|quoi|quel|quelle|quels|dans|pour|sur|pas|du|ce|cet|cette|avec|sont|comment|pourquoi|papier|article|auteurs?)\b/g) +
    count(/[éèêàùçôîœ]/g);
  const english = count(/\b(the|and|is|are|of|to|in|that|for|with|this|it|was|on|by|what|who|which|how|why|paper)\b/g);
  return french > english ? "fr" : "en";
}
