// WebLLM wrapper: model loading/switching, streaming completions and user-tunable settings.

import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from "./config.js";

export const settings = {
  temperature: DEFAULT_TEMPERATURE,
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
};

let engine = null;
let currentModel = null;

export const isModelReady = () => engine !== null;
export const getCurrentModel = () => currentModel;

export async function loadModel(modelId, onProgress) {
  if (engine) {
    try {
      await engine.unload();
    } catch (err) {
      console.warn("Failed to unload previous model:", err);
    }
    engine = null;
  }
  currentModel = modelId;
  engine = await CreateMLCEngine(modelId, {
    initProgressCallback: progress => onProgress?.(Math.round(progress.progress * 100)),
  });
}

// One completion. With onToken the answer is streamed as it is generated.
export async function complete(messages, { maxTokens = 700, temperature = settings.temperature, onToken } = {}) {
  if (!engine) throw new Error("The language model is not loaded yet.");

  if (!onToken) {
    const reply = await engine.chat.completions.create({ messages, temperature, max_tokens: maxTokens });
    return (reply.choices[0].message.content || "").trim();
  }

  const stream = await engine.chat.completions.create({ messages, temperature, max_tokens: maxTokens, stream: true });
  let text = "";
  for await (const part of stream) {
    const delta = part.choices[0]?.delta?.content || "";
    if (delta) {
      text += delta;
      onToken(text);
    }
  }
  return text.trim();
}

// System prompt + a user message: the shape every one-shot task uses
export const ask = (system, user, options) =>
  complete([{ role: "system", content: system }, { role: "user", content: user }], options);
