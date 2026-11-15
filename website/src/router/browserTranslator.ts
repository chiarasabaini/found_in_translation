/**
 * Translation to Italian in the browser (transformers.js + a Marian model, e.g.
 * Xenova/opus-mt-en-it, ~130 MB quantized, downloaded once and cached). The LIS router and the
 * dataset are Italian: other source languages go through this first.
 */
import { loadPipeline, type ModelProgress } from './transformersCdn';

export interface Translator {
  readonly model: string;
  toItalian(text: string): Promise<string>;
}

type TranslationPipeline = (text: string) => Promise<Array<{ translation_text: string }>>;

export async function loadBrowserTranslator(model: string, onProgress?: (p: ModelProgress) => void): Promise<Translator> {
  const translate = await loadPipeline<TranslationPipeline>('translation', model, onProgress);
  return {
    model,
    async toItalian(text: string) {
      const [out] = await translate(text);
      return restoreNames(text, (out?.translation_text ?? '').trim());
    },
  };
}

/** Capitalized words that do not start a sentence (likely proper names; "I" excluded). */
export function properNames(text: string): string[] {
  const names: string[] = [];
  let sentenceStart = true;
  for (const part of text.split(/\s+/).filter(Boolean)) {
    const m = part.match(/^[("'“]*(\p{Lu}[\p{L}'-]*)/u);
    if (m && !sentenceStart && m[1] !== 'I' && !/^I'/.test(m[1])) names.push(m[1]);
    sentenceStart = /[.!?:]["')”]*$/.test(part);
  }
  return names;
}

/**
 * Proper names must come out unchanged, but the translation model sometimes changes them
 * ("Chiara" -> "Calara"). A name missing from the output replaces the most similar
 * capitalized word of the output.
 */
export function restoreNames(source: string, translated: string): string {
  let out = translated;
  for (const name of properNames(source)) {
    if (new RegExp(`\\b${name}\\b`, 'u').test(out)) continue;
    const words = out.match(/\p{Lu}[\p{L}'-]*/gu) ?? [];
    let best: string | null = null;
    let bestScore = 0.5;
    for (const w of words) {
      const score = similarity(w.toLowerCase(), name.toLowerCase());
      if (score > bestScore) [best, bestScore] = [w, score];
    }
    if (best) out = out.replace(best, name);
  }
  return out;
}

function similarity(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return 1 - prev[b.length] / Math.max(a.length, b.length, 1);
}
