/**
 * Optional sentence encoder for the router, running in the browser (transformers.js + ONNX).
 *
 * The model (~120 MB, quantized) downloads once and stays in the browser cache.
 * Same scheme as model/lis_router.py: e5 models get the "query: " prefix and mean-pooled,
 * normalized embeddings. The model is multilingual, so English text also matches the Italian
 * dataset sentences. The browser uses the small e5 model; the Python router defaults to the
 * base one, so the cosine scores differ a little: calibrate tau for the dataset if needed.
 */
import type { SentenceEncoder } from './lisRouter';
import { loadPipeline, type ModelProgress } from './transformersCdn';

export const BROWSER_ENCODER_MODEL = 'Xenova/multilingual-e5-small';

type Extractor = (texts: string[], opts: { pooling: 'mean'; normalize: boolean }) => Promise<{ tolist(): number[][] }>;

export async function loadBrowserEncoder(onProgress?: (p: ModelProgress) => void): Promise<SentenceEncoder> {
  const extractor = await loadPipeline<Extractor>('feature-extraction', BROWSER_ENCODER_MODEL, onProgress);
  return {
    name: BROWSER_ENCODER_MODEL,
    async embed(texts: string[]) {
      const out: Float32Array[] = [];
      for (let i = 0; i < texts.length; i += 32) {
        const batch = texts.slice(i, i + 32).map((t) => `query: ${t}`);
        const res = await extractor(batch, { pooling: 'mean', normalize: true });
        out.push(...res.tolist().map((v) => Float32Array.from(v)));
      }
      return out;
    },
  };
}
