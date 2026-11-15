/**
 * Optional sentence encoder for the router, running in the browser (transformers.js + ONNX).
 *
 * transformers.js is loaded from the CDN only when the user turns semantic matching on, so the
 * page stays light and the site stays static (GitHub Pages). The model (~120 MB, quantized)
 * downloads once and stays in the browser cache.
 *
 * Same scheme as model/lis_router.py: e5 models get the "query: " prefix and mean-pooled,
 * normalized embeddings. The browser uses the small e5 model; the Python router defaults to the
 * base one, so the cosine scores differ a little: calibrate tau for the dataset if needed.
 */
import type { SentenceEncoder } from './lisRouter';

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
export const BROWSER_ENCODER_MODEL = 'Xenova/multilingual-e5-small';

export interface EncoderProgress {
  /** 0..1 over the model files, when known. */
  fraction: number | null;
  status: string;
}

type Extractor = (texts: string[], opts: { pooling: 'mean'; normalize: boolean }) => Promise<{ tolist(): number[][] }>;

export async function loadBrowserEncoder(onProgress?: (p: EncoderProgress) => void): Promise<SentenceEncoder> {
  const tf = (await import(/* @vite-ignore */ TRANSFORMERS_CDN)) as {
    pipeline: (task: string, model: string, opts: Record<string, unknown>) => Promise<Extractor>;
  };
  const loaded = new Map<string, number>();
  const total = new Map<string, number>();
  const extractor = await tf.pipeline('feature-extraction', BROWSER_ENCODER_MODEL, {
    dtype: 'q8',
    progress_callback: (e: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (e.file && e.total) {
        total.set(e.file, e.total);
        loaded.set(e.file, e.loaded ?? 0);
      }
      const t = [...total.values()].reduce((a, b) => a + b, 0);
      const l = [...loaded.values()].reduce((a, b) => a + b, 0);
      onProgress?.({ fraction: t ? l / t : null, status: e.status });
    },
  });
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
