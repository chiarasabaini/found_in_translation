/**
 * transformers.js, loaded from the CDN on first use (not an npm dependency: it would pull
 * native packages into the build). Used by the optional sentence encoder and the translator.
 */
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

export interface ModelProgress {
  /** 0..1 over the model files, when known. */
  fraction: number | null;
  status: string;
}

type Pipeline = (task: string, model: string, opts: Record<string, unknown>) => Promise<unknown>;

let modulePromise: Promise<{ pipeline: Pipeline }> | null = null;

/** Creates a transformers.js pipeline; the model downloads once and stays in the browser cache. */
export async function loadPipeline<T>(task: string, model: string, onProgress?: (p: ModelProgress) => void): Promise<T> {
  modulePromise ??= import(/* @vite-ignore */ TRANSFORMERS_CDN) as Promise<{ pipeline: Pipeline }>;
  let tf: { pipeline: Pipeline };
  try {
    tf = await modulePromise;
  } catch (err) {
    modulePromise = null;
    throw err;
  }
  const loaded = new Map<string, number>();
  const total = new Map<string, number>();
  return (await tf.pipeline(task, model, {
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
  })) as T;
}
