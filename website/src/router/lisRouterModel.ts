/**
 * Motion generator backed by the LIS router and the exported dataset (public/lis/).
 *
 * text -> (translation to Italian) -> LisRouter (whole sentence, or words in LIS order) ->
 * clips of the dataset samples (already retargeted at export time) -> joined by the site with
 * smooth transitions. Everything runs in the browser: it works on a static host (GitHub Pages).
 */
import type { SignAnimationPayload } from '../types/lis';
import type { GeneratedClip, GeneratedMotion, GlossSegment, MotionGenerator, MotionRequest } from '../generation/motionModel';
import { LisRouter, type LisIndex, type LisIndexEntry, type RouterOptions, type SentenceEncoder } from './lisRouter';
import type { Translator } from './browserTranslator';

/** State of a model loaded in the browser (sentence encoder, translator). */
export type ModelState =
  | { status: 'off' }
  | { status: 'loading'; fraction: number | null }
  | { status: 'on' }
  | { status: 'error'; message: string };

export class LisRouterModel implements MotionGenerator {
  readonly isMock = false;
  readonly router: LisRouter;
  private readonly baseUrl: string;
  private readonly clipCache = new Map<string, Promise<SignAnimationPayload>>();
  private readonly translators = new Map<string, Translator>();

  constructor(readonly index: LisIndex, baseUrl: string, options: Partial<RouterOptions> = {}) {
    this.router = new LisRouter(index, options);
    this.baseUrl = baseUrl;
  }

  /** Loads the exported index; null when the site has no exported dataset. */
  static async load(indexUrl: string, options: Partial<RouterOptions> = {}): Promise<LisRouterModel | null> {
    try {
      const res = await fetch(indexUrl);
      if (!res.ok) return null;
      const index = (await res.json()) as LisIndex;
      if (index?.version !== 1 || !Array.isArray(index.entries) || !index.entries.length) return null;
      return new LisRouterModel(index, indexUrl.replace(/[^/]*$/, ''), options);
    } catch {
      return null;
    }
  }

  get name(): string {
    const enc = this.router.encoderName;
    return `LIS router · ${this.index.source} (${this.index.entries.length} clips)${enc ? ` · ${enc}` : ''}`;
  }

  setEncoder(encoder: SentenceEncoder | null): Promise<void> {
    return this.router.setEncoder(encoder);
  }

  /** Translator for a source language (to Italian). */
  setTranslator(sourceLanguage: string, translator: Translator | null): void {
    if (translator) this.translators.set(sourceLanguage, translator);
    else this.translators.delete(sourceLanguage);
  }

  hasTranslator(sourceLanguage: string): boolean {
    return sourceLanguage === 'it' || this.translators.has(sourceLanguage);
  }

  private clip(entry: LisIndexEntry): Promise<SignAnimationPayload> {
    let p = this.clipCache.get(entry.id);
    if (!p) {
      p = fetch(`${this.baseUrl}${entry.clip}`).then((res) => {
        if (!res.ok) throw new Error(`Cannot load the clip ${entry.clip}`);
        return res.json() as Promise<SignAnimationPayload>;
      });
      p.catch(() => this.clipCache.delete(entry.id));
      this.clipCache.set(entry.id, p);
    }
    return p;
  }

  async generate(request: MotionRequest, signal?: AbortSignal): Promise<GeneratedMotion> {
    let italian = request.text;
    if (request.sourceLanguage !== 'it') {
      const translator = this.translators.get(request.sourceLanguage);
      if (!translator) throw new Error('The translator to Italian is still loading.');
      italian = await translator.toItalian(request.text);
    }
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    // The encoder is multilingual: the original text also takes part in the sentence match.
    const res = await this.router.translate(italian, request.sourceLanguage !== 'it' ? [request.text] : []);
    const payloads = await Promise.all(res.entries.map((e) => this.clip(e)));
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const gloss = (s: string) => s.toUpperCase().replace(/[?!.,]/g, '').trim().replace(/\s+/g, '-');
    const clips: GeneratedClip[] = res.entries.map((e, i) => ({
      text: e.sentence,
      gloss: gloss(e.sentence),
      classId: null,
      files: [],
      payload: payloads[i],
      strokeStartSec: e.strokeStartSec,
      strokeEndSec: e.strokeEndSec,
    }));
    const segments: GlossSegment[] = clips.map((c) => ({ text: c.text, gloss: c.gloss, classId: null, startSec: 0, endSec: 0 }));
    return {
      files: [],
      clips,
      segments,
      unmatched: res.skipped,
      modifiers: res.modifiers,
      model: this.name,
      route: { mode: res.mode, matcher: res.matcher, score: res.score, matchedLabel: res.matchedLabel, italian },
    };
  }
}
