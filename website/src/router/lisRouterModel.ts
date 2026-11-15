/**
 * Motion generator backed by the LIS router and the exported dataset (public/lis/).
 *
 * text -> LisRouter (whole sentence, or words in LIS order) -> clips of the dataset samples
 * (already retargeted at export time) -> joined by the site with smooth transitions.
 * Everything runs in the browser: it works on a static host (GitHub Pages).
 *
 * The router is Italian (like model/lis_router.py). Other source languages are first mapped
 * to Italian through the dictionary phrases (planSigns); words with no dictionary phrase are
 * passed through and usually end up not signed.
 */
import type { DictionaryClass, SignAnimationPayload } from '../types/lis';
import type { GeneratedClip, GeneratedMotion, GlossSegment, MotionGenerator, MotionRequest } from '../generation/motionModel';
import { planSigns } from '../generation/signPlanner';
import { LisRouter, type LisIndex, type LisIndexEntry, type RouterOptions, type SentenceEncoder } from './lisRouter';

export const LIS_INDEX_PATH = 'lis/index.json';

/** State of the optional semantic sentence match (browser encoder). */
export type SemanticState =
  | { status: 'off' }
  | { status: 'loading'; fraction: number | null }
  | { status: 'on' }
  | { status: 'error'; message: string };

export class LisRouterModel implements MotionGenerator {
  readonly isMock = false;
  readonly router: LisRouter;
  private readonly baseUrl: string;
  private readonly clipCache = new Map<string, Promise<SignAnimationPayload>>();
  /** Italian dictionary classes (for non-Italian input), and the classes in the request language. */
  private dictionary: { base: DictionaryClass[]; bySource: (code: string) => DictionaryClass[] } | null = null;

  constructor(readonly index: LisIndex, baseUrl: string, options: Partial<RouterOptions> = {}) {
    this.router = new LisRouter(index, options);
    this.baseUrl = baseUrl;
  }

  /** Loads the exported index; null when the site has no exported dataset. */
  static async load(baseUrl: string, options: Partial<RouterOptions> = {}): Promise<LisRouterModel | null> {
    try {
      const res = await fetch(`${baseUrl}${LIS_INDEX_PATH}`);
      if (!res.ok) return null;
      const index = (await res.json()) as LisIndex;
      if (index?.version !== 1 || !Array.isArray(index.entries) || !index.entries.length) return null;
      return new LisRouterModel(index, `${baseUrl}lis/`, options);
    } catch {
      return null;
    }
  }

  get name(): string {
    const enc = this.router.encoderName;
    return `LIS router · ${this.index.source} dataset (${this.index.entries.length} clips)${enc ? ` · ${enc}` : ''}`;
  }

  setDictionary(base: DictionaryClass[], bySource: (code: string) => DictionaryClass[]): void {
    this.dictionary = { base, bySource };
  }

  setEncoder(encoder: SentenceEncoder | null): Promise<void> {
    return this.router.setEncoder(encoder);
  }

  /** Non-Italian text -> Italian, phrase by phrase through the dictionary. */
  toItalian(text: string, sourceLanguage: string): string {
    if (sourceLanguage === 'it' || !this.dictionary) return text;
    const classes = this.dictionary.bySource(sourceLanguage);
    const { signs, unmatched } = planSigns(text, classes);
    if (!signs.length) return text;
    const baseById = new Map(this.dictionary.base.map((c) => [c.id, c]));
    const parts = signs.map((s) => (baseById.get(s.cls.id)?.phrase ?? s.cls.phrase).replace(/\.\.\.|…/g, '').toLowerCase());
    return [...parts, ...unmatched].join(' ').trim() + (text.trim().endsWith('?') ? '?' : '');
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
    const italian = this.toItalian(request.text, request.sourceLanguage);
    const res = await this.router.translate(italian);
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const payloads = await Promise.all(res.entries.map((e) => this.clip(e)));
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const gloss = (s: string) => s.toUpperCase().replace(/\s+/g, '-');
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
