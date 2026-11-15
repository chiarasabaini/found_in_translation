/**
 * Text-to-motion model interface.
 *
 * The site sends the typed phrase to a motion model. The model returns the same files a
 * glove recording session produces (see README, "Motion model API"):
 *   left_hand.csv, right_hand.csv  (required)  solved hands, 26 joints, positions in meters
 *   body.csv                       (optional)  shoulders and elbows
 *   face.csv                       (optional)  ARKit blendshapes and a few face landmarks
 * The files go through the same parser and retargeting as real recordings.
 *
 * The model can answer in two forms:
 *   - `files`: one continuous recording of the whole phrase;
 *   - `clips`: one recording per sign, each starting and ending at rest. The site joins them
 *     with smooth transitions (src/generation/stitchClips.ts).
 *
 * The site uses the LIS router over the exported dataset (src/router/lisRouterModel.ts).
 * Set VITE_MOTION_MODEL_URL to use a model server over HTTP instead.
 */

import type { RecordingFileInput } from '../recording/rokokoRecording';
import type { SignAnimationPayload } from '../types/lis';

export interface MotionRequest {
  /** The phrase the user typed. */
  text: string;
  /** Source language code, e.g. "en" or "it". */
  sourceLanguage: string;
  /** Target sign language code, e.g. "LIS". */
  signLanguage: string;
}

/** One sign in the generated sequence. */
export interface GlossSegment {
  /** Words of the input phrase that produced this sign. */
  text: string;
  /** Gloss of the sign, e.g. "GRAZIE". */
  gloss: string;
  /** Dataset class or dictionary id of the sign, when known. */
  classId: string | null;
  startSec: number;
  endSec: number;
}

/** One sign as a separate rest-to-rest recording. */
export interface GeneratedClip {
  /** Words of the input phrase that produced this sign. */
  text: string;
  gloss: string;
  classId: string | null;
  /** Recording-format CSV files of this sign (empty when `payload` is given). */
  files: RecordingFileInput[];
  /** Already retargeted animation (exported dataset clips), instead of `files`. */
  payload?: SignAnimationPayload;
  /** Stroke inside the clip (s): after leaving rest, before returning. Detected when missing. */
  strokeStartSec?: number;
  strokeEndSec?: number;
}

export interface GeneratedMotion {
  /** Recording-format CSV files of the whole phrase (empty when the model returns clips). */
  files: RecordingFileInput[];
  /** One clip per sign, to be joined by the site. */
  clips?: GeneratedClip[];
  /** Signs in order, with timing (optional for real models). */
  segments: GlossSegment[];
  /** Words the model could not sign. */
  unmatched: string[];
  /** Model name and version, shown in the UI. */
  model: string;
  /** Face channels to overlay on the whole phrase: "negation", "question_wh", "question_yn". */
  modifiers?: string[];
  /** How the router handled the phrase (router-based generators only). */
  route?: { mode: string; matcher: string; score: number; matchedLabel: string | null; italian: string };
}

export interface MotionGenerator {
  readonly name: string;
  readonly isMock: boolean;
  generate(request: MotionRequest, signal?: AbortSignal): Promise<GeneratedMotion>;
}

/**
 * Client for a real model served over HTTP.
 * Request:  POST <url>  {"text", "sourceLanguage", "signLanguage"}
 * Response: {"files"?: [{"name", "text"}], "clips"?: GeneratedClip[], "segments"?: GlossSegment[],
 *            "unmatched"?: string[], "model"?: string}   ("files" or "clips" is required)
 */
export class HttpMotionModel implements MotionGenerator {
  readonly isMock = false;
  readonly name: string;
  private readonly url: string;

  constructor(url: string) {
    this.url = url;
    this.name = `HTTP model (${new URL(url, window.location.href).host})`;
  }

  async generate(request: MotionRequest, signal?: AbortSignal): Promise<GeneratedMotion> {
    const res = await fetch(this.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    });
    if (!res.ok) throw new Error(`Motion model error: HTTP ${res.status}`);
    const data = (await res.json()) as Partial<GeneratedMotion>;
    const validFiles = (files: unknown): files is RecordingFileInput[] =>
      Array.isArray(files) && files.every((f) => typeof f?.name === 'string' && typeof f?.text === 'string');
    const clips = Array.isArray(data.clips)
      ? data.clips.filter((c) => c && validFiles(c.files)).map((c) => ({
          text: String(c.text ?? ''),
          gloss: String(c.gloss ?? ''),
          classId: typeof c.classId === 'string' ? c.classId : null,
          files: c.files,
          strokeStartSec: Number.isFinite(c.strokeStartSec) ? c.strokeStartSec : undefined,
          strokeEndSec: Number.isFinite(c.strokeEndSec) ? c.strokeEndSec : undefined,
        }))
      : undefined;
    if (!validFiles(data.files) && !clips?.length) {
      throw new Error('Motion model response has no valid "files" or "clips" list.');
    }
    return {
      files: validFiles(data.files) ? data.files : [],
      clips,
      segments: Array.isArray(data.segments) ? data.segments : [],
      unmatched: Array.isArray(data.unmatched) ? data.unmatched : [],
      model: typeof data.model === 'string' ? data.model : this.name,
    };
  }
}

/** URL of the real model, if configured at build time. */
export function configuredModelUrl(): string | null {
  const url = (import.meta.env?.VITE_MOTION_MODEL_URL as string | undefined)?.trim();
  return url ? url : null;
}
