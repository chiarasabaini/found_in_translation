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
 * Until the real model exists, MockMotionModel generates these files in the browser.
 * Set VITE_MOTION_MODEL_URL to use a real model over HTTP instead.
 */

import type { RecordingFileInput } from '../recording/rokokoRecording';

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
  /** Dictionary class, when the sign is in the dictionary. */
  classId: string | null;
  startSec: number;
  endSec: number;
}

export interface GeneratedMotion {
  /** Recording-format CSV files. */
  files: RecordingFileInput[];
  /** Signs in order, with timing (optional for real models). */
  segments: GlossSegment[];
  /** Words the model could not sign. */
  unmatched: string[];
  /** Model name and version, shown in the UI. */
  model: string;
}

export interface MotionGenerator {
  readonly name: string;
  readonly isMock: boolean;
  generate(request: MotionRequest, signal?: AbortSignal): Promise<GeneratedMotion>;
}

/**
 * Client for a real model served over HTTP.
 * Request:  POST <url>  {"text", "sourceLanguage", "signLanguage"}
 * Response: {"files": [{"name", "text"}], "segments"?: GlossSegment[], "unmatched"?: string[], "model"?: string}
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
    if (!Array.isArray(data.files) || !data.files.every((f) => typeof f?.name === 'string' && typeof f?.text === 'string')) {
      throw new Error('Motion model response has no valid "files" list.');
    }
    return {
      files: data.files,
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
