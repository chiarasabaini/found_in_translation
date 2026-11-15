/**
 * Dataset -> static files for the site (Node only: used by scripts/export-lis-dataset.ts and
 * the tests, never by the browser).
 *
 * Reads every sample folder under the dataset root (metadata.json + left_hand.csv,
 * right_hand.csv, face.csv, body.csv, the layout of dataset/ and model/lis_router.py) and
 * retargets it with the same code the site uses. The result is an index (sentences, word
 * lemmas and POS, clip paths, stroke times) and one compact clip per sample.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join, relative } from 'node:path';
import { createHash } from 'node:crypto';
import { parseRokokoRecording, type RecordingFileInput } from '../recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, retargetRecording } from '../recording/recordingRetarget';
import { detectStroke } from '../generation/stitchClips';
import type { SignAnimationPayload } from '../types/lis';
import { guessPos, normalizeText, tokens, type LisIndex, type LisIndexEntry } from './lisRouter';

export interface ExportOptions {
  /** Name shown in the UI. */
  source: string;
  /** Glove yaw in degrees, or null for the automatic estimate per sample. */
  gloveYawDeg: number | null;
  /** Only the first N samples (tests). */
  limit?: number;
  /** Only samples whose sentence is in this list (tests). */
  sentences?: string[];
  onWarning?: (message: string) => void;
}

export interface ExportResult {
  index: LisIndex;
  /** Clip path (as in the index) -> retargeted animation. */
  clips: Map<string, SignAnimationPayload>;
}

/** Sample folders (folders with a metadata.json), sorted. */
export function findSamples(dir: string): string[] {
  const found: string[] = [];
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (!statSync(p).isDirectory()) continue;
    if (existsSync(join(p, 'metadata.json'))) found.push(p);
    else found.push(...findSamples(p));
  }
  return found;
}

const round = (x: number, k: number) => Math.round(x * k) / k;

export function exportDataset(root: string, opts: ExportOptions): ExportResult {
  const warn = opts.onWarning ?? (() => undefined);
  const entries: LisIndexEntry[] = [];
  const clips = new Map<string, SignAnimationPayload>();
  const wanted = opts.sentences ? new Set(opts.sentences.map(normalizeText)) : null;

  for (const dir of findSamples(root)) {
    if (opts.limit !== undefined && entries.length >= opts.limit) break;
    let meta: Record<string, unknown>;
    try {
      meta = JSON.parse(readFileSync(join(dir, 'metadata.json'), 'utf8'));
    } catch {
      warn(`skip ${dir}: invalid metadata.json`);
      continue;
    }
    const sentence = typeof meta.sentence === 'string' ? meta.sentence.trim() : '';
    const norm = normalizeText(sentence);
    if (!norm) {
      warn(`skip ${dir}: no "sentence"`);
      continue;
    }
    if (wanted && !wanted.has(norm)) continue;
    const files: RecordingFileInput[] = ['left_hand.csv', 'right_hand.csv', 'face.csv', 'body.csv']
      .filter((f) => existsSync(join(dir, f)))
      .map((f) => ({ name: f, text: readFileSync(join(dir, f), 'utf8') }));
    const rec = parseRokokoRecording(files, basename(dir));
    const error = rec.diagnostics.find((d) => d.level === 'error');
    if (error) {
      warn(`skip ${dir}: ${error.message}`);
      continue;
    }
    const { payload } = retargetRecording(rec, {
      ...DEFAULT_RETARGET_OPTIONS,
      gloveYawDeg: opts.gloveYawDeg,
      startSec: 0,
      endSec: rec.durationSec,
      classId: basename(dir),
      phrase: sentence,
      category: 'Dataset',
      description: '',
    });
    // Smaller files: no per-frame QA data, 4 decimals for quaternions, 3 for morphs.
    payload.qaComparison = { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] };
    delete payload.source;
    for (const k of Object.keys(payload.boneTracks)) payload.boneTracks[k] = payload.boneTracks[k].map((x) => round(x, 1e4));
    for (const k of Object.keys(payload.morphTracks)) payload.morphTracks[k] = payload.morphTracks[k].map((x) => round(x, 1e3));

    const hash = createHash('sha1').update(relative(root, dir)).digest('hex').slice(0, 6);
    const id = `${basename(dir).replace(/[^\w-]+/g, '_')}-${hash}`;
    const kind = meta.type === 'word' || meta.type === 'sentence' ? meta.type : tokens(norm).length === 1 ? 'word' : 'sentence';
    const stroke = detectStroke(payload);
    const entry: LisIndexEntry = {
      id,
      sentence,
      norm,
      kind,
      clip: `clips/${id}.json`,
      duration: payload.duration,
      strokeStartSec: round(stroke.startSec, 1e3),
      strokeEndSec: round(stroke.endSec, 1e3),
    };
    if (kind === 'word') {
      entry.lemma = typeof meta.lemma === 'string' ? meta.lemma.toLowerCase() : norm;
      entry.pos = typeof meta.pos === 'string' ? meta.pos.toUpperCase() : guessPos(entry.lemma);
    }
    entries.push(entry);
    clips.set(entry.clip, payload);
  }
  return { index: { version: 1, source: opts.source, entries }, clips };
}
