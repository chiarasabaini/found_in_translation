/**
 * Model output -> avatar animation.
 *
 * Continuous output (`files`) goes through the recording import and retargeting as one clip.
 * Per-sign output (`clips`) is retargeted clip by clip with one shared glove yaw, then joined
 * by stitchClips with smooth transitions.
 */

import { parseRokokoRecording, type RecordingFileInput, type RokokoRecording } from '../recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, estimateGloveYawDeg, retargetRecording } from '../recording/recordingRetarget';
import type { SignAnimationPayload } from '../types/lis';
import type { GeneratedMotion, GlossSegment } from './motionModel';
import { stitchClips, type StitchOptions, type TransitionCurve } from './stitchClips';

export interface MotionAnimation {
  payload: SignAnimationPayload;
  /** Signs in the timeline of `payload`. */
  segments: GlossSegment[];
  /** Transitions added between signs (empty for continuous output). */
  transitions: Array<{ startSec: number; endSec: number }>;
}

function parseModelFiles(files: RecordingFileInput[], name: string): RokokoRecording {
  const rec = parseRokokoRecording(files, name);
  const error = rec.diagnostics.find((d) => d.level === 'error');
  if (error) throw new Error(`Model output not valid (${name}): ${error.message}`);
  return rec;
}

function retarget(rec: RokokoRecording, text: string, description: string, gloveYawDeg: number | null): SignAnimationPayload {
  return retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    gloveYawDeg,
    startSec: 0,
    endSec: rec.durationSec,
    classId: 'generated',
    phrase: text,
    category: 'Generated',
    description,
  }).payload;
}

export function motionToAnimation(
  motion: GeneratedMotion,
  text: string,
  curve: TransitionCurve = 'quintic',
  stitch: Partial<StitchOptions> = {},
): MotionAnimation | null {
  const glossLine = motion.segments.map((s) => s.gloss).join(' · ');
  const clips = motion.clips ?? [];

  if (!clips.length) {
    if (!motion.files.length) return null;
    const payload = retarget(parseModelFiles(motion.files, 'generated'), text, glossLine, null);
    payload.prototypeClipId = motion.model;
    applyNonManualMarkers(payload, motion.modifiers ?? [], 0, payload.duration);
    return { payload, segments: motion.segments, transitions: [] };
  }

  // Clips with files are retargeted here (one yaw for all, so a change of estimate between
  // clips does not turn the hands); exported dataset clips come already retargeted.
  const recs = clips.map((c, i) => (c.payload ? null : parseModelFiles(c.files, `${i + 1}-${c.gloss || 'sign'}`)));
  const firstRec = recs.find((r) => r !== null);
  const yaw = firstRec ? estimateGloveYawDeg(firstRec) : null;
  const payloads = clips.map((c, i) => c.payload ?? retarget(recs[i]!, text, glossLine, yaw));
  const result = stitchClips(
    payloads.map((payload, i) => ({
      payload,
      strokeStartSec: clips[i].strokeStartSec,
      strokeEndSec: clips[i].strokeEndSec,
    })),
    { ...stitch, curve },
  );
  result.payload.prototypeClipId = motion.model;
  result.payload.phrase = text;
  result.payload.description = glossLine;
  if (result.segments.length) {
    applyNonManualMarkers(
      result.payload,
      motion.modifiers ?? [],
      result.segments[0].startSec,
      result.segments[result.segments.length - 1].endSec,
    );
  }
  const segments: GlossSegment[] = result.segments.map((s) => ({
    text: clips[s.clip].text,
    gloss: clips[s.clip].gloss,
    classId: clips[s.clip].classId,
    startSec: Math.round(s.startSec * 100) / 100,
    endSec: Math.round(s.endSec * 100) / 100,
  }));
  return { payload: result.payload, segments, transitions: result.transitions };
}

/** All files of the model output, with a clip prefix for per-sign output. */
export function motionOutputFiles(motion: GeneratedMotion): RecordingFileInput[] {
  if (motion.files.length) return motion.files;
  return (motion.clips ?? []).flatMap((c, i) =>
    c.files.map((f) => ({ name: `${i + 1}-${c.gloss || 'sign'}-${f.name}`, text: f.text })),
  );
}

/**
 * LIS non-manual markers over the signed part of the phrase (face channels of the avatar):
 * yes/no question -> raised brows; wh-question -> furrowed brows; negation -> furrowed brows
 * and a frown. Each channel takes the maximum of the clip value and the marker.
 */
export function applyNonManualMarkers(payload: SignAnimationPayload, modifiers: string[], startSec: number, endSec: number): void {
  const targets: Record<string, number> = {};
  const add = (name: string, v: number) => (targets[name] = Math.max(targets[name] ?? 0, v));
  if (modifiers.includes('question_yn')) add('browUp', 0.6);
  if (modifiers.includes('question_wh')) add('browDown', 0.5);
  if (modifiers.includes('negation')) {
    add('browDown', 0.35);
    add('mouthFrown', 0.35);
  }
  if (!Object.keys(targets).length) return;
  const ramp = 0.25;
  const from = Math.max(0, startSec - 0.2);
  const to = Math.min(payload.duration, endSec + 0.2);
  const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const env = (t: number) => Math.min(smooth((t - from) / ramp), smooth((to - t) / ramp));
  for (const [name, value] of Object.entries(targets)) {
    const track = (payload.morphTracks[name] ??= payload.times.map(() => 0));
    payload.times.forEach((t, k) => {
      track[k] = Math.round(Math.max(track[k] ?? 0, value * env(t)) * 1e4) / 1e4;
    });
  }
}
