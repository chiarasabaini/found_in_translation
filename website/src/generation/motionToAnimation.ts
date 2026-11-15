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
    return { payload, segments: motion.segments, transitions: [] };
  }

  const recs = clips.map((c, i) => parseModelFiles(c.files, `${i + 1}-${c.gloss || 'sign'}`));
  // One yaw for all clips, so a change of estimate between clips does not turn the hands.
  const yaw = estimateGloveYawDeg(recs[0]);
  const payloads = recs.map((rec) => retarget(rec, text, glossLine, yaw));
  const result = stitchClips(
    payloads.map((payload, i) => ({
      payload,
      strokeStartSec: clips[i].strokeStartSec,
      strokeEndSec: clips[i].strokeEndSec,
    })),
    { ...stitch, curve },
  );
  result.payload.prototypeClipId = motion.model;
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
