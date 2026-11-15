/**
 * Joins per-sign clips into one continuous animation.
 *
 * The motion model returns one clip per sign. Every clip starts and ends at the rest pose, so
 * playing them one after the other makes the avatar drop its hands between signs. This module:
 *   1. keeps the stroke of each sign (the part between leaving and returning to rest) unchanged,
 *      plus the lead-in of the first sign and the lead-out of the last one;
 *   2. replaces each rest gap with a short transition that starts at the end pose of sign A and
 *      ends at the start pose of sign B, matching pose, velocity and (quintic) acceleration at
 *      both ends, so the joint is smooth;
 *   3. sets the duration of each transition from how far the arms must move.
 *
 * Transitions run on the retargeted tracks, not on the CSV joint positions:
 *   - bone rotations: Hermite curve on the rotation vector relative to the start pose,
 *     q(t) = q0 · exp(h(t)), with the end velocity mapped through the inverse right Jacobian
 *     of SO(3), so the angular velocity at both ends matches the clips;
 *   - blendshapes: scalar Hermite curve, clamped to [0, 1].
 * Curves: 'quintic' (minimum-jerk profile, C2: pose, velocity and acceleration continuous) or
 * 'cubic' (C1: pose and velocity continuous).
 *
 * A hand that does nothing in a sign (one-handed sign between two signs that use that hand)
 * does not drop to rest: one long transition bridges it from the previous sign to the next.
 */

import { IDENTITY, qConj, qDot, qMul, qNormalize, type Quat } from '../recording/math3d';
import type { SignAnimationPayload } from '../types/lis';

export type TransitionCurve = 'quintic' | 'cubic';

export interface StitchClipInput {
  payload: SignAnimationPayload;
  /** Stroke start inside the clip (s). Detected from the arm motion when missing. */
  strokeStartSec?: number;
  /** Stroke end inside the clip (s). Detected from the arm motion when missing. */
  strokeEndSec?: number;
}

export interface StitchOptions {
  curve: TransitionCurve;
  /** Output frame rate; default: the frame rate of the first clip. */
  fps?: number;
  minTransitionSec: number;
  maxTransitionSec: number;
  /** Transition time per radian of the largest arm rotation between the two poses. */
  secPerRadian: number;
  /** Peak angular speed allowed in a transition (rad/s): arm bones, other bones (fingers, head). */
  maxArmSpeed: number;
  maxOtherSpeed: number;
  /** Bridge a hand that is idle in a sign between two signs that use it. */
  bridgeIdleHand: boolean;
  /** Arm rotation (rad) below which a hand counts as idle during a stroke. */
  idleAngleRad: number;
}

export const DEFAULT_STITCH_OPTIONS: StitchOptions = {
  curve: 'quintic',
  minTransitionSec: 0.15,
  maxTransitionSec: 0.6,
  secPerRadian: 0.22,
  maxArmSpeed: 7,
  maxOtherSpeed: 12,
  bridgeIdleHand: true,
  idleAngleRad: 0.26,
};

export interface StitchedSegment {
  /** Index of the source clip. */
  clip: number;
  startSec: number;
  endSec: number;
}

export interface StitchResult {
  payload: SignAnimationPayload;
  /** Stroke of each clip in the output timeline. */
  segments: StitchedSegment[];
  /** Transitions in the output timeline. */
  transitions: Array<{ startSec: number; endSec: number }>;
}

// ---------------------------------------------------------------------------
// Rotation helpers (rotation vectors: axis * angle)
// ---------------------------------------------------------------------------

type V3 = [number, number, number];

export function qLog(q: Quat): V3 {
  let [x, y, z, w] = q;
  if (w < 0) [x, y, z, w] = [-x, -y, -z, -w];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const angle = 2 * Math.atan2(s, w);
  const k = angle / s;
  return [x * k, y * k, z * k];
}

export function qExp(v: V3): Quat {
  const angle = Math.hypot(v[0], v[1], v[2]);
  if (angle < 1e-9) return qNormalize([v[0] / 2, v[1] / 2, v[2] / 2, 1]);
  const s = Math.sin(angle / 2) / angle;
  return [v[0] * s, v[1] * s, v[2] * s, Math.cos(angle / 2)];
}

const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Inverse right Jacobian of SO(3) at theta applied to w: rate of the rotation vector for body rate w. */
function invRightJacobian(theta: V3, w: V3): V3 {
  const a = Math.hypot(theta[0], theta[1], theta[2]);
  const tw = cross(theta, w);
  const ttw = cross(theta, tw);
  const c = a < 1e-4 ? 1 / 12 : 1 / (a * a) - (1 + Math.cos(a)) / (2 * a * Math.sin(a));
  return [w[0] + 0.5 * tw[0] + c * ttw[0], w[1] + 0.5 * tw[1] + c * ttw[1], w[2] + 0.5 * tw[2] + c * ttw[2]];
}

/** Body-frame rotation from a to b, as a rotation vector. */
function relRot(a: Quat, b: Quat): V3 {
  return qLog(qMul(qConj(a), b));
}

function angleBetween(a: Quat, b: Quat): number {
  return 2 * Math.acos(Math.min(1, Math.abs(qDot(qNormalize(a), qNormalize(b)))));
}

// ---------------------------------------------------------------------------
// Hermite bases. u in [0, 1], T = duration; returns weights for
// cubic:   [p0, v0, p1, v1]
// quintic: [p0, v0, a0, p1, v1, a1]
// ---------------------------------------------------------------------------

function hermite(curve: TransitionCurve, u: number, T: number): number[] {
  const u2 = u * u;
  const u3 = u2 * u;
  if (curve === 'cubic') {
    return [2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * T, -2 * u3 + 3 * u2, (u3 - u2) * T];
  }
  const u4 = u3 * u;
  const u5 = u4 * u;
  return [
    1 - 10 * u3 + 15 * u4 - 6 * u5,
    (u - 6 * u3 + 8 * u4 - 3 * u5) * T,
    (0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5) * T * T,
    10 * u3 - 15 * u4 + 6 * u5,
    (-4 * u3 + 7 * u4 - 3 * u5) * T,
    (0.5 * u3 - u4 + 0.5 * u5) * T * T,
  ];
}

interface Boundary<T> {
  p: T;
  v: T;
  a: T;
}

function curveScalar(curve: TransitionCurve, b0: Boundary<number>, b1: Boundary<number>, u: number, T: number): number {
  const w = hermite(curve, u, T);
  return curve === 'cubic'
    ? w[0] * b0.p + w[1] * b0.v + w[2] * b1.p + w[3] * b1.v
    : w[0] * b0.p + w[1] * b0.v + w[2] * b0.a + w[3] * b1.p + w[4] * b1.v + w[5] * b1.a;
}

/** Rotation transition: q(u) = q0 · exp(h(u)), h Hermite from 0 to log(q0⁻¹ q1). */
interface RotationBoundary {
  p: Quat;
  /** Body-frame angular velocity (rad/s). */
  w: V3;
  /** Body-frame angular acceleration (rad/s²). */
  alpha: V3;
}

function curveRotation(curve: TransitionCurve, b0: RotationBoundary, b1: RotationBoundary, u: number, T: number): Quat {
  const theta = relRot(b0.p, b1.p);
  const v1 = invRightJacobian(theta, b1.w);
  const a1 = invRightJacobian(theta, b1.alpha);
  const h: V3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    h[i] = curveScalar(curve, { p: 0, v: b0.w[i], a: b0.alpha[i] }, { p: theta[i], v: v1[i], a: a1[i] }, u, T);
  }
  return qNormalize(qMul(b0.p, qExp(h)));
}

// ---------------------------------------------------------------------------
// Clip sampling
// ---------------------------------------------------------------------------

class ClipSampler {
  readonly fps: number;
  readonly frames: number;
  readonly duration: number;

  constructor(readonly payload: SignAnimationPayload) {
    this.fps = payload.fps;
    this.frames = payload.times.length;
    this.duration = payload.times[this.frames - 1] ?? 0;
  }

  private locate(t: number): [number, number] {
    const f = Math.max(0, Math.min(this.frames - 1, t * this.fps));
    const i = Math.min(this.frames - 2, Math.floor(f));
    return i < 0 ? [0, 0] : [i, f - i];
  }

  bone(name: string, t: number): Quat {
    const flat = this.payload.boneTracks[name];
    if (!flat || flat.length < this.frames * 4) return IDENTITY;
    if (this.frames < 2) return qNormalize(flat.slice(0, 4) as Quat);
    const [i, f] = this.locate(t);
    const a = flat.slice(i * 4, i * 4 + 4) as Quat;
    let b = flat.slice(i * 4 + 4, i * 4 + 8) as Quat;
    if (qDot(a, b) < 0) b = [-b[0], -b[1], -b[2], -b[3]];
    return nlerp(a, b, f);
  }

  morph(name: string, t: number): number {
    const tr = this.payload.morphTracks[name];
    if (!tr || tr.length < this.frames) return 0;
    if (this.frames < 2) return tr[0] ?? 0;
    const [i, f] = this.locate(t);
    return tr[i] * (1 - f) + tr[i + 1] * f;
  }
}

function nlerp(a: Quat, b: Quat, f: number): Quat {
  return qNormalize([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f]);
}

const ARM_BONES = ['UpperArm', 'LowerArm', 'Hand'];

/** Arm rotation of one side away from the clip's first frame (the rest pose), per frame. */
function sideActivity(s: ClipSampler, side: 'right' | 'left', t: number): number {
  let sum = 0;
  for (const b of ARM_BONES) {
    const name = `${side}${b}`;
    if (!s.payload.boneTracks[name]) continue;
    sum += angleBetween(s.bone(name, 0), s.bone(name, t)) * (b === 'Hand' ? 0.5 : 1);
  }
  return sum;
}

/**
 * Stroke of a rest-to-rest clip: where the arms hold the sign, away from the first frame (rest).
 * The sign level is the median arm rotation over the frames clearly away from rest; the stroke
 * is where the rotation stays above 75% of that level, so the raise from rest and the drop
 * back to rest are left out.
 */
export function detectStroke(payload: SignAnimationPayload): { startSec: number; endSec: number } {
  const s = new ClipSampler(payload);
  const act = payload.times.map((t) => Math.max(sideActivity(s, 'right', t), sideActivity(s, 'left', t)));
  const peak = Math.max(0, ...act);
  if (peak < 0.1) return { startSec: 0, endSec: s.duration };
  const core = act.filter((a) => a >= 0.45 * peak).sort((a, b) => a - b);
  const level = core[Math.floor(core.length / 2)] ?? peak;
  const thr = Math.max(0.1, 0.75 * level);
  let i0 = act.findIndex((a) => a >= thr);
  let i1 = act.length - 1 - [...act].reverse().findIndex((a) => a >= thr);
  if (i0 < 0) [i0, i1] = [0, act.length - 1];
  // Move the cuts inward while the arms are still rising from rest or already dropping back:
  // the velocity at the cut feeds the transition, and the motion to or from rest must not.
  const fast = 1.2 / s.fps; // rad per frame (1.2 rad/s)
  const maxMove = Math.round(0.25 * s.fps);
  for (let n = 0; n < maxMove && i0 + 1 < i1 && act[i0 + 1] - act[i0] > fast; n++) i0++;
  for (let n = 0; n < maxMove && i1 - 1 > i0 && act[i1 - 1] - act[i1] > fast; n++) i1--;
  return { startSec: payload.times[i0], endSec: payload.times[i1] };
}

// ---------------------------------------------------------------------------
// Stitching
// ---------------------------------------------------------------------------

interface KeptRange {
  sampler: ClipSampler;
  /** Times sampled in the clip, on the output frame grid. */
  t0: number;
  frames: number;
  strokeStart: number;
  strokeEnd: number;
}

export function stitchClips(clips: StitchClipInput[], options: Partial<StitchOptions> = {}): StitchResult {
  if (!clips.length) throw new Error('stitchClips: no clips');
  const opts: StitchOptions = { ...DEFAULT_STITCH_OPTIONS, ...options };
  const fps = opts.fps ?? clips[0].payload.fps;
  const h = 1 / fps;

  const samplers = clips.map((c) => new ClipSampler(c.payload));
  const boneNames = Array.from(new Set(clips.flatMap((c) => Object.keys(c.payload.boneTracks)))).sort();
  const morphNames = Array.from(new Set(clips.flatMap((c) => Object.keys(c.payload.morphTracks)))).sort();

  // 1. Kept range of every clip.
  const ranges: KeptRange[] = clips.map((c, i) => {
    const s = samplers[i];
    const detected = c.strokeStartSec === undefined || c.strokeEndSec === undefined ? detectStroke(c.payload) : null;
    const margin = 2 * h; // room for the one-sided derivatives
    let strokeStart = Math.max(margin, Math.min(s.duration - margin, c.strokeStartSec ?? detected!.startSec));
    let strokeEnd = Math.max(strokeStart, Math.min(s.duration - margin, c.strokeEndSec ?? detected!.endSec));
    if (strokeEnd - strokeStart < h) strokeEnd = Math.min(s.duration, strokeStart + h);
    const from = i === 0 ? 0 : strokeStart;
    const to = i === clips.length - 1 ? s.duration : strokeEnd;
    const frames = Math.max(1, Math.round((to - from) * fps) + 1);
    return { sampler: s, t0: from, frames, strokeStart, strokeEnd };
  });
  const endTime = (r: KeptRange) => r.t0 + (r.frames - 1) * h;

  // Boundary state of a bone at time t in a clip; backward differences at the end of a range,
  // forward differences at the start.
  const boneBoundary = (s: ClipSampler, name: string, t: number, dir: -1 | 1): RotationBoundary => {
    const q0 = s.bone(name, t);
    const q1 = s.bone(name, t + dir * h);
    const q2 = s.bone(name, t + 2 * dir * h);
    // Body-frame angular velocity: forward in time from the earlier to the later sample.
    const wNear = dir < 0 ? relRot(q1, q0) : relRot(q0, q1);
    const wFar = dir < 0 ? relRot(q2, q1) : relRot(q1, q2);
    const w: V3 = [wNear[0] / h, wNear[1] / h, wNear[2] / h];
    const wOther: V3 = [wFar[0] / h, wFar[1] / h, wFar[2] / h];
    const sign = dir < 0 ? 1 : -1;
    const alpha: V3 = [0, 1, 2].map((i) => clampAbs((sign * (w[i] - wOther[i])) / h, MAX_ACCEL)) as V3;
    return { p: q0, w, alpha };
  };
  const morphBoundary = (s: ClipSampler, name: string, t: number, dir: -1 | 1): Boundary<number> => {
    const x0 = s.morph(name, t);
    const x1 = s.morph(name, t + dir * h);
    const x2 = s.morph(name, t + 2 * dir * h);
    const v = (dir < 0 ? x0 - x1 : x1 - x0) / h;
    const a = clampAbs((x0 - 2 * x1 + x2) / (h * h), MAX_ACCEL);
    return { p: x0, v, a };
  };

  // 2. Transition durations: from the largest arm rotation, then long enough that no bone
  //    exceeds its peak speed (peak / mean speed: 1.875 quintic, 1.5 cubic).
  const peakFactor = opts.curve === 'quintic' ? 1.875 : 1.5;
  const transitionFrames: number[] = [];
  for (let i = 0; i + 1 < ranges.length; i++) {
    const A = ranges[i];
    const B = ranges[i + 1];
    let maxArm = 0;
    let minBySpeed = 0;
    for (const name of boneNames) {
      const a = angleBetween(A.sampler.bone(name, endTime(A)), B.sampler.bone(name, B.t0));
      const isArm = /(UpperArm|LowerArm|Hand)$/.test(name);
      if (isArm) maxArm = Math.max(maxArm, a);
      minBySpeed = Math.max(minBySpeed, (peakFactor * a) / (isArm ? opts.maxArmSpeed : opts.maxOtherSpeed));
    }
    const base = Math.min(opts.maxTransitionSec, opts.minTransitionSec + opts.secPerRadian * maxArm);
    const T = Math.max(opts.minTransitionSec, base, minBySpeed);
    transitionFrames.push(Math.max(2, Math.ceil(T * fps - 1e-6)));
  }

  // 3. Output timeline: kept frames, transition frames (exclusive of both ends), kept frames, ...
  const total = ranges.reduce((n, r) => n + r.frames, 0) + transitionFrames.reduce((n, k) => n + k - 1, 0);
  const bones: Record<string, Quat[]> = Object.fromEntries(boneNames.map((b) => [b, new Array<Quat>(total)]));
  const morphs: Record<string, number[]> = Object.fromEntries(morphNames.map((m) => [m, new Array<number>(total).fill(0)]));
  const rangeStart: number[] = [];
  const segments: StitchedSegment[] = [];
  const transitions: StitchResult['transitions'] = [];

  let k = 0;
  ranges.forEach((r, i) => {
    rangeStart.push(k);
    for (let j = 0; j < r.frames; j++, k++) {
      const t = r.t0 + j * h;
      for (const b of boneNames) bones[b][k] = r.sampler.bone(b, t);
      for (const m of morphNames) morphs[m][k] = r.sampler.morph(m, t);
    }
    segments.push({
      clip: i,
      startSec: (rangeStart[i] + Math.round((r.strokeStart - r.t0) * fps)) * h,
      endSec: (rangeStart[i] + Math.round((r.strokeEnd - r.t0) * fps)) * h,
    });
    if (i + 1 < ranges.length) {
      const B = ranges[i + 1];
      const N = transitionFrames[i];
      const T = N * h;
      const tA = endTime(r);
      const boneEnds = boneNames.map((b) => [boneBoundary(r.sampler, b, tA, -1), boneBoundary(B.sampler, b, B.t0, 1)] as const);
      const morphEnds = morphNames.map((m) => [morphBoundary(r.sampler, m, tA, -1), morphBoundary(B.sampler, m, B.t0, 1)] as const);
      transitions.push({ startSec: (k - 1) * h, endSec: (k - 1 + N) * h });
      for (let m = 1; m < N; m++, k++) {
        const u = m / N;
        boneNames.forEach((b, bi) => {
          bones[b][k] = curveRotation(opts.curve, boneEnds[bi][0], boneEnds[bi][1], u, T);
        });
        morphNames.forEach((name, mi) => {
          morphs[name][k] = clamp01(curveScalar(opts.curve, morphEnds[mi][0], morphEnds[mi][1], u, T));
        });
      }
    }
  });

  // 4. Idle hand bridging: hand idle during the stroke of clip i, used in clips i-1 and i+1.
  if (opts.bridgeIdleHand && ranges.length >= 3) {
    const idle = (r: KeptRange, side: 'right' | 'left') => {
      let maxA = 0;
      for (let t = r.strokeStart; t <= r.strokeEnd + 1e-9; t += h) maxA = Math.max(maxA, sideActivity(r.sampler, side, t));
      return maxA < opts.idleAngleRad;
    };
    for (const side of ['right', 'left'] as const) {
      const sideBones = boneNames.filter((b) => b.startsWith(side));
      let i = 1;
      while (i < ranges.length - 1) {
        if (!idle(ranges[i], side) || idle(ranges[i - 1], side)) {
          i++;
          continue;
        }
        // Run of idle clips i..j, with clip j+1 using the hand again.
        let j = i;
        while (j + 1 < ranges.length - 1 && idle(ranges[j + 1], side)) j++;
        if (idle(ranges[j + 1], side)) break;
        const A = ranges[i - 1];
        const C = ranges[j + 1];
        const kA = rangeStart[i - 1] + A.frames - 1;
        const kC = rangeStart[j + 1];
        const N = kC - kA;
        const T = N * h;
        for (const b of sideBones) {
          const b0 = boneBoundary(A.sampler, b, endTime(A), -1);
          const b1 = boneBoundary(C.sampler, b, C.t0, 1);
          for (let m = 1; m < N; m++) bones[b][kA + m] = curveRotation(opts.curve, b0, b1, m / N, T);
        }
        i = j + 1;
      }
    }
  }

  // 5. Payload.
  const boneTracks: Record<string, number[]> = {};
  for (const b of boneNames) {
    const flat: number[] = [];
    let prev: Quat | null = null;
    for (let q of bones[b]) {
      q = qNormalize(q);
      if (prev && qDot(prev, q) < 0) q = [-q[0], -q[1], -q[2], -q[3]];
      prev = q;
      flat.push(...q.map((x) => Math.round(x * 1e5) / 1e5));
    }
    boneTracks[b] = flat;
  }
  const morphTracks: Record<string, number[]> = {};
  for (const m of morphNames) morphTracks[m] = morphs[m].map((x) => Math.round(x * 1e4) / 1e4);

  const first = clips[0].payload;
  const times = Array.from({ length: total }, (_, i) => Math.round(i * h * 1e4) / 1e4);
  const payload: SignAnimationPayload = {
    classId: first.classId,
    phrase: first.phrase,
    category: first.category,
    description: first.description,
    prototypeClipId: clips.map((c) => c.payload.prototypeClipId).join('+'),
    fps,
    duration: times[times.length - 1],
    times,
    boneTracks,
    morphTracks,
    qaComparison: { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] },
  };
  return { payload, segments, transitions };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/** Finite-difference accelerations are noisy: limit them (rad/s², or 1/s² for blendshapes). */
const MAX_ACCEL = 60;

function clampAbs(x: number, max: number): number {
  return Math.max(-max, Math.min(max, x));
}
