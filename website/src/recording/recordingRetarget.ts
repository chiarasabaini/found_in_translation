/**
 * Retargets a synchronized Rokoko + MediaPipe recording onto the app's humanoid rig.
 *
 * Output is the same SignAnimationPayload the offline Python pipeline exports, so the
 * clip plays on the procedural avatar and on imported VRM/GLB avatars unchanged.
 *
 * Conventions (source rig, same as pipeline/retargeting.py):
 *   avatar world: Y-up, character faces +Z, character's left = +X;
 *   every arm/finger bone points along its local -Y axis in the rest pose;
 *   tracks are local bone quaternions (x, y, z, w).
 *
 * Method: direction-based retargeting. Bone directions come from joint positions, so the
 * result does not depend on the solver's per-joint axis conventions:
 *   - fingers and hand frame: solved Smartgloves joints (26 per hand);
 *   - forearm: glove ForeArm -> Hand vector;
 *   - upper arm and torso: MediaPipe shoulders/elbows from body.csv (optional);
 *   - head: MediaPipe face landmarks relative to the neutral (median) face;
 *   - face morphs: ARKit blendshapes from face.csv.
 * The gloves report yaw relative to magnetic north, so one global yaw aligns them with
 * the body; it is estimated automatically and can be fine-tuned per hand.
 */

import type { QaFrame, QaHandFrame, SignAnimationPayload } from '../types/lis.ts';
import {
  IDENTITY,
  degToRad,
  median,
  orthoFrame,
  qClampAngle,
  qConj,
  qDot,
  qFrameToFrame,
  qFromAxisAngle,
  qFromEulerXYZ,
  qFromUnitVectors,
  qMul,
  qNormalize,
  qRotate,
  qSlerp,
  radToDeg,
  vAdd,
  vDot,
  vLen,
  vLerp,
  vNorm,
  vScale,
  vSub,
  type Quat,
  type Vec3,
} from './math3d.ts';
import type { HandSide, HandStream, RokokoJoint, RokokoRecording } from './rokokoRecording.ts';

export interface RecordingRetargetOptions {
  startSec: number;
  endSec: number;
  fps: number;
  /** Global glove yaw in degrees; null = automatic estimate. */
  gloveYawDeg: number | null;
  leftYawOffsetDeg: number;
  rightYawOffsetDeg: number;
  useBodyArms: boolean;
  useTorso: boolean;
  useHead: boolean;
  useFaceMorphs: boolean;
  headGain: number;
  /** 0 = raw, 1 = strong smoothing. */
  smoothing: number;
  /** Blend from/to the rest pose at clip start/end (seconds). */
  fadeSec: number;
  classId: string;
  phrase: string;
  category: string;
  description: string;
}

export const DEFAULT_RETARGET_OPTIONS: Omit<
  RecordingRetargetOptions,
  'startSec' | 'endSec' | 'classId' | 'phrase' | 'category' | 'description'
> = {
  fps: 30,
  gloveYawDeg: null,
  leftYawOffsetDeg: 0,
  rightYawOffsetDeg: 0,
  useBodyArms: true,
  useTorso: true,
  useHead: true,
  useFaceMorphs: true,
  headGain: 1,
  smoothing: 0.35,
  fadeSec: 0.2,
};

/** ARKit blendshape -> avatar morph target (same as config/sensors_and_retargeting.yaml). */
export const FACE_BLENDSHAPE_MAP: Record<string, string> = {
  jawOpen: 'mouthOpen',
  mouthSmileLeft: 'mouthSmile',
  mouthSmileRight: 'mouthSmile',
  mouthPucker: 'mouthPucker',
  mouthFrownLeft: 'mouthFrown',
  mouthFrownRight: 'mouthFrown',
  browInnerUp: 'browUp',
  browDownLeft: 'browDown',
  browDownRight: 'browDown',
  eyeBlinkLeft: 'eyeBlinkLeft',
  eyeBlinkRight: 'eyeBlinkRight',
  cheekPuff: 'cheekPuff',
};

const DOWN: Vec3 = [0, -1, 0];
const FINGERS = ['thumb', 'index', 'middle', 'ring', 'little'] as const;
type FingerName = (typeof FINGERS)[number];
const PHALANGES = ['Proximal', 'Intermediate', 'Distal'] as const;

/** Rokoko joint segments that drive the avatar's 3 phalanx bones per finger. */
const FINGER_SEGMENTS: Record<FingerName, Array<[RokokoJoint, RokokoJoint]>> = {
  thumb: [
    ['Finger1Metacarpal', 'Finger1Proximal'],
    ['Finger1Proximal', 'Finger1Distal'],
    ['Finger1Distal', 'Finger1Tip'],
  ],
  index: [
    ['Finger2Proximal', 'Finger2Medial'],
    ['Finger2Medial', 'Finger2Distal'],
    ['Finger2Distal', 'Finger2Tip'],
  ],
  middle: [
    ['Finger3Proximal', 'Finger3Medial'],
    ['Finger3Medial', 'Finger3Distal'],
    ['Finger3Distal', 'Finger3Tip'],
  ],
  ring: [
    ['Finger4Proximal', 'Finger4Medial'],
    ['Finger4Medial', 'Finger4Distal'],
    ['Finger4Distal', 'Finger4Tip'],
  ],
  little: [
    ['Finger5Proximal', 'Finger5Medial'],
    ['Finger5Medial', 'Finger5Distal'],
    ['Finger5Distal', 'Finger5Tip'],
  ],
};

const FINGER_CURL_SEGMENT: Record<FingerName, [RokokoJoint, RokokoJoint]> = {
  thumb: ['Finger1Proximal', 'Finger1Tip'],
  index: ['Finger2Proximal', 'Finger2Tip'],
  middle: ['Finger3Proximal', 'Finger3Tip'],
  ring: ['Finger4Proximal', 'Finger4Tip'],
  little: ['Finger5Proximal', 'Finger5Tip'],
};

/** Rest pose used for fades, matching pipeline/retargeting.py REST_POSE_BONES. */
const REST_POSE: Record<string, Quat> = {
  rightUpperArm: qFromEulerXYZ(0.1, 0.05, -0.18),
  rightLowerArm: qFromEulerXYZ(-0.22, 0, 0),
  leftUpperArm: qFromEulerXYZ(0.1, -0.05, 0.18),
  leftLowerArm: qFromEulerXYZ(-0.22, 0, 0),
};

/**
 * Two-frame animation holding the rest pose (arms along the sides), for an idle avatar.
 * Through the same retargeting as the clips, so T-pose and A-pose models end up alike.
 */
export function restPosePayload(): SignAnimationPayload {
  const boneTracks: Record<string, number[]> = {};
  for (const [name, q] of Object.entries(REST_POSE)) boneTracks[name] = [...q, ...q];
  return {
    classId: 'rest',
    phrase: '',
    category: 'Rest',
    description: 'Rest pose',
    prototypeClipId: 'rest',
    fps: 10,
    duration: 0.1,
    times: [0, 0.1],
    boneTracks,
    morphTracks: {},
    qaComparison: { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] },
  };
}

/** Procedural rig dimensions, used only for the QA wrist trace. */
const SHOULDER_POS: Record<HandSide, Vec3> = { right: [-0.195, 1.42, 0], left: [0.195, 1.42, 0] };
const UPPER_ARM_LEN = 0.28;
const LOWER_ARM_LEN = 0.25;

/** Hand bone frame (finger direction, radial side, normal) in the rest pose. */
function restHandFrame(side: HandSide): [Vec3, Vec3, Vec3] {
  const s: Vec3 = side === 'right' ? [-1, 0, 0] : [1, 0, 0];
  return orthoFrame(DOWN, s);
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

function locate(times: number[], t: number, hint: { i: number }): [number, number] {
  const n = times.length;
  if (n === 0) return [0, 0];
  if (t <= times[0]) return [0, 0];
  if (t >= times[n - 1]) return [n - 1, 0];
  let i = Math.min(Math.max(0, hint.i), n - 2);
  while (i < n - 2 && times[i + 1] <= t) i++;
  while (i > 0 && times[i] > t) i--;
  hint.i = i;
  const span = times[i + 1] - times[i];
  return [i, span > 0 ? (t - times[i]) / span : 0];
}

function sampleVec(arr: Vec3[], i: number, f: number): Vec3 {
  if (!arr.length) return [0, 0, 0];
  const a = arr[i];
  const b = arr[Math.min(arr.length - 1, i + 1)];
  return vLerp(a, b, f);
}

function sampleNum(arr: number[], i: number, f: number): number {
  if (!arr.length) return 0;
  const a = arr[i];
  const b = arr[Math.min(arr.length - 1, i + 1)];
  return a + (b - a) * f;
}

/** Index map that replaces undetected or degenerate frames with the nearest valid one. */
function validIndexMap(valid: boolean[]): number[] {
  const n = valid.length;
  const map = new Array<number>(n).fill(-1);
  let last = -1;
  for (let i = 0; i < n; i++) {
    if (valid[i]) last = i;
    map[i] = last;
  }
  let next = -1;
  for (let i = n - 1; i >= 0; i--) {
    if (valid[i]) next = i;
    if (map[i] === -1) map[i] = next === -1 ? i : next;
  }
  return map;
}

function handFrameValid(stream: HandStream, i: number): boolean {
  if (!stream.detected[i]) return false;
  const w = stream.positions.Hand[i];
  const fa = stream.positions.ForeArm[i];
  return vLen(vSub(w, fa)) > 0.05 && vLen(vSub(stream.positions.Finger2Proximal[i], stream.positions.Finger5Proximal[i])) > 0.01;
}

// ---------------------------------------------------------------------------
// One Euro filter over quaternion / vector tracks
// ---------------------------------------------------------------------------

function oneEuroSeries(values: number[][], fps: number, minCutoff: number, beta: number): number[][] {
  if (values.length < 2) return values;
  const dCutoff = 1.0;
  const te = 1 / fps;
  const alpha = (cutoff: number) => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / te);
  };
  const dim = values[0].length;
  const out: number[][] = [values[0].slice()];
  let prev = values[0].slice();
  let dPrev = new Array<number>(dim).fill(0);
  for (let k = 1; k < values.length; k++) {
    const x = values[k];
    const dx = x.map((v, j) => (v - prev[j]) / te);
    const aD = alpha(dCutoff);
    const dHat = dx.map((v, j) => aD * v + (1 - aD) * dPrev[j]);
    const speed = Math.sqrt(dHat.reduce((s, v) => s + v * v, 0));
    const a = alpha(minCutoff + beta * speed);
    const xHat = x.map((v, j) => a * v + (1 - a) * prev[j]);
    out.push(xHat);
    prev = xHat;
    dPrev = dHat;
  }
  return out;
}

function smoothQuatTrack(track: Quat[], fps: number, minCutoff: number, beta: number): Quat[] {
  // Enforce hemisphere continuity first so component-wise filtering is valid.
  const cont: Quat[] = [];
  for (const q of track) {
    const prev = cont[cont.length - 1];
    cont.push(prev && qDot(prev, q) < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q);
  }
  return oneEuroSeries(cont, fps, minCutoff, beta).map((q) => qNormalize(q as Quat));
}

function smoothVecTrack(track: Vec3[], fps: number, minCutoff: number, beta: number): Vec3[] {
  return oneEuroSeries(track, fps, minCutoff, beta).map((v) => vNorm(v as Vec3));
}

// ---------------------------------------------------------------------------
// Calibration estimates
// ---------------------------------------------------------------------------

/**
 * Estimates the yaw that turns the gloves' magnetic-north frame into the body frame.
 * Assumption: while signing, the forearms point mostly forward (+Z), so the summed
 * horizontal forearm direction of both hands over raised frames defines "forward".
 */
export function estimateGloveYawDeg(rec: RokokoRecording): number {
  let hx = 0;
  let hz = 0;
  let hxAll = 0;
  let hzAll = 0;
  for (const s of [rec.left, rec.right]) {
    if (!s) continue;
    for (let i = 0; i < s.times.length; i++) {
      if (!handFrameValid(s, i)) continue;
      const d = vNorm(vSub(s.positions.Hand[i], s.positions.ForeArm[i]));
      hxAll += d[0];
      hzAll += d[2];
      if (d[1] > -0.4) {
        // weight raised forearms by their horizontal extent
        hx += d[0];
        hz += d[2];
      }
    }
  }
  if (Math.hypot(hx, hz) < 1e-3) {
    hx = hxAll;
    hz = hzAll;
  }
  if (Math.hypot(hx, hz) < 1e-6) return 0;
  return radToDeg(Math.atan2(-hx, hz));
}

interface BodyAxisSigns {
  x: number;
  y: number;
  z: number;
}

interface BodyTorsoSeries {
  leftShoulder: Vec3[] | null;
  rightShoulder: Vec3[] | null;
  centralTop: Vec3[] | null;
  lowerJoint: Vec3[] | null;
}

const LOWER_TORSO_KEYS = [
  'torso',
  'hips',
  'pelvis',
  'mid_hip',
  'hip',
  'waist',
  'spine',
  'lower_spine',
  'spine_base',
  'chest',
  'root',
] as const;

/**
 * Resolves upper-body and 3-bone torso landmarks (1 central + 2 lateral joined at a lower torso
 * joint, or MediaPipe left_hip/right_hip midpoint) from body.csv across common naming conventions.
 */
function resolveBodyTorsoSeries(body: RokokoRecording['body']): BodyTorsoSeries {
  if (!body) {
    return { leftShoulder: null, rightShoulder: null, centralTop: null, lowerJoint: null };
  }
  const pts = body.points;
  const leftShoulder = pts.left_shoulder ?? pts.shoulder_l ?? pts.LeftShoulder ?? null;
  const rightShoulder = pts.right_shoulder ?? pts.shoulder_r ?? pts.RightShoulder ?? null;
  const centralTop = pts.neck ?? pts.cervical ?? pts.upper_spine ?? pts.head_base ?? pts.Neck ?? null;

  let lowerJoint: Vec3[] | null = null;
  for (const key of LOWER_TORSO_KEYS) {
    if (pts[key] && pts[key].length > 0) {
      lowerJoint = pts[key];
      break;
    }
  }
  if (!lowerJoint) {
    const lh = pts.left_hip ?? pts.hip_l ?? pts.LeftUpLeg;
    const rh = pts.right_hip ?? pts.hip_r ?? pts.RightUpLeg;
    if (lh && rh && lh.length > 0 && lh.length === rh.length) {
      lowerJoint = lh.map((p, i) => vScale(vAdd(p, rh[i]), 0.5));
    }
  }

  return { leftShoulder, rightShoulder, centralTop, lowerJoint };
}

/** Detects axis signs of body.csv so that left shoulder = +X, up = +Y, forward = +Z. */
function detectBodyAxes(rec: RokokoRecording): BodyAxisSigns {
  const b = rec.body;
  const signs: BodyAxisSigns = { x: -1, y: 1, z: 1 };
  if (!b) return signs;
  const torsoSeries = resolveBodyTorsoSeries(b);
  const ls = torsoSeries.leftShoulder;
  const rs = torsoSeries.rightShoulder;
  const le = b.points.left_elbow;
  const re = b.points.right_elbow;
  if (ls && rs && ls.length) {
    const dx = median(ls.map((p, i) => p[0] - rs[i][0]));
    if (Math.abs(dx) > 1e-4) signs.x = dx > 0 ? 1 : -1;
  }
  if (ls && rs && torsoSeries.lowerJoint && torsoSeries.lowerJoint.length === ls.length) {
    const dyLow = median(
      torsoSeries.lowerJoint.map((low, i) => low[1] - 0.5 * (ls[i][1] + rs[i][1])),
    );
    if (Math.abs(dyLow) > 1e-4) signs.y = dyLow < 0 ? 1 : -1;
  } else if (ls && le && ls.length) {
    // Elbows sit below the shoulders most of the time.
    const dy = median(le.map((p, i) => p[1] - ls[i][1]).concat(re && rs ? re.map((p, i) => p[1] - rs[i][1]) : []));
    if (Math.abs(dy) > 1e-4) signs.y = dy < 0 ? 1 : -1;
  }
  if (ls && le && ls.length) {
    // Raising an elbow for signing moves it forward: correlate height with depth.
    const pairs: Array<[number, number]> = [];
    for (const [sh, el] of [
      [ls, le],
      [rs, re],
    ] as const) {
      if (!sh || !el) continue;
      for (let i = 0; i < el.length; i++) pairs.push([(el[i][1] - sh[i][1]) * signs.y, el[i][2] - sh[i][2]]);
    }
    if (pairs.length > 10) {
      const my = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
      const mz = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
      const cov = pairs.reduce((s, p) => s + (p[0] - my) * (p[1] - mz), 0);
      const vy = pairs.reduce((s, p) => s + (p[0] - my) ** 2, 0);
      const vz = pairs.reduce((s, p) => s + (p[1] - mz) ** 2, 0);
      const r = vy > 0 && vz > 0 ? cov / Math.sqrt(vy * vz) : 0;
      // Flip only on clear evidence; +Z forward is the usual convention.
      if (r < -0.2) signs.z = -1;
    }
  }
  return signs;
}

function mapBody(p: Vec3, s: BodyAxisSigns): Vec3 {
  return [p[0] * s.x, p[1] * s.y, p[2] * s.z];
}

/** Face landmarks: image x right, y down, z toward camera negative -> avatar frame. */
function detectFaceMirror(rec: RokokoRecording): number {
  const f = rec.face;
  if (!f || !f.landmarks.leftEye.length) return 1;
  const dx = median(f.landmarks.leftEye.map((p, i) => p[0] - f.landmarks.rightEye[i][0]));
  return dx >= 0 ? 1 : -1;
}

function mapFace(p: Vec3, mirror: number): Vec3 {
  return [p[0] * mirror, -p[1], -p[2]];
}

function headFrame(
  le: Vec3,
  re: Vec3,
  forehead: Vec3,
  chin: Vec3,
  mirror: number,
): [Vec3, Vec3, Vec3] {
  const side = vSub(mapFace(le, mirror), mapFace(re, mirror));
  const up = vSub(mapFace(forehead, mirror), mapFace(chin, mirror));
  return orthoFrame(up, side);
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

export interface ActivitySegment {
  startSec: number;
  endSec: number;
  peak: number;
}

export interface ActivityProfile {
  times: number[];
  /** 0 = arms at rest, 1 = forearm raised (max over both hands). */
  activity: number[];
  segments: ActivitySegment[];
}

/**
 * Finds signing segments from forearm elevation (yaw independent). A sign is
 * performed with at least one forearm raised; rest = forearms hanging down.
 */
export function detectSigningSegments(rec: RokokoRecording, threshold = 0.4): ActivityProfile {
  const ref = rec.right ?? rec.left;
  if (!ref) return { times: [], activity: [], segments: [] };
  const times = ref.times;
  const activity = times.map((_, i) => {
    let a = 0;
    for (const s of [rec.left, rec.right]) {
      if (!s || i >= s.times.length || !handFrameValid(s, i)) continue;
      const d = vNorm(vSub(s.positions.Hand[i], s.positions.ForeArm[i]));
      // d.y = -1 hanging, 0 horizontal, +1 straight up
      a = Math.max(a, Math.min(1, Math.max(0, (d[1] + 0.85) / 1.1)));
    }
    return a;
  });

  const raw: ActivitySegment[] = [];
  let start = -1;
  let peak = 0;
  for (let i = 0; i < times.length; i++) {
    const on = activity[i] >= threshold;
    if (on && start < 0) {
      start = i;
      peak = 0;
    }
    if (on) peak = Math.max(peak, activity[i]);
    if ((!on || i === times.length - 1) && start >= 0) {
      const end = on ? i : i - 1;
      raw.push({ startSec: times[start], endSec: times[end], peak });
      start = -1;
    }
  }
  // Merge short gaps, drop blips, pad edges.
  const merged: ActivitySegment[] = [];
  for (const seg of raw) {
    const last = merged[merged.length - 1];
    if (last && seg.startSec - last.endSec < 0.35) {
      last.endSec = seg.endSec;
      last.peak = Math.max(last.peak, seg.peak);
    } else {
      merged.push({ ...seg });
    }
  }
  const total = times[times.length - 1] ?? 0;
  const segments = merged
    .filter((s) => s.endSec - s.startSec >= 0.4)
    .map((s) => ({
      startSec: Math.max(0, s.startSec - 0.25),
      endSec: Math.min(total, s.endSec + 0.25),
      peak: s.peak,
    }));
  return { times, activity, segments };
}

// ---------------------------------------------------------------------------
// Retargeting
// ---------------------------------------------------------------------------

interface SideContext {
  side: HandSide;
  stream: HandStream | null;
  validMap: number[];
  yaw: Quat;
  hint: { i: number };
}

/** Swing-twist split of q around unit axis; returns the twist part. */
function twistAround(q: Quat, axis: Vec3): Quat {
  const p = vDot([q[0], q[1], q[2]], axis);
  const t: Quat = [axis[0] * p, axis[1] * p, axis[2] * p, q[3]];
  const l = Math.hypot(t[0], t[1], t[2], t[3]);
  if (l < 1e-9) return [0, 0, 0, 1];
  return [t[0] / l, t[1] / l, t[2] / l, t[3] / l];
}

function swingTo(parentWorld: Quat, dirWorld: Vec3): Quat {
  const dirLocal = qRotate(qConj(parentWorld), dirWorld);
  return qFromUnitVectors(DOWN, dirLocal);
}

function fallbackUpperArmDir(side: HandSide, forearmDir: Vec3): Vec3 {
  const out = side === 'right' ? -1 : 1;
  const raise = Math.min(1, Math.max(0, (forearmDir[1] + 0.2) / 0.9));
  return vNorm(vLerp([0.12 * out, -1, 0.05], [0.45 * out, -0.7, 0.55], raise));
}

export interface RetargetResult {
  payload: SignAnimationPayload;
  gloveYawDeg: number;
  notes: string[];
}

export function retargetRecording(rec: RokokoRecording, opts: RecordingRetargetOptions): RetargetResult {
  const notes: string[] = [];
  const fps = Math.max(5, Math.min(120, opts.fps));
  const start = Math.max(0, Math.min(opts.startSec, rec.durationSec));
  const end = Math.max(start + 1 / fps, Math.min(opts.endSec, rec.durationSec));
  const nFrames = Math.max(2, Math.round((end - start) * fps) + 1);
  const times = Array.from({ length: nFrames }, (_, k) => Math.min(end - start, k / fps));

  const gloveYawDeg = opts.gloveYawDeg ?? estimateGloveYawDeg(rec);
  const sides: SideContext[] = (['right', 'left'] as const).map((side) => {
    const stream = side === 'right' ? rec.right : rec.left;
    const offset = side === 'right' ? opts.rightYawOffsetDeg : opts.leftYawOffsetDeg;
    return {
      side,
      stream,
      validMap: stream ? validIndexMap(stream.times.map((_, i) => handFrameValid(stream, i))) : [],
      yaw: qFromAxisAngle([0, 1, 0], degToRad(gloveYawDeg + offset)),
      hint: { i: 0 },
    };
  });

  const bodyAxes = detectBodyAxes(rec);
  const faceMirror = detectFaceMirror(rec);
  const body = opts.useBodyArms || opts.useTorso ? rec.body : null;
  const face = rec.face;

  const torsoSeries = resolveBodyTorsoSeries(body);
  const medVec3 = (arr: Vec3[], fallback: Vec3): Vec3 => [
    median(arr.map((p) => p[0])),
    median(arr.map((p) => p[1])),
    median(arr.map((p) => p[2])),
  ].some((x) => !Number.isFinite(x))
    ? fallback
    : [median(arr.map((p) => p[0])), median(arr.map((p) => p[1])), median(arr.map((p) => p[2]))];

  // Neutral references (median over the whole recording) for torso (3-bone or shoulder-line) and head.
  let neutralShoulderDir: Vec3 | null = null;
  let hasThreeBoneTorso = false;
  let torsoPivotMode: 'lower' | 'centralBelow' | 'centralAbove' | 'none' = 'none';
  let neutralTorsoFrame: [Vec3, Vec3, Vec3] | null = null;
  const neutralLateralDir: Partial<Record<HandSide, Vec3>> = {};

  if (torsoSeries.leftShoulder && torsoSeries.rightShoulder && torsoSeries.leftShoulder.length > 0) {
    const lsArr = torsoSeries.leftShoulder.map((p) => mapBody(p, bodyAxes));
    const rsArr = torsoSeries.rightShoulder.map((p) => mapBody(p, bodyAxes));
    const dirs = lsArr.map((l, i) => vNorm(vSub(l, rsArr[i]), [1, 0, 0]));
    neutralShoulderDir = vNorm(medVec3(dirs, [1, 0, 0]), [1, 0, 0]);

    if (torsoSeries.lowerJoint && torsoSeries.lowerJoint.length === lsArr.length) {
      const lowArr = torsoSeries.lowerJoint.map((p) => mapBody(p, bodyAxes));
      const verticalSpans = lsArr.map((l, i) => 0.5 * (l[1] + rsArr[i][1]) - lowArr[i][1]);
      if (median(verticalSpans) > 0.04) {
        hasThreeBoneTorso = true;
        torsoPivotMode = 'lower';
        const topArr =
          torsoSeries.centralTop && torsoSeries.centralTop.length === lsArr.length
            ? torsoSeries.centralTop.map((p) => mapBody(p, bodyAxes))
            : lsArr.map((l, i) => vScale(vAdd(l, rsArr[i]), 0.5));

        const upVecs: Vec3[] = [];
        const leftLatVecs: Vec3[] = [];
        const rightLatVecs: Vec3[] = [];
        for (let i = 0; i < lsArr.length; i++) {
          const bCentral = vNorm(vSub(topArr[i], lowArr[i]), [0, 1, 0]);
          const bLeft = vSub(lsArr[i], lowArr[i]);
          const bRight = vSub(rsArr[i], lowArr[i]);
          const bLatSym = vNorm(vAdd(bLeft, bRight), [0, 1, 0]);
          upVecs.push(vNorm(vAdd(vScale(bCentral, 0.65), vScale(bLatSym, 0.35)), [0, 1, 0]));
          leftLatVecs.push(vNorm(bLeft, [0.4, 0.9, 0]));
          rightLatVecs.push(vNorm(bRight, [-0.4, 0.9, 0]));
        }
        const neutralUp = vNorm(medVec3(upVecs, [0, 1, 0]), [0, 1, 0]);
        neutralTorsoFrame = orthoFrame(neutralUp, neutralShoulderDir);
        neutralLateralDir.left = vNorm(medVec3(leftLatVecs, [0.4, 0.9, 0]), [0.4, 0.9, 0]);
        neutralLateralDir.right = vNorm(medVec3(rightLatVecs, [-0.4, 0.9, 0]), [-0.4, 0.9, 0]);
      }
    } else if (torsoSeries.centralTop && torsoSeries.centralTop.length === lsArr.length) {
      const topArr = torsoSeries.centralTop.map((p) => mapBody(p, bodyAxes));
      const neckOffsets = lsArr.map((l, i) => topArr[i][1] - 0.5 * (l[1] + rsArr[i][1]));
      const medOffset = median(neckOffsets);
      if (Math.abs(medOffset) > 0.04) {
        torsoPivotMode = medOffset > 0 ? 'centralAbove' : 'centralBelow';
        const upVecs = lsArr.map((l, i) => {
          const mid = vScale(vAdd(l, rsArr[i]), 0.5);
          return medOffset > 0
            ? vNorm(vSub(topArr[i], mid), [0, 1, 0])
            : vNorm(vSub(mid, topArr[i]), [0, 1, 0]);
        });
        const neutralUp = vNorm(medVec3(upVecs, [0, 1, 0]), [0, 1, 0]);
        neutralTorsoFrame = orthoFrame(neutralUp, neutralShoulderDir);
      }
    }
  }
  let neutralHeadQ: Quat | null = null;
  const faceOk = !!face && face.landmarks.leftEye.length > 0 && face.landmarks.chin.length > 0;
  if (opts.useHead && faceOk && face) {
    const med = (arr: Vec3[]): Vec3 => [median(arr.map((p) => p[0])), median(arr.map((p) => p[1])), median(arr.map((p) => p[2]))];
    const fr = headFrame(
      med(face.landmarks.leftEye),
      med(face.landmarks.rightEye),
      med(face.landmarks.forehead),
      med(face.landmarks.chin),
      faceMirror,
    );
    neutralHeadQ = qFrameToFrame(orthoFrame([0, 1, 0], [1, 0, 0]), fr);
  }
  if (opts.useHead && !faceOk) notes.push('Head rotation not available (no face.csv or no landmarks).');

  const bodyHint = { i: 0 };
  const faceHint = { i: 0 };
  const faceValid = face ? validIndexMap(face.detected) : [];
  const bodyValid = body ? validIndexMap(body.detected) : [];

  // Raw per-frame world-space quantities (before smoothing).
  const raw = {
    torso: [] as Quat[],
    clavicle: { right: [] as Vec3[], left: [] as Vec3[] },
    head: [] as Quat[],
    upper: { right: [] as Vec3[], left: [] as Vec3[] },
    fore: { right: [] as Vec3[], left: [] as Vec3[] },
    hand: { right: [] as Quat[], left: [] as Quat[] },
    finger: { right: {} as Record<string, Vec3[]>, left: {} as Record<string, Vec3[]> },
    curls: { right: {} as Record<FingerName, number[]>, left: {} as Record<FingerName, number[]> },
  };
  for (const side of ['right', 'left'] as const) {
    for (const f of FINGERS) {
      raw.curls[side][f] = [];
      for (const ph of PHALANGES) raw.finger[side][`${f}${ph}`] = [];
    }
  }

  const morphTargets = Array.from(new Set(Object.values(FACE_BLENDSHAPE_MAP))).sort();
  const morphTracks: Record<string, number[]> = {};
  for (const m of morphTargets) morphTracks[m] = [];

  for (let k = 0; k < nFrames; k++) {
    const t = start + times[k];

    // --- 3-bone torso (1 central + 2 lateral from lower joint) or shoulder line ---
    let torso: Quat = IDENTITY;
    let latLeft: Vec3 = neutralLateralDir.left ?? [0.4, 0.9, 0];
    let latRight: Vec3 = neutralLateralDir.right ?? [-0.4, 0.9, 0];
    let bodyI = 0;
    let bodyF = 0;
    if (body) {
      [bodyI, bodyF] = locate(body.times, t, bodyHint);
      if ((bodyValid[bodyI] ?? bodyI) !== bodyI) {
        bodyI = bodyValid[bodyI];
        bodyF = 0;
      }
    }
    if (opts.useTorso && body && neutralShoulderDir && torsoSeries.leftShoulder && torsoSeries.rightShoulder) {
      const l = mapBody(sampleVec(torsoSeries.leftShoulder, bodyI, bodyF), bodyAxes);
      const r = mapBody(sampleVec(torsoSeries.rightShoulder, bodyI, bodyF), bodyAxes);
      const sideDir = vNorm(vSub(l, r), [1, 0, 0]);
      if (torsoPivotMode === 'lower' && hasThreeBoneTorso && neutralTorsoFrame && torsoSeries.lowerJoint) {
        const low = mapBody(sampleVec(torsoSeries.lowerJoint, bodyI, bodyF), bodyAxes);
        const top = torsoSeries.centralTop
          ? mapBody(sampleVec(torsoSeries.centralTop, bodyI, bodyF), bodyAxes)
          : vScale(vAdd(l, r), 0.5);
        const bCentral = vNorm(vSub(top, low), [0, 1, 0]);
        const bLeft = vSub(l, low);
        const bRight = vSub(r, low);
        const bLatSym = vNorm(vAdd(bLeft, bRight), [0, 1, 0]);
        const upDir = vNorm(vAdd(vScale(bCentral, 0.65), vScale(bLatSym, 0.35)), [0, 1, 0]);
        const curTorsoFrame = orthoFrame(upDir, sideDir);
        torso = qClampAngle(qFrameToFrame(neutralTorsoFrame, curTorsoFrame), degToRad(28));
        latLeft = vNorm(bLeft, latLeft);
        latRight = vNorm(bRight, latRight);
      } else if (
        (torsoPivotMode === 'centralAbove' || torsoPivotMode === 'centralBelow') &&
        neutralTorsoFrame &&
        torsoSeries.centralTop
      ) {
        const top = mapBody(sampleVec(torsoSeries.centralTop, bodyI, bodyF), bodyAxes);
        const mid = vScale(vAdd(l, r), 0.5);
        const upDir =
          torsoPivotMode === 'centralAbove'
            ? vNorm(vSub(top, mid), [0, 1, 0])
            : vNorm(vSub(mid, top), [0, 1, 0]);
        const curTorsoFrame = orthoFrame(upDir, sideDir);
        torso = qClampAngle(qFrameToFrame(neutralTorsoFrame, curTorsoFrame), degToRad(28));
      } else {
        torso = qClampAngle(qFromUnitVectors(neutralShoulderDir, sideDir), degToRad(25));
      }
    }
    raw.torso.push(torso);
    raw.clavicle.left.push(latLeft);
    raw.clavicle.right.push(latRight);

    // --- head from face landmarks ------------------------------------------
    let head: Quat = IDENTITY;
    let fi = 0;
    let ff = 0;
    if (face) {
      [fi, ff] = locate(face.times, t, faceHint);
      if ((faceValid[fi] ?? fi) !== fi) {
        fi = faceValid[fi];
        ff = 0;
      }
    }
    if (neutralHeadQ && face) {
      const fr = headFrame(
        sampleVec(face.landmarks.leftEye, fi, ff),
        sampleVec(face.landmarks.rightEye, fi, ff),
        sampleVec(face.landmarks.forehead, fi, ff),
        sampleVec(face.landmarks.chin, fi, ff),
        faceMirror,
      );
      const cur = qFrameToFrame(orthoFrame([0, 1, 0], [1, 0, 0]), fr);
      const rel = qMul(cur, qConj(neutralHeadQ));
      head = qClampAngle(qSlerp(IDENTITY, rel, Math.max(0, opts.headGain)), degToRad(40));
    }
    raw.head.push(head);

    // --- face morphs --------------------------------------------------------
    if (face) {
      const frameMorphs: Record<string, number> = {};
      for (const m of morphTargets) frameMorphs[m] = 0;
      if (opts.useFaceMorphs) {
        for (const [arkit, target] of Object.entries(FACE_BLENDSHAPE_MAP)) {
          const series = face.blendshapes[arkit];
          if (!series) continue;
          frameMorphs[target] = Math.max(frameMorphs[target], sampleNum(series, fi, ff));
        }
      }
      for (const m of morphTargets) morphTracks[m].push(frameMorphs[m]);
    } else {
      for (const m of morphTargets) morphTracks[m].push(0);
    }

    // --- arms, hands and fingers -------------------------------------------
    for (const ctx of sides) {
      const side = ctx.side;
      const s = ctx.stream;
      if (!s) {
        raw.fore[side].push(DOWN);
        raw.upper[side].push(fallbackUpperArmDir(side, DOWN));
        raw.hand[side].push(IDENTITY);
        for (const f of FINGERS) {
          raw.curls[side][f].push(0);
          for (const ph of PHALANGES) raw.finger[side][`${f}${ph}`].push(DOWN);
        }
        continue;
      }
      let [hi, hf] = locate(s.times, t, ctx.hint);
      const hiValid = ctx.validMap[hi] ?? hi;
      if (hiValid !== hi) {
        hi = hiValid;
        hf = 0;
      }
      const P = (j: RokokoJoint): Vec3 => qRotate(ctx.yaw, sampleVec(s.positions[j], hi, hf));

      const fore = vNorm(vSub(P('Hand'), P('ForeArm')));
      raw.fore[side].push(fore);

      // Upper arm: camera body when available, blended by landmark visibility.
      let upper = fallbackUpperArmDir(side, fore);
      if (opts.useBodyArms && body) {
        const shoulderKey = `${side}_shoulder`;
        const elbowKey = `${side}_elbow`;
        const sh = body.points[shoulderKey] ?? (side === 'left' ? torsoSeries.leftShoulder : torsoSeries.rightShoulder);
        const el = body.points[elbowKey] ?? body.points[side === 'left' ? 'elbow_l' : 'elbow_r'];
        if (sh && el) {
          const dir = vNorm(vSub(mapBody(sampleVec(el, bodyI, bodyF), bodyAxes), mapBody(sampleVec(sh, bodyI, bodyF), bodyAxes)));
          const vis = Math.min(sampleNum(body.visibility[elbowKey] ?? [], bodyI, bodyF) || 1, sampleNum(body.visibility[shoulderKey] ?? [], bodyI, bodyF) || 1);
          const w = Math.min(1, Math.max(0, (vis - 0.25) / 0.35));
          upper = vNorm(vLerp(upper, dir, w));
        }
      }
      raw.upper[side].push(upper);

      // Hand frame from the palm joints.
      const wrist = P('Hand');
      const i2 = P('Finger2Proximal');
      const i5 = P('Finger5Proximal');
      const knuckles = vScale(vAdd(vAdd(i2, P('Finger3Proximal')), vAdd(P('Finger4Proximal'), i5)), 0.25);
      const handWorldFrame = orthoFrame(vSub(knuckles, wrist), vSub(i2, i5));
      const handWorld = qFrameToFrame(restHandFrame(side), handWorldFrame);
      raw.hand[side].push(handWorld);

      for (const f of FINGERS) {
        FINGER_SEGMENTS[f].forEach(([a, b], idx) => {
          raw.finger[side][`${f}${PHALANGES[idx]}`].push(vNorm(vSub(P(b), P(a)), handWorldFrame[0]));
        });
        const [ca, cb] = FINGER_CURL_SEGMENT[f];
        const fd = vNorm(vSub(P(cb), P(ca)));
        const ref = f === 'thumb' ? vNorm(vSub(P('Finger1Proximal'), P('Finger1Metacarpal'))) : handWorldFrame[0];
        raw.curls[side][f].push(radToDeg(Math.acos(Math.max(-1, Math.min(1, vDot(fd, ref))))));
      }
    }
  }

  // ---- smoothing ------------------------------------------------------------
  const sm = Math.max(0, Math.min(1, opts.smoothing));
  const minCutoff = 8 - 7 * sm; // 8 Hz (light) .. 1 Hz (strong)
  const beta = 0.3;
  if (sm > 0) {
    raw.torso = smoothQuatTrack(raw.torso, fps, minCutoff * 0.6, beta);
    raw.head = smoothQuatTrack(raw.head, fps, minCutoff, beta);
    for (const side of ['right', 'left'] as const) {
      if (hasThreeBoneTorso) {
        raw.clavicle[side] = smoothVecTrack(raw.clavicle[side], fps, minCutoff * 0.6, beta);
      }
      // Camera elbows jitter more than the gloves: filter them harder.
      raw.upper[side] = smoothVecTrack(raw.upper[side], fps, minCutoff * 0.6, beta);
      raw.fore[side] = smoothVecTrack(raw.fore[side], fps, minCutoff, beta);
      raw.hand[side] = smoothQuatTrack(raw.hand[side], fps, minCutoff, beta);
      for (const key of Object.keys(raw.finger[side])) {
        raw.finger[side][key] = smoothVecTrack(raw.finger[side][key], fps, minCutoff * 1.2, beta);
      }
    }
  }

  // ---- forward kinematics into local bone tracks ----------------------------
  const tracks: Record<string, Quat[]> = {};
  const push = (name: string, q: Quat) => {
    (tracks[name] ??= []).push(q);
  };
  const qaFrames: QaFrame[] = [];
  const fadeFrames = Math.max(0, Math.round(opts.fadeSec * fps));
  const prevTwistAngle: Record<HandSide, number | null> = { right: null, left: null };

  for (let k = 0; k < nFrames; k++) {
    let fade = 1;
    if (fadeFrames > 0) {
      if (k < fadeFrames) fade = 0.5 - 0.5 * Math.cos(Math.PI * (k / fadeFrames));
      else if (k >= nFrames - fadeFrames) fade = 0.5 - 0.5 * Math.cos(Math.PI * ((nFrames - 1 - k) / fadeFrames));
    }
    const blend = (name: string, q: Quat) => push(name, qSlerp(REST_POSE[name] ?? IDENTITY, q, fade));

    const torsoHalf = qSlerp(IDENTITY, raw.torso[k], 0.5);
    const spineWorld = torsoHalf;
    const chestWorld = qMul(spineWorld, torsoHalf);
    push('hips', IDENTITY);
    blend('spine', torsoHalf);
    blend('chest', torsoHalf);

    // Head relative to the chest, split 35/65 between neck and head.
    const headLocalTotal = qNormalize(qMul(qConj(chestWorld), raw.head[k]));
    const neck = qSlerp(IDENTITY, headLocalTotal, 0.35);
    blend('neck', neck);
    blend('head', qNormalize(qMul(qConj(neck), headLocalTotal)));

    const qaHands: Partial<Record<HandSide, QaHandFrame>> = {};
    for (const side of ['right', 'left'] as const) {
      let shoulderLocal: Quat = IDENTITY;
      const neutralLat = neutralLateralDir[side];
      if (opts.useTorso && hasThreeBoneTorso && neutralLat) {
        const expectedLatWorld = qRotate(chestWorld, neutralLat);
        const actualLatWorld = raw.clavicle[side][k];
        const chestInv = qConj(chestWorld);
        const rawClav = qFromUnitVectors(
          qRotate(chestInv, expectedLatWorld),
          qRotate(chestInv, actualLatWorld),
        );
        shoulderLocal = qClampAngle(rawClav, degToRad(14));
      }
      blend(`${side}Shoulder`, shoulderLocal);
      const shoulderLocalFaded = tracks[`${side}Shoulder`][k];
      const shoulderWorld = qMul(chestWorld, shoulderLocalFaded);

      const upperLocal = swingTo(shoulderWorld, raw.upper[side][k]);
      const upperWorld = qMul(shoulderWorld, upperLocal);
      let lowerLocal = swingTo(upperWorld, raw.fore[side][k]);
      let lowerWorld = qMul(upperWorld, lowerLocal);
      let handLocal = qNormalize(qMul(qConj(lowerWorld), raw.hand[side][k]));

      // Move 70% of the forearm twist (pronation) from the wrist into the forearm.
      // Unwrap the twist angle over time: near 180° the shortest-path twist flips between
      // +180° and -180°, and 70% of it would make the forearm jump by ~110° in one frame.
      const twist = twistAround(handLocal, [0, 1, 0]);
      let twistAngle = 2 * Math.atan2(twist[1], twist[3]);
      twistAngle = Math.atan2(Math.sin(twistAngle), Math.cos(twistAngle));
      const prevTwist = prevTwistAngle[side];
      if (prevTwist !== null) {
        while (twistAngle - prevTwist > Math.PI) twistAngle -= 2 * Math.PI;
        while (twistAngle - prevTwist < -Math.PI) twistAngle += 2 * Math.PI;
      }
      prevTwistAngle[side] = twistAngle;
      const share = qFromAxisAngle([0, 1, 0], 0.7 * twistAngle);
      lowerLocal = qNormalize(qMul(lowerLocal, share));
      lowerWorld = qMul(upperWorld, lowerLocal);
      handLocal = qNormalize(qMul(qConj(share), handLocal));
      const handWorld = qMul(lowerWorld, handLocal);

      blend(`${side}UpperArm`, upperLocal);
      blend(`${side}LowerArm`, lowerLocal);
      blend(`${side}Hand`, handLocal);

      for (const f of FINGERS) {
        let parentWorld = handWorld;
        for (const ph of PHALANGES) {
          const name = `${side}${f[0].toUpperCase()}${f.slice(1)}${ph}`;
          const local = swingTo(parentWorld, raw.finger[side][`${f}${ph}`][k]);
          parentWorld = qMul(parentWorld, local);
          blend(name, local);
        }
      }

      const wristPos = vAdd(
        vAdd(SHOULDER_POS[side], vScale(raw.upper[side][k], UPPER_ARM_LEN)),
        vScale(raw.fore[side][k], LOWER_ARM_LEN),
      );
      const round3 = (v: Vec3) => v.map((x) => Math.round(x * 1e4) / 1e4);
      const curls: Record<string, number> = {};
      for (const f of FINGERS) curls[f] = Math.round(raw.curls[side][f][k] * fade * 100) / 100;
      qaHands[side] = {
        target_wrist: round3(wristPos) as [number, number, number],
        achieved_wrist: round3(wristPos) as [number, number, number],
        curls_deg_g2: curls,
        curls_deg_g1: curls,
        mean_emf_residual: 0,
      };
    }

    const morphs: Record<string, number> = {};
    for (const m of morphTargets) {
      morphTracks[m][k] = Math.round(Math.max(0, Math.min(1, morphTracks[m][k] * fade)) * 1e4) / 1e4;
      morphs[m] = morphTracks[m][k];
    }
    qaFrames.push({
      t: Math.round(times[k] * 1e4) / 1e4,
      right: qaHands.right!,
      left: qaHands.left!,
      morphs,
    });
  }

  const boneTracks: Record<string, number[]> = {};
  for (const [name, quats] of Object.entries(tracks)) {
    const flat: number[] = [];
    let prev: Quat | null = null;
    for (let q of quats) {
      q = qNormalize(q);
      if (prev && qDot(prev, q) < 0) q = [-q[0], -q[1], -q[2], -q[3]];
      prev = q;
      flat.push(...q.map((x) => Math.round(x * 1e5) / 1e5));
    }
    boneTracks[name] = flat;
  }

  const clipId = `${rec.name}@${start.toFixed(2)}-${end.toFixed(2)}`;
  const payload: SignAnimationPayload = {
    classId: opts.classId,
    phrase: opts.phrase,
    category: opts.category,
    description: opts.description,
    prototypeClipId: clipId,
    fps,
    duration: Math.round((end - start) * 1e4) / 1e4,
    times: times.map((t) => Math.round(t * 1e4) / 1e4),
    boneTracks,
    morphTracks,
    qaComparison: {
      group1VsGroup2Right: [],
      group1VsGroup2Left: [],
      frames: qaFrames,
    },
    source: {
      kind: 'recording',
      recordingName: rec.name,
      startSec: Math.round(start * 1000) / 1000,
      endSec: Math.round(end * 1000) / 1000,
      gloveYawDeg: Math.round(gloveYawDeg * 10) / 10,
      leftYawOffsetDeg: opts.leftYawOffsetDeg,
      rightYawOffsetDeg: opts.rightYawOffsetDeg,
      sourceFrameIntervalMs: Math.round(rec.medianFrameIntervalMs * 10) / 10,
      streams: {
        leftHand: !!rec.left,
        rightHand: !!rec.right,
        face: !!rec.face,
        body: !!rec.body,
      },
    },
  };
  return { payload, gloveYawDeg, notes };
}
