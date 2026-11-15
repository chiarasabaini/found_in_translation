/**
 * Placeholder for the text-to-motion model.
 *
 * It splits the phrase into dictionary signs and synthesizes recording-format CSV files
 * (solved hands, body, face) at 30 Hz with simple kinematics: per-sign handshapes, palm and
 * finger directions, the body-relative anchors and motion arcs of config/dictionary.yaml,
 * and 2-bone arm IK. The output has the format the real model must produce, so the rest of
 * the site (parser, retargeting, avatar) runs exactly as it will with the real model.
 *
 * The handshapes are illustrative only: they are NOT validated LIS signs.
 */

import type { AnchorConfig, DictionaryClass } from '../types/lis';

export { glossOf, planSigns, type PlannedSign };
import { glossOf, planSigns, type PlannedSign } from './signPlanner';
import { ROKOKO_HAND_JOINTS, type RecordingFileInput, type RokokoJoint } from '../recording/rokokoRecording';
import {
  qFromAxisAngle,
  qFromBasis,
  qRotate,
  vAdd,
  vCross,
  vDot,
  vLen,
  vLerp,
  vNorm,
  vScale,
  vSub,
  type Quat,
  type Vec3,
} from '../recording/math3d';
import type { GeneratedClip, GeneratedMotion, GlossSegment, MotionGenerator, MotionRequest } from './motionModel';

type Side = 'right' | 'left';

interface HandShape {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  little: number;
  /** Finger abduction, 0 = together, 1 = spread. */
  spread: number;
}

const SHAPES = {
  open: { thumb: 0.05, index: 0, middle: 0, ring: 0, little: 0, spread: 1 },
  flat: { thumb: 0.45, index: 0, middle: 0, ring: 0, little: 0, spread: 0.1 },
  fist: { thumb: 0.75, index: 1, middle: 1, ring: 1, little: 1, spread: 0 },
  point: { thumb: 0.8, index: 0, middle: 1, ring: 1, little: 1, spread: 0 },
  cup: { thumb: 0.35, index: 0.45, middle: 0.45, ring: 0.45, little: 0.45, spread: 0.3 },
  y: { thumb: 0, index: 1, middle: 1, ring: 1, little: 0, spread: 0.8 },
  relaxed: { thumb: 0.2, index: 0.15, middle: 0.2, ring: 0.25, little: 0.3, spread: 0.2 },
} satisfies Record<string, HandShape>;

/** Hand pose of one sign. Directions are in the avatar frame (X = signer's left, Y up, Z forward). */
interface HandPose {
  shape: HandShape;
  /** Optional second shape: the hand moves between `shape` and `shapeTo` while signing. */
  shapeTo?: HandShape;
  fingers: Vec3;
  palm: Vec3;
}

interface MockSign {
  right: HandPose;
  /** Left hand pose for two-handed signs. */
  left?: HandPose;
  face: Record<string, number>;
  head?: 'nod' | 'shake' | 'tilt';
}

/** Placeholder sign descriptions per dictionary class (see class descriptions). */
const MOCK_SIGNS: Record<string, MockSign> = {
  ciao: {
    right: { shape: SHAPES.open, fingers: [0, 1, 0.1], palm: [0, 0, 1] },
    face: { mouthSmileLeft: 0.6, mouthSmileRight: 0.6, browInnerUp: 0.2 },
    head: 'tilt',
  },
  grazie: {
    right: { shape: SHAPES.flat, fingers: [0.1, 1, 0.2], palm: [0, 0, -1] },
    face: { mouthSmileLeft: 0.35, mouthSmileRight: 0.35 },
    head: 'nod',
  },
  per_favore: {
    right: { shape: SHAPES.flat, fingers: [1, 0.4, 0], palm: [0, 0, -1] },
    face: { browInnerUp: 0.45, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
  },
  come_stai: {
    right: { shape: SHAPES.cup, fingers: [0.2, 0.3, 1], palm: [0, 1, 0] },
    left: { shape: SHAPES.cup, fingers: [-0.2, 0.3, 1], palm: [0, 1, 0] },
    face: { browInnerUp: 0.7 },
    head: 'tilt',
  },
  mi_chiamo: {
    right: { shape: SHAPES.point, shapeTo: SHAPES.open, fingers: [0.3, 0.2, -1], palm: [0, -1, 0] },
    face: { mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
  },
  non_capisco: {
    right: { shape: SHAPES.point, fingers: [0.2, 1, 0], palm: [1, 0, 0.3] },
    face: { browDownLeft: 0.75, browDownRight: 0.75, mouthFrownLeft: 0.4, mouthFrownRight: 0.4 },
    head: 'shake',
  },
  aiuto: {
    right: { shape: SHAPES.fist, fingers: [0.3, 0.2, 1], palm: [1, 0, 0] },
    left: { shape: SHAPES.flat, fingers: [-0.3, 0, 1], palm: [0, 1, 0] },
    face: { browInnerUp: 0.85, jawOpen: 0.25 },
  },
  acqua: {
    right: { shape: SHAPES.point, fingers: [0, 1, 0.2], palm: [1, 0, 0] },
    face: { jawOpen: 0.15 },
  },
  bagno: {
    right: { shape: SHAPES.y, fingers: [0, 1, 0.1], palm: [0, 0, 1] },
    face: { browInnerUp: 0.5 },
    head: 'tilt',
  },
  arrivederci: {
    right: { shape: SHAPES.open, shapeTo: { ...SHAPES.cup, thumb: 0.5 }, fingers: [0, 1, 0.2], palm: [0, 0, 1] },
    face: { mouthSmileLeft: 0.5, mouthSmileRight: 0.5 },
  },
};

const DEFAULT_SIGN: MockSign = {
  right: { shape: SHAPES.open, fingers: [0, 1, 0.2], palm: [0, 0, 1] },
  face: {},
};

const FPS = 30;
const SHOULDER: Record<Side, Vec3> = { right: [-0.195, 1.42, 0], left: [0.195, 1.42, 0] };
const NECK: Vec3 = [0, 1.42, 0];
const UPPER_LEN = 0.28;
const LOWER_LEN = 0.25;
/** Glove forearm length (ForeArm -> Hand), as in the Rokoko solver output. */
const GLOVE_FOREARM_LEN = 0.289;
const REST_SEC = 0.35;
const TRANSITION_SEC = 0.35;

interface HandState {
  wrist: Vec3;
  shape: HandShape;
  fingers: Vec3;
  palm: Vec3;
}

interface FrameState {
  right: HandState;
  left: HandState;
  face: Record<string, number>;
  head: Quat;
}

const outSign = (side: Side) => (side === 'right' ? -1 : 1);

function restHand(side: Side): HandState {
  const o = outSign(side);
  return {
    wrist: vAdd(SHOULDER[side], [0.03 * o, -0.52, 0.06]),
    shape: SHAPES.relaxed,
    fingers: [0, -1, 0.05],
    palm: [-o, 0, 0],
  };
}

/** Port of pipeline/retargeting.py _compute_anchor_trajectory. */
function anchorTrajectory(anchor: AnchorConfig, u: number, side: Side, active: boolean): Vec3 {
  const base = side === 'right' ? anchor.right_hand : anchor.left_hand;
  if (!active) return [base[0], base[1], base[2]];
  const amp = anchor.motion_amplitude ?? [0, 0, 0];
  const env = Math.sin(Math.PI * u);
  const phase = 2 * Math.PI * 2 * u;
  let dx = 0;
  let dy = 0;
  let dz = 0;
  switch (anchor.motion_arc) {
    case 'wave_side':
      dx = amp[0] * Math.sin(phase) * env;
      dy = amp[1] * Math.cos(phase) * env;
      break;
    case 'chin_forward':
      dz = amp[2] * Math.sin(Math.PI * u);
      dy = amp[1] * Math.sin(Math.PI * u);
      break;
    case 'chest_circle':
      dx = amp[0] * Math.cos(phase) * env;
      dy = amp[1] * Math.sin(phase) * env;
      break;
    case 'symmetric_open': {
      const sign = side === 'right' ? -1 : 1;
      dx = sign * amp[0] * Math.sin(Math.PI * u);
      dz = amp[2] * Math.sin(Math.PI * u);
      break;
    }
    case 'chest_to_front':
      dx = amp[0] * u;
      dz = amp[2] * Math.sin(Math.PI * u);
      break;
    case 'temple_twist':
      dx = amp[0] * Math.sin(phase) * env;
      dz = amp[2] * env;
      break;
    case 'lift_together':
      dy = amp[1] * Math.sin(Math.PI * u);
      dz = amp[2] * Math.sin(Math.PI * u);
      break;
    case 'chin_tap':
      dy = amp[1] * Math.abs(Math.sin(phase)) * env;
      break;
    case 'wrist_shake':
      dx = amp[0] * Math.sin(phase * 1.4) * env;
      break;
    case 'wave_farewell':
      dx = amp[0] * Math.sin(phase) * env;
      dy = amp[1] * Math.sin(Math.PI * u);
      break;
  }
  return [base[0] + dx, base[1] + dy, base[2] + dz];
}

function mixShape(a: HandShape, b: HandShape, t: number): HandShape {
  const m = (x: number, y: number) => x + (y - x) * t;
  return {
    thumb: m(a.thumb, b.thumb),
    index: m(a.index, b.index),
    middle: m(a.middle, b.middle),
    ring: m(a.ring, b.ring),
    little: m(a.little, b.little),
    spread: m(a.spread, b.spread),
  };
}

function mixHand(a: HandState, b: HandState, t: number): HandState {
  return {
    wrist: vLerp(a.wrist, b.wrist, t),
    shape: mixShape(a.shape, b.shape, t),
    fingers: vNorm(vLerp(a.fingers, b.fingers, t)),
    palm: vNorm(vLerp(a.palm, b.palm, t), [0, 0, 1]),
  };
}

function signHand(cls: DictionaryClass, sign: MockSign, side: Side, u: number): HandState {
  const active = side === 'right' || cls.twoHanded;
  const pose = side === 'right' ? sign.right : sign.left;
  if (!active || !pose) {
    // The passive hand rests near its dictionary anchor (or at the side of the body).
    const rest = restHand(side);
    return cls.twoHanded ? { ...rest, wrist: anchorTrajectory(cls.anchor, u, side, false) } : rest;
  }
  const shape = pose.shapeTo ? mixShape(pose.shape, pose.shapeTo, 0.5 - 0.5 * Math.cos(2 * Math.PI * 2 * u)) : pose.shape;
  return {
    wrist: anchorTrajectory(cls.anchor, u, side, true),
    shape,
    fingers: vNorm(pose.fingers),
    palm: vNorm(pose.palm),
  };
}

function headRotation(kind: MockSign['head'], u: number): Quat {
  const env = Math.sin(Math.PI * u);
  if (kind === 'nod') return qFromAxisAngle([1, 0, 0], 0.18 * env * Math.sin(2 * Math.PI * 2 * u) + 0.06 * env);
  if (kind === 'shake') return qFromAxisAngle([0, 1, 0], 0.22 * env * Math.sin(2 * Math.PI * 3 * u));
  if (kind === 'tilt') return qFromAxisAngle([0, 0, 1], 0.1 * env);
  return [0, 0, 0, 1];
}

/** Elbow position from a 2-bone IK with the elbow out, down and slightly forward. */
function solveElbow(side: Side, wrist: Vec3): { elbow: Vec3; wrist: Vec3 } {
  const sh = SHOULDER[side];
  const diff = vSub(wrist, sh);
  const dist = Math.max(Math.abs(UPPER_LEN - LOWER_LEN) + 0.05, Math.min((UPPER_LEN + LOWER_LEN) * 0.985, vLen(diff)));
  const dir = vNorm(diff);
  const cosA = (UPPER_LEN * UPPER_LEN + dist * dist - LOWER_LEN * LOWER_LEN) / (2 * UPPER_LEN * dist);
  const a = Math.acos(Math.max(-1, Math.min(1, cosA)));
  const pole: Vec3 = [0.7 * outSign(side), -0.6, 0.15];
  const bend = vNorm(vSub(pole, vScale(dir, vDot(pole, dir))), [0, -1, 0]);
  const elbow = vAdd(sh, vAdd(vScale(dir, UPPER_LEN * Math.cos(a)), vScale(bend, UPPER_LEN * Math.sin(a))));
  return { elbow, wrist: vAdd(sh, vScale(dir, dist)) };
}

/**
 * Joint positions of one hand relative to the ForeArm joint (the elbow), as the Rokoko solver
 * reports them: Y-up, right-handed, signer facing +Z.
 */
function handJoints(side: Side, state: HandState, forearmDir: Vec3): Record<RokokoJoint, Vec3> {
  const s = side === 'right' ? 1 : -1; // radial (thumb side) along local +X for the right hand
  // Hand orientation: local Y = fingers, local Z = palm, local X = Y x Z.
  const y = vNorm(state.fingers);
  let z = vSub(state.palm, vScale(y, vDot(state.palm, y)));
  if (vLen(z) < 1e-3) z = vCross(y, [1, 0, 0]);
  z = vNorm(z);
  const x = vCross(y, z);
  const rot = qFromBasis(x, y, z);
  const hand = vScale(forearmDir, GLOVE_FOREARM_LEN);
  const P = (local: Vec3): Vec3 => vAdd(hand, qRotate(rot, local));

  const out = { ForeArm: [0, 0, 0] as Vec3, Hand: hand } as Record<RokokoJoint, Vec3>;
  const sh = state.shape;

  // Fingers 2..5: index, middle, ring, little.
  const fingerCurl = [sh.index, sh.middle, sh.ring, sh.little];
  const baseX = [0.024, 0.004, -0.015, -0.031];
  const knuckleY = [0.088, 0.092, 0.086, 0.078];
  const lengths = [
    [0.042, 0.024, 0.02],
    [0.047, 0.028, 0.022],
    [0.044, 0.027, 0.021],
    [0.034, 0.019, 0.018],
  ];
  const abduction = [0.12, 0.03, -0.06, -0.15];
  for (let k = 0; k < 4; k++) {
    const n = k + 2;
    const c = fingerCurl[k];
    const a = sh.spread * abduction[k];
    const d0: Vec3 = [Math.sin(a) * s, Math.cos(a), 0];
    const flex = [c * 1.45, c * 1.7, c * 1.1];
    out[`Finger${n}Metacarpal` as RokokoJoint] = P([baseX[k] * 0.5 * s, 0.025, 0]);
    let p: Vec3 = [baseX[k] * s, knuckleY[k], 0];
    out[`Finger${n}Proximal` as RokokoJoint] = P(p);
    const names = ['Medial', 'Distal', 'Tip'];
    let phi = 0;
    for (let j = 0; j < 3; j++) {
      phi += flex[j];
      const dir: Vec3 = vAdd(vScale(d0, Math.cos(phi)), [0, 0, Math.sin(phi)]);
      p = vAdd(p, vScale(dir, lengths[k][j]));
      out[`Finger${n}${names[j]}` as RokokoJoint] = P(p);
    }
  }

  // Thumb: from abducted (open) toward across-the-palm (closed).
  const t = sh.thumb;
  const open = vNorm([(0.75 + 0.15 * sh.spread) * s, 0.6, 0.2]);
  const closed = vNorm([-0.35 * s, 0.55, 0.75]);
  let p: Vec3 = [0.02 * s, 0.02, 0.012];
  out.Finger1Metacarpal = P(p);
  const thumbLen = [0.045, 0.032, 0.027];
  const thumbNames = ['Proximal', 'Distal', 'Tip'];
  for (let j = 0; j < 3; j++) {
    const dir = vNorm(vAdd(vLerp(open, closed, Math.min(1, t * (0.6 + 0.3 * j))), [0, 0, 0.25 * t * j]));
    p = vAdd(p, vScale(dir, thumbLen[j]));
    out[`Finger1${thumbNames[j]}` as RokokoJoint] = P(p);
  }
  return out;
}

/** Face landmarks (MediaPipe indices) in normalized image coordinates for a head rotation. */
const FACE_POINTS: Array<[number, Vec3]> = [
  [33, [-0.045, 0.03, 0.02]], // signer's right eye, outer corner
  [263, [0.045, 0.03, 0.02]], // signer's left eye, outer corner
  [10, [0, 0.09, 0.02]], // forehead
  [152, [0, -0.11, 0]], // chin
  [1, [0, 0, 0.06]], // nose tip
];

function faceLandmarks(head: Quat): Array<[number, Vec3]> {
  const scale = 2.2;
  return FACE_POINTS.map(([i, p]) => {
    const r = qRotate(head, p);
    // avatar (X left, Y up, Z toward camera) -> image (x right, y down, z negative toward camera)
    return [i, [0.5 + r[0] * scale, 0.42 - r[1] * scale, -r[2] * scale]];
  });
}

const f5 = (v: number) => (Math.abs(v) < 5e-6 ? '0' : v.toFixed(5));

/** Synthesizes recording-format CSV files for a sequence of signs. */
export function synthesizeRecording(signs: PlannedSign[]): { files: RecordingFileInput[]; segments: GlossSegment[] } {
  // Timeline: rest, sign, transition, sign, ..., rest.
  interface Item {
    start: number;
    end: number;
    sign: PlannedSign | null;
  }
  const items: Item[] = [];
  let t = REST_SEC;
  signs.forEach((s, idx) => {
    const dur = Math.max(0.8, s.cls.anchor.duration_sec || s.cls.duration || 1.5);
    items.push({ start: t, end: t + dur, sign: s });
    t += dur + (idx < signs.length - 1 ? TRANSITION_SEC : 0);
  });
  const total = t + REST_SEC + 0.1;
  const nFrames = Math.round(total * FPS) + 1;

  const stateAt = (time: number): FrameState => {
    const signState = (item: Item, u: number): FrameState => {
      const def = MOCK_SIGNS[item.sign!.cls.id] ?? DEFAULT_SIGN;
      const env = Math.min(1, Math.sin(Math.PI * Math.min(1, Math.max(0, u))) * 3);
      const face: Record<string, number> = {};
      for (const [k, v] of Object.entries(def.face)) face[k] = v * env;
      return {
        right: signHand(item.sign!.cls, def, 'right', u),
        left: signHand(item.sign!.cls, def, 'left', u),
        face,
        head: headRotation(def.head, u),
      };
    };
    const rest: FrameState = { right: restHand('right'), left: restHand('left'), face: {}, head: [0, 0, 0, 1] };
    const smooth = (x: number) => x * x * (3 - 2 * x);
    const mix = (a: FrameState, b: FrameState, k: number): FrameState => {
      const face: Record<string, number> = {};
      for (const key of new Set([...Object.keys(a.face), ...Object.keys(b.face)])) {
        face[key] = (a.face[key] ?? 0) * (1 - k) + (b.face[key] ?? 0) * k;
      }
      const hq = vNorm(vLerp([a.head[0], a.head[1], a.head[2]], [b.head[0], b.head[1], b.head[2]], k), [0, 0, 0]);
      const angle = (q: Quat) => 2 * Math.acos(Math.min(1, Math.abs(q[3])));
      const ang = angle(a.head) * (1 - k) + angle(b.head) * k;
      return {
        right: mixHand(a.right, b.right, k),
        left: mixHand(a.left, b.left, k),
        face,
        head: vLen(hq) > 0 ? qFromAxisAngle(hq, ang) : [0, 0, 0, 1],
      };
    };
    if (!items.length) return rest;
    const first = items[0];
    const last = items[items.length - 1];
    if (time <= first.start) {
      const k = smooth(Math.max(0, Math.min(1, time / first.start)));
      return mix(rest, signState(first, 0), k);
    }
    if (time >= last.end) {
      const k = smooth(Math.max(0, Math.min(1, (time - last.end) / REST_SEC)));
      return mix(signState(last, 1), rest, k);
    }
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      if (time >= it.start && time <= it.end) return signState(it, (time - it.start) / (it.end - it.start));
      const next = items[i + 1];
      if (next && time > it.end && time < next.start) {
        const k = smooth((time - it.end) / (next.start - it.end));
        return mix(signState(it, 1), signState(next, 0), k);
      }
    }
    return rest;
  };

  const handHeader = (side: Side) => {
    const prefix = side === 'right' ? 'Right' : 'Left';
    const cols = ['frame', 'timestamp_ms', 'glove_timestamp_us', 'hand_detected'];
    for (const j of ROKOKO_HAND_JOINTS) cols.push(`${prefix}${j}_pos_x`, `${prefix}${j}_pos_y`, `${prefix}${j}_pos_z`);
    return cols.join(',');
  };
  const bodyPoints = ['neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'];
  const bodyHeader = ['frame', 'timestamp_ms', 'body_detected', ...bodyPoints.flatMap((p) => [`${p}_x`, `${p}_y`, `${p}_z`, `${p}_vis`])].join(',');
  const blendNames = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'cheekPuff', 'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker', 'mouthSmileLeft', 'mouthSmileRight',
  ];
  const faceHeader = [
    'frame', 'timestamp_ms', 'face_detected',
    ...FACE_POINTS.flatMap(([i]) => [`face_${i}_x`, `face_${i}_y`, `face_${i}_z`]),
    ...blendNames.map((b) => `bs_${b}`),
  ].join(',');

  const lines = { right: [handHeader('right')], left: [handHeader('left')], body: [bodyHeader], face: [faceHeader] };
  for (let k = 0; k < nFrames; k++) {
    const time = k / FPS;
    const ms = Math.round(time * 1000);
    const st = stateAt(time);
    const body: Record<string, Vec3> = { neck: [0, 0, 0] };
    for (const side of ['right', 'left'] as const) {
      const hs = st[side];
      const { elbow, wrist } = solveElbow(side, hs.wrist);
      const fore = vNorm(vSub(wrist, elbow));
      const joints = handJoints(side, hs, fore);
      const row: string[] = [String(k + 1), String(ms), String(Math.round(time * 1e6)), '1'];
      for (const j of ROKOKO_HAND_JOINTS) row.push(...joints[j].map(f5));
      lines[side].push(row.join(','));
      body[`${side}_shoulder`] = vSub(SHOULDER[side], NECK);
      body[`${side}_elbow`] = vSub(elbow, NECK);
    }
    // body.csv uses the camera convention of the capture tool: X mirrored (signer's left = -X).
    const brow = [String(k + 1), String(ms), '1'];
    for (const p of bodyPoints) {
      const v = body[p];
      brow.push(f5(-v[0]), f5(v[1]), f5(v[2]), '0.99');
    }
    lines.body.push(brow.join(','));
    // Blink every ~3.3 s.
    const blink = Math.max(0, 1 - Math.abs(((time + 1.1) % 3.3) - 0.1) / 0.08);
    const frow = [String(k + 1), String(ms), '1'];
    for (const [, p] of faceLandmarks(st.head)) frow.push(f5(p[0]), f5(p[1]), f5(p[2]));
    for (const b of blendNames) {
      const v = b.startsWith('eyeBlink') ? Math.max(blink, st.face[b] ?? 0) : st.face[b] ?? 0;
      frow.push(f5(Math.max(0, Math.min(1, v))));
    }
    lines.face.push(frow.join(','));
  }

  const segments: GlossSegment[] = items.map((it) => ({
    text: it.sign!.text,
    gloss: glossOf(it.sign!.cls),
    classId: it.sign!.cls.id,
    startSec: Math.round(it.start * 100) / 100,
    endSec: Math.round(it.end * 100) / 100,
  }));
  return {
    files: [
      { name: 'right_hand.csv', text: lines.right.join('\n') + '\n' },
      { name: 'left_hand.csv', text: lines.left.join('\n') + '\n' },
      { name: 'body.csv', text: lines.body.join('\n') + '\n' },
      { name: 'face.csv', text: lines.face.join('\n') + '\n' },
    ],
    segments,
  };
}

/** One rest-to-rest clip per sign, as the real model returns them. */
export function synthesizeSignClips(signs: PlannedSign[]): GeneratedClip[] {
  return signs.map((s) => {
    const { files, segments } = synthesizeRecording([s]);
    return {
      text: s.text,
      gloss: glossOf(s.cls),
      classId: s.cls.id,
      files,
      strokeStartSec: segments[0].startSec,
      strokeEndSec: segments[0].endSec,
    };
  });
}

export type MockOutput = 'clips' | 'continuous';

export class MockMotionModel implements MotionGenerator {
  readonly name = 'Mock motion model v0 (in-browser placeholder)';
  readonly isMock = true;
  private readonly classes: DictionaryClass[];
  private readonly output: MockOutput;

  /**
   * `classes` must carry the text of the request's source language.
   * `output`: 'clips' (default) returns one clip per sign, like the real model;
   * 'continuous' returns the whole phrase as one recording.
   */
  constructor(classes: DictionaryClass[], output: MockOutput = 'clips') {
    this.classes = classes;
    this.output = output;
  }

  async generate(request: MotionRequest, signal?: AbortSignal): Promise<GeneratedMotion> {
    // Simulate model latency so the UI shows its loading state.
    await new Promise((r) => setTimeout(r, 350));
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const { signs, unmatched } = planSigns(request.text, this.classes);
    if (!signs.length) {
      return { files: [], segments: [], unmatched, model: this.name };
    }
    if (this.output === 'clips') {
      const clips = synthesizeSignClips(signs);
      const segments: GlossSegment[] = clips.map((c) => ({
        text: c.text,
        gloss: c.gloss,
        classId: c.classId,
        startSec: c.strokeStartSec ?? 0,
        endSec: c.strokeEndSec ?? 0,
      }));
      return { files: [], clips, segments, unmatched, model: this.name };
    }
    const { files, segments } = synthesizeRecording(signs);
    return { files, segments, unmatched, model: this.name };
  }
}
