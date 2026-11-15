import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import type { SignAnimationPayload } from '../types/lis';
import { IDENTITY, qConj, qFromAxisAngle, qMul, qNormalize, type Quat, type Vec3 } from '../recording/math3d';
import { detectStroke, qExp, qLog, stitchClips, type TransitionCurve } from './stitchClips';
import { motionToAnimation } from './motionToAnimation';
import { exportDataset } from '../router/exportDataset';
import type { GeneratedMotion } from './motionModel';

const FPS = 30;
const H = 1 / FPS;
const ARM = ['UpperArm', 'LowerArm', 'Hand'];

interface SideMotion {
  /** Pose held during the stroke (start), per arm bone. */
  pose: Quat;
  /** Constant body-frame angular velocity during the stroke (rad/s). */
  w: Vec3;
}

/**
 * Rest-to-rest clip: rest 0.4 s, raise 0.3 s, stroke 0.8 s with constant angular velocity,
 * lower 0.3 s, rest 0.3 s. A side without motion stays at rest.
 */
function makeClip(right: SideMotion | null, left: SideMotion | null, jaw = 0.5): { payload: SignAnimationPayload; stroke: [number, number] } {
  const t0 = 0.7;
  const t1 = 1.5;
  const dur = 2.1;
  const n = Math.round(dur * FPS) + 1;
  const times = Array.from({ length: n }, (_, k) => k * H);
  const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const env = (t: number) => (t < t0 ? smooth((t - 0.4) / 0.3) : t > t1 ? 1 - smooth((t - t1) / 0.3) : 1);
  const strokeQ = (m: SideMotion, t: number) => qNormalize(qMul(m.pose, qExp(m.w.map((x) => x * (Math.min(t1, Math.max(t0, t)) - t0)) as Vec3)));
  const boneTracks: Record<string, number[]> = {};
  for (const [side, m] of [['right', right], ['left', left]] as const) {
    for (const b of ARM) {
      const flat: number[] = [];
      for (const t of times) {
        const q = m ? slerp(IDENTITY, strokeQ(m, t), env(t)) : IDENTITY;
        flat.push(...q);
      }
      boneTracks[`${side}${b}`] = flat;
    }
  }
  const morphTracks = { jawOpen: times.map((t) => jaw * env(t)) };
  const payload: SignAnimationPayload = {
    classId: 'c', phrase: 'p', category: 'c', description: '', prototypeClipId: 'clip', fps: FPS, duration: dur, times,
    boneTracks, morphTracks, qaComparison: { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] },
  };
  return { payload, stroke: [t0, t1] };
}

function slerp(a: Quat, b: Quat, t: number): Quat {
  return qNormalize(qMul(a, qExp(qLog(qMul(qConj(a), b)).map((x) => x * t) as Vec3)));
}

const quat = (flat: number[], k: number): Quat => flat.slice(k * 4, k * 4 + 4) as Quat;
const rel = (a: Quat, b: Quat) => qLog(qMul(qConj(a), b));
const norm = (v: number[]) => Math.hypot(...v);
const angle = (a: Quat, b: Quat) => norm(rel(a, b));

const RIGHT_A: SideMotion = { pose: qFromAxisAngle([1, 0, 0], -1.2), w: [0, 0.9, 0] };
const RIGHT_B: SideMotion = { pose: qFromAxisAngle(qNormalize([0, 0.3, 1, 0] as unknown as Quat).slice(0, 3) as Vec3, 1.4), w: [0.6, 0, -0.5] };
const LEFT_A: SideMotion = { pose: qFromAxisAngle([0, 0, 1], 0.9), w: [0.3, 0, 0] };

test('qExp / qLog round trip', () => {
  for (const v of [[0, 0, 0], [0.1, -0.2, 0.3], [2.5, 0.5, -1], [0, 3.1, 0]] as Vec3[]) {
    const back = qLog(qExp(v));
    assert.ok(norm(back.map((x, i) => x - v[i])) < 1e-9, `${v} -> ${back}`);
  }
});

test('detectStroke finds the part away from rest', () => {
  const { payload, stroke } = makeClip(RIGHT_A, null);
  const s = detectStroke(payload);
  assert.ok(Math.abs(s.startSec - stroke[0]) < 0.2 && Math.abs(s.endSec - stroke[1]) < 0.2, JSON.stringify(s));
});

for (const curve of ['quintic', 'cubic'] as TransitionCurve[]) {
  test(`${curve}: strokes unchanged, rest removed, pose and velocity continuous at the joins`, () => {
    const A = makeClip(RIGHT_A, LEFT_A, 0.8);
    const B = makeClip(RIGHT_B, LEFT_A, 0.1);
    const r = stitchClips(
      [A, B].map((c) => ({ payload: c.payload, strokeStartSec: c.stroke[0], strokeEndSec: c.stroke[1] })),
      { curve },
    );
    const p = r.payload;
    const n = p.times.length;
    assert.equal(r.transitions.length, 1);
    // Shorter than playing both clips: the rest-out of A and the rest-in of B are gone.
    assert.ok(p.duration < A.payload.duration + B.payload.duration - 0.4, `duration ${p.duration}`);
    for (const flat of Object.values(p.boneTracks)) assert.equal(flat.length, n * 4);

    // The stroke of B is copied unchanged.
    const segB = r.segments[1];
    const kB = Math.round(segB.startSec * FPS);
    for (let j = 0; j < 20; j++) {
      const src = quat(B.payload.boneTracks.rightUpperArm, Math.round(B.stroke[0] * FPS) + j);
      assert.ok(angle(quat(p.boneTracks.rightUpperArm, kB + j), src) < 1e-3, `frame ${j}`);
    }

    // Around each join: frame-to-frame body rate on both sides of the boundary matches.
    const tr = r.transitions[0];
    for (const kb of [Math.round(tr.startSec * FPS), Math.round(tr.endSec * FPS)]) {
      for (const bone of ['rightUpperArm', 'rightHand', 'leftLowerArm']) {
        const f = p.boneTracks[bone];
        const wl = rel(quat(f, kb - 1), quat(f, kb)).map((x) => x / H);
        const wr = rel(quat(f, kb), quat(f, kb + 1)).map((x) => x / H);
        const dw = norm(wl.map((x, i) => x - wr[i]));
        // One-frame differences include the acceleration: the cubic curve has an acceleration
        // step at the joins (C1 only), the quintic one does not (C2).
        const tol = curve === 'quintic' ? 0.35 : 1.2;
        assert.ok(dw < tol, `${curve} ${bone} at frame ${kb}: |Δω| = ${dw.toFixed(3)} rad/s (${wl} vs ${wr})`);
      }
    }
    // No jumps: the largest frame-to-frame rotation is not larger than in the source clips.
    const maxStep = (pl: SignAnimationPayload) => {
      let m = 0;
      for (const f of Object.values(pl.boneTracks)) for (let k = 1; k < pl.times.length; k++) m = Math.max(m, angle(quat(f, k - 1), quat(f, k)));
      return m;
    };
    const srcMax = Math.max(maxStep(A.payload), maxStep(B.payload));
    assert.ok(maxStep(p) <= srcMax + 0.02, `max step ${maxStep(p).toFixed(3)} vs source ${srcMax.toFixed(3)} rad/frame`);
    // During the transition the right arm does not go back to rest.
    for (let k = Math.round(tr.startSec * FPS); k <= Math.round(tr.endSec * FPS); k++) {
      assert.ok(angle(quat(p.boneTracks.rightUpperArm, k), IDENTITY) > 0.6, `frame ${k} near rest`);
    }
    // Blendshapes stay in range and continuous.
    const jaw = p.morphTracks.jawOpen;
    assert.ok(jaw.every((x) => x >= 0 && x <= 1));
    for (let k = 1; k < n; k++) assert.ok(Math.abs(jaw[k] - jaw[k - 1]) < 0.15, `jaw step at ${k}`);
  });
}

test('quintic keeps the acceleration continuous better than cubic', () => {
  const A = makeClip(RIGHT_A, null);
  const B = makeClip(RIGHT_B, null);
  const jump = (curve: TransitionCurve) => {
    const r = stitchClips([A, B].map((c) => ({ payload: c.payload, strokeStartSec: c.stroke[0], strokeEndSec: c.stroke[1] })), { curve });
    const f = r.payload.boneTracks.rightUpperArm;
    const kb = Math.round(r.transitions[0].startSec * FPS);
    // Second difference just before vs just after the join (strokes have zero acceleration).
    const acc = (k: number) => norm(rel(quat(f, k), quat(f, k + 1)).map((x, i) => x - rel(quat(f, k - 1), quat(f, k))[i])) / (H * H);
    return Math.max(acc(kb), acc(kb + 1));
  };
  const q = jump('quintic');
  const c = jump('cubic');
  assert.ok(q < c, `quintic ${q.toFixed(2)} vs cubic ${c.toFixed(2)} rad/s²`);
});

test('transition time grows with the distance between the poses and stays in range', () => {
  const near = makeClip({ ...RIGHT_A, pose: qMul(RIGHT_A.pose, qFromAxisAngle([0, 1, 0], 0.1)) }, null);
  const far = makeClip(RIGHT_B, null);
  const A = makeClip(RIGHT_A, null);
  const dur = (B: ReturnType<typeof makeClip>) => {
    const tr = stitchClips([A, B].map((c) => ({ payload: c.payload, strokeStartSec: c.stroke[0], strokeEndSec: c.stroke[1] }))).transitions[0];
    return tr.endSec - tr.startSec;
  };
  const dNear = dur(near);
  const dFar = dur(far);
  assert.ok(dNear < dFar, `near ${dNear} far ${dFar}`);
  assert.ok(dNear >= 0.15 - 1e-9 && dFar <= 1.0, `near ${dNear} far ${dFar}`);
});

test('a hand idle in the middle sign is bridged instead of dropping to rest', () => {
  const A = makeClip(RIGHT_A, LEFT_A);
  const B = makeClip(RIGHT_B, null); // left hand idle
  const C = makeClip(RIGHT_A, LEFT_A);
  const clips = [A, B, C].map((c) => ({ payload: c.payload, strokeStartSec: c.stroke[0], strokeEndSec: c.stroke[1] }));
  const bridged = stitchClips(clips);
  const plain = stitchClips(clips, { bridgeIdleHand: false });
  const mid = Math.round(((bridged.segments[1].startSec + bridged.segments[1].endSec) / 2) * FPS);
  const midPlain = Math.round(((plain.segments[1].startSec + plain.segments[1].endSec) / 2) * FPS);
  const away = (r: typeof bridged, k: number) => angle(quat(r.payload.boneTracks.leftUpperArm, k), IDENTITY);
  assert.ok(away(plain, midPlain) < 0.05, 'without bridging the left arm is at rest in B');
  assert.ok(away(bridged, mid) > 0.5, `with bridging the left arm stays up (${away(bridged, mid).toFixed(2)} rad)`);
  // The right hand (active in B) is not touched by the bridge.
  assert.ok(angle(quat(bridged.payload.boneTracks.rightUpperArm, mid), quat(plain.payload.boneTracks.rightUpperArm, midPlain)) < 1e-3);
});

const DATASET = '../dataset/data/processed';

test('real dataset clips are joined: strokes detected, no drop to rest, no jumps', { skip: !existsSync(DATASET) }, () => {
  const { index, clips } = exportDataset(DATASET, { source: 'test', gloveYawDeg: null, sentences: ['ciao mi chiamo Chiara', 'dove abiti?', 'grazie'] });
  const pick = (sentence: string) => index.entries.find((e) => e.sentence === sentence)!;
  const entries = [pick('grazie'), pick('dove abiti?'), pick('ciao mi chiamo Chiara')];
  const payloads = entries.map((e) => clips.get(e.clip)!);
  const motion: GeneratedMotion = {
    files: [],
    clips: entries.map((e, i) => ({
      text: e.sentence, gloss: e.sentence, classId: null, files: [], payload: payloads[i],
      strokeStartSec: e.strokeStartSec, strokeEndSec: e.strokeEndSec,
    })),
    segments: [],
    unmatched: [],
    model: 'test',
  };
  const anim = motionToAnimation(motion, 'test', 'quintic');
  assert.ok(anim);
  const p = anim.payload;
  assert.equal(anim.transitions.length, 2);
  for (let i = 1; i < anim.segments.length; i++) assert.ok(anim.segments[i].startSec > anim.segments[i - 1].endSec);
  for (const [bone, f] of Object.entries(p.boneTracks)) {
    assert.ok(f.every(Number.isFinite), bone);
    for (const tr of anim.transitions) {
      for (let k = Math.round(tr.startSec * FPS); k <= Math.round(tr.endSec * FPS); k++) {
        assert.ok(angle(quat(f, k - 1), quat(f, k)) < 0.3, `${bone}: jump at frame ${k}`);
      }
    }
  }
  // The detected strokes keep most of each recording (the dataset is already trimmed).
  for (const e of entries) {
    assert.ok(e.strokeEndSec! - e.strokeStartSec! > 0.4 * e.duration, `${e.sentence}: stroke too short`);
  }
});
