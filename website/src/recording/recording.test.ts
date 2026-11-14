/**
 * Tests for the recording import path. Run with:
 *   node --test src/recording/recording.test.ts
 * Set LIS_RECORDING_DIR to a folder with left_hand.csv / right_hand.csv / face.csv / body.csv
 * to also check a real recording.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  ROKOKO_HAND_JOINTS,
  classifyRecordingFile,
  parseRokokoRecording,
  type RecordingFileInput,
} from './rokokoRecording.ts';
import {
  DEFAULT_RETARGET_OPTIONS,
  detectSigningSegments,
  estimateGloveYawDeg,
  retargetRecording,
} from './recordingRetarget.ts';
import {
  orthoFrame,
  qAngle,
  qFrameToFrame,
  qFromAxisAngle,
  qMul,
  qRotate,
  radToDeg,
  vDot,
  type Quat,
  type Vec3,
} from './math3d.ts';

/** Builds a synthetic hand CSV: forearm along `fore`, fingers straight or curled. */
function syntheticHandCsv(side: 'Left' | 'Right', frames: number, curl: (k: number) => number, fore: (k: number) => Vec3): string {
  const header = ['frame', 'timestamp_ms', 'glove_timestamp_us', 'hand_detected'];
  for (const j of ROKOKO_HAND_JOINTS) {
    for (const a of ['x', 'y', 'z']) header.push(`${side}${j}_pos_${a}`);
    for (const a of ['x', 'y', 'z', 'w']) header.push(`${side}${j}_rot_${a}`);
  }
  const lines = [header.join(',')];
  const radial = side === 'Right' ? -1 : 1; // index side in avatar X for palm-forward hands
  for (let k = 0; k < frames; k++) {
    const f = fore(k);
    const wrist: Vec3 = [f[0] * 0.29, f[1] * 0.29, f[2] * 0.29];
    // Hand in the forearm direction, palm facing +Z when the forearm hangs down.
    const pos: Record<string, Vec3> = { ForeArm: [0, 0, 0], Hand: wrist };
    const c = curl(k);
    const fingerX = [0, 0.03, 0.01, -0.01, -0.03];
    for (let n = 1; n <= 5; n++) {
      const x = wrist[0] + radial * fingerX[n - 1];
      const base: Vec3 = [x, wrist[1] - 0.08, wrist[2]];
      const segs = n === 1 ? ['Metacarpal', 'Proximal', 'Distal', 'Tip'] : ['Metacarpal', 'Proximal', 'Medial', 'Distal', 'Tip'];
      let p: Vec3 = n === 1 ? [wrist[0] + radial * 0.04, wrist[1] - 0.03, wrist[2] + 0.02] : [x, wrist[1] - 0.04, wrist[2]];
      segs.forEach((seg, idx) => {
        if (n !== 1 && idx === 1) p = base;
        pos[`Finger${n}${seg}`] = p;
        // each later segment bends toward the palm (+Z) by `c` radians per joint
        const ang = c * idx;
        const len = 0.025;
        p = [p[0], p[1] - len * Math.cos(ang), p[2] + len * Math.sin(ang)];
      });
    }
    const row: (string | number)[] = [k + 1, 1000 + k * 33, 5000000 + k * 33000, 1];
    for (const j of ROKOKO_HAND_JOINTS) {
      const p = pos[j];
      row.push(p[0], p[1], p[2], 0, 0, 0, 1);
    }
    lines.push(row.join(','));
  }
  return lines.join('\n');
}

function syntheticRecording(curl: (k: number) => number, fore: (k: number) => Vec3, frames = 60) {
  const files: RecordingFileInput[] = [
    { name: 'left_hand.csv', text: syntheticHandCsv('Left', frames, curl, fore) },
    { name: 'right_hand.csv', text: syntheticHandCsv('Right', frames, curl, fore) },
  ];
  return parseRokokoRecording(files, 'synthetic');
}

function boneQuat(payload: { boneTracks: Record<string, number[]> }, bone: string, frame: number): Quat {
  const t = payload.boneTracks[bone];
  return [t[frame * 4], t[frame * 4 + 1], t[frame * 4 + 2], t[frame * 4 + 3]];
}

test('orthoFrame + qFrameToFrame map one frame onto another', () => {
  const from = orthoFrame([0, -1, 0], [-1, 0, 0]);
  const q = qFromAxisAngle([0.3, 1, -0.2], 1.1);
  const to = orthoFrame(qRotate(q, from[0]), qRotate(q, from[1]));
  const r = qFrameToFrame(from, to);
  const diff = qMul(r, [-q[0], -q[1], -q[2], q[3]]);
  assert.ok(radToDeg(qAngle(diff)) < 1e-3);
});

test('classifyRecordingFile uses headers, not only names', () => {
  assert.equal(classifyRecordingFile({ name: 'x.csv', text: 'frame,timestamp_ms,LeftHand_rot_x\n' }), 'left');
  assert.equal(classifyRecordingFile({ name: 'x.csv', text: 'frame,timestamp_ms,face_detected,face_0_x\n' }), 'face');
  assert.equal(classifyRecordingFile({ name: 'pose.csv', text: 'frame,timestamp_ms\n' }), 'body');
});

test('missing hand files produce an error diagnostic', () => {
  const rec = parseRokokoRecording([{ name: 'face.csv', text: 'frame,timestamp_ms,face_detected\n1,0,1\n' }]);
  assert.ok(rec.diagnostics.some((d) => d.level === 'error'));
});

test('straight hanging hand retargets to (near) rest pose', () => {
  const rec = syntheticRecording(() => 0, () => [0, -1, 0]);
  const { payload } = retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    gloveYawDeg: 0,
    smoothing: 0,
    fadeSec: 0,
    startSec: 0,
    endSec: rec.durationSec,
    classId: 'test',
    phrase: 'Test',
    category: 'Test',
    description: '',
  });
  for (const bone of ['rightHand', 'rightIndexProximal', 'rightIndexIntermediate', 'leftMiddleDistal']) {
    const ang = radToDeg(qAngle(boneQuat(payload, bone, 10)));
    assert.ok(ang < 3, `${bone} should be near identity, got ${ang.toFixed(1)}°`);
  }
  assert.equal(payload.times.length, payload.boneTracks.rightHand.length / 4);
});

test('curled fingers bend the phalanges toward the palm', () => {
  const rec = syntheticRecording(() => 0.6, () => [0, -1, 0]);
  const { payload } = retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    gloveYawDeg: 0,
    smoothing: 0,
    fadeSec: 0,
    startSec: 0,
    endSec: rec.durationSec,
    classId: 'test',
    phrase: 'Test',
    category: 'Test',
    description: '',
  });
  for (const side of ['right', 'left']) {
    const q = boneQuat(payload, `${side}IndexIntermediate`, 5);
    const ang = radToDeg(qAngle(q));
    assert.ok(ang > 25 && ang < 45, `${side} intermediate flexion ~34°, got ${ang.toFixed(1)}`);
    // The bent bone tip must move toward +Z (palm side of the rest hand).
    const tip = qRotate(q, [0, -1, 0]);
    assert.ok(tip[2] > 0.3, `${side} finger should curl to the palm, tip=${tip.map((x) => x.toFixed(2))}`);
  }
});

test('raised forearms are detected as a signing segment', () => {
  const rec = syntheticRecording(
    () => 0,
    (k) => (k > 20 && k < 45 ? [0, 0.2, 0.98] : [0, -1, 0]),
    70,
  );
  const prof = detectSigningSegments(rec);
  assert.equal(prof.segments.length, 1);
  const seg = prof.segments[0];
  assert.ok(seg.startSec < 21 * 0.033 && seg.endSec > 44 * 0.033);
  // forearms point +Z while raised, so the automatic yaw stays near 0
  assert.ok(Math.abs(estimateGloveYawDeg(rec)) < 5);
});

const realDir = process.env.LIS_RECORDING_DIR;
test('real recording imports and retargets', { skip: !realDir || !existsSync(realDir) }, () => {
  const dir = realDir as string;
  const files = readdirSync(dir)
    .filter((n) => n.endsWith('.csv'))
    .map((n) => ({ name: n, text: readFileSync(join(dir, n), 'utf8') }));
  const rec = parseRokokoRecording(files, 'real');
  assert.ok(rec.left && rec.right, 'both hands parsed');
  assert.ok(!rec.diagnostics.some((d) => d.level === 'error'), JSON.stringify(rec.diagnostics));
  const prof = detectSigningSegments(rec);
  const yaw = estimateGloveYawDeg(rec);
  const { payload } = retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    startSec: 0,
    endSec: rec.durationSec,
    classId: 'real',
    phrase: 'Real',
    category: 'Test',
    description: '',
  });
  const n = payload.times.length;
  for (const [bone, flat] of Object.entries(payload.boneTracks)) {
    assert.equal(flat.length, n * 4, bone);
    assert.ok(flat.every(Number.isFinite), `${bone} finite`);
  }
  // Sanity: palm normal of the solved hands should face mostly the same way the curl goes.
  const q = boneQuat(payload, 'rightIndexIntermediate', Math.floor(n / 2));
  assert.ok(vDot(qRotate(q, [0, -1, 0]), [0, -1, 0]) > -0.5);
  console.log(
    `real: ${rec.frameCount} frames, ${rec.durationSec.toFixed(1)} s, ~${(1000 / rec.medianFrameIntervalMs).toFixed(1)} Hz, ` +
      `yaw ${yaw.toFixed(1)}°, segments ${prof.segments.map((s) => `${s.startSec.toFixed(1)}-${s.endSec.toFixed(1)}`).join(' ')}`,
  );
  console.log('diagnostics:', rec.diagnostics.map((d) => `${d.level}: ${d.message}`).join(' | '));
});
