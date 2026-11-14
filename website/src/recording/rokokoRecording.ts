/**
 * Parser for multi-stream LIS recordings captured with Rokoko Smartgloves II
 * (solved 26-joint hands) plus camera-based MediaPipe face and upper-body tracking.
 *
 * Expected files in one recording folder (names are matched loosely):
 *   left_hand.csv / right_hand.csv
 *     frame, timestamp_ms, glove_timestamp_us, hand_detected,
 *     <Side><Joint>_pos_{x,y,z}, <Side><Joint>_rot_{x,y,z,w}   (26 joints)
 *   face.csv   frame, timestamp_ms, face_detected, face_<i>_{x,y,z} (478), bs_<ARKit name> (52)
 *   body.csv   frame, timestamp_ms, body_detected, <landmark>_{x,y,z,vis}
 *
 * All streams share the `timestamp_ms` clock. Only the hand files are required.
 */

import type { Quat, Vec3 } from './math3d.ts';

export type HandSide = 'left' | 'right';

/** Rokoko joint names without the Left/Right prefix, in solver order. */
export const ROKOKO_HAND_JOINTS = [
  'ForeArm',
  'Hand',
  'Finger1Metacarpal',
  'Finger1Proximal',
  'Finger1Distal',
  'Finger1Tip',
  'Finger2Metacarpal',
  'Finger2Proximal',
  'Finger2Medial',
  'Finger2Distal',
  'Finger2Tip',
  'Finger3Metacarpal',
  'Finger3Proximal',
  'Finger3Medial',
  'Finger3Distal',
  'Finger3Tip',
  'Finger4Metacarpal',
  'Finger4Proximal',
  'Finger4Medial',
  'Finger4Distal',
  'Finger4Tip',
  'Finger5Metacarpal',
  'Finger5Proximal',
  'Finger5Medial',
  'Finger5Distal',
  'Finger5Tip',
] as const;

export type RokokoJoint = (typeof ROKOKO_HAND_JOINTS)[number];

export interface HandStream {
  side: HandSide;
  /** Seconds from the start of the recording (shared clock). */
  times: number[];
  detected: boolean[];
  /** Positions per joint per frame, in the glove world frame (meters, Y-up). */
  positions: Record<RokokoJoint, Vec3[]>;
  /** World orientations per joint per frame (x, y, z, w). */
  rotations: Record<RokokoJoint, Quat[]>;
  gloveTimestampsUs: number[] | null;
}

export interface FaceStream {
  times: number[];
  detected: boolean[];
  /** A few landmarks used for head pose, in normalized image coordinates. */
  landmarks: Record<'rightEye' | 'leftEye' | 'forehead' | 'chin' | 'noseTip', Vec3[]>;
  /** ARKit blendshape name -> per-frame weight (0..1). */
  blendshapes: Record<string, number[]>;
}

export interface BodyStream {
  times: number[];
  detected: boolean[];
  points: Record<string, Vec3[]>;
  visibility: Record<string, number[]>;
}

export interface RecordingDiagnostics {
  level: 'info' | 'warning' | 'error';
  message: string;
}

export interface RokokoRecording {
  name: string;
  left: HandStream | null;
  right: HandStream | null;
  face: FaceStream | null;
  body: BodyStream | null;
  durationSec: number;
  frameCount: number;
  medianFrameIntervalMs: number;
  diagnostics: RecordingDiagnostics[];
}

export interface RecordingFileInput {
  name: string;
  text: string;
}

/** MediaPipe Face Mesh landmark indices used for head pose. */
const FACE_LANDMARKS = {
  rightEye: 33,
  leftEye: 263,
  forehead: 10,
  chin: 152,
  noseTip: 1,
} as const;

interface CsvTable {
  header: string[];
  index: Map<string, number>;
  rows: string[][];
}

/** Minimal CSV reader: comma separated, optional double quotes, CRLF tolerant. */
export function parseCsv(text: string): CsvTable {
  const lines = text.split(/\r?\n/);
  const parseLine = (line: string): string[] => {
    if (!line.includes('"')) return line.split(',');
    const out: string[] = [];
    let cur = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (quoted) {
        if (ch === '"' && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else if (ch === '"') {
          quoted = false;
        } else {
          cur += ch;
        }
      } else if (ch === '"') {
        quoted = true;
      } else if (ch === ',') {
        out.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out;
  };

  let headerIdx = 0;
  while (headerIdx < lines.length && !lines[headerIdx].trim()) headerIdx++;
  const header = headerIdx < lines.length ? parseLine(lines[headerIdx]).map((h) => h.trim()) : [];
  const index = new Map<string, number>();
  header.forEach((h, i) => {
    if (!index.has(h)) index.set(h, i);
  });
  const rows: string[][] = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    rows.push(parseLine(lines[i]));
  }
  return { header, index, rows };
}

function num(row: string[], col: number | undefined, fallback = 0): number {
  if (col === undefined) return fallback;
  const v = Number(row[col]);
  return Number.isFinite(v) ? v : fallback;
}

function requireCol(table: CsvTable, name: string, file: string): number {
  const c = table.index.get(name);
  if (c === undefined) throw new Error(`${file}: missing column "${name}"`);
  return c;
}

function readTimes(table: CsvTable, file: string): number[] {
  const c = requireCol(table, 'timestamp_ms', file);
  return table.rows.map((r) => num(r, c) / 1000);
}

function readDetected(table: CsvTable, name: string): boolean[] {
  const c = table.index.get(name);
  return table.rows.map((r) => (c === undefined ? true : num(r, c, 1) > 0));
}

export function detectHandSide(header: string[]): HandSide | null {
  if (header.includes('LeftHand_pos_x') || header.includes('LeftHand_rot_x')) return 'left';
  if (header.includes('RightHand_pos_x') || header.includes('RightHand_rot_x')) return 'right';
  return null;
}

export function parseHandCsv(text: string, file = 'hand.csv'): HandStream {
  const table = parseCsv(text);
  const side = detectHandSide(table.header);
  if (!side) {
    throw new Error(
      `${file}: unknown format. Expected columns such as "LeftHand_pos_x" or "RightHand_pos_x" (Rokoko solved hand, 26 joints).`,
    );
  }
  const prefix = side === 'left' ? 'Left' : 'Right';
  const times = readTimes(table, file);
  const detected = readDetected(table, 'hand_detected');
  const positions = {} as Record<RokokoJoint, Vec3[]>;
  const rotations = {} as Record<RokokoJoint, Quat[]>;
  for (const joint of ROKOKO_HAND_JOINTS) {
    const base = `${prefix}${joint}`;
    const px = requireCol(table, `${base}_pos_x`, file);
    const py = requireCol(table, `${base}_pos_y`, file);
    const pz = requireCol(table, `${base}_pos_z`, file);
    // Rotations are optional: retargeting uses joint positions only.
    const rx = table.index.get(`${base}_rot_x`);
    const ry = table.index.get(`${base}_rot_y`);
    const rz = table.index.get(`${base}_rot_z`);
    const rw = table.index.get(`${base}_rot_w`);
    positions[joint] = table.rows.map((r) => [num(r, px), num(r, py), num(r, pz)] as Vec3);
    rotations[joint] = table.rows.map((r) => [num(r, rx), num(r, ry), num(r, rz), num(r, rw, 1)] as Quat);
  }
  const gc = table.index.get('glove_timestamp_us');
  return {
    side,
    times,
    detected,
    positions,
    rotations,
    gloveTimestampsUs: gc === undefined ? null : table.rows.map((r) => num(r, gc)),
  };
}

export function parseFaceCsv(text: string, file = 'face.csv'): FaceStream {
  const table = parseCsv(text);
  const times = readTimes(table, file);
  const detected = readDetected(table, 'face_detected');
  const landmarks = {} as FaceStream['landmarks'];
  for (const [key, idx] of Object.entries(FACE_LANDMARKS) as Array<[keyof FaceStream['landmarks'], number]>) {
    const cx = table.index.get(`face_${idx}_x`);
    const cy = table.index.get(`face_${idx}_y`);
    const cz = table.index.get(`face_${idx}_z`);
    if (cx === undefined || cy === undefined || cz === undefined) {
      landmarks[key] = [];
      continue;
    }
    landmarks[key] = table.rows.map((r) => [num(r, cx), num(r, cy), num(r, cz)] as Vec3);
  }
  const blendshapes: Record<string, number[]> = {};
  table.header.forEach((h, c) => {
    if (!h.startsWith('bs_')) return;
    const name = h.slice(3);
    if (!name || name.startsWith('_')) return; // skip "_neutral"
    blendshapes[name] = table.rows.map((r) => Math.max(0, Math.min(1, num(r, c))));
  });
  return { times, detected, landmarks, blendshapes };
}

export function parseBodyCsv(text: string, file = 'body.csv'): BodyStream {
  const table = parseCsv(text);
  const times = readTimes(table, file);
  const detected = readDetected(table, 'body_detected');
  const points: Record<string, Vec3[]> = {};
  const visibility: Record<string, number[]> = {};
  for (const h of table.header) {
    if (!h.endsWith('_x')) continue;
    const name = h.slice(0, -2);
    const cx = table.index.get(`${name}_x`);
    const cy = table.index.get(`${name}_y`);
    const cz = table.index.get(`${name}_z`);
    if (cx === undefined || cy === undefined || cz === undefined) continue;
    points[name] = table.rows.map((r) => [num(r, cx), num(r, cy), num(r, cz)] as Vec3);
    const cv = table.index.get(`${name}_vis`);
    visibility[name] = table.rows.map((r) => (cv === undefined ? 1 : num(r, cv, 1)));
  }
  return { times, detected, points, visibility };
}

type StreamKind = 'left' | 'right' | 'face' | 'body' | 'unknown';

/** Classifies a CSV file by its header (the file name is only a hint). */
export function classifyRecordingFile(file: RecordingFileInput): StreamKind {
  const firstLine = file.text.slice(0, 4096).split(/\r?\n/, 1)[0] ?? '';
  if (firstLine.includes('LeftHand_pos_x') || firstLine.includes('LeftHand_rot_x')) return 'left';
  if (firstLine.includes('RightHand_pos_x') || firstLine.includes('RightHand_rot_x')) return 'right';
  if (firstLine.includes('face_0_x') || firstLine.includes('bs_jawOpen')) return 'face';
  if (firstLine.includes('shoulder_x') || firstLine.includes('body_detected')) return 'body';
  const n = file.name.toLowerCase();
  if (n.includes('left')) return 'left';
  if (n.includes('right')) return 'right';
  if (n.includes('face')) return 'face';
  if (n.includes('body') || n.includes('pose')) return 'body';
  return 'unknown';
}

function medianInterval(times: number[]): number {
  if (times.length < 2) return 0;
  const d: number[] = [];
  for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
  d.sort((a, b) => a - b);
  return d[d.length >> 1];
}

function rate(flags: boolean[]): number {
  if (!flags.length) return 0;
  return flags.filter(Boolean).length / flags.length;
}

/** Parses a set of recording CSV files into one synchronized recording. */
export function parseRokokoRecording(files: RecordingFileInput[], name = 'recording'): RokokoRecording {
  const diagnostics: RecordingDiagnostics[] = [];
  let left: HandStream | null = null;
  let right: HandStream | null = null;
  let face: FaceStream | null = null;
  let body: BodyStream | null = null;

  for (const f of files) {
    if (!f.name.toLowerCase().endsWith('.csv')) continue;
    const kind = classifyRecordingFile(f);
    try {
      if (kind === 'left' || kind === 'right') {
        const stream = parseHandCsv(f.text, f.name);
        if (stream.side === 'left') left = stream;
        else right = stream;
      } else if (kind === 'face') {
        face = parseFaceCsv(f.text, f.name);
      } else if (kind === 'body') {
        body = parseBodyCsv(f.text, f.name);
      } else {
        diagnostics.push({ level: 'warning', message: `${f.name}: file skipped (unknown format).` });
      }
    } catch (err) {
      diagnostics.push({
        level: 'error',
        message: err instanceof Error ? err.message : `${f.name}: cannot read the file`,
      });
    }
  }

  if (!left && !right) {
    diagnostics.push({
      level: 'error',
      message: 'No valid glove file: left_hand.csv and/or right_hand.csv are required.',
    });
  } else {
    if (!right) diagnostics.push({ level: 'warning', message: 'right_hand.csv is missing: the right hand stays at rest.' });
    if (!left) diagnostics.push({ level: 'warning', message: 'left_hand.csv is missing: the left hand stays at rest.' });
  }
  if (!face) diagnostics.push({ level: 'info', message: 'No face.csv: no facial expressions and no head rotation.' });
  if (!body) {
    diagnostics.push({
      level: 'info',
      message: 'No body.csv: the upper arm is estimated from the glove forearm.',
    });
  }

  const ref = right ?? left;
  const times = ref?.times ?? [];
  const t0 = times.length ? times[0] : 0;
  // Re-base every stream on the same t0 so all clocks start at zero.
  const rebase = (s: { times: number[] } | null) => {
    if (s) s.times = s.times.map((t) => t - t0);
  };
  rebase(left);
  rebase(right);
  rebase(face);
  rebase(body);

  const frameIntervalMs = medianInterval(ref?.times ?? []) * 1000;
  const durationSec = ref && ref.times.length ? ref.times[ref.times.length - 1] : 0;

  if (ref && ref.times.length >= 2) {
    const hz = 1000 / Math.max(1e-6, frameIntervalMs);
    if (hz < 25) {
      diagnostics.push({
        level: 'warning',
        message: `Low glove frame rate: about ${hz.toFixed(1)} Hz (the gloves can stream 83 Hz). Fast finger movements lose detail.`,
      });
    }
  }
  for (const s of [left, right]) {
    if (!s) continue;
    const r = rate(s.detected);
    if (r < 0.95) {
      diagnostics.push({
        level: 'warning',
        message: `${s.side === 'left' ? 'Left' : 'Right'} hand detected in only ${(r * 100).toFixed(0)}% of the frames.`,
      });
    }
  }
  if (face && rate(face.detected) < 0.9) {
    diagnostics.push({ level: 'warning', message: `Face detected in ${(rate(face.detected) * 100).toFixed(0)}% of the frames.` });
  }
  if (body && rate(body.detected) < 0.9) {
    diagnostics.push({ level: 'warning', message: `Body detected in ${(rate(body.detected) * 100).toFixed(0)}% of the frames.` });
  }

  return {
    name,
    left,
    right,
    face,
    body,
    durationSec,
    frameCount: ref?.times.length ?? 0,
    medianFrameIntervalMs: frameIntervalMs,
    diagnostics,
  };
}
