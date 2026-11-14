/**
 * Automatic facial rig (mixed: facial bones + blendshapes).
 *
 * Third level of the facial pipeline, after (1) the model's own blendshapes and (2) its facial
 * bones (Rigify, generic jaw). For every LIS face channel still missing it fits a rig to the
 * head geometry of the loaded avatar:
 *
 *   1. Collect the vertices weighted to the head bone (world bind pose: +Y up, +Z forward,
 *      +X = character's left, as set by normalizeAvatarRootTransform).
 *   2. Detect face landmarks on the face mesh: nose tip, lip line, eye sockets, brows, chin,
 *      face width.
 *   3. Define a displacement field per channel (jaw rotation, lid closure, brow raise, mouth
 *      corners, lips, cheeks), scaled to the detected face size.
 *   4. Add facial bones under the head (jaw, eyelids, brows, mouth corners, cheeks) at the
 *      landmarks, and give the face vertices skin weights toward them, fitted to the fields.
 *      The bones carry the main motion and appear in the skeleton view.
 *   5. Store what the bones cannot reproduce (the residual) as relative morph targets with the
 *      canonical channel names (mouthOpen, eyeBlinkLeft, ...).
 *
 * Bone tracks and morph tracks are driven from the same payload channels.
 */

import * as THREE from 'three';

/** The 9 face channels the LIS animation payloads drive. */
export const FACE_CHANNELS = [
  'mouthOpen',
  'mouthSmile',
  'mouthFrown',
  'mouthPucker',
  'browUp',
  'browDown',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'cheekPuff',
] as const;
export type FaceChannel = (typeof FACE_CHANNELS)[number];

export interface FaceLandmarks {
  /** Face midline X. */
  centerX: number;
  /** Half width of the head at the cheeks. */
  halfWidth: number;
  topY: number;
  chinY: number;
  noseTip: THREE.Vector3;
  mouthY: number;
  /** Z of the face surface on the midline at the mouth line. */
  mouthZ: number;
  eyeY: number;
  /** Horizontal distance of each eye center from the midline. */
  eyeOffsetX: number;
  /** Z of the face surface at the eye centers. */
  eyeZ: number;
  browY: number;
  /** Size factor relative to a reference adult face (eye line to mouth line = 72 mm). */
  scale: number;
  /** True when the mouth line came from the lip profile, false when it is a proportion estimate. */
  mouthDetected: boolean;
  eyesDetected: boolean;
}

export interface AutoFaceRigResult {
  channels: FaceChannel[];
  landmarks: FaceLandmarks;
  /** Facial bones added under the head bone. */
  bones: THREE.Bone[];
  /** Track key of each mesh that received residual morph targets, with its morph names. */
  meshes: Array<{ meshName: string; morphs: FaceChannel[] }>;
  buildFacialBoneTracks: (morphTracks: Record<string, number[]>, times: number[]) => THREE.KeyframeTrack[];
}

export interface HeadVertex {
  mesh: THREE.Mesh;
  index: number;
  weight: number;
  world: THREE.Vector3;
}

const smoothstep = (e0: number, e1: number, x: number): number => {
  if (e0 === e1) return x < e0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};
const gauss = (d: number, r: number): number => Math.exp(-(d * d) / (r * r));

function median(values: number[]): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Skinned (or rigid) world position of a vertex in the current pose. */
function vertexWorld(mesh: THREE.Mesh, index: number, local: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
  out.copy(local);
  if ((mesh as THREE.SkinnedMesh).isSkinnedMesh) (mesh as THREE.SkinnedMesh).applyBoneTransform(index, out);
  return out.applyMatrix4(mesh.matrixWorld);
}

/** Head weight of every vertex of every mesh that follows the head bone. */
function collectHeadVertices(root: THREE.Object3D, headNode: THREE.Object3D): Map<THREE.Mesh, HeadVertex[]> {
  const headSet = new Set<THREE.Object3D>();
  headNode.traverse((o) => headSet.add(o));
  const isUnderHead = (o: THREE.Object3D): boolean => {
    for (let cur: THREE.Object3D | null = o; cur; cur = cur.parent) if (headSet.has(cur)) return true;
    return false;
  };

  const out = new Map<THREE.Mesh, HeadVertex[]>();
  const local = new THREE.Vector3();
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry?.attributes?.position) return;
    const pos = mesh.geometry.attributes.position;
    const verts: HeadVertex[] = [];
    const sm = mesh as THREE.SkinnedMesh;
    if (sm.isSkinnedMesh && sm.skeleton) {
      const si = sm.geometry.attributes.skinIndex;
      const sw = sm.geometry.attributes.skinWeight;
      if (!si || !sw) return;
      const headBoneIdx = new Set<number>();
      sm.skeleton.bones.forEach((b, i) => {
        if (b && headSet.has(b)) headBoneIdx.add(i);
      });
      if (!headBoneIdx.size) return;
      const comps = Math.min(4, si.itemSize);
      for (let i = 0; i < pos.count; i++) {
        let w = 0;
        for (let c = 0; c < comps; c++) if (headBoneIdx.has(si.getComponent(i, c))) w += sw.getComponent(i, c);
        if (w < 0.02) continue;
        local.fromBufferAttribute(pos, i);
        verts.push({ mesh, index: i, weight: Math.min(1, w), world: vertexWorld(mesh, i, local, new THREE.Vector3()) });
      }
    } else if (isUnderHead(mesh)) {
      for (let i = 0; i < pos.count; i++) {
        local.fromBufferAttribute(pos, i);
        verts.push({ mesh, index: i, weight: 1, world: vertexWorld(mesh, i, local, new THREE.Vector3()) });
      }
    }
    if (verts.length) out.set(mesh, verts);
  });
  return out;
}

/** Front-most Z per height bin, for points inside an X window. */
function depthProfile(points: THREE.Vector3[], x0: number, x1: number, bin: number): Map<number, number> {
  const prof = new Map<number, number>();
  for (const p of points) {
    if (p.x < x0 || p.x > x1) continue;
    const b = Math.round(p.y / bin);
    const cur = prof.get(b);
    if (cur === undefined || p.z > cur) prof.set(b, p.z);
  }
  return prof;
}

/** Profile sampled on a regular grid from yLow to yHigh, gaps filled linearly, lightly smoothed. */
function sampleProfile(prof: Map<number, number>, bin: number, yLow: number, yHigh: number): { ys: number[]; zs: number[]; gaps: boolean[] } {
  const ys: number[] = [];
  const raw: Array<number | null> = [];
  for (let b = Math.round(yLow / bin); b <= Math.round(yHigh / bin); b++) {
    ys.push(b * bin);
    raw.push(prof.get(b) ?? null);
  }
  const gaps = raw.map((z) => z === null);
  const filled = raw.slice();
  for (let i = 0; i < filled.length; i++) {
    if (filled[i] !== null) continue;
    let a = i - 1;
    while (a >= 0 && raw[a] === null) a--;
    let b = i + 1;
    while (b < raw.length && raw[b] === null) b++;
    const za = a >= 0 ? (raw[a] as number) : null;
    const zb = b < raw.length ? (raw[b] as number) : null;
    filled[i] = za !== null && zb !== null ? za + ((zb - za) * (i - a)) / (b - a) : (za ?? zb ?? 0);
  }
  const zs = filled.map((_, i) => {
    let s = 0;
    let n = 0;
    for (let k = Math.max(0, i - 1); k <= Math.min(filled.length - 1, i + 1); k++) {
      s += filled[k] as number;
      n++;
    }
    return s / n;
  });
  return { ys, zs, gaps };
}

/** Detects face landmarks from the world-space vertices of the face mesh. */
export function detectFaceLandmarks(points: THREE.Vector3[]): FaceLandmarks | null {
  if (points.length < 200) return null;
  const centerX = median(points.map((p) => p.x));
  const topY = Math.max(...points.map((p) => p.y));
  const bottomY = Math.min(...points.map((p) => p.y));
  const height = topY - bottomY;
  if (!(height > 0.05)) return null;

  const bin = Math.max(0.0015, height / 90);
  const mid = sampleProfile(depthProfile(points, centerX - height * 0.03, centerX + height * 0.03, bin), bin, bottomY, topY);

  // Nose tip: most forward midline point in the middle half of the head.
  let noseIdx = -1;
  for (let i = 0; i < mid.ys.length; i++) {
    const y = mid.ys[i];
    if (y < bottomY + 0.25 * height || y > bottomY + 0.75 * height) continue;
    if (noseIdx < 0 || mid.zs[i] > mid.zs[noseIdx]) noseIdx = i;
  }
  if (noseIdx < 0) return null;
  const noseTip = new THREE.Vector3(centerX, mid.ys[noseIdx], mid.zs[noseIdx]);
  // Reference: nose tip to chin is ~0.4 of the head height on adult faces.
  let scale = Math.max(0.5, Math.min(1.8, height / 0.21));

  // Mouth line. Below the nose the midline profile reads: subnasale (hollow), upper lip
  // (bulge), lip line (hollow, often very shallow on closed-mouth meshes), lower lip (bulge),
  // labiomental groove (hollow, usually deeper), chin (bulge). The lip line sits at about 1/3 of
  // the way from the subnasale to the chin bottom; pick the hollow closest to that estimate.
  let mouthY = noseTip.y - 0.036 * scale;
  let mouthDetected = false;
  {
    // Raw midline profile: one sample per vertex row, front-most Z, top to bottom.
    const rowTol = 0.0006 * scale;
    const raw = points
      .filter(
        (p) =>
          Math.abs(p.x - centerX) < 0.0025 * scale &&
          p.y < noseTip.y &&
          p.y > bottomY &&
          p.z > noseTip.z - 0.07 * scale, // front of the face only, not the back of the head
      )
      .sort((a, b) => b.y - a.y);
    const prof: Array<{ y: number; z: number }> = [];
    for (const p of raw) {
      const last = prof[prof.length - 1];
      if (last && last.y - p.y < rowTol) last.z = Math.max(last.z, p.z);
      else prof.push({ y: p.y, z: p.z });
    }
    // A hollow is the lowest Z within +-2 mm and at least 0.3 mm below the highest Z there.
    const reach = 0.002 * scale;
    const isMin = (i: number) => {
      const { y, z } = prof[i];
      let lower = false;
      let upper = false;
      let top = z;
      for (const q of prof) {
        if (Math.abs(q.y - y) > reach || q === prof[i]) continue;
        if (q.z < z) return false;
        if (q.y > y) upper = true;
        else lower = true;
        top = Math.max(top, q.z);
      }
      return upper && lower && top - z > 0.0003 * scale;
    };
    // Subnasale: first hollow below the nose tip.
    let sub = prof.findIndex((_, i) => isMin(i) && noseTip.y - prof[i].y < 0.03 * scale);
    const subY = sub >= 0 ? prof[sub].y : noseTip.y - 0.014 * scale;
    // Chin bottom: below the chin bulge, where the profile turns back under the jaw.
    const chinZone = prof.filter((p) => p.y < subY - 0.025 * scale && p.y > subY - 0.09 * scale);
    let mentonY = bottomY;
    if (chinZone.length) {
      const chinZ = Math.max(...chinZone.map((p) => p.z));
      const chinIdx = prof.findIndex((p) => p.z === chinZ && p.y < subY - 0.025 * scale);
      const under = prof.slice(chinIdx).find((p) => p.z < chinZ - 0.015 * scale);
      if (under) mentonY = under.y;
    }
    const lowerFace = subY - mentonY;
    if (lowerFace > 0.02 * scale) {
      const prior = subY - 0.31 * lowerFace;
      let best = -1;
      for (let i = 0; i < prof.length; i++) {
        if (!isMin(i) || i === sub) continue;
        const y = prof[i].y;
        if (y > subY - 0.12 * lowerFace || y < subY - 0.6 * lowerFace) continue;
        if (best < 0 || Math.abs(y - prior) < Math.abs(prof[best].y - prior)) best = i;
      }
      if (best >= 0 && Math.abs(prof[best].y - prior) < 0.35 * lowerFace) {
        mouthY = prof[best].y;
        mouthDetected = true;
      } else {
        mouthY = prior;
      }
    }
  }
  const zAtY = (prof: { ys: number[]; zs: number[] }, y: number) => {
    let bestI = 0;
    for (let i = 0; i < prof.ys.length; i++) if (Math.abs(prof.ys[i] - y) < Math.abs(prof.ys[bestI] - y)) bestI = i;
    return prof.zs[bestI];
  };
  const mouthZ = zAtY(mid, mouthY);

  // Face half width at cheek height (between mouth and eyes), front half only.
  const cheekY = (noseTip.y + mouthY) / 2;
  const cheekXs = points.filter((p) => Math.abs(p.y - cheekY) < 0.01 * scale && p.z > noseTip.z - 0.1 * scale).map((p) => Math.abs(p.x - centerX));
  const halfWidth = cheekXs.length ? Math.max(...cheekXs) : 0.075 * scale;

  // Eyes: recess of the sockets above the nose tip, around 0.4 of the half width.
  const eyeOffsetX = Math.max(0.022 * scale, Math.min(0.042 * scale, 0.4 * halfWidth));
  let eyeY = noseTip.y + 0.024 * scale;
  let eyesDetected = false;
  const eyeProfiles = [1, -1].map((side) => {
    const x = centerX + side * eyeOffsetX;
    return sampleProfile(depthProfile(points, x - 0.005 * scale, x + 0.005 * scale, bin), bin, noseTip.y - 0.02 * scale, noseTip.y + 0.08 * scale);
  });
  {
    // Concavity of the vertical profile through the eyes: depth of z(y) below the mean of
    // z(y - d) and z(y + d). Unlike the plain minimum, it ignores the backward slope of the
    // forehead, so it finds the center of the socket (cheek below, brow above).
    const avg = eyeProfiles[0].zs.map((z, i) => (z + eyeProfiles[1].zs[i]) / 2);
    const ys = eyeProfiles[0].ys;
    const zAt = (y: number) => {
      if (y <= ys[0]) return avg[0];
      if (y >= ys[ys.length - 1]) return avg[avg.length - 1];
      const f = (y - ys[0]) / (ys[1] - ys[0]);
      const i = Math.floor(f);
      return avg[i] + (avg[i + 1] - avg[i]) * (f - i);
    };
    const d = 0.018 * scale;
    let bestY = eyeY;
    let bestDepth = 0;
    for (let y = noseTip.y + 0.008 * scale; y <= noseTip.y + 0.05 * scale; y += bin / 2) {
      const depth = (zAt(y - d) + zAt(y + d)) / 2 - zAt(y);
      if (depth > bestDepth) {
        bestDepth = depth;
        bestY = y;
      }
    }
    if (bestDepth > 0.001 * scale) {
      eyeY = bestY;
      eyesDetected = true;
    }
  }
  // Refine the eye X: center of the flat eye area between the nose bridge slope and the
  // outer fall-off of the face, along the eye line (both sides mirrored).
  let eyeX = eyeOffsetX;
  {
    const xBin = 0.0025 * scale;
    const row = new Map<number, number>();
    for (const p of points) {
      if (Math.abs(p.y - eyeY) > 0.003 * scale) continue;
      const b = Math.round(Math.abs(p.x - centerX) / xBin);
      const cur = row.get(b);
      if (cur === undefined || p.z > cur) row.set(b, p.z);
    }
    const lo = Math.round((0.01 * scale) / xBin);
    const hi = Math.round((0.05 * scale) / xBin);
    const zs: number[] = [];
    for (let b = lo; b <= hi; b++) if (row.has(b)) zs.push(row.get(b)!);
    if (zs.length >= 5) {
      const plateau = median(zs);
      let start = -1;
      let end = -1;
      for (let b = lo; b <= hi; b++) {
        const z = row.get(b);
        if (z === undefined) continue;
        const flat = Math.abs(z - plateau) <= 0.0025 * scale;
        if (flat && start < 0) start = b;
        if (flat) end = b;
        else if (start >= 0 && z < plateau) break; // outer fall-off reached
      }
      if (start >= 0 && end > start) {
        const x = ((start + end) / 2) * xBin;
        if (x > 0.018 * scale && x < 0.045 * scale) eyeX = x;
      }
    }
  }
  const eyeZ = (zAtY(eyeProfiles[0], eyeY) + zAtY(eyeProfiles[1], eyeY)) / 2;

  // Refine the size factor with the eye-to-mouth distance when both were detected (stylized
  // faces vary a lot in proportions, so average it with the head height estimate).
  if (mouthDetected && eyesDetected) {
    scale = Math.max(0.5, Math.min(1.8, (scale + (eyeY - mouthY) / 0.072) / 2));
  }

  return {
    centerX,
    halfWidth,
    topY,
    chinY: bottomY,
    noseTip,
    mouthY,
    mouthZ,
    eyeY,
    eyeOffsetX: eyeX,
    eyeZ,
    browY: eyeY + 0.025 * scale,
    scale,
    mouthDetected,
    eyesDetected,
  };
}

/** World-space displacement of one channel at full weight for a point on the face. */
export function faceChannelDisplacement(channel: FaceChannel, p: THREE.Vector3, L: FaceLandmarks, out: THREE.Vector3): THREE.Vector3 {
  const s = L.scale;
  const dx = p.x - L.centerX;
  const ax = Math.abs(dx);
  const side = dx >= 0 ? 1 : -1;
  out.set(0, 0, 0);
  // Weight of the front of the face (ignores the back of the head and the ears).
  const front = smoothstep(L.noseTip.z - 0.075 * s, L.noseTip.z - 0.045 * s, p.z);

  switch (channel) {
    case 'mouthOpen': {
      // Jaw rotation around a hinge near the ears, lower lip and chin only.
      const pivot = new THREE.Vector3(L.centerX, L.mouthY + 0.012 * s, L.noseTip.z - 0.095 * s);
      const below =
        smoothstep(L.mouthY + 0.0012 * s, L.mouthY - 0.0035 * s, p.y) * smoothstep(L.chinY - 0.035 * s, L.chinY - 0.005 * s, p.y);
      const lateral = smoothstep(0.08 * s, 0.045 * s, ax);
      const ahead = smoothstep(pivot.z - 0.01 * s, pivot.z + 0.03 * s, p.z);
      const w = below * lateral * ahead;
      if (w > 0) {
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.14 * w);
        out.copy(p).sub(pivot).applyQuaternion(q).add(pivot).sub(p);
      }
      // Upper lip lifts a little.
      const upper = smoothstep(L.mouthY - 0.0005 * s, L.mouthY + 0.002 * s, p.y) * gauss(p.y - L.mouthY - 0.004 * s, 0.008 * s);
      out.y += 0.0018 * s * upper * gauss(dx, 0.022 * s) * front;
      break;
    }
    case 'eyeBlinkLeft':
    case 'eyeBlinkRight': {
      const eyeSide = channel === 'eyeBlinkLeft' ? 1 : -1;
      const ex = L.centerX + eyeSide * L.eyeOffsetX;
      const lateral = smoothstep(0.02 * s, 0.012 * s, Math.abs(p.x - ex));
      const onEye = lateral * smoothstep(L.eyeZ - 0.016 * s, L.eyeZ - 0.007 * s, p.z);
      if (onEye <= 0) break;
      const closeY = L.eyeY - 0.002 * s;
      if (p.y >= closeY) {
        // Upper lid folds down onto the closing line.
        const c = smoothstep(L.eyeY + 0.016 * s, L.eyeY + 0.008 * s, p.y);
        out.y = -(p.y - closeY) * 0.95 * c * onEye;
        out.z = 0.0014 * s * c * onEye;
      } else {
        // Lower lid rises a little.
        const c = smoothstep(L.eyeY - 0.012 * s, L.eyeY - 0.005 * s, p.y);
        out.y = (closeY - p.y) * 0.3 * c * onEye;
      }
      break;
    }
    case 'browUp':
    case 'browDown': {
      const ex = L.centerX + side * L.eyeOffsetX;
      const lateral = smoothstep(0.034 * s, 0.022 * s, Math.abs(p.x - ex));
      const w = gauss(p.y - L.browY, 0.009 * s) * lateral * front;
      if (w <= 0) break;
      if (channel === 'browUp') out.set(0, 0.0055 * s, 0.0008 * s).multiplyScalar(w);
      else {
        const inner = smoothstep(L.eyeOffsetX + 0.01 * s, L.eyeOffsetX - 0.012 * s, ax);
        out.set(-side * 0.0016 * s * inner, -0.0042 * s, 0.0006 * s).multiplyScalar(w);
      }
      break;
    }
    case 'mouthSmile':
    case 'mouthFrown': {
      const cx = L.centerX + side * 0.025 * s;
      const g = gauss(Math.hypot(p.x - cx, p.y - L.mouthY), 0.017 * s) * front;
      if (channel === 'mouthSmile') {
        out.set(side * 0.0035 * s, 0.004 * s, -0.0015 * s).multiplyScalar(g);
        // cheeks rise with the smile
        const cheek = gauss(Math.hypot(p.x - (L.centerX + side * 0.034 * s), p.y - (L.mouthY + 0.022 * s)), 0.014 * s) * front;
        out.y += 0.0016 * s * cheek;
        out.z += 0.0008 * s * cheek;
      } else {
        out.set(side * 0.0008 * s, -0.0036 * s, 0).multiplyScalar(g);
      }
      break;
    }
    case 'mouthPucker': {
      const g = Math.exp(-((dx / (0.024 * s)) ** 2 + ((p.y - L.mouthY) / (0.012 * s)) ** 2)) * front;
      out.set(-dx * 0.28 * g, 0, 0.0055 * s * g);
      break;
    }
    case 'cheekPuff': {
      const cx = L.centerX + side * 0.04 * s;
      const g = gauss(Math.hypot(p.x - cx, p.y - (L.mouthY + 0.012 * s)), 0.02 * s) * smoothstep(L.noseTip.z - 0.11 * s, L.noseTip.z - 0.07 * s, p.z);
      out.set(side * 0.7, 0, 0.7).multiplyScalar(0.006 * s * g);
      break;
    }
  }
  return out;
}

/** Score of a mesh as "the face": head vertices on the front of the lower face. */
function faceScore(verts: HeadVertex[]): number {
  if (verts.length < 50) return 0;
  const ys = verts.map((v) => v.world.y);
  const zs = verts.map((v) => v.world.z);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const zMid = (Math.min(...zs) + Math.max(...zs)) / 2;
  const cx = median(verts.map((v) => v.world.x));
  return verts.filter(
    (v) => v.weight > 0.5 && v.world.y < yMin + 0.45 * (yMax - yMin) && v.world.z > zMid && Math.abs(v.world.x - cx) < 0.03,
  ).length;
}

// ---------------------------------------------------------------------------
// Facial bones
// ---------------------------------------------------------------------------

/** Full motion of a facial bone for one channel at weight 1 (world space, bind pose). */
export type FaceBoneMotion =
  | { kind: 'rotate'; axis: THREE.Vector3; angle: number }
  | { kind: 'translate'; delta: THREE.Vector3 };

export interface FaceBoneDef {
  name: string;
  /** Bone position (and rotation pivot) in world space. */
  pivot: THREE.Vector3;
  motions: Partial<Record<FaceChannel, FaceBoneMotion>>;
  /** Channel whose displacement field defines the bone's skin weights. */
  primary: FaceChannel;
}

/** Displacement of point p when the bone performs `motion` at weight 1. */
export function faceBoneMotionDelta(def: FaceBoneDef, motion: FaceBoneMotion, p: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
  if (motion.kind === 'translate') return out.copy(motion.delta);
  const q = new THREE.Quaternion().setFromAxisAngle(motion.axis, motion.angle);
  return out.copy(p).sub(def.pivot).applyQuaternion(q).add(def.pivot).sub(p);
}

/**
 * Facial bones fitted to the detected landmarks: jaw, upper/lower eyelids, brows, mouth corners
 * and cheeks. Only the bones that serve one of `channels` are returned.
 */
export function faceBoneDefsFromLandmarks(L: FaceLandmarks, channels: readonly FaceChannel[]): FaceBoneDef[] {
  const s = L.scale;
  const X = new THREE.Vector3(1, 0, 0);
  const defs: FaceBoneDef[] = [
    {
      name: 'AutoFace_Jaw',
      pivot: new THREE.Vector3(L.centerX, L.mouthY + 0.012 * s, L.noseTip.z - 0.095 * s),
      motions: { mouthOpen: { kind: 'rotate', axis: X, angle: 0.14 } },
      primary: 'mouthOpen',
    },
  ];
  for (const [side, suffix, blink] of [
    [1, 'L', 'eyeBlinkLeft'],
    [-1, 'R', 'eyeBlinkRight'],
  ] as const) {
    const ex = L.centerX + side * L.eyeOffsetX;
    const eyeCenter = new THREE.Vector3(ex, L.eyeY, L.eyeZ - 0.012 * s);
    defs.push(
      {
        name: `AutoFace_EyelidUpper_${suffix}`,
        pivot: eyeCenter.clone(),
        motions: { [blink]: { kind: 'rotate', axis: X, angle: 0.75 } },
        primary: blink,
      },
      {
        name: `AutoFace_EyelidLower_${suffix}`,
        pivot: eyeCenter.clone(),
        motions: { [blink]: { kind: 'rotate', axis: X, angle: -0.25 } },
        primary: blink,
      },
      {
        name: `AutoFace_Brow_${suffix}`,
        pivot: new THREE.Vector3(ex, L.browY, L.eyeZ + 0.004 * s),
        motions: {
          browUp: { kind: 'translate', delta: new THREE.Vector3(0, 0.0055 * s, 0.0008 * s) },
          browDown: { kind: 'translate', delta: new THREE.Vector3(-side * 0.0008 * s, -0.0042 * s, 0.0006 * s) },
        },
        primary: 'browUp',
      },
      {
        name: `AutoFace_MouthCorner_${suffix}`,
        pivot: new THREE.Vector3(L.centerX + side * 0.025 * s, L.mouthY, L.mouthZ - 0.006 * s),
        motions: {
          mouthSmile: { kind: 'translate', delta: new THREE.Vector3(side * 0.0035 * s, 0.004 * s, -0.0015 * s) },
          mouthFrown: { kind: 'translate', delta: new THREE.Vector3(side * 0.0008 * s, -0.0036 * s, 0) },
          mouthPucker: { kind: 'translate', delta: new THREE.Vector3(-side * 0.006 * s, 0, 0.0045 * s) },
        },
        primary: 'mouthSmile',
      },
      {
        name: `AutoFace_Cheek_${suffix}`,
        pivot: new THREE.Vector3(L.centerX + side * 0.04 * s, L.mouthY + 0.012 * s, L.noseTip.z - 0.045 * s),
        motions: { cheekPuff: { kind: 'translate', delta: new THREE.Vector3(side * 0.7, 0, 0.7).multiplyScalar(0.006 * s) } },
        primary: 'cheekPuff',
      },
    );
  }
  const wanted = new Set<FaceChannel>(channels);
  return defs
    .map((d) => {
      const motions: FaceBoneDef['motions'] = {};
      for (const [ch, m] of Object.entries(d.motions) as Array<[FaceChannel, FaceBoneMotion]>) if (wanted.has(ch)) motions[ch] = m;
      return { ...d, motions };
    })
    .filter((d) => Object.keys(d.motions).length > 0 && wanted.has(d.primary));
}

interface CreatedFaceBone {
  bone: THREE.Bone;
  def: FaceBoneDef;
  bindPosition: THREE.Vector3;
  bindQuaternion: THREE.Quaternion;
  /** Per channel: rotation axis (bone local) or translation (parent local). */
  localAxis: Partial<Record<FaceChannel, THREE.Vector3>>;
  localDelta: Partial<Record<FaceChannel, THREE.Vector3>>;
}

function uniqueName(root: THREE.Object3D, base: string): string {
  let name = base;
  for (let i = 2; root.getObjectByName(name); i++) name = `${base}_${i}`;
  return name;
}

function createFaceBones(root: THREE.Object3D, headNode: THREE.Object3D, defs: FaceBoneDef[]): CreatedFaceBone[] {
  root.updateMatrixWorld(true);
  const headQuatInv = headNode.getWorldQuaternion(new THREE.Quaternion()).invert();
  const headScale = headNode.getWorldScale(new THREE.Vector3());
  return defs.map((def) => {
    const bone = new THREE.Bone();
    bone.name = uniqueName(root, def.name);
    bone.position.copy(headNode.worldToLocal(def.pivot.clone()));
    headNode.add(bone);
    bone.updateMatrixWorld(true);
    const localAxis: CreatedFaceBone['localAxis'] = {};
    const localDelta: CreatedFaceBone['localDelta'] = {};
    for (const [ch, m] of Object.entries(def.motions) as Array<[FaceChannel, FaceBoneMotion]>) {
      // The bone keeps the head's orientation (identity local rotation).
      if (m.kind === 'rotate') localAxis[ch] = m.axis.clone().applyQuaternion(headQuatInv).normalize();
      else localDelta[ch] = m.delta.clone().applyQuaternion(headQuatInv).divide(headScale);
    }
    return { bone, def, bindPosition: bone.position.clone(), bindQuaternion: bone.quaternion.clone(), localAxis, localDelta };
  });
}

/** Keyframe tracks of the facial bones for a payload's morph channels. */
function buildFaceBoneTracks(
  bones: CreatedFaceBone[],
  morphTracks: Record<string, number[]>,
  times: number[],
): THREE.KeyframeTrack[] {
  const n = times.length;
  const tracks: THREE.KeyframeTrack[] = [];
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
  const active = (ch: string) => {
    const s = morphTracks[ch];
    return s && s.length === n && s.some((v) => Math.abs(v) > 1e-4) ? s : null;
  };
  const qStep = new THREE.Quaternion();
  for (const fb of bones) {
    const rotChannels = (Object.keys(fb.localAxis) as FaceChannel[]).filter((c) => active(c));
    if (rotChannels.length) {
      const buf: number[] = [];
      const q = new THREE.Quaternion();
      for (let f = 0; f < n; f++) {
        q.copy(fb.bindQuaternion);
        for (const ch of rotChannels) {
          const m = fb.def.motions[ch] as Extract<FaceBoneMotion, { kind: 'rotate' }>;
          q.multiply(qStep.setFromAxisAngle(fb.localAxis[ch]!, m.angle * clamp01(morphTracks[ch][f])));
        }
        q.normalize();
        if (buf.length >= 4 && buf[buf.length - 4] * q.x + buf[buf.length - 3] * q.y + buf[buf.length - 2] * q.z + buf[buf.length - 1] * q.w < 0) {
          q.set(-q.x, -q.y, -q.z, -q.w);
        }
        buf.push(q.x, q.y, q.z, q.w);
      }
      tracks.push(new THREE.QuaternionKeyframeTrack(`${fb.bone.name}.quaternion`, times, buf));
    }
    const posChannels = (Object.keys(fb.localDelta) as FaceChannel[]).filter((c) => active(c));
    if (posChannels.length) {
      const buf: number[] = [];
      const p = new THREE.Vector3();
      for (let f = 0; f < n; f++) {
        p.copy(fb.bindPosition);
        for (const ch of posChannels) p.addScaledVector(fb.localDelta[ch]!, clamp01(morphTracks[ch][f]));
        buf.push(p.x, p.y, p.z);
      }
      tracks.push(new THREE.VectorKeyframeTrack(`${fb.bone.name}.position`, times, buf));
    }
  }
  return tracks;
}

// ---------------------------------------------------------------------------
// Mixed rig engine: facial bones for the main motion, blendshapes for the rest
// ---------------------------------------------------------------------------

export interface FaceRigMeshInput {
  mesh: THREE.Mesh;
  verts: HeadVertex[];
  /** 0..1: how much a vertex follows the face for a channel (1 on the face mesh). */
  follow: (v: HeadVertex, ch: FaceChannel) => number;
}

export interface MixedFaceRigResult {
  channels: FaceChannel[];
  /** Facial bones added under the head bone. */
  bones: THREE.Bone[];
  /** Track key of each mesh that received residual morph targets, with its morph names. */
  meshes: Array<{ meshName: string; morphs: FaceChannel[] }>;
  buildFacialBoneTracks: (morphTracks: Record<string, number[]>, times: number[]) => THREE.KeyframeTrack[];
}

/**
 * Builds a mixed facial rig for `channels` on the given head meshes:
 *   - creates the facial bones under `headNode`;
 *   - gives each vertex skin weights toward them, fitted so that the bone motion reproduces
 *     the channel's displacement field (`target`) as closely as linear skinning allows;
 *   - stores what the bones cannot reproduce (the residual) as a relative morph target.
 * Bone + morph together give exactly the target displacement at the bind pose.
 */
export function applyMixedFaceRig(params: {
  root: THREE.Object3D;
  headNode: THREE.Object3D;
  channels: readonly FaceChannel[];
  boneDefs: FaceBoneDef[];
  meshes: FaceRigMeshInput[];
  /** Target world displacement of a vertex for a channel at weight 1. */
  target: (ch: FaceChannel, v: HeadVertex, out: THREE.Vector3) => THREE.Vector3;
  trackKeyOf: (obj: THREE.Object3D) => string;
  /** Displacements below this length are ignored (meters). */
  minDelta?: number;
}): MixedFaceRigResult | null {
  const { root, headNode, channels, meshes, target, trackKeyOf } = params;
  const minDelta = params.minDelta ?? 1e-5;
  root.updateMatrixWorld(true);

  const headSet = new Set<THREE.Object3D>();
  headNode.traverse((o) => headSet.add(o));
  const created = params.boneDefs.length ? createFaceBones(root, headNode, params.boneDefs) : [];
  root.updateMatrixWorld(true);

  // One extended skeleton per original skeleton: original bones + facial bones. The facial
  // bones get the head's bind matrix, so moving weights from the head to them keeps the rest shape.
  const extended = new Map<THREE.Skeleton, { skeleton: THREE.Skeleton; faceIndex: number[] } | null>();
  const extend = (sk: THREE.Skeleton) => {
    if (extended.has(sk)) return extended.get(sk)!;
    const headIdx = sk.bones.indexOf(headNode as THREE.Bone);
    if (headIdx < 0 || !created.length) {
      extended.set(sk, null);
      return null;
    }
    const headBoneMatrix = new THREE.Matrix4().multiplyMatrices(headNode.matrixWorld, sk.boneInverses[headIdx]);
    const bones = [...sk.bones];
    const inverses = sk.boneInverses.map((m) => m.clone());
    const faceIndex = created.map((fb) => {
      bones.push(fb.bone);
      inverses.push(fb.bone.matrixWorld.clone().invert().multiply(headBoneMatrix));
      return bones.length - 1;
    });
    const entry = { skeleton: new THREE.Skeleton(bones, inverses), faceIndex };
    extended.set(sk, entry);
    return entry;
  };

  const used = new Set<FaceChannel>();
  const result: MixedFaceRigResult = {
    channels: [],
    bones: created.map((c) => c.bone),
    meshes: [],
    buildFacialBoneTracks: (morphTracks, times) => buildFaceBoneTracks(created, morphTracks, times),
  };

  const t = new THREE.Vector3();
  const full = new THREE.Vector3();
  const local = new THREE.Vector3();
  const base = new THREE.Vector3();
  const probe = new THREE.Vector3();
  const jac = new THREE.Matrix3();

  for (const input of meshes) {
    const { mesh, verts } = input;
    const geo = mesh.geometry as THREE.BufferGeometry;
    const pos = geo.attributes.position;
    const sm = mesh as THREE.SkinnedMesh;
    const ext = sm.isSkinnedMesh && sm.skeleton ? extend(sm.skeleton) : null;
    let skinIndex = geo.attributes.skinIndex as THREE.BufferAttribute | undefined;
    const skinWeight = geo.attributes.skinWeight as THREE.BufferAttribute | undefined;
    if (ext && skinIndex && ext.skeleton.bones.length > 255 && skinIndex.array instanceof Uint8Array) {
      skinIndex = new THREE.Uint16BufferAttribute(Uint16Array.from(skinIndex.array as Uint8Array), skinIndex.itemSize);
      geo.setAttribute('skinIndex', skinIndex);
    }
    const canSkin = Boolean(ext && skinIndex && skinWeight && skinIndex.itemSize >= 4);
    const headBoneIdx = new Set<number>();
    if (ext) ext.skeleton.bones.forEach((b, i) => headSet.has(b) && headBoneIdx.add(i));
    const buffers = new Map<FaceChannel, Float32Array>();
    let weightsChanged = false;

    for (const v of verts) {
      // Target displacement per channel.
      const targets = new Map<FaceChannel, THREE.Vector3>();
      for (const ch of channels) {
        const f = input.follow(v, ch);
        if (f <= 1e-3) continue;
        target(ch, v, t);
        if (t.lengthSq() * f * f > minDelta * minDelta) targets.set(ch, t.clone().multiplyScalar(f));
      }
      if (!targets.size) continue;

      // Skinning Jacobian at the bind pose (geometry space -> world), before weights change.
      local.fromBufferAttribute(pos, v.index);
      vertexWorld(mesh, v.index, local, base);
      const cols: THREE.Vector3[] = [];
      for (let k = 0; k < 3; k++) {
        probe.copy(local);
        probe.setComponent(k, probe.getComponent(k) + 1);
        cols.push(vertexWorld(mesh, v.index, probe, new THREE.Vector3()).sub(base));
      }
      jac.set(cols[0].x, cols[1].x, cols[2].x, cols[0].y, cols[1].y, cols[2].y, cols[0].z, cols[1].z, cols[2].z);
      if (Math.abs(jac.determinant()) < 1e-12) continue;
      const jacInv = jac.clone().invert();

      // Bone weights fitted to each bone's primary channel: w = <target, motion> / |motion|^2.
      const boneW: number[] = created.map(() => 0);
      if (canSkin) {
        created.forEach((fb, b) => {
          const tg = targets.get(fb.def.primary);
          const m = fb.def.motions[fb.def.primary];
          if (!tg || !m) return;
          faceBoneMotionDelta(fb.def, m, v.world, full);
          const len2 = full.lengthSq();
          if (len2 < 1e-12) return;
          boneW[b] = Math.max(0, Math.min(1, tg.dot(full) / len2));
        });
      }

      // Move weight from the head bone(s) to the facial bones, keep the 4 strongest influences.
      const achievedW: number[] = created.map(() => 0);
      if (canSkin && boneW.some((w) => w > 0.02)) {
        const si = skinIndex!;
        const sw = skinWeight!;
        const infl: Array<{ idx: number; w: number }> = [];
        for (let c = 0; c < 4; c++) infl.push({ idx: si.getComponent(v.index, c), w: sw.getComponent(v.index, c) });
        const hw = infl.filter((e) => headBoneIdx.has(e.idx)).reduce((s, e) => s + e.w, 0);
        const sum = boneW.reduce((s, w) => s + (w > 0.02 ? w : 0), 0);
        const k = sum > 1 ? 1 / sum : 1;
        const faceShare = Math.min(1, sum * k);
        for (const e of infl) if (headBoneIdx.has(e.idx)) e.w *= 1 - faceShare;
        boneW.forEach((w, b) => {
          if (w > 0.02) infl.push({ idx: ext!.faceIndex[b], w: w * k * hw });
        });
        const merged = new Map<number, number>();
        for (const e of infl) if (e.w > 1e-6) merged.set(e.idx, (merged.get(e.idx) ?? 0) + e.w);
        const top = [...merged.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
        const total = top.reduce((s, [, w]) => s + w, 0) || 1;
        for (let c = 0; c < 4; c++) {
          const entry = top[c];
          si.setComponent(v.index, c, entry ? entry[0] : 0);
          sw.setComponent(v.index, c, entry ? entry[1] / total : 0);
        }
        // Weight the facial bones really got, relative to how much the vertex follows the head.
        top.forEach(([idx, w]) => {
          const b = ext!.faceIndex.indexOf(idx);
          if (b >= 0 && hw > 0) achievedW[b] = w / total / hw;
        });
        weightsChanged = true;
      }

      // Residual = target - bone motion, stored as a morph target in geometry space.
      for (const [ch, tg] of targets) {
        const residual = tg.clone();
        created.forEach((fb, b) => {
          const m = fb.def.motions[ch];
          if (!m || achievedW[b] <= 0) return;
          faceBoneMotionDelta(fb.def, m, v.world, full);
          residual.addScaledVector(full, -achievedW[b] * v.weight);
        });
        if (residual.lengthSq() < minDelta * minDelta) continue;
        let buf = buffers.get(ch);
        if (!buf) {
          buf = new Float32Array(pos.count * 3);
          buffers.set(ch, buf);
        }
        residual.applyMatrix3(jacInv);
        buf[v.index * 3] = residual.x;
        buf[v.index * 3 + 1] = residual.y;
        buf[v.index * 3 + 2] = residual.z;
      }
      for (const ch of targets.keys()) used.add(ch);
    }

    if (ext && weightsChanged) {
      skinIndex!.needsUpdate = true;
      skinWeight!.needsUpdate = true;
    }
    if (ext) sm.bind(ext.skeleton, sm.bindMatrix);

    if (!buffers.size) continue;
    // Append the residual morph targets without touching the authored ones.
    const relative = geo.morphAttributes.position ? geo.morphTargetsRelative : true;
    geo.morphAttributes.position ??= [];
    geo.morphTargetsRelative = relative;
    mesh.morphTargetDictionary ??= {};
    mesh.morphTargetInfluences ??= [];
    const added: FaceChannel[] = [];
    for (const [ch, buf] of buffers) {
      if (!relative) {
        for (let i = 0; i < pos.count; i++) {
          buf[i * 3] += pos.getX(i);
          buf[i * 3 + 1] += pos.getY(i);
          buf[i * 3 + 2] += pos.getZ(i);
        }
      }
      const attr = new THREE.Float32BufferAttribute(buf, 3);
      attr.name = ch;
      geo.morphAttributes.position.push(attr);
      if (geo.morphAttributes.normal) {
        // Normals are not displaced: zero delta (relative) or the base normals (absolute).
        const nrm = new Float32Array(pos.count * 3);
        const baseNormal = geo.attributes.normal;
        if (!relative && baseNormal) for (let i = 0; i < nrm.length; i++) nrm[i] = (baseNormal.array as ArrayLike<number>)[i];
        geo.morphAttributes.normal.push(new THREE.Float32BufferAttribute(nrm, 3));
      }
      mesh.morphTargetDictionary[ch] = geo.morphAttributes.position.length - 1;
      mesh.morphTargetInfluences.push(0);
      added.push(ch);
    }
    result.meshes.push({ meshName: trackKeyOf(mesh), morphs: added });
  }

  // Meshes that were rebound but had no face vertices still need the extended skeleton,
  // otherwise two skeletons would fight over the shared bones.
  root.traverse((obj) => {
    const sm = obj as THREE.SkinnedMesh;
    if (!sm.isSkinnedMesh || !sm.skeleton) return;
    const ext = extended.get(sm.skeleton);
    if (ext) sm.bind(ext.skeleton, sm.bindMatrix);
  });

  result.channels = FACE_CHANNELS.filter((c) => used.has(c));
  return result.channels.length || created.length ? result : null;
}

/**
 * Automatic facial rig for an arbitrary avatar: detects the face, adds the facial bones and the
 * residual blendshapes for `channels`. Returns null when the model has no usable face geometry.
 */
export function buildAutomaticFaceRig(
  root: THREE.Object3D,
  headNode: THREE.Object3D | null,
  channels: FaceChannel[],
  trackKeyOf: (obj: THREE.Object3D) => string,
  options: { bones?: boolean } = {},
): AutoFaceRigResult | null {
  if (!headNode || channels.length === 0) return null;
  root.updateMatrixWorld(true);
  const byMesh = collectHeadVertices(root, headNode);
  if (!byMesh.size) return null;

  // Landmarks come from the face mesh only (not hair, hats or glasses).
  let faceMesh: THREE.Mesh | null = null;
  let bestScore = 0;
  for (const [mesh, verts] of byMesh) {
    const sc = faceScore(verts);
    if (sc > bestScore) {
      bestScore = sc;
      faceMesh = mesh;
    }
  }
  if (!faceMesh || bestScore < 30) return null;
  const facePoints = byMesh.get(faceMesh)!.filter((v) => v.weight >= 0.5).map((v) => v.world);
  const landmarks = detectFaceLandmarks(facePoints);
  if (!landmarks) return null;

  // Other head meshes (eyebrows, lashes, beard, teeth) follow the face only near its surface;
  // hair strands or accessories away from the skin stay still.
  const cell = 0.01 * landmarks.scale;
  const maxSurfaceDist = 0.02 * landmarks.scale;
  const grid = new Map<string, THREE.Vector3[]>();
  for (const p of facePoints) {
    const k = `${Math.floor(p.x / cell)},${Math.floor(p.y / cell)},${Math.floor(p.z / cell)}`;
    const list = grid.get(k);
    if (list) list.push(p);
    else grid.set(k, [p]);
  }
  const distCache = new Map<HeadVertex, number>();
  const faceSurfaceDistance = (v: HeadVertex): number => {
    const cached = distCache.get(v);
    if (cached !== undefined) return cached;
    const p = v.world;
    const r = Math.ceil(maxSurfaceDist / cell);
    const bx = Math.floor(p.x / cell);
    const by = Math.floor(p.y / cell);
    const bz = Math.floor(p.z / cell);
    let best = Infinity;
    for (let i = -r; i <= r; i++)
      for (let j = -r; j <= r; j++)
        for (let k = -r; k <= r; k++) {
          const list = grid.get(`${bx + i},${by + j},${bz + k}`);
          if (list) for (const q of list) best = Math.min(best, q.distanceToSquared(p));
        }
    const d = Math.sqrt(best);
    distCache.set(v, d);
    return d;
  };
  // Lids close only what lies on them (lashes), not hair strands in front of the eyes.
  const channelReach = (ch: FaceChannel): number =>
    ch === 'eyeBlinkLeft' || ch === 'eyeBlinkRight' ? 0.006 * landmarks.scale : maxSurfaceDist;

  const meshes: FaceRigMeshInput[] = [...byMesh].map(([mesh, verts]) => ({
    mesh,
    verts,
    follow:
      mesh === faceMesh
        ? (v: HeadVertex) => v.weight
        : (v: HeadVertex, ch: FaceChannel) => {
            const reach = channelReach(ch);
            return v.weight * smoothstep(reach, reach * 0.5, faceSurfaceDistance(v));
          },
  }));

  const rig = applyMixedFaceRig({
    root,
    headNode,
    channels,
    boneDefs: options.bones === false ? [] : faceBoneDefsFromLandmarks(landmarks, channels),
    meshes,
    target: (ch, v, out) => faceChannelDisplacement(ch, v.world, landmarks, out),
    trackKeyOf,
    minDelta: 1e-5 * landmarks.scale,
  });
  if (!rig) return null;
  return { ...rig, landmarks };
}
