/**
 * Automatic facial rig.
 *
 * Third level of the facial pipeline, after (1) the model's own blendshapes and (2) facial
 * bones (Rigify, generic jaw). For every LIS face channel still missing, it synthesizes a
 * blendshape on the meshes skinned to the head, from the head geometry itself:
 *
 *   1. Collect the vertices weighted to the head bone (world bind pose: +Y up, +Z forward,
 *      +X = character's left, as set by normalizeAvatarRootTransform).
 *   2. Detect face landmarks on the face mesh: nose tip (most forward point of the midline
 *      profile), mouth line (deepest point between the two lip bulges), eye line (recess of
 *      the eye sockets), brows, chin and face width.
 *   3. Build smooth displacement fields per channel (jaw rotation, lid closure, brow raise,
 *      mouth corners, lips, cheeks), scaled to the detected face size.
 *   4. Convert each world-space displacement to the geometry space of its vertex (inverse of
 *      the local skinning Jacobian) and store it as a relative morph target.
 *
 * The morph targets use the canonical channel names (mouthOpen, eyeBlinkLeft, ...), so the
 * clip builder drives them like any authored blendshape.
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
  /** Track key of each mesh that received morph targets, with its new morph names. */
  meshes: Array<{ meshName: string; morphs: FaceChannel[] }>;
}

interface HeadVertex {
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

/**
 * Adds procedural blendshapes for `channels` to the meshes that follow `headNode`.
 * Returns null when the model has no usable face geometry.
 */
export function buildAutomaticFaceRig(
  root: THREE.Object3D,
  headNode: THREE.Object3D | null,
  channels: FaceChannel[],
  trackKeyOf: (obj: THREE.Object3D) => string,
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
  const keyOf = (x: number, y: number, z: number) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (const p of facePoints) {
    const k = keyOf(p.x, p.y, p.z);
    const list = grid.get(k);
    if (list) list.push(p);
    else grid.set(k, [p]);
  }
  /** Distance to the face surface (Infinity beyond maxSurfaceDist). */
  const faceSurfaceDistance = (p: THREE.Vector3): number => {
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
    return Math.sqrt(best);
  };
  // Lids close only what lies on them (lashes), not hair strands in front of the eyes.
  const channelReach = (ch: FaceChannel): number =>
    ch === 'eyeBlinkLeft' || ch === 'eyeBlinkRight' ? 0.006 * landmarks.scale : maxSurfaceDist;

  const result: AutoFaceRigResult = { channels: [], landmarks, meshes: [] };
  const used = new Set<FaceChannel>();
  const delta = new THREE.Vector3();
  const local = new THREE.Vector3();
  const base = new THREE.Vector3();
  const probe = new THREE.Vector3();
  const jac = new THREE.Matrix3();
  const jacInv = new THREE.Matrix3();
  const minDelta = 1e-5 * landmarks.scale;

  for (const [mesh, verts] of byMesh) {
    const geo = mesh.geometry as THREE.BufferGeometry;
    const pos = geo.attributes.position;
    const buffers = new Map<FaceChannel, Float32Array>();
    const isFace = mesh === faceMesh;
    for (const v of verts) {
      // Displacements in world space, scaled by how much the vertex follows the head and,
      // for meshes other than the face, by how close it is to the face surface.
      const dist = isFace ? 0 : faceSurfaceDistance(v.world);
      if (dist > maxSurfaceDist) continue;
      const worldDeltas: Array<[FaceChannel, THREE.Vector3]> = [];
      for (const ch of channels) {
        const reach = channelReach(ch);
        const follow = v.weight * (isFace ? 1 : smoothstep(reach, reach * 0.5, dist));
        if (follow <= 1e-3) continue;
        faceChannelDisplacement(ch, v.world, landmarks, delta);
        if (delta.lengthSq() * follow * follow > minDelta * minDelta) worldDeltas.push([ch, delta.clone().multiplyScalar(follow)]);
      }
      if (!worldDeltas.length) continue;
      // Local skinning Jacobian: geometry space -> world space at this vertex.
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
      jacInv.copy(jac).invert();
      for (const [ch, wd] of worldDeltas) {
        let buf = buffers.get(ch);
        if (!buf) {
          buf = new Float32Array(pos.count * 3);
          buffers.set(ch, buf);
        }
        wd.applyMatrix3(jacInv);
        buf[v.index * 3] = wd.x;
        buf[v.index * 3 + 1] = wd.y;
        buf[v.index * 3 + 2] = wd.z;
      }
    }
    if (!buffers.size) continue;

    // Append the morph targets without touching the authored ones.
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
        const n = new Float32Array(pos.count * 3);
        const baseNormal = geo.attributes.normal;
        if (!relative && baseNormal) for (let i = 0; i < n.length; i++) n[i] = (baseNormal.array as ArrayLike<number>)[i];
        geo.morphAttributes.normal.push(new THREE.Float32BufferAttribute(n, 3));
      }
      mesh.morphTargetDictionary[ch] = geo.morphAttributes.position.length - 1;
      mesh.morphTargetInfluences.push(0);
      added.push(ch);
      used.add(ch);
    }
    result.meshes.push({ meshName: trackKeyOf(mesh), morphs: added });
  }

  result.channels = FACE_CHANNELS.filter((c) => used.has(c));
  return result.channels.length ? result : null;
}
