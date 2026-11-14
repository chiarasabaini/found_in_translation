import * as THREE from 'three';
import type { AvatarRigTarget } from '../utils/clipBuilder';
import { applyMixedFaceRig, type FaceBoneDef, type HeadVertex } from './autoFaceRig';

export interface ProceduralAvatarHandle {
  root: THREE.Group;
  skeleton: THREE.Skeleton;
  skeletonHelper: THREE.SkeletonHelper;
  rigTarget: AvatarRigTarget;
  faceMesh: THREE.Mesh;
  dispose: () => void;
}

const MORPH_NAMES = [
  'mouthOpen',
  'mouthSmile',
  'mouthPucker',
  'mouthFrown',
  'browUp',
  'browDown',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'cheekPuff',
] as const;

type MorphName = (typeof MORPH_NAMES)[number];

function createBone(name: string, x: number, y: number, z: number, parent?: THREE.Bone): THREE.Bone {
  const b = new THREE.Bone();
  b.name = name;
  b.position.set(x, y, z);
  if (parent) parent.add(b);
  return b;
}

/**
 * Builds an articulated, skinned studio mannequin/avatar with a VRM-compatible
 * humanoid bone hierarchy (torso, head, arms, 30 finger phalanges) and a
 * vertex-morphed facial mesh for 9 LIS expression morph targets.
 */
export function createProceduralLisAvatar(): ProceduralAvatarHandle {
  const root = new THREE.Group();
  root.name = 'ProceduralLisAvatar';

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];

  // Calibrated studio materials (matte tunic vs warm satin skin vs subtle accents)
  const skinMat = new THREE.MeshStandardMaterial({
    color: 0xe6baa3,
    roughness: 0.62,
    metalness: 0.02,
  });
  const palmMat = new THREE.MeshStandardMaterial({
    color: 0xdfb097,
    roughness: 0.58,
    metalness: 0.02,
  });
  const tunicMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.78,
    metalness: 0.05,
  });
  const collarMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.55,
    metalness: 0.12,
  });
  const hairMat = new THREE.MeshStandardMaterial({
    color: 0x27272a,
    roughness: 0.72,
    metalness: 0.06,
  });
  const featureDarkMat = new THREE.MeshStandardMaterial({
    color: 0x18181b,
    roughness: 0.35,
    metalness: 0.08,
  });
  const lipMat = new THREE.MeshStandardMaterial({
    color: 0xb96862,
    roughness: 0.45,
    metalness: 0.02,
  });
  materials.push(skinMat, palmMat, tunicMat, collarMat, hairMat, featureDarkMat, lipMat);

  // Build hierarchical bone skeleton matching VRM/retargeting.py names
  const hips = createBone('hips', 0, 0.96, 0);
  const spine = createBone('spine', 0, 0.14, 0, hips);
  const chest = createBone('chest', 0, 0.18, 0, spine);
  const neck = createBone('neck', 0, 0.22, 0, chest);
  const head = createBone('head', 0, 0.10, 0, neck);

  const allBones: THREE.Bone[] = [hips, spine, chest, neck, head];
  const boneNodeNames: Record<string, string> = {
    hips: 'hips',
    spine: 'spine',
    chest: 'chest',
    neck: 'neck',
    head: 'head',
  };

  // Helper to attach rigid/articulated visual meshes directly to bones while keeping a real Skeleton
  const addPart = (
    parent: THREE.Object3D,
    geo: THREE.BufferGeometry,
    mat: THREE.Material,
    pos: [number, number, number],
    rot?: [number, number, number],
  ) => {
    geometries.push(geo);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    if (rot) mesh.rotation.set(rot[0], rot[1], rot[2]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  };

  // Torso composite parts (pelvis/waist, chest tunic, collar) + grounded legs
  addPart(hips, new THREE.CylinderGeometry(0.145, 0.13, 0.22, 20), tunicMat, [0, 0.05, 0]);
  addPart(hips, new THREE.CylinderGeometry(0.068, 0.052, 0.48, 16), collarMat, [-0.082, -0.28, 0]);
  addPart(hips, new THREE.CylinderGeometry(0.068, 0.052, 0.48, 16), collarMat, [0.082, -0.28, 0]);
  addPart(hips, new THREE.CylinderGeometry(0.052, 0.042, 0.44, 16), collarMat, [-0.082, -0.72, 0]);
  addPart(hips, new THREE.CylinderGeometry(0.052, 0.042, 0.44, 16), collarMat, [0.082, -0.72, 0]);
  addPart(hips, new THREE.BoxGeometry(0.085, 0.06, 0.16), featureDarkMat, [-0.082, -0.93, 0.03]);
  addPart(hips, new THREE.BoxGeometry(0.085, 0.06, 0.16), featureDarkMat, [0.082, -0.93, 0.03]);
  addPart(spine, new THREE.CylinderGeometry(0.155, 0.14, 0.22, 20), tunicMat, [0, 0.09, 0]);
  addPart(chest, new THREE.CylinderGeometry(0.18, 0.155, 0.23, 24), tunicMat, [0, 0.095, 0]);
  // Sculpted shoulder/trapezius cap so the chest doesn't end in a flat cylinder top
  const shoulderCapGeo = new THREE.SphereGeometry(0.185, 24, 16);
  shoulderCapGeo.scale(1.04, 0.36, 0.76);
  addPart(chest, shoulderCapGeo, tunicMat, [0, 0.205, 0]);
  addPart(chest, new THREE.TorusGeometry(0.072, 0.016, 12, 28), collarMat, [0, 0.245, 0.01], [Math.PI / 2, 0, 0]);
  addPart(neck, new THREE.CylinderGeometry(0.048, 0.055, 0.13, 16), skinMat, [0, 0.055, 0]);

  // Sculpted face mesh, skinned to the head; its facial rig is built once the skeleton exists.
  const { mesh: faceMesh, roles: faceRoles, materials: faceMaterials } = createMorphableHeadMesh(skinMat, geometries);
  materials.push(...faceMaterials);
  head.add(faceMesh);

  // Hair cap tilted back cleanly above the forehead/eyebrows + ears & eyes
  addPart(head, new THREE.SphereGeometry(0.112, 24, 18, 0, Math.PI * 2, 0, Math.PI * 0.50), hairMat, [0, 0.125, -0.022], [-0.46, 0, 0]);
  addPart(head, new THREE.SphereGeometry(0.022, 12, 10), skinMat, [-0.102, 0.085, -0.005]);
  addPart(head, new THREE.SphereGeometry(0.022, 12, 10), skinMat, [0.102, 0.085, -0.005]);
  // Eyes (sclera + dark iris)
  const scleraMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.25 });
  materials.push(scleraMat);
  for (const sx of [-0.036, 0.036]) {
    addPart(head, new THREE.SphereGeometry(0.015, 14, 12), scleraMat, [sx, 0.102, 0.092]);
    addPart(head, new THREE.SphereGeometry(0.0085, 12, 10), featureDarkMat, [sx, 0.102, 0.102]);
  }

  // Arms & Articulated Hands
  const sides = ['right', 'left'] as const;
  const fingerSpecs = [
    { name: 'thumb', offset: [-0.042, -0.030, 0.018], lengths: [0.028, 0.024, 0.020], radius: 0.0095, rotZ: -0.45 },
    { name: 'index', offset: [-0.030, -0.085, 0.004], lengths: [0.036, 0.026, 0.020], radius: 0.0082, rotZ: -0.06 },
    { name: 'middle', offset: [-0.010, -0.088, 0.005], lengths: [0.040, 0.028, 0.022], radius: 0.0085, rotZ: 0.0 },
    { name: 'ring', offset: [0.011, -0.086, 0.004], lengths: [0.036, 0.026, 0.020], radius: 0.0080, rotZ: 0.05 },
    { name: 'little', offset: [0.031, -0.080, 0.002], lengths: [0.028, 0.021, 0.017], radius: 0.0072, rotZ: 0.11 },
  ] as const;

  for (const side of sides) {
    const sign = side === 'right' ? -1 : 1;
    const shoulder = createBone(`${side}Shoulder`, sign * 0.07, 0.14, 0, chest);
    const upperArm = createBone(`${side}UpperArm`, sign * 0.125, 0, 0, shoulder);
    const lowerArm = createBone(`${side}LowerArm`, 0, -0.28, 0, upperArm);
    const hand = createBone(`${side}Hand`, 0, -0.25, 0, lowerArm);

    allBones.push(shoulder, upperArm, lowerArm, hand);
    boneNodeNames[`${side}Shoulder`] = shoulder.name;
    boneNodeNames[`${side}UpperArm`] = upperArm.name;
    boneNodeNames[`${side}LowerArm`] = lowerArm.name;
    boneNodeNames[`${side}Hand`] = hand.name;

    // Shoulder cap + sleeve + cuff + forearm + palm
    addPart(upperArm, new THREE.SphereGeometry(0.052, 14, 12), tunicMat, [0, 0, 0]);
    addPart(upperArm, new THREE.CylinderGeometry(0.045, 0.038, 0.25, 16), tunicMat, [0, -0.135, 0]);
    addPart(upperArm, new THREE.CylinderGeometry(0.041, 0.039, 0.028, 16), collarMat, [0, -0.25, 0]);
    addPart(lowerArm, new THREE.SphereGeometry(0.035, 12, 10), skinMat, [0, 0, 0]);
    addPart(lowerArm, new THREE.CylinderGeometry(0.034, 0.026, 0.23, 16), skinMat, [0, -0.125, 0]);

    // Palm box with rounded edges (wrist joint + metacarpal block)
    addPart(hand, new THREE.SphereGeometry(0.025, 12, 10), palmMat, [0, 0, 0]);
    addPart(hand, new THREE.BoxGeometry(0.076, 0.082, 0.024), palmMat, [0, -0.045, 0]);

    // Fingers: 3 phalanges per finger
    for (const f of fingerSpecs) {
      const ox = side === 'right' ? f.offset[0] : -f.offset[0];
      const proxName = `${side}${capitalize(f.name)}Proximal`;
      const interName = `${side}${capitalize(f.name)}Intermediate`;
      const distName = `${side}${capitalize(f.name)}Distal`;

      const prox = createBone(proxName, ox, f.offset[1], f.offset[2], hand);
      if (f.name === 'thumb') {
        prox.rotation.z = side === 'right' ? -0.45 : 0.45;
      }
      const inter = createBone(interName, 0, -f.lengths[0], 0, prox);
      const dist = createBone(distName, 0, -f.lengths[1], 0, inter);

      allBones.push(prox, inter, dist);
      boneNodeNames[proxName] = proxName;
      boneNodeNames[interName] = interName;
      boneNodeNames[distName] = distName;

      // Visual phalanx capsules/cylinders + knuckle spheres
      const [l0, l1, l2] = f.lengths;
      addPart(prox, new THREE.SphereGeometry(f.radius * 1.05, 10, 8), palmMat, [0, 0, 0]);
      addPart(prox, new THREE.CylinderGeometry(f.radius, f.radius * 0.92, l0, 10), skinMat, [0, -l0 * 0.5, 0]);

      addPart(inter, new THREE.SphereGeometry(f.radius * 0.94, 10, 8), palmMat, [0, 0, 0]);
      addPart(inter, new THREE.CylinderGeometry(f.radius * 0.9, f.radius * 0.82, l1, 10), skinMat, [0, -l1 * 0.5, 0]);

      addPart(dist, new THREE.SphereGeometry(f.radius * 0.84, 10, 8), palmMat, [0, 0, 0]);
      addPart(dist, new THREE.CylinderGeometry(f.radius * 0.80, f.radius * 0.65, l2, 10), skinMat, [0, -l2 * 0.5, 0]);
      addPart(dist, new THREE.SphereGeometry(f.radius * 0.66, 10, 8), skinMat, [0, -l2, 0]);
    }
  }

  // Bind a lightweight dummy skinned anchor mesh to give THREE.Skeleton & SkeletonHelper full bone topology
  const dummyGeo = new THREE.BoxGeometry(0.01, 0.01, 0.01);
  const posCount = dummyGeo.attributes.position.count;
  const skinIndices = new Uint16Array(posCount * 4);
  const skinWeights = new Float32Array(posCount * 4);
  for (let i = 0; i < posCount; i++) {
    skinWeights[i * 4] = 1;
  }
  dummyGeo.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
  dummyGeo.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
  geometries.push(dummyGeo);

  const skinnedAnchor = new THREE.SkinnedMesh(dummyGeo, tunicMat);
  skinnedAnchor.name = 'AvatarSkeletonAnchor';
  skinnedAnchor.frustumCulled = false;
  skinnedAnchor.add(hips);
  const skeleton = new THREE.Skeleton(allBones);
  skinnedAnchor.bind(skeleton);
  root.add(skinnedAnchor);

  // Mixed facial rig, same engine as imported avatars: facial bones for the main motion,
  // residual blendshapes for the rest. The hand-made deformations below are the target.
  root.updateMatrixWorld(true);
  faceMesh.bind(new THREE.Skeleton([head]));
  const facePos = faceMesh.geometry.attributes.position;
  const headQuat = head.getWorldQuaternion(new THREE.Quaternion());
  const faceVerts: HeadVertex[] = Array.from({ length: facePos.count }, (_, i) => ({
    mesh: faceMesh,
    index: i,
    weight: 1,
    world: new THREE.Vector3().fromBufferAttribute(facePos, i).applyMatrix4(faceMesh.matrixWorld),
  }));
  const faceRig = applyMixedFaceRig({
    root,
    headNode: head,
    channels: MORPH_NAMES,
    boneDefs: proceduralFaceBoneDefs(head),
    meshes: [{ mesh: faceMesh, verts: faceVerts, follow: () => 1 }],
    target: (ch, v, out) => {
      const [dx, dy, dz] = computeMorphDelta(
        ch,
        faceRoles[v.index],
        facePos.getX(v.index),
        facePos.getY(v.index),
        facePos.getZ(v.index),
      );
      return out.set(dx, dy, dz).applyQuaternion(headQuat);
    },
    trackKeyOf: (o) => o.name,
  });

  const skeletonHelper = new THREE.SkeletonHelper(hips);
  skeletonHelper.visible = false;
  root.add(skeletonHelper);

  const rigTarget: AvatarRigTarget = {
    boneNodeNames,
    faceMeshName: faceMesh.name,
    availableMorphs: new Set<string>(faceRig?.meshes.flatMap((m) => m.morphs) ?? []),
    buildFacialBoneTracks: faceRig?.buildFacialBoneTracks,
  };

  return {
    root,
    skeleton,
    skeletonHelper,
    rigTarget,
    faceMesh,
    dispose: () => {
      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
    },
  };
}

/**
 * Facial bones of the procedural head, placed on its features (head-bone local coordinates of
 * the face mesh). Translations match the hand-made deformations, so the bones carry most of them.
 */
function proceduralFaceBoneDefs(head: THREE.Bone): FaceBoneDef[] {
  const w = (x: number, y: number, z: number) => head.localToWorld(new THREE.Vector3(x, y, z));
  const headQuat = head.getWorldQuaternion(new THREE.Quaternion());
  const d = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z).applyQuaternion(headQuat);
  const X = d(1, 0, 0);
  const defs: FaceBoneDef[] = [
    {
      name: 'AutoFace_Jaw',
      pivot: w(0, 0.06, -0.01),
      motions: { mouthOpen: { kind: 'rotate', axis: X, angle: 0.22 } },
      primary: 'mouthOpen',
    },
  ];
  for (const [side, suffix, blink] of [
    [1, 'L', 'eyeBlinkLeft'],
    [-1, 'R', 'eyeBlinkRight'],
  ] as const) {
    // Bone suffixes follow the avatar's side (+X = its left), like the GLB auto rig.
    const x = side * 0.036;
    defs.push(
      {
        name: `AutoFace_EyelidUpper_${suffix}`,
        pivot: w(x, 0.119, 0.096),
        motions: { [blink]: { kind: 'translate', delta: d(0, -0.024, 0.004) } },
        primary: blink,
      },
      {
        name: `AutoFace_Brow_${suffix}`,
        pivot: w(x, 0.132, 0.095),
        motions: {
          browUp: { kind: 'translate', delta: d(0, 0.014, 0.002) },
          browDown: { kind: 'translate', delta: d(-side * 0.004, -0.012, 0.002) },
        },
        primary: 'browUp',
      },
      {
        name: `AutoFace_MouthCorner_${suffix}`,
        pivot: w(side * 0.022, 0.033, 0.096),
        motions: {
          mouthSmile: { kind: 'translate', delta: d(side * 0.0035, 0.011, -0.003) },
          mouthFrown: { kind: 'translate', delta: d(-side * 0.0018, -0.011, 0) },
          mouthPucker: { kind: 'translate', delta: d(-side * 0.0077, 0, 0.014) },
        },
        primary: 'mouthSmile',
      },
      {
        name: `AutoFace_Cheek_${suffix}`,
        pivot: w(side * 0.05, 0.055, 0.07),
        motions: { cheekPuff: { kind: 'translate', delta: d(side * 0.016, 0, 0.01) } },
        primary: 'cheekPuff',
      },
    );
  }
  return defs;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

type FaceRole = 'head' | 'browL' | 'browR' | 'lidL' | 'lidR' | 'upperLip' | 'lowerLip' | 'mouthCav';

/**
 * Creates a sculpted head + facial landmarks mesh (head, brows, lids, lips, mouth cavity) as one
 * vertex-colored BufferGeometry, skinned to the head. `roles` tells which part each vertex
 * belongs to; the facial rig uses it to fit bones and residual blendshapes.
 */
function createMorphableHeadMesh(
  _baseSkinMat: THREE.Material,
  geometries: THREE.BufferGeometry[],
): { mesh: THREE.SkinnedMesh; roles: FaceRole[]; materials: THREE.Material[] } {
  const parts: Array<{
    geo: THREE.BufferGeometry;
    role: 'head' | 'browL' | 'browR' | 'lidL' | 'lidR' | 'upperLip' | 'lowerLip' | 'mouthCav';
    color: number;
  }> = [];

  const headGeo = new THREE.SphereGeometry(0.105, 28, 22);
  headGeo.scale(0.92, 1.12, 0.96);
  headGeo.translate(0, 0.088, 0);
  parts.push({ geo: headGeo, role: 'head', color: 0xe6baa3 });

  const browL = new THREE.BoxGeometry(0.034, 0.008, 0.012, 4, 1, 1);
  browL.translate(-0.036, 0.132, 0.095);
  parts.push({ geo: browL, role: 'browL', color: 0x27272a });

  const browR = new THREE.BoxGeometry(0.034, 0.008, 0.012, 4, 1, 1);
  browR.translate(0.036, 0.132, 0.095);
  parts.push({ geo: browR, role: 'browR', color: 0x27272a });

  const lidL = new THREE.BoxGeometry(0.030, 0.004, 0.014, 2, 1, 1);
  lidL.translate(-0.036, 0.117, 0.096);
  parts.push({ geo: lidL, role: 'lidL', color: 0xd4a188 });

  const lidR = new THREE.BoxGeometry(0.030, 0.004, 0.014, 2, 1, 1);
  lidR.translate(0.036, 0.117, 0.096);
  parts.push({ geo: lidR, role: 'lidR', color: 0xd4a188 });

  const upperLip = new THREE.BoxGeometry(0.044, 0.008, 0.014, 6, 1, 1);
  upperLip.translate(0, 0.038, 0.096);
  parts.push({ geo: upperLip, role: 'upperLip', color: 0xb96862 });

  const lowerLip = new THREE.BoxGeometry(0.042, 0.009, 0.014, 6, 1, 1);
  lowerLip.translate(0, 0.028, 0.095);
  parts.push({ geo: lowerLip, role: 'lowerLip', color: 0xb05a54 });

  const mouthCav = new THREE.BoxGeometry(0.032, 0.006, 0.008, 2, 1, 1);
  mouthCav.translate(0, 0.033, 0.090);
  parts.push({ geo: mouthCav, role: 'mouthCav', color: 0x3b1518 });

  // Merge non-indexed geometries manually so we know each vertex's semantic role
  const nonIndexed = parts.map((p) => ({
    ...p,
    geo: p.geo.index ? p.geo.toNonIndexed() : p.geo,
  }));

  let totalVerts = 0;
  for (const p of nonIndexed) {
    totalVerts += p.geo.attributes.position.count;
  }

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const colors = new Float32Array(totalVerts * 3);
  const roles: Array<(typeof parts)[number]['role']> = [];

  let offset = 0;
  const tmpColor = new THREE.Color();
  for (const p of nonIndexed) {
    const posAttr = p.geo.attributes.position;
    const normAttr = p.geo.attributes.normal;
    tmpColor.setHex(p.color);
    for (let i = 0; i < posAttr.count; i++) {
      const idx3 = (offset + i) * 3;
      positions[idx3] = posAttr.getX(i);
      positions[idx3 + 1] = posAttr.getY(i);
      positions[idx3 + 2] = posAttr.getZ(i);
      normals[idx3] = normAttr.getX(i);
      normals[idx3 + 1] = normAttr.getY(i);
      normals[idx3 + 2] = normAttr.getZ(i);
      colors[idx3] = tmpColor.r;
      colors[idx3 + 1] = tmpColor.g;
      colors[idx3 + 2] = tmpColor.b;
      roles.push(p.role);
    }
    offset += posAttr.count;
    p.geo.dispose();
  }

  const mergedGeo = new THREE.BufferGeometry();
  mergedGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  mergedGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  mergedGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Skinned to the head (index 0 of the face skeleton); the facial rig adds its bones later.
  mergedGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(new Uint16Array(totalVerts * 4), 4));
  const weights = new Float32Array(totalVerts * 4);
  for (let v = 0; v < totalVerts; v++) weights[v * 4] = 1;
  mergedGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
  geometries.push(mergedGeo);

  const faceMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.58,
    metalness: 0.02,
  });

  const mesh = new THREE.SkinnedMesh(mergedGeo, faceMat);
  mesh.name = 'AvatarFaceMesh';
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  // Skinned vertices move outside the static bounds when the jaw opens.
  mesh.frustumCulled = false;
  return { mesh, roles, materials: [faceMat] };
}

function computeMorphDelta(
  morph: MorphName,
  role: string,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  if (morph === 'mouthOpen') {
    if (role === 'lowerLip') return [0, -0.022, -0.002];
    if (role === 'mouthCav') return [0, y < 0.033 ? -0.018 : 0.003, 0.002];
    if (role === 'head' && y < 0.03 && z > 0.05) return [0, -0.012, 0];
  } else if (morph === 'mouthSmile') {
    const corner = Math.abs(x) / 0.022;
    if (role === 'upperLip' || role === 'lowerLip') {
      return [x * 0.16, 0.011 * corner * corner, -0.003 * corner];
    }
  } else if (morph === 'mouthFrown') {
    const corner = Math.abs(x) / 0.022;
    if (role === 'upperLip' || role === 'lowerLip') {
      return [-x * 0.08, -0.011 * corner * corner, 0];
    }
  } else if (morph === 'mouthPucker') {
    if (role === 'upperLip' || role === 'lowerLip') {
      return [-x * 0.35, 0, 0.014];
    }
  } else if (morph === 'browUp') {
    if (role === 'browL' || role === 'browR') {
      return [0, 0.014, 0.002];
    }
  } else if (morph === 'browDown') {
    if (role === 'browL' || role === 'browR') {
      const inner = Math.max(0, 1 - Math.abs(Math.abs(x) - 0.02) / 0.03);
      return [x > 0 ? -0.004 : 0.004, -0.010 - 0.004 * inner, 0.002];
    }
  } else if (morph === 'eyeBlinkLeft' && role === 'lidR') {
    // Part names follow the viewer's side: lidR sits at +X, the avatar's LEFT eye (ARKit eyeBlinkLeft).
    if (y < 0.117) return [0, -0.024, 0.004];
  } else if (morph === 'eyeBlinkRight' && role === 'lidL') {
    if (y < 0.117) return [0, -0.024, 0.004];
  } else if (morph === 'cheekPuff' && role === 'head') {
    if (y > 0.02 && y < 0.09 && z > 0.03 && Math.abs(x) > 0.035) {
      return [Math.sign(x) * 0.016, 0, 0.010];
    }
  }
  return [0, 0, 0];
}
