import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import * as THREE from 'three';
import { buildAutomaticFaceRig, detectFaceLandmarks, faceChannelDisplacement, FACE_CHANNELS } from './autoFaceRig';
import { loadVrmOrGlbFromArrayBuffer } from './vrmAdapter';
import { buildSignAnimationClip } from '../utils/clipBuilder';
import type { SignAnimationPayload } from '../types/lis';

/**
 * Synthetic head skinned to a "Head" bone: an ellipsoid with a nose, upper/lower lips, a chin
 * and eye sockets on the front (+Z). Feature heights are known, so detection can be checked.
 */
const FACE = { noseY: 1.655, mouthY: 1.619, eyeY: 1.68, eyeX: 0.03 };

function syntheticHead(): { root: THREE.Group; head: THREE.Bone; mesh: THREE.SkinnedMesh } {
  const geo = new THREE.SphereGeometry(1, 160, 160);
  const pos = geo.attributes.position;
  const c = new THREE.Vector3(0, 1.65, 0);
  const g = (d: number, s: number) => Math.exp(-(d * d) / (s * s));
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) * 0.075;
    const y = pos.getY(i) * 0.105 + c.y;
    let z = pos.getZ(i) * 0.09;
    if (z > 0) {
      let bump = 0;
      bump += 0.02 * g(x, 0.008) * g(y - FACE.noseY, 0.012); // nose
      bump += 0.006 * g(x, 0.02) * g(y - (FACE.mouthY + 0.0055), 0.0035); // upper lip
      bump += 0.006 * g(x, 0.02) * g(y - (FACE.mouthY - 0.0055), 0.0035); // lower lip
      bump += 0.004 * g(x, 0.025) * g(y - 1.592, 0.006); // chin
      for (const sx of [-1, 1]) bump -= 0.006 * g(x - sx * FACE.eyeX, 0.01) * g(y - FACE.eyeY, 0.008); // eye sockets
      z += bump * Math.min(1, z / 0.03);
    }
    pos.setXYZ(i, x, y, z);
  }
  geo.computeVertexNormals();
  const n = pos.count;
  geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(new Uint16Array(n * 4).map((_, k) => (k % 4 === 0 ? 1 : 0)), 4));
  geo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(new Float32Array(n * 4).map((_, k) => (k % 4 === 0 ? 1 : 0)), 4));

  const root = new THREE.Group();
  const hips = new THREE.Bone();
  hips.name = 'Hips';
  hips.position.set(0, 1.0, 0);
  const head = new THREE.Bone();
  head.name = 'Head';
  head.position.set(0, 0.55, 0);
  hips.add(head);
  const mesh = new THREE.SkinnedMesh(geo, new THREE.MeshBasicMaterial());
  mesh.name = 'HeadMesh';
  mesh.add(hips);
  root.add(mesh);
  root.updateMatrixWorld(true);
  mesh.bind(new THREE.Skeleton([hips, head]));
  return { root, head, mesh };
}

function worldPoints(mesh: THREE.SkinnedMesh): THREE.Vector3[] {
  const pos = mesh.geometry.attributes.position;
  return Array.from({ length: pos.count }, (_, i) => {
    const v = new THREE.Vector3().fromBufferAttribute(pos, i);
    mesh.applyBoneTransform(i, v);
    return v.applyMatrix4(mesh.matrixWorld);
  });
}

/**
 * Displacement of vertex i when one channel is at weight 1, as the renderer computes it:
 * facial bone pose (from the rig's bone tracks) + residual morph target + skinning.
 */
function channelDelta(
  root: THREE.Object3D,
  mesh: THREE.SkinnedMesh,
  rig: { buildFacialBoneTracks: (m: Record<string, number[]>, t: number[]) => THREE.KeyframeTrack[] },
  channel: string,
  i: number,
): THREE.Vector3 {
  const rest = mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
  const tracks = rig.buildFacialBoneTracks({ [channel]: [0, 1] }, [0, 1]);
  const mixer = new THREE.AnimationMixer(root);
  const action = mixer.clipAction(new THREE.AnimationClip('t', 1, tracks));
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = true;
  action.play();
  mixer.setTime(1);
  const idx = mesh.morphTargetDictionary?.[channel];
  if (idx !== undefined) mesh.morphTargetInfluences![idx] = 1;
  root.updateMatrixWorld(true);
  const posed = mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
  if (idx !== undefined) mesh.morphTargetInfluences![idx] = 0;
  action.stop();
  mixer.uncacheRoot(root);
  root.traverse((o) => {
    if (o.name.startsWith('AutoFace_')) {
      const b = o as THREE.Bone & { userData: { bindPos?: THREE.Vector3; bindQuat?: THREE.Quaternion } };
      if (b.userData.bindPos) b.position.copy(b.userData.bindPos);
      if (b.userData.bindQuat) b.quaternion.copy(b.userData.bindQuat);
    }
  });
  root.updateMatrixWorld(true);
  return posed.sub(rest);
}

test('detectFaceLandmarks finds nose, mouth line and eyes on a synthetic head', () => {
  const { mesh } = syntheticHead();
  const L = detectFaceLandmarks(worldPoints(mesh));
  assert.ok(L, 'landmarks found');
  assert.ok(Math.abs(L.noseTip.y - FACE.noseY) < 0.004, `nose y ${L.noseTip.y}`);
  assert.ok(L.mouthDetected, 'mouth line detected from the lip profile');
  assert.ok(Math.abs(L.mouthY - FACE.mouthY) < 0.003, `mouth y ${L.mouthY}`);
  assert.ok(Math.abs(L.eyeY - FACE.eyeY) < 0.007, `eye y ${L.eyeY}`);
  assert.ok(Math.abs(L.eyeOffsetX - FACE.eyeX) < 0.01, `eye x ${L.eyeOffsetX}`);
  assert.ok(L.browY > L.eyeY && L.browY < L.eyeY + 0.04);
});

test('buildAutomaticFaceRig adds facial bones plus residual blendshapes', () => {
  const { root, head, mesh } = syntheticHead();
  const r = buildAutomaticFaceRig(root, head, [...FACE_CHANNELS], (o) => o.name);
  assert.ok(r, 'auto face rig built');
  assert.deepEqual(r.channels, [...FACE_CHANNELS]);
  // 11 facial bones under the head, part of the mesh skeleton.
  assert.equal(r.bones.length, 11);
  for (const b of r.bones) {
    assert.equal(b.parent, head);
    assert.ok(mesh.skeleton.bones.includes(b), `${b.name} in skeleton`);
    b.userData.bindPos = b.position.clone();
    b.userData.bindQuat = b.quaternion.clone();
  }
  assert.deepEqual(
    r.bones.map((b) => b.name).sort(),
    [
      'AutoFace_Brow_L', 'AutoFace_Brow_R', 'AutoFace_Cheek_L', 'AutoFace_Cheek_R', 'AutoFace_EyelidLower_L',
      'AutoFace_EyelidLower_R', 'AutoFace_EyelidUpper_L', 'AutoFace_EyelidUpper_R', 'AutoFace_Jaw',
      'AutoFace_MouthCorner_L', 'AutoFace_MouthCorner_R',
    ],
  );
  assert.equal(mesh.geometry.morphTargetsRelative, true);

  const pts = worldPoints(mesh);
  const nearest = (x: number, y: number) => {
    let best = -1;
    for (let i = 0; i < pts.length; i++) {
      if (pts[i].z < 0.03) continue;
      if (best < 0 || Math.hypot(pts[i].x - x, pts[i].y - y) < Math.hypot(pts[best].x - x, pts[best].y - y)) best = i;
    }
    return best;
  };
  // Rest shape unchanged by the new skin weights.
  const after = worldPoints(mesh);
  assert.ok(pts.every((p, i) => p.distanceTo(after[i]) < 1e-6));

  // Jaw: the chin goes down, the forehead does not move.
  const chin = nearest(0, 1.595);
  const forehead = nearest(0, 1.72);
  assert.ok(channelDelta(root, mesh, r, 'mouthOpen', chin).y < -0.004, 'chin moves down on mouthOpen');
  assert.ok(channelDelta(root, mesh, r, 'mouthOpen', forehead).length() < 1e-5, 'forehead stays');
  // Blink: only the eye on the same side (+X = character's left) closes.
  const lidL = nearest(FACE.eyeX, FACE.eyeY + 0.006);
  const lidR = nearest(-FACE.eyeX, FACE.eyeY + 0.006);
  assert.ok(channelDelta(root, mesh, r, 'eyeBlinkLeft', lidL).y < -0.002, 'left upper lid moves down');
  assert.ok(channelDelta(root, mesh, r, 'eyeBlinkLeft', lidR).length() < 1e-5, 'right lid untouched by left blink');
  // Brows up/down, smile corners up and out.
  const brow = nearest(FACE.eyeX, FACE.eyeY + 0.022);
  assert.ok(channelDelta(root, mesh, r, 'browUp', brow).y > 0.002);
  assert.ok(channelDelta(root, mesh, r, 'browDown', brow).y < -0.001);
  const cornerL = nearest(0.025, FACE.mouthY);
  const smile = channelDelta(root, mesh, r, 'mouthSmile', cornerL);
  assert.ok(smile.y > 0.001 && smile.x > 0.001, `smile ${smile.toArray()}`);
  // Bone + residual morph reproduce the target field of the jaw at the chin within 1 mm.
  const target = faceChannelDisplacement('mouthOpen', pts[chin], r.landmarks, new THREE.Vector3());
  assert.ok(channelDelta(root, mesh, r, 'mouthOpen', chin).distanceTo(target) < 0.001);
});

test('channels already covered by the model are not generated', () => {
  const { root, head } = syntheticHead();
  const r = buildAutomaticFaceRig(root, head, ['eyeBlinkLeft', 'eyeBlinkRight'], (o) => o.name);
  assert.deepEqual(r?.channels, ['eyeBlinkLeft', 'eyeBlinkRight']);
  assert.equal(buildAutomaticFaceRig(root, head, [], (o) => o.name), null);
});

const avatarGlb = process.env.LIS_AVATAR_GLB;
test('real Avaturn avatar.glb (no blendshapes, no face bones) gets a 9-channel auto face rig', {
  skip: !avatarGlb || !existsSync(avatarGlb),
}, async () => {
  const buf = readFileSync(avatarGlb as string);
  const loaded = await loadVrmOrGlbFromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  assert.ok(loaded.autoFaceRig, 'auto face rig built');
  assert.deepEqual(loaded.autoFaceRig.channels, [...FACE_CHANNELS]);
  const L = loaded.autoFaceRig.landmarks;
  assert.ok(L.mouthDetected && L.eyesDetected);
  assert.ok(L.mouthY < L.noseTip.y && L.eyeY > L.noseTip.y && L.browY > L.eyeY);
  assert.ok(loaded.rigTarget.faceMeshName, 'face mesh set');

  // The clip builder emits morph tracks for the generated channels.
  const payload: SignAnimationPayload = {
    classId: 't', phrase: 't', category: 't', description: '', prototypeClipId: 't', fps: 30, duration: 0.5,
    times: [0, 0.5], boneTracks: {}, morphTracks: { mouthOpen: [0, 1], eyeBlinkLeft: [0, 1], mouthSmile: [0, 0.5] },
    qaComparison: { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] },
  };
  assert.equal(loaded.autoFaceRig.bones.length, 11, 'facial bones added');
  assert.ok(loaded.autoFaceRig.bones.every((b) => b.parent?.name === 'Head'));
  const clip = buildSignAnimationClip(payload, loaded.rigTarget);
  const names = clip.tracks.map((t) => t.name);
  // Bones carry the main motion...
  for (const t of ['AutoFace_Jaw.quaternion', 'AutoFace_EyelidUpper_L.quaternion', 'AutoFace_MouthCorner_L.position']) {
    assert.ok(names.includes(t), `${t} (${names.join(', ')})`);
  }
  // ...and residual blendshapes the details.
  for (const m of ['mouthOpen', 'eyeBlinkLeft', 'mouthSmile']) {
    assert.ok(names.some((n) => n.endsWith(`.morphTargetInfluences[${m}]`)), `${m} track (${names.join(', ')})`);
  }
  loaded.dispose();
});
