import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as THREE from 'three';
import { buildSignAnimationClip } from './clipBuilder';
import {
  analyzeUniversalAvatarRig,
  loadVrmOrGlbFromArrayBuffer,
  matchGlbBoneNames,
  normalizeAvatarRootTransform,
} from '../three/vrmAdapter';
import { parseRokokoRecording } from '../recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, retargetRecording } from '../recording/recordingRetarget';
import type { SignAnimationPayload } from '../types/lis';

function toArrayBuffer(buf: Buffer): ArrayBuffer {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function makeSimplePayload(
  boneOverrides: Record<string, THREE.Quaternion>,
  morphOverrides: Record<string, number> = {},
  isRecording = true,
): SignAnimationPayload {
  const boneTracks: Record<string, number[]> = {};
  for (const [canon, q] of Object.entries(boneOverrides)) {
    boneTracks[canon] = [0, 0, 0, 1, q.x, q.y, q.z, q.w];
  }
  const morphTracks: Record<string, number[]> = {};
  for (const [m, v] of Object.entries(morphOverrides)) {
    morphTracks[m] = [0, v];
  }
  return {
    classId: 'test',
    phrase: 'Test',
    category: 'Test',
    description: 'Test',
    prototypeClipId: 'test_0',
    fps: 30,
    duration: 0.5,
    times: [0, 0.5],
    boneTracks,
    morphTracks,
    qaComparison: {
      group1VsGroup2Right: [],
      group1VsGroup2Left: [],
      frames: [],
    },
    ...(isRecording
      ? {
          source: {
            kind: 'recording' as const,
            recordingName: 'unit_test',
            startSec: 0,
            endSec: 0.5,
            gloveYawDeg: 0,
            leftYawOffsetDeg: 0,
            rightYawOffsetDeg: 0,
            sourceFrameIntervalMs: 33.3,
            streams: { leftHand: true, rightHand: true, face: true, body: true },
          },
        }
      : {}),
  };
}

test('matchGlbBoneNames maps Ready Player Me and Mixamo bone names', () => {
  const rpmBones = [
    'Hips',
    'Spine',
    'Spine2',
    'Neck',
    'Head',
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'RightHandIndex1',
    'RightHandIndex2',
    'RightHandIndex3',
    'mixamorig:LeftArm',
    'mixamorig:LeftForeArm',
  ];
  const mapped = matchGlbBoneNames(rpmBones);
  assert.equal(mapped.rightUpperArm, 'RightArm');
  assert.equal(mapped.rightLowerArm, 'RightForeArm');
  assert.equal(mapped.rightIndexProximal, 'RightHandIndex1');
  assert.equal(mapped.leftUpperArm, 'mixamorig:LeftArm');
  assert.equal(mapped.leftLowerArm, 'mixamorig:LeftForeArm');
});

test('matchGlbBoneNames maps Blender Rigify DEF-*, dot-stripped GLTFLoader names, Meta-Rig, Unreal, and Biped', () => {
  // 1. Blender Rigify with Three.js GLTFLoader dot-stripped node names
  const rigifySanitized = [
    'DEF-spine',
    'DEF-spine001',
    'DEF-spine002',
    'DEF-spine003',
    'DEF-spine004',
    'DEF-spine005',
    'DEF-spine006',
    'DEF-shoulderL',
    'DEF-upper_armL',
    'DEF-upper_armL001',
    'DEF-forearmL',
    'DEF-forearmL001',
    'DEF-handL',
    'DEF-f_index01L',
    'DEF-f_index02L',
    'DEF-f_index03L',
    'DEF-thumb01L',
    'DEF-thumb02L',
    'DEF-thumb03L',
    'DEF-shoulderR',
    'DEF-upper_armR',
    'DEF-forearmR',
    'DEF-handR',
  ];
  const rigifyMap = matchGlbBoneNames(rigifySanitized);
  assert.equal(rigifyMap.hips, 'DEF-spine');
  assert.equal(rigifyMap.spine, 'DEF-spine002');
  assert.equal(rigifyMap.chest, 'DEF-spine003');
  assert.equal(rigifyMap.neck, 'DEF-spine004');
  assert.equal(rigifyMap.head, 'DEF-spine006');
  assert.equal(rigifyMap.leftShoulder, 'DEF-shoulderL');
  assert.equal(rigifyMap.leftUpperArm, 'DEF-upper_armL');
  assert.equal(rigifyMap.leftLowerArm, 'DEF-forearmL');
  assert.equal(rigifyMap.leftHand, 'DEF-handL');
  assert.equal(rigifyMap.leftIndexProximal, 'DEF-f_index01L');
  assert.equal(rigifyMap.leftIndexIntermediate, 'DEF-f_index02L');
  assert.equal(rigifyMap.leftIndexDistal, 'DEF-f_index03L');
  assert.equal(rigifyMap.leftThumbProximal, 'DEF-thumb01L');

  // 2. Blender Rigify Meta-Rig (without DEF- prefix)
  const metaRig = [
    'spine',
    'spine.001',
    'spine.002',
    'spine.003',
    'spine.004',
    'spine.005',
    'spine.006',
    'shoulder.L',
    'upper_arm.L',
    'forearm.L',
    'hand.L',
    'f_middle.01.L',
    'f_middle.02.L',
    'f_middle.03.L',
  ];
  const metaMap = matchGlbBoneNames(metaRig);
  assert.equal(metaMap.hips, 'spine');
  assert.equal(metaMap.chest, 'spine.003');
  assert.equal(metaMap.head, 'spine.006');
  assert.equal(metaMap.leftUpperArm, 'upper_arm.L');
  assert.equal(metaMap.leftMiddleIntermediate, 'f_middle.02.L');

  // 3. Unreal Mannequin
  const unreal = [
    'pelvis',
    'spine_01',
    'spine_03',
    'neck_01',
    'head',
    'clavicle_l',
    'upperarm_l',
    'lowerarm_l',
    'hand_l',
    'index_01_l',
    'index_02_l',
    'index_03_l',
  ];
  const unrealMap = matchGlbBoneNames(unreal);
  assert.equal(unrealMap.hips, 'pelvis');
  assert.equal(unrealMap.leftShoulder, 'clavicle_l');
  assert.equal(unrealMap.leftUpperArm, 'upperarm_l');
  assert.equal(unrealMap.leftLowerArm, 'lowerarm_l');
  assert.equal(unrealMap.leftIndexProximal, 'index_01_l');

  // 4. 3ds Max Biped (Finger0 = Thumb, Finger1 = Index)
  const biped = [
    'Bip01 Pelvis',
    'Bip01 Spine',
    'Bip01 Spine2',
    'Bip01 Neck',
    'Bip01 Head',
    'Bip01 L Clavicle',
    'Bip01 L UpperArm',
    'Bip01 L Forearm',
    'Bip01 L Hand',
    'Bip01 L Finger0',
    'Bip01 L Finger01',
    'Bip01 L Finger02',
    'Bip01 L Finger1',
    'Bip01 L Finger11',
    'Bip01 L Finger12',
  ];
  const bipedMap = matchGlbBoneNames(biped);
  assert.equal(bipedMap.leftThumbProximal, 'Bip01 L Finger0');
  assert.equal(bipedMap.leftThumbIntermediate, 'Bip01 L Finger01');
  assert.equal(bipedMap.leftIndexProximal, 'Bip01 L Finger1');
  assert.equal(bipedMap.leftIndexDistal, 'Bip01 L Finger12');
});

test('buildSignAnimationClip maps ARKit morph aliases across primary and secondary meshes', () => {
  const payload = makeSimplePayload(
    { rightUpperArm: new THREE.Quaternion() },
    { mouthOpen: 0.8, mouthSmile: 0.6, browUp: 0.4 },
  );

  const clip = buildSignAnimationClip(payload, {
    boneNodeNames: { rightUpperArm: 'RightArm' },
    faceMeshName: 'Wolf3D_Head',
    availableMorphs: new Set(['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp']),
    morphTargetMeshes: [
      {
        meshName: 'Wolf3D_Head',
        availableMorphs: new Set(['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp']),
      },
      {
        meshName: 'Wolf3D_Teeth',
        availableMorphs: new Set(['jawOpen']),
      },
    ],
  });

  const trackNames = clip.tracks.map((t) => t.name);
  assert.ok(trackNames.includes('RightArm.quaternion'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[jawOpen]'));
  assert.ok(trackNames.includes('Wolf3D_Teeth.morphTargetInfluences[jawOpen]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[mouthSmileLeft]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[mouthSmileRight]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[browInnerUp]'));
});

test('universal topological retargeting adapts 1-bone torso, 5-bone Meta-Rig spine, and unnamed custom skeletons', () => {
  // Case A: Minimal 1-bone torso (Hips -> Spine -> Head) receives full torso rotation without losing chest
  {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = 'Hips';
    hips.position.set(0, 1.0, 0);
    const spine = new THREE.Bone();
    spine.name = 'Spine';
    spine.position.set(0, 0.35, 0);
    const head = new THREE.Bone();
    head.name = 'Head';
    head.position.set(0, 0.25, 0);
    const lArm = new THREE.Bone();
    lArm.name = 'LeftArm';
    lArm.position.set(0.18, 0.2, 0);
    const lFore = new THREE.Bone();
    lFore.name = 'LeftForeArm';
    lFore.position.set(0.26, 0, 0);
    const lHand = new THREE.Bone();
    lHand.name = 'LeftHand';
    lHand.position.set(0.24, 0, 0);
    const rArm = new THREE.Bone();
    rArm.name = 'RightArm';
    rArm.position.set(-0.18, 0.2, 0);
    const rFore = new THREE.Bone();
    rFore.name = 'RightForeArm';
    rFore.position.set(-0.26, 0, 0);
    const rHand = new THREE.Bone();
    rHand.name = 'RightHand';
    rHand.position.set(-0.24, 0, 0);

    root.add(hips);
    hips.add(spine);
    spine.add(head, lArm, rArm);
    lArm.add(lFore);
    lFore.add(lHand);
    rArm.add(rFore);
    rFore.add(rHand);
    root.updateMatrixWorld(true);

    const analysis = analyzeUniversalAvatarRig(root, null);
    const halfTilt = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.15);
    const payload = makeSimplePayload({ spine: halfTilt, chest: halfTilt });
    const clip = buildSignAnimationClip(payload, analysis.rigTarget);

    const mixer = new THREE.AnimationMixer(root);
    const act = mixer.clipAction(clip);
    act.setLoop(THREE.LoopOnce, 1);
    act.clampWhenFinished = true;
    act.play();
    mixer.setTime(0.5);
    root.updateMatrixWorld(true);

    // Total torso tilt should equal halfTilt * halfTilt = 0.30 rad around +X
    const spineWorldQ = new THREE.Quaternion();
    spine.getWorldQuaternion(spineWorldQ);
    const expectedQ = halfTilt.clone().multiply(halfTilt);
    assert.ok(spineWorldQ.angleTo(expectedQ) < 0.02, '1-bone spine absorbs full spine*chest rotation');
  }

  // Case B: Completely unnamed custom skeleton (Joint_00..Joint_N) matched purely by graph topology
  {
    const root = new THREE.Group();
    let id = 0;
    const mkBone = (x: number, y: number, z: number, parent: THREE.Object3D): THREE.Bone => {
      const b = new THREE.Bone();
      b.name = `CustomJoint_${id++}`;
      b.position.set(x, y, z);
      parent.add(b);
      return b;
    };
    const pelvis = mkBone(0, 1.0, 0, root);
    const sp1 = mkBone(0, 0.15, 0, pelvis);
    const sp2 = mkBone(0, 0.15, 0, sp1);
    const sp3 = mkBone(0, 0.15, 0, sp2); // chest hub
    const neck = mkBone(0, 0.12, 0, sp3);
    const head = mkBone(0, 0.12, 0, neck);
    mkBone(0, 0.1, 0, head); // head tip

    for (const s of [1, -1]) {
      const clav = mkBone(s * 0.06, 0.1, 0, sp3);
      const upper = mkBone(s * 0.12, 0, 0, clav);
      const lower = mkBone(s * 0.26, 0, 0, upper);
      const hand = mkBone(s * 0.24, 0, 0, lower);
      // 5 fingers arranged from radial/anterior (thumb closer to wrist) to ulnar (little)
      const zOffsets = [0.04, 0.02, 0.0, -0.02, -0.04];
      zOffsets.forEach((dz, fIdx) => {
        const dx = fIdx === 0 ? 0.03 : 0.08;
        const f1 = mkBone(s * dx, 0, dz, hand);
        const f2 = mkBone(s * 0.03, 0, 0, f1);
        const f3 = mkBone(s * 0.025, 0, 0, f2);
        mkBone(s * 0.02, 0, 0, f3);
      });
      // Legs so pelvis is a real lower-body hub too
      const thigh = mkBone(s * 0.1, -0.05, 0, pelvis);
      const calf = mkBone(0, -0.42, 0, thigh);
      mkBone(0, -0.42, 0, calf);
    }
    root.updateMatrixWorld(true);

    const analysis = analyzeUniversalAvatarRig(root, null);
    assert.equal(analysis.detectedFormat, 'Topological Humanoid GLB');
    assert.ok(analysis.matchedBonesCount >= 40, `expected >= 40 topological matches, got ${analysis.matchedBonesCount}`);
    assert.equal(analysis.rigTarget.boneNodeNames.hips, pelvis.name);
    assert.equal(analysis.rigTarget.boneNodeNames.chest, sp3.name);
    assert.equal(analysis.rigTarget.boneNodeNames.head, head.name);
  }

  // Case C: -Z facing model is turned 180° around Y to face +Z
  {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = 'Hips';
    hips.position.set(0, 1.0, 0);
    // Left arm on -X and Right arm on +X => character faces -Z
    const lArm = new THREE.Bone();
    lArm.name = 'LeftArm';
    lArm.position.set(-0.2, 1.4, 0);
    const rArm = new THREE.Bone();
    rArm.name = 'RightArm';
    rArm.position.set(0.2, 1.4, 0);
    root.add(hips);
    hips.add(lArm, rArm);
    normalizeAvatarRootTransform(root, false);
    const worldLeftArm = new THREE.Vector3();
    lArm.getWorldPosition(worldLeftArm);
    assert.ok(worldLeftArm.x > 0.1, `expected LeftArm at +X after +Z normalization, got x=${worldLeftArm.x.toFixed(3)}`);
  }
});

const riggedGlbPath = '/home/davide/Documenti/Junction2026/rigged_model.glb';
const sampleRecordingDir = '/home/davide/Documenti/Junction2026/data_sample_test/test_1_20261114_162048';

test(
  'real Blender Rigify rigged_model.glb loads and animates real LIS recording cleanly',
  { skip: !existsSync(riggedGlbPath) },
  async () => {
    const glbBuffer = toArrayBuffer(readFileSync(riggedGlbPath));
    const loaded = await loadVrmOrGlbFromArrayBuffer(glbBuffer);

    assert.equal(loaded.detectedFormat, 'Blender Rigify GLB');
    assert.ok(loaded.matchedBonesCount >= 43, `matchedBonesCount=${loaded.matchedBonesCount}`);
    assert.ok(loaded.drivenBonesCount >= 55, `drivenBonesCount=${loaded.drivenBonesCount}`);
    assert.equal(loaded.hasFacialBoneRig, true, 'Rigify face bone rig fallback should activate');
    assert.ok(
      loaded.skeletonHelper.bones.length < 300,
      `SkeletonHelper should filter Rigify WGT/MCH clutter (got ${loaded.skeletonHelper.bones.length} of 921)`,
    );

    const defHead = loaded.root.getObjectByName('DEF-spine006')!;
    const orgHead = loaded.root.getObjectByName('ORG-spine006')!;
    const jawMaster = loaded.root.getObjectByName('jaw_master');
    const defChin = loaded.root.getObjectByName('DEF-chin');
    const defIndex1L = loaded.root.getObjectByName('DEF-f_index01L')!;
    const defIndex2L = loaded.root.getObjectByName('DEF-f_index02L')!;
    assert.ok(defHead && orgHead && defIndex1L && defIndex2L, 'key Rigify deform/ORG nodes present');

    // Capture neutral jaw/chin position in head-local space before animation
    loaded.root.updateMatrixWorld(true);
    const chinProbe = defChin ?? jawMaster!;
    const neutralChinWorld = new THREE.Vector3();
    chinProbe.getWorldPosition(neutralChinWorld);
    const neutralChinHeadLocal = defHead.worldToLocal(neutralChinWorld.clone());

    // 1. Verify controlled pose: torso pitch + yaw, head turn, straight-forward left arm + flexed index, and jaw open
    const controlledPayload = makeSimplePayload(
      {
        spine: new THREE.Quaternion().setFromEuler(new THREE.Euler(0.14, 0.1, 0.04)),
        chest: new THREE.Quaternion().setFromEuler(new THREE.Euler(0.14, 0.1, 0.04)),
        neck: new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.05, -0.12, 0.02)),
        head: new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.08, -0.18, 0.03)),
        // Bend left index proximal 45° toward +Z (canonical rest bone points -Y, so +45° around +X swings -Y toward +Z)
        leftIndexProximal: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 4),
      },
      { mouthOpen: 0.85, eyeBlinkLeft: 0.9 },
    );

    const controlledClip = buildSignAnimationClip(controlledPayload, loaded.rigTarget);
    const mixer = new THREE.AnimationMixer(loaded.root);
    const action = mixer.clipAction(controlledClip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
    mixer.setTime(0.5);
    loaded.root.updateMatrixWorld(true);

    // Check 1: DEF-spine.006 (mesh skull) and ORG-spine.006 (eyes/teeth/hair parent) stay locked together!
    const pDef = new THREE.Vector3();
    const pOrg = new THREE.Vector3();
    const qDef = new THREE.Quaternion();
    const qOrg = new THREE.Quaternion();
    defHead.getWorldPosition(pDef);
    orgHead.getWorldPosition(pOrg);
    defHead.getWorldQuaternion(qDef);
    orgHead.getWorldQuaternion(qOrg);
    const headDriftMeters = pDef.distanceTo(pOrg);
    const headAngleDriftRad = qDef.angleTo(qOrg);
    assert.ok(
      headDriftMeters < 0.001,
      `ORG-spine.006 (eyes/teeth/hair) drifted ${(headDriftMeters * 1000).toFixed(3)} mm from DEF-spine.006`,
    );
    assert.ok(
      headAngleDriftRad < 0.005,
      `ORG-spine.006 rotation drifted ${((headAngleDriftRad * 180) / Math.PI).toFixed(3)}° from DEF-spine.006`,
    );

    // Check 2: Facial bone rig opened the jaw/chin downward in head-local space
    const openChinWorld = new THREE.Vector3();
    chinProbe.getWorldPosition(openChinWorld);
    const openChinHeadLocal = defHead.worldToLocal(openChinWorld.clone());
    assert.ok(
      openChinHeadLocal.distanceTo(neutralChinHeadLocal) > 0.002,
      'facial bone rig should displace jaw/chin when mouthOpen > 0',
    );

    // Check 3: Left index proximal bone points accurately despite ORG-palm.01.L's 71° bind rotation
    mixer.stopAllAction();
    // Reset to pure finger bend toward +Z palm (-60° around +X swings canonical [0,-1,0] to [0,-0.5,+0.866])
    const fingerPayload = makeSimplePayload({
      leftUpperArm: new THREE.Quaternion(),
      leftLowerArm: new THREE.Quaternion(),
      leftHand: new THREE.Quaternion(),
      leftIndexProximal: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 3),
    });
    const fingerClip = buildSignAnimationClip(fingerPayload, loaded.rigTarget);
    const fingerAct = mixer.clipAction(fingerClip);
    fingerAct.setLoop(THREE.LoopOnce, 1);
    fingerAct.clampWhenFinished = true;
    fingerAct.play();
    mixer.setTime(0.5);
    loaded.root.updateMatrixWorld(true);

    const pIdx1 = new THREE.Vector3();
    const pIdx2 = new THREE.Vector3();
    defIndex1L.getWorldPosition(pIdx1);
    defIndex2L.getWorldPosition(pIdx2);
    const actualFingerDir = pIdx2.sub(pIdx1).normalize();
    // Canonical [0, -1, 0] rotated +60° around +X => [0, -cos(60°), +sin(60°)] = [0, -0.5, +0.866]
    const expectedFingerDir = new THREE.Vector3(0, -Math.cos(Math.PI / 3), Math.sin(Math.PI / 3)).normalize();
    const fingerDot = actualFingerDir.dot(expectedFingerDir);
    assert.ok(
      fingerDot > 0.98,
      `expected left index world dir ~[0,-0.5,0.866] despite ORG-palm twist, got [${actualFingerDir
        .toArray()
        .map((v) => v.toFixed(3))
        .join(',')}], dot=${fingerDot.toFixed(4)}`,
    );

    // 2. End-to-end test with real dataset recording test_1_20261114_162048
    if (existsSync(sampleRecordingDir)) {
      const files = readdirSync(sampleRecordingDir)
        .filter((n) => n.endsWith('.csv'))
        .map((n) => ({ name: n, text: readFileSync(join(sampleRecordingDir, n), 'utf8') }));
      const rec = parseRokokoRecording(files, 'test_1');
      const { payload } = retargetRecording(rec, {
        ...DEFAULT_RETARGET_OPTIONS,
        startSec: 0,
        endSec: Math.min(rec.durationSec, 4.0),
        classId: 'test_1',
        phrase: 'Test 1',
        category: 'Dataset',
        description: '',
      });
      const realClip = buildSignAnimationClip(payload, loaded.rigTarget);
      assert.ok(realClip.tracks.length >= 55, `expected >= 55 tracks on Rigify clip, got ${realClip.tracks.length}`);

      mixer.stopAllAction();
      mixer.clipAction(realClip).play();
      for (const sampleT of [0, 1.0, 2.0, 3.5]) {
        mixer.setTime(sampleT);
        loaded.root.updateMatrixWorld(true);
        defHead.getWorldPosition(pDef);
        orgHead.getWorldPosition(pOrg);
        assert.ok(Number.isFinite(pDef.x) && Number.isFinite(pDef.y) && Number.isFinite(pDef.z));
        assert.ok(
          pDef.distanceTo(pOrg) < 0.001,
          `at t=${sampleT}s ORG-spine.006 stayed synced within ${(pDef.distanceTo(pOrg) * 1000).toFixed(3)} mm`,
        );
      }
    }

    loaded.dispose();
  },
);

const rpmGlbPath = '/home/davide/Documenti/Junction2026/avatar.glb';

test(
  'real Ready Player Me avatar.glb loads without 90° wrist-forearm twist jump and animates cleanly',
  { skip: !existsSync(rpmGlbPath) },
  async () => {
    const glbBuffer = toArrayBuffer(readFileSync(rpmGlbPath));
    const loaded = await loadVrmOrGlbFromArrayBuffer(glbBuffer);

    assert.ok(loaded.matchedBonesCount >= 43, `matchedBonesCount=${loaded.matchedBonesCount}`);
    // Spine, Spine1, Spine2 should all be driven
    assert.ok(loaded.rigTarget.drivenBoneTrackNames?.includes('Spine1'), 'intermediate Spine1 should be driven');

    // At canonical identity rest pose, LeftHand and LeftForeArm should not suffer a 90° relative twist jump
    const restPayload = makeSimplePayload({
      leftUpperArm: new THREE.Quaternion(),
      leftLowerArm: new THREE.Quaternion(),
      leftHand: new THREE.Quaternion(),
    });
    const clip = buildSignAnimationClip(restPayload, loaded.rigTarget);
    const handTrack = clip.tracks.find((t) => t.name === 'LeftHand.quaternion');
    assert.ok(handTrack, 'LeftHand.quaternion track generated');
    const lHandNode = loaded.root.getObjectByName('LeftHand')!;
    const bindLocal = lHandNode.quaternion.clone();

    const mixer = new THREE.AnimationMixer(loaded.root);
    const restAct = mixer.clipAction(clip);
    restAct.setLoop(THREE.LoopOnce, 1);
    restAct.clampWhenFinished = true;
    restAct.play();
    mixer.setTime(0.5);
    loaded.root.updateMatrixWorld(true);

    // Relative wrist twist from bind local should remain small (< 25°), not a ~90° bind-axis mismatch
    const wristTwistDeltaDeg = (lHandNode.quaternion.angleTo(bindLocal) * 180) / Math.PI;
    assert.ok(
      wristTwistDeltaDeg < 25,
      `expected smooth wrist rest alignment (< 25° from bind), got ${wristTwistDeltaDeg.toFixed(1)}°`,
    );

    loaded.dispose();
  },
);
