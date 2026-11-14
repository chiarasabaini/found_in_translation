import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSignAnimationClip } from './clipBuilder';
import { matchGlbBoneNames } from '../three/vrmAdapter';
import type { SignAnimationPayload } from '../types/lis';

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

test('buildSignAnimationClip maps ARKit morph aliases on GLB face mesh', () => {
  const payload: SignAnimationPayload = {
    classId: 'ciao',
    phrase: 'Ciao',
    category: 'Saluti',
    description: 'Test',
    prototypeClipId: 'ciao_s00_c00',
    fps: 30,
    duration: 0.5,
    times: [0, 0.5],
    boneTracks: {
      rightUpperArm: [0, 0, 0, 1, 0, 0, 0, 1],
    },
    morphTracks: {
      mouthOpen: [0, 0.8],
      mouthSmile: [0, 0.6],
      browUp: [0, 0.4],
    },
    qaComparison: {
      group1VsGroup2Right: [],
      group1VsGroup2Left: [],
      frames: [],
    },
  };

  const clip = buildSignAnimationClip(payload, {
    boneNodeNames: { rightUpperArm: 'RightArm' },
    faceMeshName: 'Wolf3D_Head',
    availableMorphs: new Set(['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp']),
  });

  const trackNames = clip.tracks.map((t) => t.name);
  assert.ok(trackNames.includes('RightArm.quaternion'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[jawOpen]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[mouthSmileLeft]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[mouthSmileRight]'));
  assert.ok(trackNames.includes('Wolf3D_Head.morphTargetInfluences[browInnerUp]'));
});
