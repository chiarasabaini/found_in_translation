import * as THREE from 'three';
import type { SignAnimationPayload } from '../types/lis';

export interface AvatarRigTarget {
  /** Maps canonical bone names (e.g. "rightUpperArm") to actual scene Object3D node names */
  boneNodeNames: Record<string, string>;
  /**
   * Optional per-bone local quaternion retargeting transform for external GLB/VRM rigs
   * whose bind pose differs from our arms-down Y-up source skeleton (e.g. T-pose or A-pose).
   * When provided, takes a source frame's canonical local bone map and returns the target's
   * local quaternion for each mapped bone.
   */
  retargetFrameQuats?: (
    srcLocalQuats: Record<string, THREE.Quaternion>,
  ) => Record<string, THREE.Quaternion>;
  /** Name of the Mesh/SkinnedMesh holding facial morphTargetDictionary */
  faceMeshName: string | null;
  /** Available morph target names on the face mesh */
  availableMorphs: Set<string>;
}

const MORPH_ALIAS_FALLBACKS: Record<string, string[]> = {
  mouthOpen: ['mouthOpen', 'jawOpen'],
  mouthSmile: ['mouthSmile', 'mouthSmileLeft', 'mouthSmileRight', 'happy'],
  mouthFrown: ['mouthFrown', 'mouthFrownLeft', 'mouthFrownRight', 'sad'],
  mouthPucker: ['mouthPucker', 'mouthFunnel', 'ou'],
  browUp: ['browUp', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'],
  browDown: ['browDown', 'browDownLeft', 'browDownRight'],
  eyeBlinkLeft: ['eyeBlinkLeft', 'blinkLeft', 'blink'],
  eyeBlinkRight: ['eyeBlinkRight', 'blinkRight', 'blink'],
  cheekPuff: ['cheekPuff'],
};

/**
 * Builds a THREE.AnimationClip from a precomputed LIS animation JSON payload,
 * targeting either our procedural humanoid rig or an imported VRM/GLB humanoid rig.
 */
export function buildSignAnimationClip(
  payload: SignAnimationPayload,
  target: AvatarRigTarget,
): THREE.AnimationClip {
  const tracks: THREE.KeyframeTrack[] = [];
  const times = payload.times;
  const nFrames = times.length;

  if (target.retargetFrameQuats && nFrames > 0) {
    const outBuffers: Record<string, number[]> = {};
    for (const canonBone of Object.keys(target.boneNodeNames)) {
      if (payload.boneTracks[canonBone]?.length === nFrames * 4) {
        outBuffers[canonBone] = [];
      }
    }

    const srcFrame: Record<string, THREE.Quaternion> = {};
    for (let f = 0; f < nFrames; f++) {
      for (const [canonBone, flat] of Object.entries(payload.boneTracks)) {
        if (flat.length !== nFrames * 4) continue;
        const i4 = f * 4;
        srcFrame[canonBone] = new THREE.Quaternion(
          flat[i4],
          flat[i4 + 1],
          flat[i4 + 2],
          flat[i4 + 3],
        );
      }
      const mapped = target.retargetFrameQuats(srcFrame);
      for (const [canonBone, buf] of Object.entries(outBuffers)) {
        const q = mapped[canonBone] ?? srcFrame[canonBone];
        if (q) {
          // Hemisphere continuity check against previous frame
          if (buf.length >= 4) {
            const px = buf[buf.length - 4];
            const py = buf[buf.length - 3];
            const pz = buf[buf.length - 2];
            const pw = buf[buf.length - 1];
            if (px * q.x + py * q.y + pz * q.z + pw * q.w < 0) {
              q.set(-q.x, -q.y, -q.z, -q.w);
            }
          }
          buf.push(q.x, q.y, q.z, q.w);
        }
      }
    }

    for (const [canonBone, buf] of Object.entries(outBuffers)) {
      const nodeName = target.boneNodeNames[canonBone];
      if (!nodeName || buf.length !== nFrames * 4) continue;
      tracks.push(new THREE.QuaternionKeyframeTrack(`${nodeName}.quaternion`, times, buf));
    }
  } else {
    for (const [canonBone, flatQuats] of Object.entries(payload.boneTracks)) {
      const nodeName = target.boneNodeNames[canonBone];
      if (!nodeName) continue;
      if (flatQuats.length !== nFrames * 4) continue;
      tracks.push(
        new THREE.QuaternionKeyframeTrack(`${nodeName}.quaternion`, times, flatQuats),
      );
    }
  }

  if (target.faceMeshName) {
    const emittedTargets = new Set<string>();
    for (const [morphName, weights] of Object.entries(payload.morphTracks)) {
      if (weights.length !== nFrames) continue;
      const candidates = MORPH_ALIAS_FALLBACKS[morphName] ?? [morphName];
      for (const candidate of candidates) {
        if (target.availableMorphs.has(candidate) && !emittedTargets.has(candidate)) {
          emittedTargets.add(candidate);
          tracks.push(
            new THREE.NumberKeyframeTrack(
              `${target.faceMeshName}.morphTargetInfluences[${candidate}]`,
              times,
              weights,
            ),
          );
        }
      }
    }
  }

  const clip = new THREE.AnimationClip(payload.classId, payload.duration, tracks);
  clip.resetDuration();
  return clip;
}
