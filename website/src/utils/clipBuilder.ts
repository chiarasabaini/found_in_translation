import * as THREE from 'three';
import type { SignAnimationPayload } from '../types/lis';

export interface MorphTargetMeshBinding {
  meshName: string;
  availableMorphs: Set<string>;
}

export interface RetargetFrameContext {
  /** True when the payload comes from MoCap / model recordingRetarget (world-direction thumb tracks). */
  isRecordingClip: boolean;
}

export interface AvatarRigTarget {
  /** Maps canonical bone names (e.g. "rightUpperArm") to actual scene Object3D node names (or UUIDs) */
  boneNodeNames: Record<string, string>;
  /**
   * Optional per-bone local quaternion retargeting transform for external GLB/VRM rigs
   * whose bind pose or topology differs from our arms-down Y-up source skeleton.
   * May return local quaternions keyed either by canonical bone names (1-to-1 legacy mode)
   * or by `drivenBoneTrackNames` keys (universal multi-bone spine/neck/twist/Rigify mode).
   */
  retargetFrameQuats?: (
    srcLocalQuats: Record<string, THREE.Quaternion>,
    context?: RetargetFrameContext,
  ) => Record<string, THREE.Quaternion>;
  /**
   * Optional list of target scene track identifiers (sanitized node name or UUID) emitted
   * directly by `retargetFrameQuats`. When omitted, falls back to `boneNodeNames`.
   */
  drivenBoneTrackNames?: string[];
  /** Name of the primary Mesh/SkinnedMesh holding facial morphTargetDictionary */
  faceMeshName: string | null;
  /** Available morph target names (union across all morphable face meshes) */
  availableMorphs: Set<string>;
  /** Optional multi-mesh morph target bindings (e.g. Head + Teeth + Eyelashes meshes) */
  morphTargetMeshes?: MorphTargetMeshBinding[];
  /**
   * Optional generator for bone-driven facial expression tracks (jaw, eyelids, brows, lips, cheeks)
   * used when facial blendshape morph targets are absent (e.g. Blender Rigify facial bone rigs).
   */
  buildFacialBoneTracks?: (
    morphTracks: Record<string, number[]>,
    times: number[],
  ) => THREE.KeyframeTrack[];
}

export const MORPH_ALIAS_FALLBACKS: Record<string, string[]> = {
  mouthOpen: ['mouthOpen', 'jawOpen', 'viseme_aa', 'A', 'aa'],
  mouthSmile: ['mouthSmile', 'mouthSmileLeft', 'mouthSmileRight', 'happy', 'joy'],
  mouthFrown: ['mouthFrown', 'mouthFrownLeft', 'mouthFrownRight', 'sad', 'sorrow'],
  mouthPucker: ['mouthPucker', 'mouthFunnel', 'ou', 'viseme_U', 'O'],
  browUp: ['browUp', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'],
  browDown: ['browDown', 'browDownLeft', 'browDownRight', 'angry'],
  eyeBlinkLeft: ['eyeBlinkLeft', 'blinkLeft', 'eye_close_L', 'blink_L', 'blink'],
  eyeBlinkRight: ['eyeBlinkRight', 'blinkRight', 'eye_close_R', 'blink_R', 'blink'],
  cheekPuff: ['cheekPuff'],
};

function pushHemisphereContinuous(buf: number[], q: THREE.Quaternion): void {
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;
  if (buf.length >= 4) {
    const px = buf[buf.length - 4];
    const py = buf[buf.length - 3];
    const pz = buf[buf.length - 2];
    const pw = buf[buf.length - 1];
    if (px * x + py * y + pz * z + pw * w < 0) {
      x = -x;
      y = -y;
      z = -z;
      w = -w;
    }
  }
  buf.push(x, y, z, w);
}

/**
 * Builds a THREE.AnimationClip from a precomputed or live-retargeted LIS animation payload,
 * targeting either our procedural humanoid rig or any imported VRM/GLB skeleton topology.
 */
export function buildSignAnimationClip(
  payload: SignAnimationPayload,
  target: AvatarRigTarget,
): THREE.AnimationClip {
  const tracks: THREE.KeyframeTrack[] = [];
  const times = payload.times;
  const nFrames = times.length;
  const context: RetargetFrameContext = {
    isRecordingClip: payload.source?.kind === 'recording',
  };

  if (target.retargetFrameQuats && nFrames > 0) {
    const useDirectNodeKeys =
      Array.isArray(target.drivenBoneTrackNames) && target.drivenBoneTrackNames.length > 0;

    const outBuffers: Record<string, number[]> = {};
    if (useDirectNodeKeys) {
      for (const trackKey of target.drivenBoneTrackNames!) {
        outBuffers[trackKey] = [];
      }
    } else {
      for (const canonBone of Object.keys(target.boneNodeNames)) {
        if (payload.boneTracks[canonBone]?.length === nFrames * 4) {
          outBuffers[canonBone] = [];
        }
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
      const mapped = target.retargetFrameQuats(srcFrame, context);
      if (useDirectNodeKeys) {
        for (const trackKey of target.drivenBoneTrackNames!) {
          const q = mapped[trackKey];
          if (q) {
            pushHemisphereContinuous(outBuffers[trackKey], q);
          }
        }
      } else {
        for (const [canonBone, buf] of Object.entries(outBuffers)) {
          const q = mapped[canonBone] ?? srcFrame[canonBone];
          if (q) {
            pushHemisphereContinuous(buf, q);
          }
        }
      }
    }

    if (useDirectNodeKeys) {
      for (const [trackKey, buf] of Object.entries(outBuffers)) {
        if (!trackKey || buf.length !== nFrames * 4) continue;
        tracks.push(new THREE.QuaternionKeyframeTrack(`${trackKey}.quaternion`, times, buf));
      }
    } else {
      for (const [canonBone, buf] of Object.entries(outBuffers)) {
        const nodeName = target.boneNodeNames[canonBone];
        if (!nodeName || buf.length !== nFrames * 4) continue;
        tracks.push(new THREE.QuaternionKeyframeTrack(`${nodeName}.quaternion`, times, buf));
      }
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

  // Emit facial blendshape morph tracks across all morphable facial meshes (e.g. Head + Teeth + Brows)
  const morphBindings: MorphTargetMeshBinding[] =
    target.morphTargetMeshes && target.morphTargetMeshes.length > 0
      ? target.morphTargetMeshes
      : target.faceMeshName
        ? [{ meshName: target.faceMeshName, availableMorphs: target.availableMorphs }]
        : [];

  for (const binding of morphBindings) {
    if (!binding.meshName || binding.availableMorphs.size === 0) continue;
    const emittedTargets = new Set<string>();
    for (const [morphName, weights] of Object.entries(payload.morphTracks)) {
      if (weights.length !== nFrames) continue;
      const candidates = MORPH_ALIAS_FALLBACKS[morphName] ?? [morphName];
      for (const candidate of candidates) {
        if (binding.availableMorphs.has(candidate) && !emittedTargets.has(candidate)) {
          emittedTargets.add(candidate);
          tracks.push(
            new THREE.NumberKeyframeTrack(
              `${binding.meshName}.morphTargetInfluences[${candidate}]`,
              times,
              weights,
            ),
          );
        }
      }
    }
  }

  // Fallback / complementary bone-driven facial animation (e.g. Rigify jaw, eyelids, brows, lips)
  if (target.buildFacialBoneTracks && nFrames > 0) {
    const faceBoneTracks = target.buildFacialBoneTracks(payload.morphTracks, times);
    for (const ft of faceBoneTracks) {
      tracks.push(ft);
    }
  }

  const clip = new THREE.AnimationClip(payload.classId, payload.duration, tracks);
  clip.resetDuration();
  return clip;
}
