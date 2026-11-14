import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import type { AvatarRigTarget } from '../utils/clipBuilder';
import {
  analyzeUniversalAvatarRig,
  buildCleanSkeletonHelper,
  matchGlbBoneNames,
  normalizeAvatarRootTransform,
} from './universalSkeleton';

export { matchGlbBoneNames, analyzeUniversalAvatarRig, normalizeAvatarRootTransform };

export interface LoadedVrmAvatar {
  root: THREE.Group;
  vrm: VRM | null;
  skeletonHelper: THREE.SkeletonHelper;
  rigTarget: AvatarRigTarget;
  detectedFormat: string;
  matchedBonesCount: number;
  drivenBonesCount: number;
  hasFacialBoneRig: boolean;
  dispose: () => void;
}

/**
 * Parses an ArrayBuffer containing a `.glb`, `.gltf`, or `.vrm` asset and adapts any
 * skeleton topology (Blender Rigify / Rigify Meta-Rig, Ready Player Me, Avaturn, Mixamo,
 * Unreal, Character Creator, 3ds Max Biped, VRM 0.x/1.0, or custom topological humanoids)
 * into our universal `AvatarRigTarget` contract.
 */
export async function loadVrmOrGlbFromArrayBuffer(buffer: ArrayBuffer): Promise<LoadedVrmAvatar> {
  const loader = new GLTFLoader();
  if (typeof document === 'undefined' && typeof createImageBitmap === 'undefined') {
    loader.register(() => ({
      name: 'headless-texture-stub',
      loadTexture: () => Promise.resolve(new THREE.Texture()),
    }));
  }
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await new Promise<any>((resolve, reject) => {
    loader.parse(buffer, '', resolve, reject);
  });

  const vrm: VRM | null = gltf.userData?.vrm ?? null;
  // VRM 0.x models face -Z; turn them to +Z like the source rig (no-op for VRM 1.0).
  if (vrm) VRMUtils.rotateVRM0(vrm);
  const root: THREE.Group = vrm ? vrm.scene : gltf.scene;

  // Normalize scale (cm -> m), ground feet at y = 0, and orient -Z GLBs to +Z BEFORE bind-pose capture
  normalizeAvatarRootTransform(root, Boolean(vrm));

  // Universal deform skeleton detection, multi-bone spine/neck/twist distribution,
  // Rigify parallel ORG-/spine_fk synchronization, and bone-based facial rig fallback
  const analysis = analyzeUniversalAvatarRig(root, vrm);

  const skeletonHelper = buildCleanSkeletonHelper(root, analysis.visualBones);
  root.add(skeletonHelper);

  return {
    root,
    vrm,
    skeletonHelper,
    detectedFormat: analysis.detectedFormat,
    matchedBonesCount: analysis.matchedBonesCount,
    drivenBonesCount: analysis.drivenBonesCount,
    hasFacialBoneRig: analysis.hasFacialBoneRig,
    rigTarget: analysis.rigTarget,
    dispose: () => {
      skeletonHelper.dispose();
      root.traverse((obj) => {
        const m = obj as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        if (m.material) {
          if (Array.isArray(m.material)) m.material.forEach((mt) => mt.dispose());
          else m.material.dispose();
        }
      });
    },
  };
}

/**
 * Loads an external `.vrm`, `.glb`, or `.gltf` File and adapts its skeleton + morph/bone face rig.
 */
export async function loadVrmOrGlbFromFile(file: File): Promise<LoadedVrmAvatar> {
  const buffer = await file.arrayBuffer();
  return loadVrmOrGlbFromArrayBuffer(buffer);
}
