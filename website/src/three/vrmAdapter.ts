import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { AvatarRigTarget } from '../utils/clipBuilder';

export interface LoadedVrmAvatar {
  root: THREE.Group;
  vrm: VRM | null;
  skeletonHelper: THREE.SkeletonHelper;
  rigTarget: AvatarRigTarget;
  detectedFormat: string;
  matchedBonesCount: number;
  dispose: () => void;
}

const CANON_TO_VRM_BONE: Record<string, VRMHumanBoneName> = {
  hips: VRMHumanBoneName.Hips,
  spine: VRMHumanBoneName.Spine,
  chest: VRMHumanBoneName.Chest,
  neck: VRMHumanBoneName.Neck,
  head: VRMHumanBoneName.Head,
  rightShoulder: VRMHumanBoneName.RightShoulder,
  rightUpperArm: VRMHumanBoneName.RightUpperArm,
  rightLowerArm: VRMHumanBoneName.RightLowerArm,
  rightHand: VRMHumanBoneName.RightHand,
  leftShoulder: VRMHumanBoneName.LeftShoulder,
  leftUpperArm: VRMHumanBoneName.LeftUpperArm,
  leftLowerArm: VRMHumanBoneName.LeftLowerArm,
  leftHand: VRMHumanBoneName.LeftHand,
  rightThumbProximal: VRMHumanBoneName.RightThumbMetacarpal,
  rightThumbIntermediate: VRMHumanBoneName.RightThumbProximal,
  rightThumbDistal: VRMHumanBoneName.RightThumbDistal,
  rightIndexProximal: VRMHumanBoneName.RightIndexProximal,
  rightIndexIntermediate: VRMHumanBoneName.RightIndexIntermediate,
  rightIndexDistal: VRMHumanBoneName.RightIndexDistal,
  rightMiddleProximal: VRMHumanBoneName.RightMiddleProximal,
  rightMiddleIntermediate: VRMHumanBoneName.RightMiddleIntermediate,
  rightMiddleDistal: VRMHumanBoneName.RightMiddleDistal,
  rightRingProximal: VRMHumanBoneName.RightRingProximal,
  rightRingIntermediate: VRMHumanBoneName.RightRingIntermediate,
  rightRingDistal: VRMHumanBoneName.RightRingDistal,
  rightLittleProximal: VRMHumanBoneName.RightLittleProximal,
  rightLittleIntermediate: VRMHumanBoneName.RightLittleIntermediate,
  rightLittleDistal: VRMHumanBoneName.RightLittleDistal,
  leftThumbProximal: VRMHumanBoneName.LeftThumbMetacarpal,
  leftThumbIntermediate: VRMHumanBoneName.LeftThumbProximal,
  leftThumbDistal: VRMHumanBoneName.LeftThumbDistal,
  leftIndexProximal: VRMHumanBoneName.LeftIndexProximal,
  leftIndexIntermediate: VRMHumanBoneName.LeftIndexIntermediate,
  leftIndexDistal: VRMHumanBoneName.LeftIndexDistal,
  leftMiddleProximal: VRMHumanBoneName.LeftMiddleProximal,
  leftMiddleIntermediate: VRMHumanBoneName.LeftMiddleIntermediate,
  leftMiddleDistal: VRMHumanBoneName.LeftMiddleDistal,
  leftRingProximal: VRMHumanBoneName.LeftRingProximal,
  leftRingIntermediate: VRMHumanBoneName.LeftRingIntermediate,
  leftRingDistal: VRMHumanBoneName.LeftRingDistal,
  leftLittleProximal: VRMHumanBoneName.LeftLittleProximal,
  leftLittleIntermediate: VRMHumanBoneName.LeftLittleIntermediate,
  leftLittleDistal: VRMHumanBoneName.LeftLittleDistal,
};

/**
 * Bone naming aliases covering Ready Player Me, Avaturn, Mixamo, and Unity/Blender Humanoid GLBs.
 * Ordered from most specific to most general.
 */
export const GLB_BONE_ALIASES: Record<string, string[]> = {
  hips: ['hips', 'mixamorighips', 'pelvis', 'root'],
  spine: ['spine', 'mixamorigspine'],
  chest: ['spine2', 'spine1', 'chest', 'upperchest', 'mixamorigspine2', 'mixamorigspine1'],
  neck: ['neck', 'mixamorigneck'],
  head: ['head', 'mixamorighead'],
  rightShoulder: ['rightshoulder', 'mixamorigrightshoulder', 'clavicle_r', 'shoulder_r'],
  rightUpperArm: ['rightarm', 'rightupperarm', 'mixamorigrightarm', 'upperarm_r'],
  rightLowerArm: ['rightforearm', 'rightlowerarm', 'mixamorigrightforearm', 'lowerarm_r'],
  rightHand: ['righthand', 'mixamorigrighthand', 'hand_r'],
  leftShoulder: ['leftshoulder', 'mixamorigleftshoulder', 'clavicle_l', 'shoulder_l'],
  leftUpperArm: ['leftarm', 'leftupperarm', 'mixamorigleftarm', 'upperarm_l'],
  leftLowerArm: ['leftforearm', 'leftlowerarm', 'mixamorigleftforearm', 'lowerarm_l'],
  leftHand: ['lefthand', 'mixamoriglefthand', 'hand_l'],
  rightThumbProximal: ['righthandthumb1', 'rightthumbproximal', 'rightthumbmetacarpal', 'thumb_01_r'],
  rightThumbIntermediate: ['righthandthumb2', 'rightthumbintermediate', 'thumb_02_r'],
  rightThumbDistal: ['righthandthumb3', 'rightthumbdistal', 'thumb_03_r'],
  rightIndexProximal: ['righthandindex1', 'rightindexproximal', 'index_01_r'],
  rightIndexIntermediate: ['righthandindex2', 'rightindexintermediate', 'index_02_r'],
  rightIndexDistal: ['righthandindex3', 'rightindexdistal', 'index_03_r'],
  rightMiddleProximal: ['righthandmiddle1', 'rightmiddleproximal', 'middle_01_r'],
  rightMiddleIntermediate: ['righthandmiddle2', 'rightmiddleintermediate', 'middle_02_r'],
  rightMiddleDistal: ['righthandmiddle3', 'rightmiddledistal', 'middle_03_r'],
  rightRingProximal: ['righthandring1', 'rightringproximal', 'ring_01_r'],
  rightRingIntermediate: ['righthandring2', 'rightringintermediate', 'ring_02_r'],
  rightRingDistal: ['righthandring3', 'rightringdistal', 'ring_03_r'],
  rightLittleProximal: ['righthandpinky1', 'righthandlittle1', 'rightlittleproximal', 'pinky_01_r'],
  rightLittleIntermediate: ['righthandpinky2', 'righthandlittle2', 'rightlittleintermediate', 'pinky_02_r'],
  rightLittleDistal: ['righthandpinky3', 'righthandlittle3', 'rightlittledistal', 'pinky_03_r'],
  leftThumbProximal: ['lefthandthumb1', 'leftthumbproximal', 'leftthumbmetacarpal', 'thumb_01_l'],
  leftThumbIntermediate: ['lefthandthumb2', 'leftthumbintermediate', 'thumb_02_l'],
  leftThumbDistal: ['lefthandthumb3', 'leftthumbdistal', 'thumb_03_l'],
  leftIndexProximal: ['lefthandindex1', 'leftindexproximal', 'index_01_l'],
  leftIndexIntermediate: ['lefthandindex2', 'leftindexintermediate', 'index_02_l'],
  leftIndexDistal: ['lefthandindex3', 'leftindexdistal', 'index_03_l'],
  leftMiddleProximal: ['lefthandmiddle1', 'leftmiddleproximal', 'middle_01_l'],
  leftMiddleIntermediate: ['lefthandmiddle2', 'leftmiddleintermediate', 'middle_02_l'],
  leftMiddleDistal: ['lefthandmiddle3', 'leftmiddledistal', 'middle_03_l'],
  leftRingProximal: ['lefthandring1', 'leftringproximal', 'ring_01_l'],
  leftRingIntermediate: ['lefthandring2', 'leftringintermediate', 'ring_02_l'],
  leftRingDistal: ['lefthandring3', 'leftringdistal', 'ring_03_l'],
  leftLittleProximal: ['lefthandpinky1', 'lefthandlittle1', 'leftlittleproximal', 'pinky_01_l'],
  leftLittleIntermediate: ['lefthandpinky2', 'lefthandlittle2', 'leftlittleintermediate', 'pinky_02_l'],
  leftLittleDistal: ['lefthandpinky3', 'lefthandlittle3', 'leftlittledistal', 'pinky_03_l'],
};

const PARENT_MAP: Record<string, string | null> = {
  hips: null,
  spine: 'hips',
  chest: 'spine',
  neck: 'chest',
  head: 'neck',
  rightShoulder: 'chest',
  rightUpperArm: 'rightShoulder',
  rightLowerArm: 'rightUpperArm',
  rightHand: 'rightLowerArm',
  leftShoulder: 'chest',
  leftUpperArm: 'leftShoulder',
  leftLowerArm: 'leftUpperArm',
  leftHand: 'leftLowerArm',
};

const CHILD_DIRECTION_MAP: Record<string, string> = {
  rightUpperArm: 'rightLowerArm',
  rightLowerArm: 'rightHand',
  leftUpperArm: 'leftLowerArm',
  leftLowerArm: 'leftHand',
};

const FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'] as const;
for (const side of ['right', 'left'] as const) {
  for (const f of FINGER_NAMES) {
    PARENT_MAP[`${side}${f}Proximal`] = `${side}Hand`;
    PARENT_MAP[`${side}${f}Intermediate`] = `${side}${f}Proximal`;
    PARENT_MAP[`${side}${f}Distal`] = `${side}${f}Intermediate`;
  }
}

export function matchGlbBoneNames(allNodeNames: string[]): Record<string, string> {
  const normMap = new Map<string, string>();
  for (const name of allNodeNames) {
    const clean = name.toLowerCase().replace(/^mixamorig:?/, 'mixamorig').replace(/[^a-z0-9_]/g, '');
    if (clean && !normMap.has(clean)) normMap.set(clean, name);
  }

  const matched: Record<string, string> = {};
  for (const [canon, aliases] of Object.entries(GLB_BONE_ALIASES)) {
    for (const alias of aliases) {
      const exact = normMap.get(alias);
      if (exact) {
        matched[canon] = exact;
        break;
      }
    }
    if (!matched[canon]) {
      for (const alias of aliases) {
        const partial = allNodeNames.find((n) =>
          n.toLowerCase().replace(/[^a-z0-9_]/g, '').endsWith(alias),
        );
        if (partial) {
          matched[canon] = partial;
          break;
        }
      }
    }
  }
  return matched;
}

/** Rotation whose X/Y/Z columns are (f, s, n) with n = f × s, s re-orthogonalized. */
function frameQuaternion(f: THREE.Vector3, s: THREE.Vector3): THREE.Quaternion {
  const fx = f.clone().normalize();
  const nz = new THREE.Vector3().crossVectors(fx, s).normalize();
  const sy = new THREE.Vector3().crossVectors(nz, fx);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(fx, sy, nz));
}

/**
 * Builds a bind-pose compensator for external GLB rigs (e.g. T-pose Ready Player Me / Mixamo)
 * so arm bones pointing sideways in bind pose align with our arms-down (0, -1, 0) source rig.
 */
function buildBindPoseRetargeter(
  root: THREE.Group,
  boneNodeNames: Record<string, string>,
): (srcLocalQuats: Record<string, THREE.Quaternion>) => Record<string, THREE.Quaternion> {
  root.updateMatrixWorld(true);

  const bindWorldQuat: Record<string, THREE.Quaternion> = {};
  const poseAlignWorld: Record<string, THREE.Quaternion> = {};
  const nodeByCanon: Record<string, THREE.Object3D> = {};

  for (const [canon, nodeName] of Object.entries(boneNodeNames)) {
    const obj = root.getObjectByName(nodeName);
    if (!obj) continue;
    nodeByCanon[canon] = obj;
    const wq = new THREE.Quaternion();
    obj.getWorldQuaternion(wq);
    bindWorldQuat[canon] = wq;
    poseAlignWorld[canon] = new THREE.Quaternion();
  }

  // Align arm segments from their actual GLB bind direction (e.g. T-pose -X/+X) to (0, -1, 0)
  const downVec = new THREE.Vector3(0, -1, 0);
  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  for (const [canon, childCanon] of Object.entries(CHILD_DIRECTION_MAP)) {
    const a = nodeByCanon[canon];
    const b = nodeByCanon[childCanon];
    if (!a || !b) continue;
    a.getWorldPosition(pA);
    b.getWorldPosition(pB);
    const dir = pB.sub(pA);
    if (dir.lengthSq() > 1e-6) {
      dir.normalize();
      poseAlignWorld[canon] = new THREE.Quaternion().setFromUnitVectors(dir, downVec);
    }
  }
  // Propagate arm alignment down to hands and fingers so palms/fingers keep clean orientation
  for (const side of ['right', 'left'] as const) {
    const lowerAlign = poseAlignWorld[`${side}LowerArm`];
    if (lowerAlign) {
      poseAlignWorld[`${side}Hand`] = lowerAlign.clone();
      for (const f of FINGER_NAMES) {
        for (const ph of ['Proximal', 'Intermediate', 'Distal'] as const) {
          poseAlignWorld[`${side}${f}${ph}`] = lowerAlign.clone();
        }
      }
    }
  }

  // Refine hands and fingers with their full bind frame. The source rig's rest hand has
  // fingers along -Y and the index side toward -X (right) / +X (left), palm facing +Z.
  // Aligning only the forearm direction leaves T-pose palms (facing down) twisted by 90°.
  const worldPos = (canon: string): THREE.Vector3 | null =>
    nodeByCanon[canon] ? nodeByCanon[canon].getWorldPosition(new THREE.Vector3()) : null;
  for (const side of ['right', 'left'] as const) {
    const hand = worldPos(`${side}Hand`);
    const index = worldPos(`${side}IndexProximal`);
    const little = worldPos(`${side}LittleProximal`) ?? worldPos(`${side}RingProximal`);
    const middle = worldPos(`${side}MiddleProximal`);
    if (!hand || !index || !little) continue;
    const knuckles = middle ?? index.clone().add(little).multiplyScalar(0.5);
    const fingerDir = knuckles.clone().sub(hand);
    const radialDir = index.clone().sub(little);
    if (fingerDir.lengthSq() < 1e-8 || radialDir.lengthSq() < 1e-8) continue;
    const from = frameQuaternion(fingerDir, radialDir);
    const to = frameQuaternion(downVec, new THREE.Vector3(side === 'right' ? -1 : 1, 0, 0));
    const handAlign = to.multiply(from.invert());
    poseAlignWorld[`${side}Hand`] = handAlign;

    for (const f of FINGER_NAMES) {
      const chain = (['Proximal', 'Intermediate', 'Distal'] as const).map((ph) => `${side}${f}${ph}`);
      let prevDir: THREE.Vector3 | null = null;
      chain.forEach((canon, i) => {
        const p = worldPos(canon);
        const child = i < chain.length - 1 ? worldPos(chain[i + 1]) : null;
        const dir: THREE.Vector3 | null = p && child ? child.sub(p) : prevDir;
        if (!p || !dir || dir.lengthSq() < 1e-10) {
          poseAlignWorld[canon] = handAlign.clone();
          return;
        }
        prevDir = dir.clone();
        const inHandFrame = dir.clone().normalize().applyQuaternion(handAlign);
        poseAlignWorld[canon] = new THREE.Quaternion().setFromUnitVectors(inHandFrame, downVec).multiply(handAlign);
      });
    }
  }

  const orderedBones = Object.keys(PARENT_MAP);

  return (srcLocalQuats) => {
    const srcWorld: Record<string, THREE.Quaternion> = {};
    const tgtWorld: Record<string, THREE.Quaternion> = {};
    const outLocal: Record<string, THREE.Quaternion> = {};

    for (const canon of orderedBones) {
      const parent = PARENT_MAP[canon];
      const qSrc = srcLocalQuats[canon] ?? new THREE.Quaternion();
      const parentSrcWorld = parent && srcWorld[parent] ? srcWorld[parent] : new THREE.Quaternion();
      srcWorld[canon] = parentSrcWorld.clone().multiply(qSrc);

      const qBind = bindWorldQuat[canon];
      if (!qBind) continue;
      const qAlign = poseAlignWorld[canon] ?? new THREE.Quaternion();
      // Target world = srcWorld * poseAlign * bindWorld
      const wTgt = srcWorld[canon].clone().multiply(qAlign).multiply(qBind);
      tgtWorld[canon] = wTgt;

      const parentNode = nodeByCanon[canon]?.parent;
      const parentTgtWorld =
        parent && tgtWorld[parent]
          ? tgtWorld[parent]
          : parentNode
            ? parentNode.getWorldQuaternion(new THREE.Quaternion())
            : new THREE.Quaternion();
      outLocal[canon] = parentTgtWorld.clone().invert().multiply(wTgt).normalize();
    }

    return outLocal;
  };
}

/**
 * Loads an external .vrm or .glb file via @pixiv/three-vrm / GLTFLoader and maps its
 * humanoid bone nodes + ARKit/VRM facial morph targets into our AvatarRigTarget contract.
 */
export async function loadVrmOrGlbFromFile(file: File): Promise<LoadedVrmAvatar> {
  const buffer = await file.arrayBuffer();
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await new Promise<any>((resolve, reject) => {
    loader.parse(buffer, '', resolve, reject);
  });

  const vrm: VRM | null = gltf.userData?.vrm ?? null;
  // VRM 0.x models face -Z; turn them to +Z like the source rig (no-op for VRM 1.0).
  if (vrm) VRMUtils.rotateVRM0(vrm);
  const root: THREE.Group = vrm ? vrm.scene : gltf.scene;

  let boneNodeNames: Record<string, string> = {};
  let detectedFormat = 'GLB Humanoid';

  if (vrm?.humanoid) {
    detectedFormat = 'VRM Humanoid';
    for (const [canonName, vrmBone] of Object.entries(CANON_TO_VRM_BONE)) {
      const node =
        vrm.humanoid.getNormalizedBoneNode(vrmBone) ?? vrm.humanoid.getRawBoneNode(vrmBone);
      if (node && node.name) {
        boneNodeNames[canonName] = node.name;
      }
    }
  } else {
    const allNamedNodes: string[] = [];
    root.traverse((obj) => {
      if (obj.name) allNamedNodes.push(obj.name);
    });
    boneNodeNames = matchGlbBoneNames(allNamedNodes);
    if (allNamedNodes.some((n) => n.toLowerCase().includes('mixamorig'))) {
      detectedFormat = 'Mixamo GLB';
    } else if (allNamedNodes.some((n) => n.toLowerCase().includes('wolf3d') || n.toLowerCase().includes('rightforearm'))) {
      detectedFormat = 'Ready Player Me / Avaturn GLB';
    }
  }

  let faceMeshName: string | null = null;
  const availableMorphs = new Set<string>();
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.isMesh && mesh.morphTargetDictionary) {
      const keys = Object.keys(mesh.morphTargetDictionary);
      if (keys.length > availableMorphs.size) {
        faceMeshName = mesh.name || null;
        availableMorphs.clear();
        keys.forEach((k) => availableMorphs.add(k));
      }
    }
    if (mesh.isMesh) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
  });

  // Normalize scale if an imported GLB is in centimeters (e.g. 170 units tall instead of 1.7m)
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  if (size.y > 5.0) {
    const scaleFactor = 1.72 / size.y;
    root.scale.multiplyScalar(scaleFactor);
    root.updateMatrixWorld(true);
    const scaledBox = new THREE.Box3().setFromObject(root);
    root.position.y -= scaledBox.min.y;
  } else if (Math.abs(box.min.y) > 0.05) {
    root.position.y -= box.min.y;
  }

  // VRM normalized bones rest in T-pose too, so they need the same bind-pose compensation
  // as GLB rigs: the source tracks assume arms hanging down (0, -1, 0).
  const retargetFrameQuats = buildBindPoseRetargeter(root, boneNodeNames);

  const skeletonHelper = new THREE.SkeletonHelper(root);
  skeletonHelper.visible = false;
  root.add(skeletonHelper);

  return {
    root,
    vrm,
    skeletonHelper,
    detectedFormat,
    matchedBonesCount: Object.keys(boneNodeNames).length,
    rigTarget: {
      boneNodeNames,
      retargetFrameQuats,
      faceMeshName,
      availableMorphs,
    },
    dispose: () => {
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
