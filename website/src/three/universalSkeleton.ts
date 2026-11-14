import * as THREE from 'three';
import { type VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { AvatarRigTarget, MorphTargetMeshBinding, RetargetFrameContext } from '../utils/clipBuilder';

export type Side = 'left' | 'right';
export const FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'] as const;
export type FingerCanon = (typeof FINGER_NAMES)[number];
export const PHALANX_NAMES = ['Proximal', 'Intermediate', 'Distal'] as const;
export type PhalanxCanon = (typeof PHALANX_NAMES)[number];

const RESERVED_TRACK_CHARS = /[\[\].:/\s]/;

export const CANON_TO_VRM_BONE: Record<string, VRMHumanBoneName> = {
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

export const CANON_PARENT_MAP: Record<string, string | null> = {
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
for (const side of ['right', 'left'] as const) {
  for (const f of FINGER_NAMES) {
    CANON_PARENT_MAP[`${side}${f}Proximal`] = `${side}Hand`;
    CANON_PARENT_MAP[`${side}${f}Intermediate`] = `${side}${f}Proximal`;
    CANON_PARENT_MAP[`${side}${f}Distal`] = `${side}${f}Intermediate`;
  }
}

export interface ParsedBoneName {
  raw: string;
  lower: string;
  tokens: string[];
  side: Side | null;
  isDef: boolean;
  isOrg: boolean;
  isMchOrControl: boolean;
  isFacialOrLowerBody: boolean;
  isTwist: boolean;
  isEndOrTip: boolean;
  numbers: number[];
  trailingSplitIndex: number | null;
}

const CONTROL_TOKENS = new Set([
  'mch',
  'wgt',
  'vis',
  'ik',
  'pole',
  'target',
  'ctrl',
  'ctl',
  'widget',
  'handle',
  'tweak',
  'glue',
  'offset',
  'reparent',
  'parent',
  'fk',
  'master',
  'heel',
  'roll',
  'socket',
  'hinge',
]);

const LOWER_OR_FACIAL_TOKENS = new Set([
  'thigh',
  'upleg',
  'leg',
  'shin',
  'calf',
  'knee',
  'foot',
  'ankle',
  'toe',
  'toes',
  'toebase',
  'glute',
  'buttock',
  'breast',
  'pectoral',
  'boob',
  'eye',
  'eyes',
  'iris',
  'pupil',
  'cornea',
  'lid',
  'eyelid',
  'brow',
  'eyebrow',
  'jaw',
  'chin',
  'lip',
  'lips',
  'mouth',
  'teeth',
  'tooth',
  'tongue',
  'cheek',
  'nose',
  'nostril',
  'ear',
  'forehead',
  'temple',
  'hair',
  'skirt',
  'cape',
  'cloth',
  'weapon',
  'prop',
  'camera',
  'light',
]);

/** Returns the original glTF node name before GLTFLoader dot-stripping when available. */
export function getRawNodeName(obj: THREE.Object3D): string {
  const raw = obj.userData?.name;
  return typeof raw === 'string' && raw.length > 0 ? raw : obj.name || '';
}

/**
 * Tokenizes any bone naming convention (Blender Rigify DEF-/ORG-/Meta-Rig, GLTFLoader-sanitized
 * names like "DEF-upper_armL001", Mixamo, Ready Player Me, Avaturn, Unreal, Unity, CC3/CC4,
 * 3ds Max Biped, Auto-Rig Pro) into clean lowercase semantic tokens + laterality + numeric indices.
 */
export function parseBoneName(rawName: string): ParsedBoneName {
  const normalizedMixamo = rawName.replace(/^mixamorig:?/i, 'mixamorig_');
  const splitCamel = normalizedMixamo
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .replace(/([a-zA-Z])([0-9])/g, '$1_$2')
    .replace(/([0-9])([a-zA-Z])/g, '$1_$2');

  const tokens = splitCamel
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);

  const lower = rawName.toLowerCase();
  const isDef = tokens[0] === 'def';
  const isOrg = tokens[0] === 'org';

  let side: Side | null = null;
  // Scan from right to left because laterality suffixes (.L, _l, .L.001) are more specific than prefixes
  for (let i = tokens.length - 1; i >= 0; i--) {
    const t = tokens[i];
    if (t === 'left' || t === 'l') {
      // Avoid mistaking "wolf_3_d" etc.; standalone 'l' is universally Left in bone rigs
      side = 'left';
      break;
    }
    if (t === 'right' || t === 'r') {
      side = 'right';
      break;
    }
  }

  // Detect Rigify lateral pelvis bones ("DEF-pelvis.L" / "DEF-pelvis.R") -> NOT central hips!
  const isLateralPelvis = tokens.includes('pelvis') && side !== null;

  const isMchOrControl = tokens.some((t) => CONTROL_TOKENS.has(t));
  const isFacialOrLowerBody =
    isLateralPelvis ||
    tokens.some((t) => LOWER_OR_FACIAL_TOKENS.has(t)) ||
    (tokens.includes('up') && tokens.includes('leg'));

  const isEndOrTip =
    tokens.includes('end') ||
    tokens.includes('tip') ||
    tokens.includes('nub') ||
    tokens.includes('leaf');

  // Detect split/twist limb sub-bones:
  //   - Explicit "twist" / "roll" token (e.g. CC_Base_L_ForearmTwist01, lowerarm_twist_01_l)
  //   - Rigify ".L.001" / "L001" where a numeric token comes AFTER the side token on an arm bone
  let trailingSplitIndex: number | null = null;
  if (side !== null) {
    let sideIdx = -1;
    for (let i = tokens.length - 1; i >= 0; i--) {
      const t = tokens[i];
      if (t === 'l' || t === 'r' || t === 'left' || t === 'right') {
        sideIdx = i;
        break;
      }
    }
    if (sideIdx >= 0 && sideIdx === tokens.length - 2 && /^\d+$/.test(tokens[tokens.length - 1])) {
      const armLike = tokens.some((t) =>
        ['arm', 'upperarm', 'forearm', 'lowerarm', 'shoulder', 'clavicle'].includes(t),
      );
      if (armLike) {
        trailingSplitIndex = parseInt(tokens[tokens.length - 1], 10);
      }
    }
  }
  const isTwist = tokens.includes('twist') || trailingSplitIndex !== null;

  const numbers = tokens.filter((t) => /^\d+$/.test(t)).map((t) => parseInt(t, 10));

  return {
    raw: rawName,
    lower,
    tokens,
    side,
    isDef,
    isOrg,
    isMchOrControl,
    isFacialOrLowerBody,
    isTwist,
    isEndOrTip,
    numbers,
    trailingSplitIndex,
  };
}

type LimbRole =
  | 'hips'
  | 'spine'
  | 'chest'
  | 'neck'
  | 'head'
  | 'shoulder'
  | 'upperArm'
  | 'lowerArm'
  | 'hand'
  | 'palmMetacarpal'
  | { kind: 'finger'; finger: FingerCanon; phalanxIndex: 0 | 1 | 2 | null };

function classifyParsedBoneRole(p: ParsedBoneName): LimbRole | null {
  if (p.isFacialOrLowerBody || p.isEndOrTip) return null;
  const tset = new Set(p.tokens);

  // 1. Check fingers & palm metacarpals first (since names like "RightHandIndex1" contain "hand")
  // 3ds Max Biped finger convention: "Bip01 L Finger0" / "Finger01" / "Finger02" .. "Finger42"
  if (tset.has('bip') && tset.has('finger')) {
    const fIdx = p.tokens.indexOf('finger');
    const numTok = p.tokens[fIdx + 1];
    if (numTok && /^\d+$/.test(numTok)) {
      const fingerDigit = parseInt(numTok[0], 10);
      const phalanxDigit = numTok.length > 1 ? parseInt(numTok.slice(1), 10) : 0;
      const fingerMap: FingerCanon[] = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
      const finger = fingerMap[fingerDigit];
      if (finger && phalanxDigit >= 0 && phalanxDigit <= 2) {
        return { kind: 'finger', finger, phalanxIndex: phalanxDigit as 0 | 1 | 2 };
      }
    }
  }

  // Palm / carpal / non-thumb metacarpal bones (e.g. DEF-palm.01.L, ORG-palm.01.L, index_metacarpal_l)
  if (tset.has('palm') || tset.has('carpal')) {
    return 'palmMetacarpal';
  }

  let finger: FingerCanon | null = null;
  if (tset.has('thumb') || tset.has('pollex')) finger = 'Thumb';
  else if (tset.has('index') || tset.has('idx')) finger = 'Index';
  else if (tset.has('middle') || tset.has('mid')) finger = 'Middle';
  else if (tset.has('ring')) finger = 'Ring';
  else if (tset.has('pinky') || tset.has('little') || tset.has('pinkie')) finger = 'Little';
  else if (tset.has('finger')) {
    // Rokoko / OpenXR style: Finger1..Finger5
    const fIdx = p.tokens.indexOf('finger');
    const nextNum = p.tokens[fIdx + 1];
    if (nextNum && /^[1-5]$/.test(nextNum)) {
      finger = FINGER_NAMES[parseInt(nextNum, 10) - 1];
    }
  }

  if (finger) {
    if (finger !== 'Thumb' && (tset.has('metacarpal') || tset.has('meta') || p.numbers.includes(0))) {
      // Non-thumb 0-indexed / metacarpal bone sits inside the palm before Proximal
      const hasZeroOnly =
        tset.has('metacarpal') ||
        tset.has('meta') ||
        p.tokens.some((tok) => tok === '0' || tok === '00');
      if (hasZeroOnly) return 'palmMetacarpal';
    }

    let phalanxIndex: 0 | 1 | 2 | null = null;
    if (finger === 'Thumb') {
      if (tset.has('metacarpal')) phalanxIndex = 0;
      else if (tset.has('proximal')) phalanxIndex = 0;
      else if (tset.has('intermediate') || tset.has('medial')) phalanxIndex = 1;
      else if (tset.has('distal')) phalanxIndex = 2;
    } else {
      if (tset.has('proximal')) phalanxIndex = 0;
      else if (tset.has('intermediate') || tset.has('medial')) phalanxIndex = 1;
      else if (tset.has('distal')) phalanxIndex = 2;
    }

    if (phalanxIndex === null) {
      // Find the finger segment number (01/1 -> 0, 02/2 -> 1, 03/3 -> 2), ignoring "bip 01" or "finger 1..5"
      const fingerTokIdx = p.tokens.findIndex((tok) =>
        ['thumb', 'pollex', 'index', 'idx', 'middle', 'mid', 'ring', 'pinky', 'little', 'pinkie'].includes(tok),
      );
      const candidateNums =
        fingerTokIdx >= 0
          ? p.tokens
              .slice(fingerTokIdx + 1)
              .filter((tok) => /^\d+$/.test(tok))
              .map((tok) => parseInt(tok, 10))
          : p.numbers;
      const segNum = candidateNums[0];
      if (segNum === 1) phalanxIndex = 0;
      else if (segNum === 2) phalanxIndex = 1;
      else if (segNum === 3) phalanxIndex = 2;
      else if (segNum !== undefined && segNum >= 4) return null; // fingertip 4th bone
    }

    return { kind: 'finger', finger, phalanxIndex };
  }

  // 2. Hand / Wrist
  if (tset.has('hand') || tset.has('wrist')) {
    return 'hand';
  }

  // 3. Forearm / LowerArm
  if (
    tset.has('forearm') ||
    tset.has('lowerarm') ||
    (tset.has('fore') && tset.has('arm')) ||
    (tset.has('lower') && tset.has('arm')) ||
    tset.has('ulna')
  ) {
    return 'lowerArm';
  }

  // 4. UpperArm / Arm
  if (
    tset.has('upperarm') ||
    (tset.has('upper') && tset.has('arm')) ||
    (tset.has('arm') && !tset.has('fore') && !tset.has('lower'))
  ) {
    return 'upperArm';
  }

  // 5. Shoulder / Clavicle
  if (tset.has('shoulder') || tset.has('clavicle') || tset.has('collar')) {
    return 'shoulder';
  }

  // 6. Central column (only when NOT lateral .L/.R; note Auto-Rig Pro uses ".x" for center)
  if (p.side === null) {
    if (tset.has('head') || tset.has('skull') || tset.has('cranium')) return 'head';
    if (tset.has('neck') || tset.has('cervical')) return 'neck';
    if (tset.has('chest') || tset.has('upperchest') || tset.has('thorax') || tset.has('ribcage')) {
      return 'chest';
    }
    if (tset.has('hips') || tset.has('pelvis') || tset.has('hip') || tset.has('cog')) {
      return 'hips';
    }
    if (tset.has('spine') || tset.has('waist') || tset.has('abdomen') || tset.has('back')) {
      return 'spine';
    }
  }

  return null;
}

/**
 * Pure string-list matcher compatible with existing unit tests (`matchGlbBoneNames(names)`),
 * upgraded to handle Blender Rigify (`DEF-*`), Rigify Meta-Rig (`spine..spine.006`, `upper_arm.L`),
 * GLTFLoader-sanitized Rigify names, Mixamo, Ready Player Me, Avaturn, Unreal, CC3/CC4, and Biped.
 */
export function matchGlbBoneNames(allNodeNames: string[]): Record<string, string> {
  interface ParsedItem {
    original: string;
    parsed: ParsedBoneName;
    role: LimbRole | null;
    score: number;
  }

  const hasAnyDef = allNodeNames.some((n) => /^def[-_]/i.test(n));

  const items: ParsedItem[] = allNodeNames.map((original) => {
    const parsed = parseBoneName(original);
    const role = classifyParsedBoneRole(parsed);
    let score = 100;
    if (parsed.isDef) score += 300;
    else if (hasAnyDef) score -= 150;
    if (parsed.isOrg) score -= 250;
    if (parsed.isMchOrControl) score -= 600;
    if (parsed.isTwist) score -= 180;
    return { original, parsed, role, score };
  });

  const matched: Record<string, string> = {};
  const used = new Set<string>();

  const pickBest = (pred: (it: ParsedItem) => boolean, minScore = -100): ParsedItem | null => {
    let best: ParsedItem | null = null;
    for (const it of items) {
      if (used.has(it.original) || it.score < minScore) continue;
      if (!pred(it)) continue;
      if (!best || it.score > best.score) best = it;
    }
    return best;
  };

  // Central spine/hips/chest/neck/head resolution (including Rigify spine..spine.006)
  const spineItems = items
    .filter(
      (it) =>
        it.score >= 0 &&
        !it.parsed.isMchOrControl &&
        (it.role === 'hips' ||
          it.role === 'spine' ||
          it.role === 'chest' ||
          it.role === 'neck' ||
          it.role === 'head'),
    )
    .sort((a, b) => {
      if (a.parsed.isDef !== b.parsed.isDef) return a.parsed.isDef ? -1 : 1;
      const na = a.parsed.numbers[0] ?? 0;
      const nb = b.parsed.numbers[0] ?? 0;
      return na - nb;
    });

  // Check if we have a Rigify-style pure `spine` .. `spine.006` numbered chain without explicit neck/head
  const explicitHead = pickBest((it) => it.role === 'head', 0);
  const explicitNeck = pickBest((it) => it.role === 'neck', 0);
  const explicitHips = pickBest((it) => it.role === 'hips', 0);

  const numberedSpines = spineItems.filter((it) => it.role === 'spine');
  const maxSpineNum = numberedSpines.reduce((m, it) => Math.max(m, it.parsed.numbers[0] ?? 0), -1);

  if (!explicitHead && numberedSpines.length >= 5 && maxSpineNum >= 5) {
    // Rigify / Meta-Rig 7-bone chain: [spine, .001, .002, .003, .004, .005, .006]
    const ordered = [...numberedSpines].sort(
      (a, b) => (a.parsed.numbers[0] ?? 0) - (b.parsed.numbers[0] ?? 0),
    );
    matched.hips = ordered[0].original;
    used.add(ordered[0].original);
    const chestItem = ordered.find((it) => (it.parsed.numbers[0] ?? 0) === 3) ?? ordered[Math.min(3, ordered.length - 3)];
    const spineItem = ordered.find((it) => (it.parsed.numbers[0] ?? 0) === 2) ?? ordered[1];
    const neckItem = ordered.find((it) => (it.parsed.numbers[0] ?? 0) === 4) ?? ordered[ordered.length - 2];
    const headItem = ordered[ordered.length - 1];
    if (spineItem) {
      matched.spine = spineItem.original;
      used.add(spineItem.original);
    }
    if (chestItem) {
      matched.chest = chestItem.original;
      used.add(chestItem.original);
    }
    if (neckItem) {
      matched.neck = neckItem.original;
      used.add(neckItem.original);
    }
    if (headItem) {
      matched.head = headItem.original;
      used.add(headItem.original);
    }
  } else {
    if (explicitHips) {
      matched.hips = explicitHips.original;
      used.add(explicitHips.original);
    } else if (numberedSpines.length >= 3) {
      matched.hips = numberedSpines[0].original;
      used.add(numberedSpines[0].original);
    }

    if (explicitHead) {
      matched.head = explicitHead.original;
      used.add(explicitHead.original);
    }
    if (explicitNeck) {
      matched.neck = explicitNeck.original;
      used.add(explicitNeck.original);
    }

    const explicitChest = pickBest((it) => it.role === 'chest', 0);
    const remainingSpines = numberedSpines.filter((it) => !used.has(it.original));
    if (explicitChest) {
      matched.chest = explicitChest.original;
      used.add(explicitChest.original);
      if (remainingSpines.length > 0) {
        matched.spine = remainingSpines[0].original;
        used.add(remainingSpines[0].original);
      }
    } else if (remainingSpines.length >= 2) {
      matched.spine = remainingSpines[0].original;
      matched.chest = remainingSpines[remainingSpines.length - 1].original;
      used.add(matched.spine);
      used.add(matched.chest);
    } else if (remainingSpines.length === 1) {
      matched.spine = remainingSpines[0].original;
      used.add(matched.spine);
    }
  }

  for (const side of ['right', 'left'] as const) {
    for (const [canonSuffix, roleName] of [
      ['Shoulder', 'shoulder'],
      ['UpperArm', 'upperArm'],
      ['LowerArm', 'lowerArm'],
      ['Hand', 'hand'],
    ] as const) {
      const hit = pickBest((it) => it.parsed.side === side && it.role === roleName && !it.parsed.isTwist, 0);
      if (hit) {
        matched[`${side}${canonSuffix}`] = hit.original;
        used.add(hit.original);
      }
    }

    for (const f of FINGER_NAMES) {
      const fCands = items.filter(
        (it) =>
          !used.has(it.original) &&
          it.score >= 0 &&
          it.parsed.side === side &&
          !it.parsed.isTwist &&
          typeof it.role === 'object' &&
          it.role !== null &&
          it.role.kind === 'finger' &&
          it.role.finger === f,
      );
      for (let pIdx = 0 as 0 | 1 | 2; pIdx < 3; pIdx = (pIdx + 1) as 0 | 1 | 2) {
        const hit = fCands.find(
          (it) =>
            !used.has(it.original) &&
            typeof it.role === 'object' &&
            it.role !== null &&
            it.role.phalanxIndex === pIdx,
        );
        if (hit) {
          matched[`${side}${f}${PHALANX_NAMES[pIdx]}`] = hit.original;
          used.add(hit.original);
        }
      }
    }
  }

  return matched;
}

// ---------------------------------------------------------------------------
// Universal Scene-Graph Skeleton Analyzer & Retargeter
// ---------------------------------------------------------------------------

export interface UniversalRigAnalysis {
  rigTarget: AvatarRigTarget;
  detectedFormat: string;
  matchedBonesCount: number;
  drivenBonesCount: number;
  hasFacialBoneRig: boolean;
  /** Deform bones suitable for a clean SkeletonHelper visualization */
  visualBones: THREE.Bone[];
}

interface CandidateNode {
  obj: THREE.Object3D;
  rawName: string;
  parsed: ParsedBoneName;
  role: LimbRole | null;
  weight: number;
  isWeighted: boolean;
  isSkeletonBone: boolean;
  worldPos: THREE.Vector3;
  score: number;
}

function collectSkinWeights(root: THREE.Object3D): {
  weightByBone: Map<THREE.Object3D, number>;
  skeletonBones: Set<THREE.Object3D>;
} {
  const weightByBone = new Map<THREE.Object3D, number>();
  const skeletonBones = new Set<THREE.Object3D>();

  root.traverse((obj) => {
    if ((obj as THREE.Bone).isBone) {
      skeletonBones.add(obj);
    }
    const sm = obj as THREE.SkinnedMesh;
    if (!sm.isSkinnedMesh || !sm.skeleton || !sm.geometry) return;
    const bones = sm.skeleton.bones;
    for (const b of bones) {
      if (b) skeletonBones.add(b);
    }
    const skinIndex = sm.geometry.getAttribute('skinIndex');
    const skinWeight = sm.geometry.getAttribute('skinWeight');
    if (!skinIndex || !skinWeight) return;

    const maxPerIndex = new Float32Array(bones.length);
    const count = Math.min(skinIndex.count, skinWeight.count);
    const itemSize = Math.min(4, skinIndex.itemSize);
    for (let v = 0; v < count; v++) {
      for (let c = 0; c < itemSize; c++) {
        const w = skinWeight.getComponent(v, c);
        if (w <= 1e-4) continue;
        const idx = skinIndex.getComponent(v, c);
        if (idx >= 0 && idx < bones.length && w > maxPerIndex[idx]) {
          maxPerIndex[idx] = w;
        }
      }
    }
    for (let j = 0; j < bones.length; j++) {
      const b = bones[j];
      if (!b || maxPerIndex[j] <= 1e-4) continue;
      const prev = weightByBone.get(b) ?? 0;
      if (maxPerIndex[j] > prev) weightByBone.set(b, maxPerIndex[j]);
    }
  });

  return { weightByBone, skeletonBones };
}

/**
 * Normalizes scale (cm -> m), grounds feet to y = 0, and auto-rotates -Z facing GLBs to +Z
 * BEFORE bind world matrices are sampled.
 */
export function normalizeAvatarRootTransform(root: THREE.Group, isVrm: boolean): void {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  if (!box.isEmpty()) {
    const size = box.getSize(new THREE.Vector3());
    if (size.y > 5.0) {
      const scaleFactor = 1.72 / size.y;
      root.scale.multiplyScalar(scaleFactor);
      root.updateMatrixWorld(true);
    } else if (size.y > 0.01 && size.y < 0.25) {
      const scaleFactor = 1.72 / size.y;
      root.scale.multiplyScalar(scaleFactor);
      root.updateMatrixWorld(true);
    }
    const groundedBox = new THREE.Box3().setFromObject(root);
    if (Math.abs(groundedBox.min.y) > 0.02) {
      root.position.y -= groundedBox.min.y;
      root.updateMatrixWorld(true);
    }
  }

  // VRM facing is already handled by VRMUtils.rotateVRM0(vrm)
  if (isVrm) return;

  // Detect -Z facing GLBs: in our +Y up, +Z forward right-handed frame, character LEFT is +X and RIGHT is -X.
  const leftXs: number[] = [];
  const rightXs: number[] = [];
  const tmp = new THREE.Vector3();
  root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) return;
    const raw = getRawNodeName(obj);
    if (!raw) return;
    const p = parseBoneName(raw);
    if (p.isMchOrControl || p.side === null) return;
    const role = classifyParsedBoneRole(p);
    if (role === 'shoulder' || role === 'upperArm' || role === 'lowerArm' || role === 'hand') {
      obj.getWorldPosition(tmp);
      if (p.side === 'left') leftXs.push(tmp.x);
      else rightXs.push(tmp.x);
    }
  });

  if (leftXs.length >= 1 && rightXs.length >= 1) {
    const avgLeftX = leftXs.reduce((s, x) => s + x, 0) / leftXs.length;
    const avgRightX = rightXs.reduce((s, x) => s + x, 0) / rightXs.length;
    if (avgLeftX - avgRightX < -0.06) {
      root.rotation.y += Math.PI;
      root.updateMatrixWorld(true);
    }
  }
}

function isDescendantOf(node: THREE.Object3D, ancestor: THREE.Object3D): boolean {
  let cur: THREE.Object3D | null = node.parent;
  while (cur) {
    if (cur === ancestor) return true;
    cur = cur.parent;
  }
  return false;
}

function chainBetween(ancestor: THREE.Object3D, descendant: THREE.Object3D): THREE.Object3D[] | null {
  const path: THREE.Object3D[] = [];
  let cur: THREE.Object3D | null = descendant;
  while (cur) {
    path.push(cur);
    if (cur === ancestor) {
      path.reverse();
      return path;
    }
    cur = cur.parent;
  }
  return null;
}

/** Rotation whose X/Y/Z columns are (f, s, n) with n = f × s, s = n × f. */
function frameQuaternion(f: THREE.Vector3, s: THREE.Vector3): THREE.Quaternion {
  const fx = f.clone().normalize();
  const nz = new THREE.Vector3().crossVectors(fx, s);
  if (nz.lengthSq() < 1e-10) {
    const fallback = Math.abs(fx.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    nz.crossVectors(fx, fallback);
  }
  nz.normalize();
  const sy = new THREE.Vector3().crossVectors(nz, fx).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(fx, sy, nz));
}

/** Topological fallback for unnamed / custom-named humanoid skeletons. */
function detectHumanoidTopologically(
  candidates: CandidateNode[],
  hasWeightedSkin: boolean,
): {
  hips: THREE.Object3D | null;
  spineColumn: THREE.Object3D[];
  arms: Record<
    Side,
    {
      shoulder: THREE.Object3D | null;
      upperArm: THREE.Object3D | null;
      lowerArm: THREE.Object3D | null;
      hand: THREE.Object3D | null;
      fingers: Partial<Record<FingerCanon, THREE.Object3D[]>>;
    }
  >;
} {
  const usable = candidates.filter((c) =>
    hasWeightedSkin ? c.isWeighted : c.isSkeletonBone && !c.parsed.isMchOrControl,
  );
  const usableSet = new Set(usable.map((c) => c.obj));
  const posOf = new Map<THREE.Object3D, THREE.Vector3>();
  for (const c of candidates) posOf.set(c.obj, c.worldPos);

  const emptyArms = (): {
    shoulder: THREE.Object3D | null;
    upperArm: THREE.Object3D | null;
    lowerArm: THREE.Object3D | null;
    hand: THREE.Object3D | null;
    fingers: Partial<Record<FingerCanon, THREE.Object3D[]>>;
  } => ({
    shoulder: null,
    upperArm: null,
    lowerArm: null,
    hand: null,
    fingers: {},
  });

  if (usable.length < 6) {
    return { hips: null, spineColumn: [], arms: { left: emptyArms(), right: emptyArms() } };
  }

  let minY = Infinity;
  let maxY = -Infinity;
  for (const c of usable) {
    minY = Math.min(minY, c.worldPos.y);
    maxY = Math.max(maxY, c.worldPos.y);
  }
  const height = Math.max(0.2, maxY - minY);

  // Find usable children inside the skeleton tree (skipping non-usable intermediate helpers up to depth 3)
  const usableChildren = (node: THREE.Object3D): THREE.Object3D[] => {
    const out: THREE.Object3D[] = [];
    const visit = (n: THREE.Object3D, depth: number) => {
      for (const ch of n.children) {
        if (usableSet.has(ch)) out.push(ch);
        else if (depth < 3) visit(ch, depth + 1);
      }
    };
    visit(node, 0);
    return out;
  };

  // Helper: walk upward along centered children from a candidate pelvis/root bone
  const traceUpwardSpine = (startObj: THREE.Object3D): THREE.Object3D[] => {
    const col: THREE.Object3D[] = [startObj];
    let curNode: THREE.Object3D = startObj;
    while (true) {
      const pCur = posOf.get(curNode)!;
      const nextCentered = usableChildren(curNode)
        .filter((ch) => {
          const p = posOf.get(ch)!;
          return Math.abs(p.x) < 0.08 * height && p.y > pCur.y + 0.005 * height;
        })
        .sort((a, b) => Math.abs(posOf.get(a)!.x) - Math.abs(posOf.get(b)!.x));
      if (nextCentered.length === 0) break;
      curNode = nextCentered[0];
      col.push(curNode);
    }
    return col;
  };

  // Find pelvis/hips: lowest centered bone (|x| < 0.08 * height) that launches an upward central spine column
  const centered = usable
    .filter(
      (c) =>
        Math.abs(c.worldPos.x) < 0.08 * height &&
        c.worldPos.y >= minY + 0.2 * height &&
        c.worldPos.y <= minY + 0.78 * height,
    )
    .sort((a, b) => {
      const xDiff = Math.abs(a.worldPos.x) - Math.abs(b.worldPos.x);
      if (Math.abs(xDiff) > 0.02 * height) return xDiff;
      return a.worldPos.y - b.worldPos.y;
    });

  let hips: THREE.Object3D | null = null;
  let spineColumn: THREE.Object3D[] = [];
  for (const cand of centered) {
    const col = traceUpwardSpine(cand.obj);
    const topY = posOf.get(col[col.length - 1])!.y;
    if (col.length >= 3 && topY >= minY + 0.72 * height) {
      if (!hips || cand.worldPos.y < posOf.get(hips)!.y) {
        hips = cand.obj;
        spineColumn = col;
      }
    }
  }
  if (!hips) {
    return { hips: null, spineColumn: [], arms: { left: emptyArms(), right: emptyArms() } };
  }

  // Find chest junction in spineColumn that branches laterally into left (+X) and right (-X) arm chains
  const spineSet = new Set(spineColumn);
  const arms = { left: emptyArms(), right: emptyArms() };

  const traceLinearChain = (start: THREE.Object3D): THREE.Object3D[] => {
    const chain: THREE.Object3D[] = [start];
    let node = start;
    while (true) {
      const kids = usableChildren(node);
      if (kids.length !== 1) break;
      chain.push(kids[0]);
      node = kids[0];
    }
    return chain;
  };

  let chestSpineIdx = -1;
  for (let i = spineColumn.length - 1; i >= 0; i--) {
    const sp = spineColumn[i];
    const spPos = posOf.get(sp)!;
    if (spPos.y < minY + 0.55 * height) continue;
    const lateralKids = usableChildren(sp).filter((ch) => !spineSet.has(ch));
    if (lateralKids.length < 2) continue;
    if (chestSpineIdx < 0) chestSpineIdx = i;

    for (const side of ['left', 'right'] as const) {
      if (arms[side].upperArm) continue;
      const sideSign = side === 'left' ? 1 : -1;
      const branchRoot = lateralKids
        .filter((ch) => posOf.get(ch)!.x * sideSign > 0.01 * height)
        .sort((a, b) => Math.abs(posOf.get(a)!.x) - Math.abs(posOf.get(b)!.x))[0];
      if (!branchRoot) continue;

      const armChain = traceLinearChain(branchRoot);
      if (armChain.length < 3) continue;
      const handNode = armChain[armChain.length - 1];
      const armBeforeHand = armChain.slice(0, -1);

      let shoulderNode: THREE.Object3D | null = null;
      let upperNode: THREE.Object3D | null = null;
      let lowerNode: THREE.Object3D | null = null;
      if (armBeforeHand.length >= 3) {
        shoulderNode = armBeforeHand[0];
        upperNode = armBeforeHand[1];
        lowerNode = armBeforeHand[armBeforeHand.length - 1];
      } else {
        upperNode = armBeforeHand[0];
        lowerNode = armBeforeHand[1];
      }

      arms[side].shoulder = shoulderNode;
      arms[side].upperArm = upperNode;
      arms[side].lowerArm = lowerNode;
      arms[side].hand = handNode;

      // Check for 4..5 finger branches under handNode
      const fingerRoots = usableChildren(handNode);
      if (fingerRoots.length >= 4) {
        const handPos = posOf.get(handNode)!;
        const fingerChains = fingerRoots
          .map((fr) => traceLinearChain(fr).slice(0, 3))
          .filter((fc) => fc.length >= 2);

        if (fingerChains.length >= 4) {
          // Thumb proximal base sits closest to the wrist joint
          fingerChains.sort(
            (a, b) => posOf.get(a[0])!.distanceTo(handPos) - posOf.get(b[0])!.distanceTo(handPos),
          );
          const thumbChain = fingerChains[0];
          const otherFour = fingerChains.slice(1, 5);
          arms[side].fingers.Thumb = thumbChain;

          const palmCenter = new THREE.Vector3();
          for (const fc of otherFour) palmCenter.add(posOf.get(fc[0])!);
          palmCenter.multiplyScalar(1 / otherFour.length);

          const radialAxis = posOf.get(thumbChain[0])!.clone().sub(palmCenter).normalize();
          otherFour.sort(
            (a, b) =>
              posOf.get(b[0])!.clone().sub(palmCenter).dot(radialAxis) -
              posOf.get(a[0])!.clone().sub(palmCenter).dot(radialAxis),
          );
          const nonThumbNames: FingerCanon[] = ['Index', 'Middle', 'Ring', 'Little'];
          otherFour.forEach((fc, idx) => {
            if (idx < nonThumbNames.length) {
              arms[side].fingers[nonThumbNames[idx]] = fc;
            }
          });
        }
      }
    }
  }

  // Drop leaf skull-tip end bone (e.g. HeadTop_End) when >= 3 nodes sit above the chest hub
  if (
    chestSpineIdx >= 0 &&
    spineColumn.length - 1 - chestSpineIdx >= 3 &&
    usableChildren(spineColumn[spineColumn.length - 1]).length === 0
  ) {
    spineColumn.pop();
  }

  return { hips, spineColumn, arms };
}

/**
 * Builds bone-driven facial animation tracks (jaw rotation + eyelid/brow/cheek/lip translations)
 * for rigs without facial morph targets (or missing specific morph channels), such as Blender Rigify.
 */
function createFacialBoneTrackBuilder(
  root: THREE.Object3D,
  headNode: THREE.Object3D | null,
  coveredMorphNames: Set<string>,
  hasWeightedSkin: boolean,
  weightByBone: Map<THREE.Object3D, number>,
  trackKeyOf: (obj: THREE.Object3D) => string,
  modelHeight: number,
): {
  hasFacialBoneRig: boolean;
  buildFacialBoneTracks: (
    morphTracks: Record<string, number[]>,
    times: number[],
  ) => THREE.KeyframeTrack[];
} {
  const heightScale = Math.max(0.5, Math.min(2.0, modelHeight / 1.72));

  // Search nodes by raw glTF name
  const byRawName = new Map<string, THREE.Object3D>();
  const allObjects: THREE.Object3D[] = [];
  root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) return;
    allObjects.push(obj);
    const raw = getRawNodeName(obj);
    if (raw && !byRawName.has(raw)) byRawName.set(raw, obj);
  });

  // 1. Jaw hinge candidates (for mouthOpen)
  interface JawDriver {
    node: THREE.Object3D;
    gain: number;
    bindLocalQuat: THREE.Quaternion;
    localHingeAxis: THREE.Vector3;
  }
  const jawDrivers: JawDriver[] = [];

  const addJawDriver = (node: THREE.Object3D, gain: number) => {
    const bindWorldQuat = node.getWorldQuaternion(new THREE.Quaternion());
    // Head rest +X = (1, 0, 0); positive rotation around +X swings forward-down chin downward & back
    const localHingeAxis = new THREE.Vector3(1, 0, 0)
      .applyQuaternion(bindWorldQuat.clone().invert())
      .normalize();
    jawDrivers.push({
      node,
      gain,
      bindLocalQuat: node.quaternion.clone(),
      localHingeAxis,
    });
  };

  if (!coveredMorphNames.has('mouthOpen')) {
    const rigifyJaw = byRawName.get('jaw_master') ?? byRawName.get('ORG-jaw_master');
    if (rigifyJaw) {
      addJawDriver(rigifyJaw, 1.0);
      const jawBottom = byRawName.get('MCH-jaw_master_bottom');
      if (jawBottom && !isDescendantOf(jawBottom, rigifyJaw)) addJawDriver(jawBottom, 1.0);
      const jawMiddle = byRawName.get('MCH-jaw_master_middle');
      if (jawMiddle && !isDescendantOf(jawMiddle, rigifyJaw)) addJawDriver(jawMiddle, 0.45);
    } else {
      // Generic jaw bone (e.g. Jaw, CC_Base_JawRoot, DEF-jaw)
      const headY = headNode ? headNode.getWorldPosition(new THREE.Vector3()).y : -Infinity;
      const genericJaw = allObjects.find((obj) => {
        if (hasWeightedSkin && !weightByBone.has(obj)) return false;
        const p = parseBoneName(getRawNodeName(obj));
        if (p.side !== null || p.isMchOrControl) return false;
        if (Number.isFinite(headY)) {
          const jy = obj.getWorldPosition(new THREE.Vector3()).y;
          if (jy < headY - 0.18 * heightScale) return false;
        }
        return p.tokens.includes('jaw') || p.tokens.includes('jawroot');
      });
      if (genericJaw) addJawDriver(genericJaw, 1.0);
    }
  }

  // 2. Translational deform bone helpers (eyelids, brows, cheeks/lips)
  interface PosInfluence {
    morph: string;
    worldDelta: THREE.Vector3; // in meters at 1.72m reference height, in rest +X left / +Y up / +Z forward
  }
  interface PosDriver {
    node: THREE.Object3D;
    bindLocalPos: THREE.Vector3;
    parentBindWorldQuatInv: THREE.Quaternion;
    parentBindWorldScale: THREE.Vector3;
    influences: PosInfluence[];
  }
  const posDrivers: PosDriver[] = [];

  const registerPosDriver = (node: THREE.Object3D, influences: PosInfluence[]) => {
    const active = influences.filter((inf) => !coveredMorphNames.has(inf.morph));
    if (active.length === 0 || !node.parent) return;
    const parentQuat = node.parent.getWorldQuaternion(new THREE.Quaternion());
    const parentScale = node.parent.getWorldScale(new THREE.Vector3());
    if (Math.abs(parentScale.x) < 1e-6) parentScale.x = 1;
    if (Math.abs(parentScale.y) < 1e-6) parentScale.y = 1;
    if (Math.abs(parentScale.z) < 1e-6) parentScale.z = 1;
    posDrivers.push({
      node,
      bindLocalPos: node.position.clone(),
      parentBindWorldQuatInv: parentQuat.invert(),
      parentBindWorldScale: parentScale,
      influences: active,
    });
  };

  // Rigify & generic eyelid roots
  for (const [sideSuffix, morphName] of [
    ['L', 'eyeBlinkLeft'],
    ['R', 'eyeBlinkRight'],
  ] as const) {
    if (coveredMorphNames.has(morphName)) continue;
    const upperLid = byRawName.get(`DEF-lid.T.${sideSuffix}`);
    const lowerLid = byRawName.get(`DEF-lid.B.${sideSuffix}`);
    if (upperLid) {
      registerPosDriver(upperLid, [
        { morph: morphName, worldDelta: new THREE.Vector3(0, -0.0085 * heightScale, 0.0015 * heightScale) },
      ]);
    }
    if (lowerLid) {
      registerPosDriver(lowerLid, [
        { morph: morphName, worldDelta: new THREE.Vector3(0, 0.0025 * heightScale, 0.0008 * heightScale) },
      ]);
    }
  }

  // Rigify brow roots (skip .002 since DEF-brow.T.L.002 is a child of DEF-brow.T.L.001)
  if (!coveredMorphNames.has('browUp') || !coveredMorphNames.has('browDown')) {
    const browCandidates = allObjects.filter((obj) => {
      if (hasWeightedSkin && !weightByBone.has(obj)) return false;
      const raw = getRawNodeName(obj);
      return /^DEF-brow\.T\.(L|R)(\.\d+)?$/.test(raw);
    });
    const browSet = new Set(browCandidates);
    for (const brow of browCandidates) {
      if (brow.parent && browSet.has(brow.parent)) continue; // avoid double-translating child segment
      const raw = getRawNodeName(brow);
      const sideSign = raw.includes('.L') ? 1 : -1;
      registerPosDriver(brow, [
        { morph: 'browUp', worldDelta: new THREE.Vector3(0, 0.0075 * heightScale, 0.001 * heightScale) },
        {
          morph: 'browDown',
          worldDelta: new THREE.Vector3(-0.002 * sideSign * heightScale, -0.0055 * heightScale, 0.001 * heightScale),
        },
      ]);
    }
  }

  const isDrivenByJaw = (node: THREE.Object3D): boolean =>
    jawDrivers.some((jd) => node === jd.node || isDescendantOf(node, jd.node));

  // If Rigify DEF-chin / lower teeth are parented to ORG-face via unexported Blender constraints
  // rather than parented under jaw_master, drive them directly on mouthOpen so the weighted chin/jaw mesh opens!
  if (!coveredMorphNames.has('mouthOpen')) {
    const defChin = byRawName.get('DEF-chin');
    if (defChin && !isDrivenByJaw(defChin)) {
      registerPosDriver(defChin, [
        {
          morph: 'mouthOpen',
          worldDelta: new THREE.Vector3(0, -0.014 * heightScale, -0.003 * heightScale),
        },
      ]);
    }
    const lowerTeeth =
      byRawName.get('DEF-teeth.B') ?? byRawName.get('ORG-teeth.B') ?? byRawName.get('teeth.B');
    if (lowerTeeth && !isDrivenByJaw(lowerTeeth)) {
      registerPosDriver(lowerTeeth, [
        {
          morph: 'mouthOpen',
          worldDelta: new THREE.Vector3(0, -0.01 * heightScale, -0.002 * heightScale),
        },
      ]);
    }
  }

  // Rigify mouth corners / lower cheeks & lips
  for (const [sideSuffix, sideSign] of [
    ['L', 1],
    ['R', -1],
  ] as const) {
    const cheekB = byRawName.get(`DEF-cheek.B.${sideSuffix}`);
    if (cheekB) {
      registerPosDriver(cheekB, [
        {
          morph: 'mouthSmile',
          worldDelta: new THREE.Vector3(0.0055 * sideSign * heightScale, 0.006 * heightScale, -0.002 * heightScale),
        },
        {
          morph: 'mouthFrown',
          worldDelta: new THREE.Vector3(-0.0015 * sideSign * heightScale, -0.005 * heightScale, 0),
        },
        {
          morph: 'mouthPucker',
          worldDelta: new THREE.Vector3(-0.0055 * sideSign * heightScale, 0, 0.0045 * heightScale),
        },
        {
          morph: 'cheekPuff',
          worldDelta: new THREE.Vector3(0.0065 * sideSign * heightScale, 0.001 * heightScale, 0.0055 * heightScale),
        },
      ]);
    }
    for (const tb of ['T', 'B'] as const) {
      const lip = byRawName.get(`DEF-lip.${tb}.${sideSuffix}`);
      if (lip) {
        const lipInfluences: PosInfluence[] = [
          {
            morph: 'mouthPucker',
            worldDelta: new THREE.Vector3(-0.0025 * sideSign * heightScale, 0, 0.005 * heightScale),
          },
          {
            morph: 'mouthSmile',
            worldDelta: new THREE.Vector3(0.002 * sideSign * heightScale, 0.0015 * heightScale, -0.001 * heightScale),
          },
        ];
        if (tb === 'B' && !isDrivenByJaw(lip)) {
          lipInfluences.push({
            morph: 'mouthOpen',
            worldDelta: new THREE.Vector3(0, -0.01 * heightScale, -0.002 * heightScale),
          });
        }
        registerPosDriver(lip, lipInfluences);
      }
    }
  }

  const hasFacialBoneRig = jawDrivers.length > 0 || posDrivers.length > 0;

  const buildFacialBoneTracks = (
    morphTracks: Record<string, number[]>,
    times: number[],
  ): THREE.KeyframeTrack[] => {
    const nFrames = times.length;
    if (!hasFacialBoneRig || nFrames === 0) return [];
    const outTracks: THREE.KeyframeTrack[] = [];

    // Jaw quaternion tracks
    const mouthOpen = morphTracks.mouthOpen;
    if (jawDrivers.length > 0 && mouthOpen && mouthOpen.length === nFrames) {
      const hasNonZero = mouthOpen.some((w) => Math.abs(w) > 1e-4);
      if (hasNonZero) {
        const qDelta = new THREE.Quaternion();
        const qOut = new THREE.Quaternion();
        for (const jd of jawDrivers) {
          const buf: number[] = [];
          for (let f = 0; f < nFrames; f++) {
            const w = Math.max(0, Math.min(1, mouthOpen[f]));
            qDelta.setFromAxisAngle(jd.localHingeAxis, 0.34 * jd.gain * w);
            qOut.copy(jd.bindLocalQuat).multiply(qDelta).normalize();
            if (buf.length >= 4) {
              const dot =
                buf[buf.length - 4] * qOut.x +
                buf[buf.length - 3] * qOut.y +
                buf[buf.length - 2] * qOut.z +
                buf[buf.length - 1] * qOut.w;
              if (dot < 0) qOut.set(-qOut.x, -qOut.y, -qOut.z, -qOut.w);
            }
            buf.push(qOut.x, qOut.y, qOut.z, qOut.w);
          }
          outTracks.push(
            new THREE.QuaternionKeyframeTrack(`${trackKeyOf(jd.node)}.quaternion`, times, buf),
          );
        }
      }
    }

    // Eyelid / brow / cheek / lip position tracks
    const worldOffset = new THREE.Vector3();
    for (const pd of posDrivers) {
      const activeInfs = pd.influences.filter((inf) => {
        const s = morphTracks[inf.morph];
        return s && s.length === nFrames && s.some((w) => Math.abs(w) > 1e-4);
      });
      if (activeInfs.length === 0) continue;

      const buf: number[] = [];
      for (let f = 0; f < nFrames; f++) {
        worldOffset.set(0, 0, 0);
        for (const inf of activeInfs) {
          const w = Math.max(0, Math.min(1, morphTracks[inf.morph][f]));
          if (w > 0) worldOffset.addScaledVector(inf.worldDelta, w);
        }
        worldOffset.applyQuaternion(pd.parentBindWorldQuatInv);
        buf.push(
          pd.bindLocalPos.x + worldOffset.x / pd.parentBindWorldScale.x,
          pd.bindLocalPos.y + worldOffset.y / pd.parentBindWorldScale.y,
          pd.bindLocalPos.z + worldOffset.z / pd.parentBindWorldScale.z,
        );
      }
      outTracks.push(new THREE.VectorKeyframeTrack(`${trackKeyOf(pd.node)}.position`, times, buf));
    }

    return outTracks;
  };

  return { hasFacialBoneRig, buildFacialBoneTracks };
}

/**
 * Analyzes ANY loaded Three.js GLB / GLTF / VRM scene graph and builds an exact
 * multi-bone, bind-pose-compensated, hierarchy-aware `AvatarRigTarget`.
 */
export function analyzeUniversalAvatarRig(
  root: THREE.Group,
  vrm: VRM | null = null,
): UniversalRigAnalysis {
  root.updateMatrixWorld(true);

  // Build unique PropertyBinding-safe track identifiers for every Object3D
  const nameCounts = new Map<string, number>();
  const allSceneObjects: THREE.Object3D[] = [];
  root.traverse((obj) => {
    allSceneObjects.push(obj);
    if (obj.name) {
      nameCounts.set(obj.name, (nameCounts.get(obj.name) ?? 0) + 1);
    }
  });
  const trackKeyOf = (obj: THREE.Object3D): string => {
    if (
      obj.name &&
      nameCounts.get(obj.name) === 1 &&
      !RESERVED_TRACK_CHARS.test(obj.name)
    ) {
      return obj.name;
    }
    return obj.uuid;
  };

  // Collect skin vertex weights + skeleton bones + morph target meshes
  const { weightByBone, skeletonBones } = collectSkinWeights(root);
  const hasWeightedSkin = weightByBone.size >= 8;

  const morphTargetMeshes: MorphTargetMeshBinding[] = [];
  let primaryFaceMeshName: string | null = null;
  let primaryMorphCount = 0;
  const availableMorphs = new Set<string>();

  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (mesh.morphTargetDictionary) {
      const keys = Object.keys(mesh.morphTargetDictionary);
      if (keys.length > 0) {
        const meshKey = trackKeyOf(mesh);
        morphTargetMeshes.push({
          meshName: meshKey,
          availableMorphs: new Set(keys),
        });
        keys.forEach((k) => availableMorphs.add(k));
        if (keys.length > primaryMorphCount) {
          primaryMorphCount = keys.length;
          primaryFaceMeshName = meshKey;
        }
      }
    }
  });

  // Build candidate bone descriptors
  const candidates: CandidateNode[] = [];
  const candidateByObj = new Map<THREE.Object3D, CandidateNode>();
  let minY = Infinity;
  let maxY = -Infinity;

  for (const obj of allSceneObjects) {
    if (obj === root) continue;
    if ((obj as THREE.Mesh).isMesh || (obj as THREE.Light).isLight || (obj as THREE.Camera).isCamera) {
      continue;
    }
    const rawName = getRawNodeName(obj);
    const parsed = parseBoneName(rawName);
    const role = classifyParsedBoneRole(parsed);
    const weight = weightByBone.get(obj) ?? 0;
    const isWeighted = weight > 1e-4;
    const isSkeletonBone = skeletonBones.has(obj) || Boolean((obj as THREE.Bone).isBone);
    const worldPos = obj.getWorldPosition(new THREE.Vector3());

    if (isWeighted || isSkeletonBone) {
      minY = Math.min(minY, worldPos.y);
      maxY = Math.max(maxY, worldPos.y);
    }

    let score = 0;
    if (isWeighted) score += 1000;
    else if (hasWeightedSkin) score -= 500;
    if (parsed.isDef) score += 250;
    if (isSkeletonBone) score += 80;
    if (parsed.isOrg) score -= 400;
    if (parsed.isMchOrControl) score -= 900;
    if (parsed.isTwist) score -= 220;

    // Confirm side via world X when name lacked side on bilateral roles
    if (
      parsed.side === null &&
      (role === 'shoulder' || role === 'upperArm' || role === 'lowerArm' || role === 'hand')
    ) {
      if (worldPos.x > 0.03) parsed.side = 'left';
      else if (worldPos.x < -0.03) parsed.side = 'right';
    }

    const cand: CandidateNode = {
      obj,
      rawName,
      parsed,
      role,
      weight,
      isWeighted,
      isSkeletonBone,
      worldPos,
      score,
    };
    candidates.push(cand);
    candidateByObj.set(obj, cand);
  }

  const modelHeight = Number.isFinite(maxY - minY) && maxY > minY ? maxY - minY : 1.72;

  // Detect rig format label
  let detectedFormat = 'GLB Humanoid';
  const rawNamesLower = candidates.map((c) => c.rawName.toLowerCase());
  if (vrm?.humanoid) {
    detectedFormat = 'VRM Humanoid';
  } else if (candidates.some((c) => c.parsed.isDef) && rawNamesLower.some((n) => n.includes('org-spine'))) {
    detectedFormat = 'Blender Rigify GLB';
  } else if (rawNamesLower.some((n) => n.startsWith('f_index') || n === 'spine.006' || n === 'spine006')) {
    detectedFormat = 'Blender Meta-Rig GLB';
  } else if (rawNamesLower.some((n) => n.includes('mixamorig'))) {
    detectedFormat = 'Mixamo GLB';
  } else if (rawNamesLower.some((n) => n.includes('cc_base_'))) {
    detectedFormat = 'Character Creator GLB';
  } else if (rawNamesLower.some((n) => n.includes('clavicle_l') || n.includes('upperarm_l'))) {
    detectedFormat = 'Unreal Humanoid GLB';
  } else if (rawNamesLower.some((n) => n.includes('wolf3d') || n.includes('avaturn') || n.includes('rightforearm'))) {
    detectedFormat = 'Ready Player Me / Avaturn GLB';
  }

  // Canonical bone node map + multi-bone chain structures
  const nodeByCanon: Record<string, THREE.Object3D> = {};
  let hipsNode: THREE.Object3D | null = null;
  let torsoChain: THREE.Object3D[] = [];
  let neckChain: THREE.Object3D[] = [];
  let headNode: THREE.Object3D | null = null;
  const upperArmSplits: Record<Side, THREE.Object3D[]> = { left: [], right: [] };
  const lowerArmSplits: Record<Side, THREE.Object3D[]> = { left: [], right: [] };

  if (vrm?.humanoid) {
    for (const [canonName, vrmBone] of Object.entries(CANON_TO_VRM_BONE)) {
      const node =
        vrm.humanoid.getNormalizedBoneNode(vrmBone) ?? vrm.humanoid.getRawBoneNode(vrmBone);
      if (node) nodeByCanon[canonName] = node;
    }
    hipsNode = nodeByCanon.hips ?? null;
    headNode = nodeByCanon.head ?? null;
    const upperChest =
      vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.UpperChest) ??
      vrm.humanoid.getRawBoneNode(VRMHumanBoneName.UpperChest);
    torsoChain = [nodeByCanon.spine, nodeByCanon.chest, upperChest].filter(
      (n, idx, arr): n is THREE.Object3D => !!n && arr.indexOf(n) === idx,
    );
    neckChain = nodeByCanon.neck ? [nodeByCanon.neck] : [];
  } else {
    const usedObjs = new Set<THREE.Object3D>();
    const pickBestNode = (
      pred: (c: CandidateNode) => boolean,
      allowUnweightedFallback = false,
    ): CandidateNode | null => {
      let best: CandidateNode | null = null;
      for (const c of candidates) {
        if (usedObjs.has(c.obj) || c.parsed.isMchOrControl) continue;
        if (hasWeightedSkin && !c.isWeighted && !allowUnweightedFallback) continue;
        if (!pred(c)) continue;
        if (!best || c.score > best.score) best = c;
      }
      return best;
    };

    // 1. Match bilateral shoulders, arms, hands, and fingers first so shoulder height is known!
    for (const side of ['right', 'left'] as const) {
      const shoulder = pickBestNode((c) => c.parsed.side === side && c.role === 'shoulder' && !c.parsed.isTwist);
      const upperArm = pickBestNode((c) => c.parsed.side === side && c.role === 'upperArm' && !c.parsed.isTwist);
      const lowerArm = pickBestNode((c) => c.parsed.side === side && c.role === 'lowerArm' && !c.parsed.isTwist);
      const hand = pickBestNode((c) => c.parsed.side === side && c.role === 'hand' && !c.parsed.isTwist);

      if (shoulder) {
        nodeByCanon[`${side}Shoulder`] = shoulder.obj;
        usedObjs.add(shoulder.obj);
      }
      if (upperArm) {
        nodeByCanon[`${side}UpperArm`] = upperArm.obj;
        usedObjs.add(upperArm.obj);
      }
      if (lowerArm) {
        nodeByCanon[`${side}LowerArm`] = lowerArm.obj;
        usedObjs.add(lowerArm.obj);
      }
      if (hand) {
        nodeByCanon[`${side}Hand`] = hand.obj;
        usedObjs.add(hand.obj);
      }

      // Detect in-series split twist bones along upperArm -> lowerArm and lowerArm -> hand
      if (upperArm && lowerArm) {
        const seg = chainBetween(upperArm.obj, lowerArm.obj);
        if (seg && seg.length > 2) {
          upperArmSplits[side] = seg.slice(1, -1).filter((o) => {
            const c = candidateByObj.get(o);
            return c ? !c.parsed.isMchOrControl && (c.isWeighted || !hasWeightedSkin) : false;
          });
          upperArmSplits[side].forEach((o) => usedObjs.add(o));
        }
      }
      if (lowerArm && hand) {
        const seg = chainBetween(lowerArm.obj, hand.obj);
        if (seg && seg.length > 2) {
          lowerArmSplits[side] = seg.slice(1, -1).filter((o) => {
            const c = candidateByObj.get(o);
            return c ? !c.parsed.isMchOrControl && (c.isWeighted || !hasWeightedSkin) : false;
          });
          lowerArmSplits[side].forEach((o) => usedObjs.add(o));
        }
      }

      // Match 5 fingers x 3 phalanges (allowing unweighted distal phalanx if descendant of hand)
      for (const f of FINGER_NAMES) {
        const fingerCands = candidates.filter((c) => {
          if (usedObjs.has(c.obj) || c.parsed.isMchOrControl || c.parsed.isTwist) return false;
          if (c.parsed.side !== side) return false;
          if (hand && !isDescendantOf(c.obj, hand.obj)) return false;
          if (hasWeightedSkin && !c.isWeighted && !c.isSkeletonBone) return false;
          if (!c.role || typeof c.role !== 'object' || c.role.kind !== 'finger') return false;
          return c.role.finger === f;
        });

        // Prefer weighted / DEF- bones if duplicate ORG-/DEF- exist
        fingerCands.sort((a, b) => {
          if (b.score !== a.score) return b.score - a.score;
          const pa = typeof a.role === 'object' && a.role ? (a.role.phalanxIndex ?? 9) : 9;
          const pb = typeof b.role === 'object' && b.role ? (b.role.phalanxIndex ?? 9) : 9;
          return pa - pb;
        });

        const chosenByIdx: Array<CandidateNode | undefined> = [undefined, undefined, undefined];
        for (const fc of fingerCands) {
          if (typeof fc.role !== 'object' || !fc.role) continue;
          const idx = fc.role.phalanxIndex;
          if (idx !== null && chosenByIdx[idx] === undefined) {
            // Avoid unweighted ORG- finger duplicates when weighted skin exists
            if (hasWeightedSkin && !fc.isWeighted && fc.parsed.isOrg) continue;
            chosenByIdx[idx] = fc;
          }
        }

        // If phalanx indices weren't parsed from numbers, sort topologically from hand -> tip
        if (!chosenByIdx[0] && fingerCands.length >= 2) {
          const filtered = hasWeightedSkin ? fingerCands.filter((c) => c.isWeighted) : fingerCands;
          filtered.sort((a, b) => {
            if (isDescendantOf(b.obj, a.obj)) return -1;
            if (isDescendantOf(a.obj, b.obj)) return 1;
            return 0;
          });
          for (let i = 0; i < Math.min(3, filtered.length); i++) chosenByIdx[i] = filtered[i];
        }

        for (let i = 0; i < 3; i++) {
          const hit = chosenByIdx[i];
          if (hit) {
            nodeByCanon[`${side}${f}${PHALANX_NAMES[i]}`] = hit.obj;
            usedObjs.add(hit.obj);
          }
        }
      }
    }

    // 2. Match & partition the central spine / torso / neck / head column
    const centralPool = candidates
      .filter((c) => {
        if (usedObjs.has(c.obj) || c.parsed.isMchOrControl || c.parsed.side !== null) return false;
        if (hasWeightedSkin && !c.isWeighted) return false;
        return (
          c.role === 'hips' ||
          c.role === 'spine' ||
          c.role === 'chest' ||
          c.role === 'neck' ||
          c.role === 'head'
        );
      })
      .sort((a, b) => a.worldPos.y - b.worldPos.y);

    if (centralPool.length >= 2) {
      const explicitHips = centralPool.find((c) => c.role === 'hips') ?? centralPool[0];
      const explicitHead =
        [...centralPool].reverse().find((c) => c.role === 'head') ??
        centralPool[centralPool.length - 1];

      hipsNode = explicitHips.obj;
      headNode = explicitHead.obj !== hipsNode ? explicitHead.obj : null;

      // Build ordered spine path from hipsNode to headNode
      let middleNodes: CandidateNode[] = [];
      if (hipsNode && headNode) {
        const treePath = chainBetween(hipsNode, headNode);
        if (treePath && treePath.length >= 2) {
          middleNodes = treePath
            .slice(1, -1)
            .map((o) => candidateByObj.get(o))
            .filter((c): c is CandidateNode => !!c && (!hasWeightedSkin || c.isWeighted));
        } else {
          middleNodes = centralPool.filter((c) => c.obj !== hipsNode && c.obj !== headNode);
        }
      } else {
        middleNodes = centralPool.filter((c) => c.obj !== hipsNode);
      }

      // Compute shoulder junction reference height & check topological shoulder parent
      const shoulderAncestors = [
        nodeByCanon.leftShoulder,
        nodeByCanon.rightShoulder,
        nodeByCanon.leftUpperArm,
        nodeByCanon.rightUpperArm,
      ].filter((n): n is THREE.Object3D => !!n);

      let shoulderRefY = -Infinity;
      for (const sa of shoulderAncestors) {
        const c = candidateByObj.get(sa);
        if (c) shoulderRefY = Math.max(shoulderRefY, c.worldPos.y);
      }

      // Highest central node that is a scene-graph ancestor of the shoulders/arms (if attached directly)
      let highestArmParentIdx = -1;
      for (let i = 0; i < middleNodes.length; i++) {
        const mn = middleNodes[i].obj;
        if (shoulderAncestors.some((sa) => isDescendantOf(sa, mn))) {
          highestArmParentIdx = i;
        }
      }

      for (let i = 0; i < middleNodes.length; i++) {
        const mn = middleNodes[i];
        if (mn.role === 'neck') {
          neckChain.push(mn.obj);
        } else if (mn.role === 'chest') {
          torsoChain.push(mn.obj);
        } else if (highestArmParentIdx >= 0) {
          if (i <= highestArmParentIdx) torsoChain.push(mn.obj);
          else neckChain.push(mn.obj);
        } else if (
          Number.isFinite(shoulderRefY) &&
          mn.worldPos.y > shoulderRefY + 0.006 * modelHeight
        ) {
          // Rigify DEF-spine.004 & DEF-spine.005 sit above shoulder height -> neck chain!
          neckChain.push(mn.obj);
        } else {
          torsoChain.push(mn.obj);
        }
      }
    }

    // 3. Pure topological fallback if semantic names matched fewer than 5 core upper-body bones
    const semanticCoreCount =
      (hipsNode ? 1 : 0) +
      torsoChain.length +
      (headNode ? 1 : 0) +
      (nodeByCanon.leftUpperArm ? 1 : 0) +
      (nodeByCanon.rightUpperArm ? 1 : 0);

    if (semanticCoreCount < 5) {
      const topo = detectHumanoidTopologically(candidates, hasWeightedSkin);
      if (topo.hips && topo.spineColumn.length >= 3) {
        detectedFormat = 'Topological Humanoid GLB';
        hipsNode = topo.hips;
        headNode = topo.spineColumn[topo.spineColumn.length - 1];
        const midCol = topo.spineColumn.slice(1, -1);

        const armY = Math.max(
          topo.arms.left.upperArm ? candidateByObj.get(topo.arms.left.upperArm)?.worldPos.y ?? -Infinity : -Infinity,
          topo.arms.right.upperArm ? candidateByObj.get(topo.arms.right.upperArm)?.worldPos.y ?? -Infinity : -Infinity,
        );
        torsoChain = [];
        neckChain = [];
        for (const m of midCol) {
          const y = candidateByObj.get(m)?.worldPos.y ?? 0;
          if (Number.isFinite(armY) && y > armY + 0.01 * modelHeight) neckChain.push(m);
          else torsoChain.push(m);
        }

        for (const side of ['left', 'right'] as const) {
          const ta = topo.arms[side];
          if (ta.shoulder) nodeByCanon[`${side}Shoulder`] = ta.shoulder;
          if (ta.upperArm) nodeByCanon[`${side}UpperArm`] = ta.upperArm;
          if (ta.lowerArm) nodeByCanon[`${side}LowerArm`] = ta.lowerArm;
          if (ta.hand) nodeByCanon[`${side}Hand`] = ta.hand;
          for (const f of FINGER_NAMES) {
            const fc = ta.fingers[f];
            if (!fc) continue;
            for (let i = 0; i < Math.min(3, fc.length); i++) {
              nodeByCanon[`${side}${f}${PHALANX_NAMES[i]}`] = fc[i];
            }
          }
        }
      }
    }

    // Populate representative canonical bone entries for spine/chest/neck/head/hips
    if (hipsNode) nodeByCanon.hips = hipsNode;
    if (torsoChain.length === 1) {
      nodeByCanon.spine = torsoChain[0];
    } else if (torsoChain.length >= 2) {
      nodeByCanon.spine = torsoChain[0];
      nodeByCanon.chest = torsoChain[torsoChain.length - 1];
    }
    if (neckChain.length >= 1) {
      nodeByCanon.neck = neckChain[0];
    }
    if (headNode) nodeByCanon.head = headNode;
  }

  // ---------------------------------------------------------------------------
  // Rigify / Disconnected Parallel Hierarchy Detection & Pivot Co-location
  // ---------------------------------------------------------------------------
  // In exported Blender Rigify GLBs, DEF-spine..006 sit under `root`, while arms and ORG-face
  // sit under `MCH-torso.parent -> torso -> spine_fk.002 -> spine_fk.003 -> ORG-spine.004..006`.
  const downstreamRoots: THREE.Object3D[] = [
    nodeByCanon.leftShoulder,
    nodeByCanon.leftUpperArm,
    nodeByCanon.rightShoulder,
    nodeByCanon.rightUpperArm,
  ].filter((n): n is THREE.Object3D => !!n);

  for (const c of candidates) {
    if (!c.isWeighted) continue;
    if (
      c.rawName.includes('eye') ||
      c.rawName.includes('jaw') ||
      c.rawName.includes('brow') ||
      c.rawName.includes('lip')
    ) {
      downstreamRoots.push(c.obj);
    }
  }

  const parallelAncestorSet = new Set<THREE.Object3D>();
  for (const dr of downstreamRoots) {
    if (hipsNode && isDescendantOf(dr, hipsNode)) continue;
    let p = dr.parent;
    while (p && p !== root) {
      parallelAncestorSet.add(p);
      p = p.parent;
    }
  }

  const syncWorldDeltaPairs: Array<{ syncNode: THREE.Object3D; sourceDeformNode: THREE.Object3D }> = [];
  let activeTorsoChain = [...torsoChain];
  let hipsLockedTorsoPrefix: THREE.Object3D[] = [];

  if (parallelAncestorSet.size > 0 && !vrm) {
    const parallelAncestors = Array.from(parallelAncestorSet);
    const findParallelCounterpart = (deformObj: THREE.Object3D): THREE.Object3D | null => {
      const defRaw = getRawNodeName(deformObj);
      const defPos = deformObj.getWorldPosition(new THREE.Vector3());
      const base = defRaw.replace(/^DEF[-_]/i, ''); // e.g. "spine.003", "shoulder.L"
      const fkName = base.replace(/^spine/i, 'spine_fk'); // e.g. "spine_fk.003"
      const orgName = `ORG-${base}`; // e.g. "ORG-spine.004", "ORG-shoulder.L"

      // Prefer spine_fk.* over ORG-spine.* when both exist (in Rigify, spine_fk.003 parents BOTH ORG-spine.003 and ORG-spine.004!)
      for (const targetRaw of [fkName, orgName]) {
        const hit = parallelAncestors.find(
          (anc) =>
            getRawNodeName(anc).toLowerCase() === targetRaw.toLowerCase() &&
            anc.getWorldPosition(new THREE.Vector3()).distanceTo(defPos) < 0.03 * modelHeight,
        );
        if (hit) return hit;
      }
      return null;
    };

    // Start active torso bending at the lowest torsoChain bone that has a parallel FK/ORG pivot
    // so DEF-spine..006 and ORG-spine..006 / ORG-face share the exact same 3D world pivot (0.0mm drift!).
    let firstSyncedTorsoIdx = -1;
    for (let i = 0; i < torsoChain.length; i++) {
      const sync = findParallelCounterpart(torsoChain[i]);
      if (sync) {
        if (firstSyncedTorsoIdx < 0) firstSyncedTorsoIdx = i;
        syncWorldDeltaPairs.push({ syncNode: sync, sourceDeformNode: torsoChain[i] });
      }
    }
    if (firstSyncedTorsoIdx > 0) {
      hipsLockedTorsoPrefix = torsoChain.slice(0, firstSyncedTorsoIdx);
      activeTorsoChain = torsoChain.slice(firstSyncedTorsoIdx);
      nodeByCanon.spine = activeTorsoChain[0];
      if (activeTorsoChain.length >= 2) {
        nodeByCanon.chest = activeTorsoChain[activeTorsoChain.length - 1];
      }
    }

    for (const nk of neckChain) {
      const sync = findParallelCounterpart(nk);
      if (sync) syncWorldDeltaPairs.push({ syncNode: sync, sourceDeformNode: nk });
    }
    if (headNode) {
      const sync = findParallelCounterpart(headNode);
      if (sync) syncWorldDeltaPairs.push({ syncNode: sync, sourceDeformNode: headNode });
    }
    for (const side of ['left', 'right'] as const) {
      const sh = nodeByCanon[`${side}Shoulder`];
      if (sh) {
        const sync = findParallelCounterpart(sh);
        if (sync) syncWorldDeltaPairs.push({ syncNode: sync, sourceDeformNode: sh });
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Precompute Bind World Quaternions & Pose-Alignment Frames
  // ---------------------------------------------------------------------------
  const bindLocalQuat = new Map<THREE.Object3D, THREE.Quaternion>();
  const bindWorldQuat = new Map<THREE.Object3D, THREE.Quaternion>();
  const bindWorldQuatInv = new Map<THREE.Object3D, THREE.Quaternion>();
  for (const obj of allSceneObjects) {
    bindLocalQuat.set(obj, obj.quaternion.clone());
    const wq = obj.getWorldQuaternion(new THREE.Quaternion());
    bindWorldQuat.set(obj, wq);
    bindWorldQuatInv.set(obj, wq.clone().invert());
  }

  const downVec = new THREE.Vector3(0, -1, 0);
  const poseAlignByObj = new Map<THREE.Object3D, THREE.Quaternion>();
  const thumbAlignRecording = new Map<THREE.Object3D, THREE.Quaternion>();
  const thumbAlignDictionary = new Map<THREE.Object3D, THREE.Quaternion>();

  const worldPosOf = (obj: THREE.Object3D | undefined | null): THREE.Vector3 | null =>
    obj ? obj.getWorldPosition(new THREE.Vector3()) : null;

  for (const side of ['right', 'left'] as const) {
    const upperObj = nodeByCanon[`${side}UpperArm`];
    const lowerObj = nodeByCanon[`${side}LowerArm`];
    const handObj = nodeByCanon[`${side}Hand`];

    const pUpper = worldPosOf(upperObj);
    const pLower = worldPosOf(lowerObj);
    const pHand = worldPosOf(handObj);

    let upperAlign = new THREE.Quaternion();
    if (upperObj && pUpper && pLower) {
      const dir = pLower.clone().sub(pUpper);
      if (dir.lengthSq() > 1e-8) {
        upperAlign = new THREE.Quaternion().setFromUnitVectors(dir.normalize(), downVec);
      }
      poseAlignByObj.set(upperObj, upperAlign);
      for (const sp of upperArmSplits[side]) poseAlignByObj.set(sp, upperAlign.clone());
    }

    // Compute full 3D hand alignment frame FIRST so lowerArm rest twist aligns seamlessly with the wrist!
    let handAlign: THREE.Quaternion | null = null;
    const pIndex = worldPosOf(nodeByCanon[`${side}IndexProximal`]);
    const pLittle =
      worldPosOf(nodeByCanon[`${side}LittleProximal`]) ??
      worldPosOf(nodeByCanon[`${side}RingProximal`]);
    const pMiddle = worldPosOf(nodeByCanon[`${side}MiddleProximal`]);

    if (pHand && pIndex && pLittle) {
      const knuckles = pMiddle ?? pIndex.clone().add(pLittle).multiplyScalar(0.5);
      const fingerDir = knuckles.clone().sub(pHand);
      const radialDir = pIndex.clone().sub(pLittle);
      if (fingerDir.lengthSq() > 1e-8 && radialDir.lengthSq() > 1e-8) {
        const from = frameQuaternion(fingerDir, radialDir);
        const to = frameQuaternion(downVec, new THREE.Vector3(side === 'right' ? -1 : 1, 0, 0));
        handAlign = to.multiply(from.invert()).normalize();
      }
    }

    // LowerArm (+ split forearm bones): swing forearm bind vector -> (0, -1, 0) WITH wrist-aligned rest twist
    // (eliminates the 90° forearm-to-wrist pronation jump on T-pose palm-down GLBs!).
    let lowerAlign = new THREE.Quaternion();
    if (lowerObj && pLower && pHand) {
      const foreDir = pHand.clone().sub(pLower);
      if (foreDir.lengthSq() > 1e-8) {
        foreDir.normalize();
        if (handAlign) {
          const foreInHandFrame = foreDir.clone().applyQuaternion(handAlign);
          lowerAlign = new THREE.Quaternion()
            .setFromUnitVectors(foreInHandFrame, downVec)
            .multiply(handAlign)
            .normalize();
        } else {
          lowerAlign = new THREE.Quaternion().setFromUnitVectors(foreDir, downVec);
        }
      }
      poseAlignByObj.set(lowerObj, lowerAlign);
      for (const sp of lowerArmSplits[side]) poseAlignByObj.set(sp, lowerAlign.clone());
    }

    const effectiveHandAlign = handAlign ?? lowerAlign.clone();
    if (handObj) {
      poseAlignByObj.set(handObj, effectiveHandAlign.clone());
    }

    // Finger & thumb phalanx alignment
    for (const f of FINGER_NAMES) {
      const phObjs = PHALANX_NAMES.map((ph) => nodeByCanon[`${side}${f}${ph}`]);
      let prevDir: THREE.Vector3 | null = null;
      for (let i = 0; i < phObjs.length; i++) {
        const curObj = phObjs[i];
        if (!curObj) continue;
        const pCur = worldPosOf(curObj);
        const nextObj = i + 1 < phObjs.length ? phObjs[i + 1] : undefined;
        let pNext = worldPosOf(nextObj);
        if (!pNext && curObj.children.length > 0) {
          // Use fingertip child bone position if available for accurate distal direction
          const childPos = curObj.children[0].getWorldPosition(new THREE.Vector3());
          if (pCur && childPos.distanceTo(pCur) > 0.003) pNext = childPos;
        }
        const dir: THREE.Vector3 | null = pCur && pNext ? pNext.clone().sub(pCur) : prevDir;
        let recAlign = effectiveHandAlign.clone();
        if (dir && dir.lengthSq() > 1e-10) {
          prevDir = dir.clone();
          const inHandFrame = dir.clone().normalize().applyQuaternion(effectiveHandAlign);
          recAlign = new THREE.Quaternion()
            .setFromUnitVectors(inHandFrame, downVec)
            .multiply(effectiveHandAlign)
            .normalize();
        }
        if (f === 'Thumb') {
          thumbAlignRecording.set(curObj, recAlign);
          thumbAlignDictionary.set(curObj, effectiveHandAlign.clone());
          poseAlignByObj.set(curObj, recAlign);
        } else {
          poseAlignByObj.set(curObj, recAlign);
        }
      }
    }
  }

  // Collect all driven scene objects in deterministic order and register their track keys
  const drivenNodesSet = new Set<THREE.Object3D>();
  const addDriven = (obj: THREE.Object3D | null | undefined) => {
    if (obj && obj !== root) drivenNodesSet.add(obj);
  };
  addDriven(hipsNode);
  hipsLockedTorsoPrefix.forEach(addDriven);
  activeTorsoChain.forEach(addDriven);
  neckChain.forEach(addDriven);
  addDriven(headNode);
  for (const side of ['right', 'left'] as const) {
    addDriven(nodeByCanon[`${side}Shoulder`]);
    addDriven(nodeByCanon[`${side}UpperArm`]);
    upperArmSplits[side].forEach(addDriven);
    addDriven(nodeByCanon[`${side}LowerArm`]);
    lowerArmSplits[side].forEach(addDriven);
    addDriven(nodeByCanon[`${side}Hand`]);
    for (const f of FINGER_NAMES) {
      for (const ph of PHALANX_NAMES) {
        addDriven(nodeByCanon[`${side}${f}${ph}`]);
      }
    }
  }
  for (const pair of syncWorldDeltaPairs) {
    addDriven(pair.syncNode);
  }

  const drivenNodes = Array.from(drivenNodesSet);
  const drivenBoneTrackNames = drivenNodes.map((obj) => trackKeyOf(obj));

  const boneNodeNames: Record<string, string> = {};
  for (const [canon, obj] of Object.entries(nodeByCanon)) {
    boneNodeNames[canon] = trackKeyOf(obj);
  }

  const orderedCanonBones = Object.keys(CANON_PARENT_MAP);
  const identityQuat = new THREE.Quaternion();

  const applyAlignedWorld = (
    obj: THREE.Object3D,
    wSrc: THREE.Quaternion,
    overrideAlign?: THREE.Quaternion,
  ): THREE.Quaternion => {
    const qAlign = overrideAlign ?? poseAlignByObj.get(obj) ?? identityQuat;
    const qBind = bindWorldQuat.get(obj) ?? identityQuat;
    return wSrc.clone().multiply(qAlign).multiply(qBind).normalize();
  };

  // Pre-build top-down traversal order of the minimal subtree spanning all drivenNodes
  const relevantSubtree = new Set<THREE.Object3D>();
  for (const dn of drivenNodes) {
    let cur: THREE.Object3D | null = dn;
    while (cur && cur !== root) {
      if (relevantSubtree.has(cur)) break;
      relevantSubtree.add(cur);
      cur = cur.parent;
    }
  }
  const topDownSubtreeNodes: THREE.Object3D[] = [];
  const collectTopDown = (node: THREE.Object3D) => {
    for (const ch of node.children) {
      if (relevantSubtree.has(ch)) {
        topDownSubtreeNodes.push(ch);
        collectTopDown(ch);
      }
    }
  };
  collectTopDown(root);

  const rootWorldQuat = root.getWorldQuaternion(new THREE.Quaternion());

  const retargetFrameQuats = (
    srcLocalQuats: Record<string, THREE.Quaternion>,
    context?: RetargetFrameContext,
  ): Record<string, THREE.Quaternion> => {
    const isRecordingClip = context?.isRecordingClip ?? true;

    // 1. Forward kinematics on the canonical 54-bone source rig
    const srcWorld: Record<string, THREE.Quaternion> = {};
    for (const canon of orderedCanonBones) {
      const parentCanon = CANON_PARENT_MAP[canon];
      const qLocal = srcLocalQuats[canon] ?? identityQuat;
      const pWorld = parentCanon ? (srcWorld[parentCanon] ?? identityQuat) : identityQuat;
      srcWorld[canon] = pWorld.clone().multiply(qLocal).normalize();
    }

    const wHips = srcWorld.hips ?? identityQuat;
    const wMidSpine = srcWorld.spine ?? wHips;
    const wTorsoTop = srcWorld.chest ?? wMidSpine;
    const wNeckTop = srcWorld.neck ?? wTorsoTop;
    const wHead = srcWorld.head ?? wNeckTop;

    // Piecewise spine curve u in [0..1]: 0 -> wHips, 0.5 -> wMidSpine, 1.0 -> wTorsoTop
    const sampleTorsoCurve = (u: number): THREE.Quaternion => {
      if (u <= 0.5) return wHips.clone().slerp(wMidSpine, u * 2).normalize();
      return wMidSpine.clone().slerp(wTorsoTop, (u - 0.5) * 2).normalize();
    };

    const desiredWorld = new Map<THREE.Object3D, THREE.Quaternion>();

    // 2. Central column distribution (arbitrary K torso spine bones, M neck bones)
    if (hipsNode) {
      desiredWorld.set(hipsNode, applyAlignedWorld(hipsNode, wHips));
    }
    for (const locked of hipsLockedTorsoPrefix) {
      desiredWorld.set(locked, applyAlignedWorld(locked, wHips));
    }
    for (let i = 0; i < activeTorsoChain.length; i++) {
      const u = (i + 1) / activeTorsoChain.length;
      const node = activeTorsoChain[i];
      desiredWorld.set(node, applyAlignedWorld(node, sampleTorsoCurve(u)));
    }
    for (let j = 0; j < neckChain.length; j++) {
      const v = (j + 1) / neckChain.length;
      const node = neckChain[j];
      const wNk = wTorsoTop.clone().slerp(wNeckTop, v).normalize();
      desiredWorld.set(node, applyAlignedWorld(node, wNk));
    }
    if (headNode) {
      desiredWorld.set(headNode, applyAlignedWorld(headNode, wHead));
    }

    // 3. Bilateral shoulders, arms (+ split twist bones), hands, and fingers
    for (const side of ['right', 'left'] as const) {
      const shNode = nodeByCanon[`${side}Shoulder`];
      if (shNode) {
        desiredWorld.set(shNode, applyAlignedWorld(shNode, srcWorld[`${side}Shoulder`] ?? wTorsoTop));
      }

      const wUpper = srcWorld[`${side}UpperArm`] ?? wTorsoTop;
      const upperNode = nodeByCanon[`${side}UpperArm`];
      if (upperNode) {
        desiredWorld.set(upperNode, applyAlignedWorld(upperNode, wUpper));
      }
      for (const sp of upperArmSplits[side]) {
        desiredWorld.set(sp, applyAlignedWorld(sp, wUpper));
      }

      const wLowerFull = srcWorld[`${side}LowerArm`] ?? wUpper;
      const lowerNode = nodeByCanon[`${side}LowerArm`];
      const splits = lowerArmSplits[side];
      if (lowerNode) {
        if (splits.length > 0) {
          // Distribute forearm axial pronation/supination twist smoothly across split forearm segments!
          const qLowerLocal = srcLocalQuats[`${side}LowerArm`] ?? identityQuat;
          const swungDir = downVec.clone().applyQuaternion(qLowerLocal).normalize();
          const qSwingOnly = new THREE.Quaternion().setFromUnitVectors(downVec, swungDir);
          const wLowerSwing = wUpper.clone().multiply(qSwingOnly).normalize();
          const totalSegs = splits.length + 1;
          const wFirst = wLowerSwing.clone().slerp(wLowerFull, 1 / totalSegs).normalize();
          desiredWorld.set(lowerNode, applyAlignedWorld(lowerNode, wFirst));
          for (let sIdx = 0; sIdx < splits.length; sIdx++) {
            const frac = (sIdx + 2) / totalSegs;
            const wSeg = wLowerSwing.clone().slerp(wLowerFull, frac).normalize();
            desiredWorld.set(splits[sIdx], applyAlignedWorld(splits[sIdx], wSeg));
          }
        } else {
          desiredWorld.set(lowerNode, applyAlignedWorld(lowerNode, wLowerFull));
        }
      }

      const handNode = nodeByCanon[`${side}Hand`];
      if (handNode) {
        desiredWorld.set(handNode, applyAlignedWorld(handNode, srcWorld[`${side}Hand`] ?? wLowerFull));
      }

      for (const f of FINGER_NAMES) {
        for (const ph of PHALANX_NAMES) {
          const canon = `${side}${f}${ph}`;
          const fNode = nodeByCanon[canon];
          if (!fNode) continue;
          const thumbAlign =
            f === 'Thumb'
              ? isRecordingClip
                ? thumbAlignRecording.get(fNode)
                : thumbAlignDictionary.get(fNode)
              : undefined;
          desiredWorld.set(fNode, applyAlignedWorld(fNode, srcWorld[canon] ?? identityQuat, thumbAlign));
        }
      }
    }

    // 4. Synchronize Rigify parallel structural ancestors (spine_fk.*, ORG-spine.*, ORG-shoulder.*)
    // via exact world rotational delta: W_anim(S) = (W_anim(D) * W_bind(D)^-1) * W_bind(S)
    for (const { syncNode, sourceDeformNode } of syncWorldDeltaPairs) {
      const wAnimDef = desiredWorld.get(sourceDeformNode);
      const wBindDefInv = bindWorldQuatInv.get(sourceDeformNode);
      const wBindSync = bindWorldQuat.get(syncNode);
      if (wAnimDef && wBindDefInv && wBindSync) {
        const wSync = wAnimDef.clone().multiply(wBindDefInv).multiply(wBindSync).normalize();
        desiredWorld.set(syncNode, wSync);
      }
    }

    // 5. Exact Top-Down Scene-Graph World-to-Local Quaternion Propagation
    // Accounts for EVERY unmapped intermediate transform (e.g. ORG-palm.01.L's 71° bind rotation,
    // MCH-spine.*, tweak_spine.*, Armature scale/rotation containers).
    const animWorld = new Map<THREE.Object3D, THREE.Quaternion>();
    animWorld.set(root, rootWorldQuat);

    const outByTrackKey: Record<string, THREE.Quaternion> = {};
    for (const node of topDownSubtreeNodes) {
      const parentWorld = (node.parent && animWorld.get(node.parent)) ?? rootWorldQuat;
      const wTarget = desiredWorld.get(node);
      if (wTarget) {
        const qLocal = parentWorld.clone().invert().multiply(wTarget).normalize();
        outByTrackKey[trackKeyOf(node)] = qLocal;
        animWorld.set(node, wTarget);
      } else {
        const qBindLocal = bindLocalQuat.get(node) ?? node.quaternion;
        animWorld.set(node, parentWorld.clone().multiply(qBindLocal).normalize());
      }
    }

    return outByTrackKey;
  };

  // Build facial bone fallback tracks when morph targets don't cover all 9 LIS facial channels
  const coveredCanonicalMorphs = new Set<string>();
  const morphChecks: Record<string, string[]> = {
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
  for (const [canonMorph, aliases] of Object.entries(morphChecks)) {
    if (aliases.some((a) => availableMorphs.has(a))) {
      coveredCanonicalMorphs.add(canonMorph);
    }
  }

  const { hasFacialBoneRig, buildFacialBoneTracks } = createFacialBoneTrackBuilder(
    root,
    headNode,
    coveredCanonicalMorphs,
    hasWeightedSkin,
    weightByBone,
    trackKeyOf,
    modelHeight,
  );

  // Filter visual deform bones for clean SkeletonHelper visualization (hiding Rigify VIS_/WGT-/MCH-/IK clutter)
  const visualBones: THREE.Bone[] = [];
  for (const c of candidates) {
    if (!(c.obj as THREE.Bone).isBone) continue;
    if (hasWeightedSkin) {
      if (c.isWeighted || drivenNodesSet.has(c.obj)) {
        visualBones.push(c.obj as THREE.Bone);
      }
    } else if (!c.parsed.isMchOrControl) {
      visualBones.push(c.obj as THREE.Bone);
    }
  }

  return {
    rigTarget: {
      boneNodeNames,
      retargetFrameQuats,
      drivenBoneTrackNames,
      faceMeshName: primaryFaceMeshName,
      availableMorphs,
      morphTargetMeshes,
      buildFacialBoneTracks: hasFacialBoneRig ? buildFacialBoneTracks : undefined,
    },
    detectedFormat,
    matchedBonesCount: Object.keys(boneNodeNames).length,
    drivenBonesCount: drivenBoneTrackNames.length,
    hasFacialBoneRig,
    visualBones,
  };
}

/**
 * Builds a clean `THREE.SkeletonHelper` that omits Rigify IK pole / widget / mechanism clutter
 * when a weighted deform skeleton is present.
 */
export function buildCleanSkeletonHelper(
  root: THREE.Object3D,
  visualBones: THREE.Bone[],
): THREE.SkeletonHelper {
  const helper = new THREE.SkeletonHelper(root);
  if (visualBones.length >= 10 && helper.bones.length > visualBones.length * 1.35) {
    // Include each visual bone's immediate Bone parent if weighted/ORG-palm so finger/limb lines stay connected
    const keep = new Set<THREE.Bone>(visualBones);
    for (const b of visualBones) {
      if (b.parent && (b.parent as THREE.Bone).isBone) {
        const rawParent = getRawNodeName(b.parent);
        if (!/^VIS_|^WGT-|ik_pole/i.test(rawParent)) {
          keep.add(b.parent as THREE.Bone);
        }
      }
    }
    const filtered = helper.bones.filter((b) => keep.has(b));
    let segVerts = 0;
    for (const b of filtered) {
      if (b.parent && (b.parent as THREE.Bone).isBone) {
        segVerts += 2;
      }
    }
    if (segVerts >= 10) {
      helper.bones = filtered;
      helper.geometry.dispose();
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(segVerts * 3), 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(segVerts * 3), 3));
      helper.geometry = geo;
      helper.setColors(new THREE.Color(0x0000ff), new THREE.Color(0x00ff00));
    }
  }
  helper.visible = false;
  return helper;
}
