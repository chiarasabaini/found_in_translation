export interface AnchorConfig {
  right_hand: [number, number, number];
  left_hand: [number, number, number];
  right_wrist_euler_deg: [number, number, number];
  left_wrist_euler_deg: [number, number, number];
  motion_arc: string;
  motion_amplitude: [number, number, number];
  duration_sec: number;
}

export interface DictionaryClass {
  id: string;
  phrase: string;
  category: string;
  description: string;
  twoHanded: boolean;
  synonyms: string[];
  anchor: AnchorConfig;
  prototypeClipId: string;
  duration: number;
  animationUrl: string;
  /** Text in other source languages; the base fields above use metadata.base_language. */
  translations?: Record<string, DictionaryClassText>;
}

export interface DictionaryClassText {
  phrase: string;
  category?: string;
  description?: string;
  synonyms?: string[];
}

export interface DictionaryPayload {
  metadata: {
    language: string;
    sign_language: string;
    /** Language of the base phrase/synonyms/description fields (default "it"). */
    base_language?: string;
    /** Source languages that have dictionary text. */
    source_languages?: string[];
    target_fps: number;
    match_threshold: number;
    max_suggestions: number;
  };
  classes: DictionaryClass[];
}

export interface MatchCandidate {
  classId: string;
  phrase: string;
  category: string;
  matchedText: string;
  score: number;
  viaSynonym: boolean;
  matchType: 'exact' | 'synonym' | 'fuzzy';
}

export interface MatchResult {
  matched: boolean;
  normalizedQuery: string;
  best: MatchCandidate | null;
  suggestions: MatchCandidate[];
}

export interface SensorComparisonStat {
  sensor_name: string;
  rms_position_diff_m: number;
  rms_angle_diff_deg: number;
  mean_emf_residual: number;
  max_emf_residual: number;
  group1_dropout_rate: number;
  group2_dropout_rate: number;
}

export interface QaHandFrame {
  target_wrist: [number, number, number];
  achieved_wrist: [number, number, number];
  curls_deg_g2: Record<string, number>;
  curls_deg_g1: Record<string, number>;
  mean_emf_residual: number;
}

export interface QaFrame {
  t: number;
  right: QaHandFrame;
  left: QaHandFrame;
  morphs: Record<string, number>;
}

export interface SignAnimationPayload {
  classId: string;
  phrase: string;
  category: string;
  description: string;
  prototypeClipId: string;
  fps: number;
  duration: number;
  times: number[];
  boneTracks: Record<string, number[]>;
  morphTracks: Record<string, number[]>;
  prototypeSelection?: {
    method: string;
    candidates: string[];
    meanDtwDistance: number;
  };
  qaComparison: {
    group1VsGroup2Right: SensorComparisonStat[];
    group1VsGroup2Left: SensorComparisonStat[];
    frames: QaFrame[];
  };
  /** Present when the clip comes from an imported recording instead of the offline pipeline. */
  source?: RecordingSourceInfo;
}

export interface RecordingSourceInfo {
  kind: 'recording';
  recordingName: string;
  startSec: number;
  endSec: number;
  gloveYawDeg: number;
  leftYawOffsetDeg: number;
  rightYawOffsetDeg: number;
  sourceFrameIntervalMs: number;
  streams: {
    leftHand: boolean;
    rightHand: boolean;
    face: boolean;
    body: boolean;
  };
}

export interface QaSummaryClass {
  classId: string;
  phrase: string;
  prototypeClipId: string;
  frameCount: number;
  durationSec: number;
  rmsWristReachErrorCm: number;
  rmsGroup1VsGroup2CurlDeg: number;
  maxMeanEmfResidual: number;
}

export interface QaSummaryPayload {
  totalClasses: number;
  classes: QaSummaryClass[];
}
