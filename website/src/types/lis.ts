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

