from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

# Suppress verbose MediaPipe C++ logs before importing mediapipe
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

from core.config import MODELS_DIR

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

NUM_FACE_JOINTS = 478
FACE_CHIN_IDX = 152

# Pip thumbnail dimensions matching core/renderer_3d.py sidebar card
PIP_HALF_W = 188
PIP_RIGHT_W = 188
PIP_H = 220

# MediaPipe Pose indices extracted from the distant body camera
RAW_BODY_POSE_INDICES: dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
}

# Exported body joints (neck is derived as the midpoint between left_shoulder and right_shoulder)
BODY_JOINT_NAMES: tuple[str, ...] = (
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)

# Standard 52 MediaPipe FaceLandmarker blendshape categories
FACE_BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)


def _ensure_model(path: Path, url: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...", flush=True)
        urllib.request.urlretrieve(url, path)


def create_face_landmarker() -> Any:
    """Create MediaPipe FaceLandmarker for the close-up face camera."""
    _ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(face_opts)


def create_body_landmarker() -> Any:
    """Create MediaPipe PoseLandmarker for the distant body camera."""
    _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(pose_opts)


def _downscale_for_inference(bgr: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """Downscale high-resolution camera frames to max_dim (MediaPipe internally uses 256x256)."""
    h, w = bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return bgr
    scale = max_dim / float(largest)
    return cv2.resize(
        bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def _make_pip_box(
    bgr: np.ndarray, box_w: int, box_h: int
) -> tuple[np.ndarray, int, int, int, int]:
    fh, fw = bgr.shape[:2]
    scale = min(box_w / float(fw), box_h / float(fh))
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((box_h, box_w, 3), (20, 22, 26), dtype=np.uint8)
    ox, oy = (box_w - nw) // 2, (box_h - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out, ox, oy, nw, nh


def extract_face_joints(
    rgb_frame: np.ndarray,
    ts_ms: int,
    face_landmarker: Any,
) -> tuple[list[tuple[float, float, float]], dict[str, float]]:
    """Extract ONLY the 478 face joints and 52 facial blendshapes from the face camera."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    face_result = face_landmarker.detect_for_video(mp_image, ts_ms)

    face_joints: list[tuple[float, float, float]] = []
    if face_result.face_landmarks:
        face_joints = [
            (float(lm.x), float(lm.y), float(lm.z))
            for lm in face_result.face_landmarks[0]
        ]

    face_blendshapes: dict[str, float] = {}
    if face_result.face_blendshapes:
        face_blendshapes = {
            bs.category_name: float(bs.score)
            for bs in face_result.face_blendshapes[0]
        }

    return face_joints, face_blendshapes


def extract_body_joints(
    rgb_frame: np.ndarray,
    ts_ms: int,
    body_landmarker: Any,
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, tuple[float, float, float, float]],
]:
    """Extract ONLY neck, shoulders, and elbows from the distant body camera using metric 3D pose_world_landmarks."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    pose_result = body_landmarker.detect_for_video(mp_image, ts_ms)

    if not pose_result.pose_landmarks or not pose_result.pose_world_landmarks:
        return {}, {}

    norm_lms = pose_result.pose_landmarks[0]
    world_lms = pose_result.pose_world_landmarks[0]

    raw_norm: dict[str, tuple[float, float, float, float]] = {}
    raw_world: dict[str, tuple[float, float, float, float]] = {}

    for name, idx in RAW_BODY_POSE_INDICES.items():
        if idx < len(norm_lms) and idx < len(world_lms):
            nlm = norm_lms[idx]
            wlm = world_lms[idx]
            vis = float(getattr(nlm, "visibility", 1.0))
            raw_norm[name] = (float(nlm.x), float(nlm.y), float(nlm.z), vis)
            raw_world[name] = (float(wlm.x), float(wlm.y), float(wlm.z), vis)

    ls_n = raw_norm.get("left_shoulder")
    rs_n = raw_norm.get("right_shoulder")
    ls_w = raw_world.get("left_shoulder")
    rs_w = raw_world.get("right_shoulder")
    if not ls_n or not rs_n or not ls_w or not rs_w:
        return {}, {}

    neck_vis = min(ls_n[3], rs_n[3])
    raw_norm["neck"] = (
        (ls_n[0] + rs_n[0]) * 0.5,
        (ls_n[1] + rs_n[1]) * 0.5,
        (ls_n[2] + rs_n[2]) * 0.5,
        neck_vis,
    )
    cx, cy, cz = (
        (ls_w[0] + rs_w[0]) * 0.5,
        (ls_w[1] + rs_w[1]) * 0.5,
        (ls_w[2] + rs_w[2]) * 0.5,
    )
    raw_world["neck"] = (cx, cy, cz, neck_vis)

    body_joints: dict[str, tuple[float, float, float, float]] = {}
    for name in BODY_JOINT_NAMES:
        wx, wy, wz, vis = raw_world[name]
        body_joints[name] = (
            -(wx - cx),
            -(wy - cy),
            -(wz - cz),
            vis,
        )

    return body_joints, raw_norm


class FaceTrackerWorker:
    """Dedicated background thread that captures Camera #0, runs MediaPipe FaceLandmarker, and prepares the PiP box."""

    def __init__(self, camera_index: int) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 60)
        self.is_opened = self._cap.isOpened()
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_opened else 0
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_opened else 0

        self._lock = threading.Lock()
        self._face_joints: list[tuple[float, float, float]] = []
        self._face_blendshapes: dict[str, float] = {}
        self._pip_box: np.ndarray = np.full((PIP_H, PIP_HALF_W, 3), (20, 22, 26), dtype=np.uint8)
        self._frame_seq: int = 0
        self._ready = False
        self._running = False
        self._thread: threading.Thread | None = None

        if self.is_opened:
            _ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def _worker_loop(self) -> None:
        start_t = time.monotonic()
        last_ts_ms = 0
        with create_face_landmarker() as landmarker:
            while self._running:
                ok, raw = self._cap.read()
                if not ok:
                    time.sleep(0.002)
                    continue

                small = _downscale_for_inference(raw, max_dim=640)
                rotated = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

                ts_ms = int((time.monotonic() - start_t) * 1000)
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms

                joints, blendshapes = extract_face_joints(rgb, ts_ms, landmarker)

                pip_box, ox, oy, nw, nh = _make_pip_box(rotated, PIP_HALF_W, PIP_H)
                if joints:
                    pts = np.asarray(joints, dtype=np.float32)
                    u = np.clip((ox + pts[:, 0] * nw).astype(np.int32), 0, PIP_HALF_W - 1)
                    v = np.clip((oy + pts[:, 1] * nh).astype(np.int32), 0, PIP_H - 1)
                    pip_box[v, u] = (0, 255, 180)

                with self._lock:
                    self._face_joints = joints
                    self._face_blendshapes = blendshapes
                    self._pip_box = pip_box
                    self._frame_seq += 1
                    self._ready = True

    def snapshot(
        self,
    ) -> tuple[bool, int, list[tuple[float, float, float]], dict[str, float], np.ndarray]:
        with self._lock:
            return (
                self._ready,
                self._frame_seq,
                self._face_joints,
                self._face_blendshapes,
                self._pip_box,
            )

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.8)
        if self._cap.isOpened():
            self._cap.release()


class BodyTrackerWorker:
    """Dedicated background thread that captures Camera #1, runs MediaPipe PoseLandmarker, and prepares the PiP box."""

    def __init__(self, camera_index: int) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 60)
        self.is_opened = self._cap.isOpened()
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_opened else 0
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_opened else 0

        self._lock = threading.Lock()
        self._body_joints: dict[str, tuple[float, float, float, float]] = {}
        self._pip_box: np.ndarray = np.full((PIP_H, PIP_RIGHT_W, 3), (20, 22, 26), dtype=np.uint8)
        self._frame_seq: int = 0
        self._ready = False
        self._running = False
        self._thread: threading.Thread | None = None

        if self.is_opened:
            _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def _worker_loop(self) -> None:
        start_t = time.monotonic()
        last_ts_ms = 0
        with create_body_landmarker() as landmarker:
            while self._running:
                ok, raw = self._cap.read()
                if not ok:
                    time.sleep(0.002)
                    continue

                small = _downscale_for_inference(raw, max_dim=640)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                ts_ms = int((time.monotonic() - start_t) * 1000)
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms

                joints, norm_2d = extract_body_joints(rgb, ts_ms, landmarker)

                pip_box, ox, oy, nw, nh = _make_pip_box(small, PIP_RIGHT_W, PIP_H)
                for _name, (nx, ny, _nz, vis) in norm_2d.items():
                    if vis >= 0.30:
                        px = int(ox + nx * nw)
                        py = int(oy + ny * nh)
                        cv2.circle(pip_box, (px, py), 4, (0, 140, 255), -1, cv2.LINE_AA)

                with self._lock:
                    self._body_joints = joints
                    self._pip_box = pip_box
                    self._frame_seq += 1
                    self._ready = True

    def snapshot(
        self,
    ) -> tuple[bool, int, dict[str, tuple[float, float, float, float]], np.ndarray]:
        with self._lock:
            return self._ready, self._frame_seq, self._body_joints, self._pip_box

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.8)
        if self._cap.isOpened():
            self._cap.release()
