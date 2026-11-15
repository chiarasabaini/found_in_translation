from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.camera_tracker import (
    BODY_JOINT_NAMES,
    FACE_BLENDSHAPE_NAMES,
    FACE_CHIN_IDX,
    NUM_FACE_JOINTS,
)
from core.config import MannequinConfig
from core.csv_exporter import (
    BODY_HEADER,
    FACE_HEADER,
    LEFT_HAND_HEADER,
    MANIFEST_HEADER,
    RIGHT_HAND_HEADER,
    RecordedFrame,
    SessionCsvExporter,
    load_session,
)
from core.gloves_tracker import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES

DEFAULT_PROCESSED_DIR = Path("./output/processed/")
DEFAULT_TARGET_FPS = 30.0


@dataclass
class PostProcessStats:
    raw_frames: int
    raw_duration_ms: int
    raw_fps: float
    trimmed_start_frames: int
    trimmed_end_frames: int
    interpolated_face_frames: int
    interpolated_body_frames: int
    interpolated_left_frames: int
    interpolated_right_frames: int
    processed_frames: int
    processed_duration_ms: int
    target_fps: float


def _quat_normalize(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _quat_slerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation with shortest-arc hemisphere alignment."""
    x0, y0, z0, w0 = _quat_normalize(q0)
    x1, y1, z1, w1 = _quat_normalize(q1)

    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        dot = -dot
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1

    if dot > 0.9995:
        rx = x0 + alpha * (x1 - x0)
        ry = y0 + alpha * (y1 - y0)
        rz = z0 + alpha * (z1 - z0)
        rw = w0 + alpha * (w1 - w0)
        return _quat_normalize((rx, ry, rz, rw))

    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize(
        (
            s0 * x0 + s1 * x1,
            s0 * y0 + s1 * y1,
            s0 * z0 + s1 * z1,
            s0 * w0 + s1 * w1,
        )
    )


def _lerp_vec3(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return (
        p0[0] + alpha * (p1[0] - p0[0]),
        p0[1] + alpha * (p1[1] - p0[1]),
        p0[2] + alpha * (p1[2] - p0[2]),
    )


def _lerp_face(
    f0: list[tuple[float, float, float]],
    bs0: dict[str, float],
    f1: list[tuple[float, float, float]],
    bs1: dict[str, float],
    alpha: float,
) -> tuple[list[tuple[float, float, float]], dict[str, float]]:
    if not f0 and not f1:
        return [], {}
    if not f0:
        return list(f1), dict(bs1)
    if not f1:
        return list(f0), dict(bs0)

    arr0 = np.asarray(f0, dtype=np.float32)
    arr1 = np.asarray(f1, dtype=np.float32)
    arr = arr0 + np.float32(alpha) * (arr1 - arr0)
    joints = [(float(p[0]), float(p[1]), float(p[2])) for p in arr]

    blendshapes: dict[str, float] = {}
    for name in FACE_BLENDSHAPE_NAMES:
        v0 = bs0.get(name, 0.0)
        v1 = bs1.get(name, 0.0)
        blendshapes[name] = float(v0 + alpha * (v1 - v0))
    return joints, blendshapes


def _lerp_body(
    b0: dict[str, tuple[float, float, float, float]],
    b1: dict[str, tuple[float, float, float, float]],
    alpha: float,
) -> dict[str, tuple[float, float, float, float]]:
    if not b0 and not b1:
        return {}
    if not b0:
        return dict(b1)
    if not b1:
        return dict(b0)

    out: dict[str, tuple[float, float, float, float]] = {}
    for name in BODY_JOINT_NAMES:
        j0 = b0.get(name)
        j1 = b1.get(name)
        if j0 is not None and j1 is not None:
            out[name] = (
                j0[0] + alpha * (j1[0] - j0[0]),
                j0[1] + alpha * (j1[1] - j0[1]),
                j0[2] + alpha * (j1[2] - j0[2]),
                j0[3] + alpha * (j1[3] - j0[3]),
            )
        elif j0 is not None:
            out[name] = j0
        elif j1 is not None:
            out[name] = j1
    return out


def _interp_hand(
    h0: dict[str, Any] | None,
    h1: dict[str, Any] | None,
    alpha: float,
    joint_names: list[str],
) -> dict[str, Any] | None:
    if h0 is None and h1 is None:
        return None
    if h0 is None:
        return h1
    if h1 is None:
        return h0

    j0_map = h0.get("joints", {})
    j1_map = h1.get("joints", {})
    joints: dict[str, Any] = {}
    for name in joint_names:
        j0 = j0_map.get(name)
        j1 = j1_map.get(name)
        if j0 is not None and j1 is not None:
            pos = _lerp_vec3(j0["position"], j1["position"], alpha)
            rot = _quat_slerp(j0["rotation"], j1["rotation"], alpha)
            joints[name] = {"position": pos, "rotation": rot}
        elif j0 is not None:
            joints[name] = {
                "position": j0["position"],
                "rotation": _quat_normalize(j0["rotation"]),
            }
        elif j1 is not None:
            joints[name] = {
                "position": j1["position"],
                "rotation": _quat_normalize(j1["rotation"]),
            }

    ts0 = int(h0.get("timestamp_us", 0))
    ts1 = int(h1.get("timestamp_us", 0))
    ts_us = int(round(ts0 + alpha * (ts1 - ts0)))
    return {"timestamp_us": ts_us, "joints": joints}


def _find_valid_bounds(valid_flags: list[bool], idx: int) -> tuple[int | None, int | None]:
    prev_i = None
    for i in range(idx - 1, -1, -1):
        if valid_flags[i]:
            prev_i = i
            break
    next_i = None
    for i in range(idx + 1, len(valid_flags)):
        if valid_flags[i]:
            next_i = i
            break
    return prev_i, next_i


def _fill_missing_modalities(
    frames: list[RecordedFrame],
) -> tuple[list[RecordedFrame], int, int, int, int]:
    """Fill missing face, body, or hand detections across frames via LERP/SLERP."""
    n = len(frames)
    if n == 0:
        return [], 0, 0, 0, 0

    face_valid = [len(f.face_joints) == NUM_FACE_JOINTS for f in frames]
    body_valid = [bool(f.body_joints) for f in frames]
    left_valid = ["left" in f.hands and bool(f.hands["left"].get("joints")) for f in frames]
    right_valid = ["right" in f.hands and bool(f.hands["right"].get("joints")) for f in frames]

    interp_f = 0
    interp_b = 0
    interp_l = 0
    interp_r = 0

    out: list[RecordedFrame] = []
    for i, f in enumerate(frames):
        f_joints = f.face_joints
        f_bs = f.face_blendshapes
        if not face_valid[i] and any(face_valid):
            p_i, n_i = _find_valid_bounds(face_valid, i)
            if p_i is not None and n_i is not None:
                dt = max(1, frames[n_i].timestamp_ms - frames[p_i].timestamp_ms)
                alpha = (f.timestamp_ms - frames[p_i].timestamp_ms) / dt
                f_joints, f_bs = _lerp_face(
                    frames[p_i].face_joints,
                    frames[p_i].face_blendshapes,
                    frames[n_i].face_joints,
                    frames[n_i].face_blendshapes,
                    alpha,
                )
            elif p_i is not None:
                f_joints = list(frames[p_i].face_joints)
                f_bs = dict(frames[p_i].face_blendshapes)
            elif n_i is not None:
                f_joints = list(frames[n_i].face_joints)
                f_bs = dict(frames[n_i].face_blendshapes)
            interp_f += 1

        b_joints = f.body_joints
        if not body_valid[i] and any(body_valid):
            p_i, n_i = _find_valid_bounds(body_valid, i)
            if p_i is not None and n_i is not None:
                dt = max(1, frames[n_i].timestamp_ms - frames[p_i].timestamp_ms)
                alpha = (f.timestamp_ms - frames[p_i].timestamp_ms) / dt
                b_joints = _lerp_body(frames[p_i].body_joints, frames[n_i].body_joints, alpha)
            elif p_i is not None:
                b_joints = dict(frames[p_i].body_joints)
            elif n_i is not None:
                b_joints = dict(frames[n_i].body_joints)
            interp_b += 1

        hands = dict(f.hands)
        if not left_valid[i] and any(left_valid):
            p_i, n_i = _find_valid_bounds(left_valid, i)
            if p_i is not None and n_i is not None:
                dt = max(1, frames[n_i].timestamp_ms - frames[p_i].timestamp_ms)
                alpha = (f.timestamp_ms - frames[p_i].timestamp_ms) / dt
                h_l = _interp_hand(
                    frames[p_i].hands.get("left"),
                    frames[n_i].hands.get("left"),
                    alpha,
                    LEFT_HAND_JOINT_NAMES,
                )
            elif p_i is not None:
                h_l = frames[p_i].hands.get("left")
            else:
                h_l = frames[n_i].hands.get("left") if n_i is not None else None
            if h_l is not None:
                hands["left"] = h_l
                interp_l += 1

        if not right_valid[i] and any(right_valid):
            p_i, n_i = _find_valid_bounds(right_valid, i)
            if p_i is not None and n_i is not None:
                dt = max(1, frames[n_i].timestamp_ms - frames[p_i].timestamp_ms)
                alpha = (f.timestamp_ms - frames[p_i].timestamp_ms) / dt
                h_r = _interp_hand(
                    frames[p_i].hands.get("right"),
                    frames[n_i].hands.get("right"),
                    alpha,
                    RIGHT_HAND_JOINT_NAMES,
                )
            elif p_i is not None:
                h_r = frames[p_i].hands.get("right")
            else:
                h_r = frames[n_i].hands.get("right") if n_i is not None else None
            if h_r is not None:
                hands["right"] = h_r
                interp_r += 1

        out.append(
            RecordedFrame(
                frame_id=f.frame_id,
                timestamp_ms=f.timestamp_ms,
                face_joints=f_joints,
                face_blendshapes=f_bs,
                body_joints=b_joints,
                hands=hands,
            )
        )
    return out, interp_f, interp_b, interp_l, interp_r


_TRIM_HAND_JOINTS = (
    ("left", "LeftHand"),
    ("right", "RightHand"),
    ("left", "LeftMiddleTip"),
    ("right", "RightMiddleTip"),
)


def _max_hand_distance(f0: RecordedFrame, f1: RecordedFrame) -> float:
    """Compute max 3D Euclidean distance in meters across wrists and middle fingertips."""
    best = 0.0
    for side, jn in _TRIM_HAND_JOINTS:
        j0 = f0.hands.get(side, {}).get("joints", {}).get(jn)
        j1 = f1.hands.get(side, {}).get("joints", {}).get(jn)
        if j0 is None or j1 is None:
            continue
        p0, p1 = j0["position"], j1["position"]
        dist = math.sqrt(
            (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2
        )
        if dist > best:
            best = dist
    return best


def _trim_head_and_tail(
    frames: list[RecordedFrame],
    trim_idle: bool = True,
) -> tuple[list[RecordedFrame], int, int]:
    """Trim initial and final calibration poses based on max single-hand 3D displacement."""
    n = len(frames)
    if not trim_idle or n <= 12:
        return frames, 0, 0

    dt0 = max(0.01, (frames[1].timestamp_ms - frames[0].timestamp_ms) / 1000.0)
    if n >= 3 and (_max_hand_distance(frames[0], frames[1]) / dt0) > 2.5:
        frames[0].hands = dict(frames[1].hands)

    ref_start = frames[min(2, n - 1)]
    ref_end = frames[max(0, n - 3)]

    active_indices = [
        i
        for i in range(n)
        if _max_hand_distance(frames[i], ref_start) > 0.06
        and _max_hand_distance(frames[i], ref_end) > 0.06
    ]

    start_idx = 0
    end_idx = n - 1
    if active_indices:
        new_start = active_indices[0]
        new_end = active_indices[-1]
        if new_end - new_start + 1 >= 10:
            start_idx = new_start
            end_idx = new_end

    trimmed = frames[start_idx : end_idx + 1]
    trimmed_start = start_idx
    trimmed_end = (n - 1) - end_idx
    return trimmed, trimmed_start, trimmed_end


def _canonicalize_face_landmarks(
    face_joints: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Center 478 face landmarks on Chin (#152) and normalize scale into canonical [0..1] space.

    This makes face.csv invariant to webcam head translation and camera distance while remaining
    100% compatible with preview.py's 3D mannequin face projection.
    """
    if len(face_joints) != NUM_FACE_JOINTS:
        return face_joints

    pts = np.asarray(face_joints, dtype=np.float32)
    chin_x, chin_y, chin_z = pts[FACE_CHIN_IDX]
    span_x = max(float(pts[:, 0].max() - pts[:, 0].min()), 1e-4)
    span_y = max(float(pts[:, 1].max() - pts[:, 1].min()), 1e-4)

    canon_span_x = np.float32(0.50)
    canon_span_y = np.float32(0.70)
    canon_chin_x = np.float32(0.50)
    canon_chin_y = np.float32(0.85)

    out = np.empty_like(pts)
    out[:, 0] = canon_chin_x + ((pts[:, 0] - chin_x) / span_x) * canon_span_x
    out[:, 1] = canon_chin_y + ((pts[:, 1] - chin_y) / span_y) * canon_span_y
    out[:, 2] = ((pts[:, 2] - chin_z) / span_x) * canon_span_x
    return [(float(p[0]), float(p[1]), float(p[2])) for p in out]


def _resample_constant_fps(
    frames: list[RecordedFrame],
    target_fps: float = DEFAULT_TARGET_FPS,
) -> list[RecordedFrame]:
    """Resample frames onto a uniform timestamp grid (0, dt, 2*dt, ...) via LERP + SLERP."""
    if not frames:
        return []
    if len(frames) == 1 or target_fps <= 0.0:
        f0 = frames[0]
        return [
            RecordedFrame(
                frame_id=1,
                timestamp_ms=0,
                face_joints=_canonicalize_face_landmarks(f0.face_joints),
                face_blendshapes=dict(f0.face_blendshapes),
                body_joints=dict(f0.body_joints),
                hands=dict(f0.hands),
            )
        ]

    t_start = frames[0].timestamp_ms
    t_end = frames[-1].timestamp_ms
    duration_ms = max(0, t_end - t_start)
    dt_ms = 1000.0 / target_fps
    num_out = max(1, int(round(duration_ms / dt_ms)) + 1)

    src_times = [f.timestamp_ms - t_start for f in frames]
    resampled: list[RecordedFrame] = []
    cursor = 0

    for out_idx in range(num_out):
        t_ms_float = min(float(duration_ms), out_idx * dt_ms)
        t_ms_int = int(round(out_idx * dt_ms))

        while cursor + 1 < len(src_times) and src_times[cursor + 1] < t_ms_float:
            cursor += 1

        if cursor + 1 >= len(frames) or src_times[cursor] >= t_ms_float:
            f_src = frames[cursor]
            f_joints = _canonicalize_face_landmarks(f_src.face_joints)
            f_bs = dict(f_src.face_blendshapes)
            b_joints = dict(f_src.body_joints)
            hands: dict[str, dict[str, Any]] = {}
            if "left" in f_src.hands:
                hands["left"] = _interp_hand(
                    f_src.hands["left"], None, 0.0, LEFT_HAND_JOINT_NAMES
                )  # type: ignore[assignment]
            if "right" in f_src.hands:
                hands["right"] = _interp_hand(
                    f_src.hands["right"], None, 0.0, RIGHT_HAND_JOINT_NAMES
                )  # type: ignore[assignment]
        else:
            f0 = frames[cursor]
            f1 = frames[cursor + 1]
            seg_dt = max(1e-6, float(src_times[cursor + 1] - src_times[cursor]))
            alpha = max(0.0, min(1.0, (t_ms_float - src_times[cursor]) / seg_dt))

            raw_f_joints, f_bs = _lerp_face(
                f0.face_joints, f0.face_blendshapes, f1.face_joints, f1.face_blendshapes, alpha
            )
            f_joints = _canonicalize_face_landmarks(raw_f_joints)
            b_joints = _lerp_body(f0.body_joints, f1.body_joints, alpha)

            hands = {}
            h_l = _interp_hand(
                f0.hands.get("left"), f1.hands.get("left"), alpha, LEFT_HAND_JOINT_NAMES
            )
            if h_l is not None:
                hands["left"] = h_l
            h_r = _interp_hand(
                f0.hands.get("right"), f1.hands.get("right"), alpha, RIGHT_HAND_JOINT_NAMES
            )
            if h_r is not None:
                hands["right"] = h_r

        resampled.append(
            RecordedFrame(
                frame_id=out_idx + 1,
                timestamp_ms=t_ms_int,
                face_joints=f_joints,
                face_blendshapes=f_bs,
                body_joints=b_joints,
                hands=hands,
            )
        )

    return resampled


def _enforce_quaternion_continuity(frames: list[RecordedFrame]) -> None:
    """Enforce dot(q_t, q_{t-1}) >= 0 across time for every hand joint so quaternions never flip sign."""
    for side, joint_names in (
        ("left", LEFT_HAND_JOINT_NAMES),
        ("right", RIGHT_HAND_JOINT_NAMES),
    ):
        prev_quats: dict[str, tuple[float, float, float, float]] = {}
        for f in frames:
            hand = f.hands.get(side)
            if not hand:
                continue
            jmap = hand.get("joints", {})
            for jname in joint_names:
                j = jmap.get(jname)
                if j is None:
                    continue
                q = _quat_normalize(j["rotation"])
                prev_q = prev_quats.get(jname)
                if prev_q is not None:
                    dot = (
                        q[0] * prev_q[0]
                        + q[1] * prev_q[1]
                        + q[2] * prev_q[2]
                        + q[3] * prev_q[3]
                    )
                    if dot < 0.0:
                        q = (-q[0], -q[1], -q[2], -q[3])
                j["rotation"] = q
                prev_quats[jname] = q


def postprocess_frames(
    raw_frames: list[RecordedFrame],
    target_fps: float = DEFAULT_TARGET_FPS,
    trim_idle: bool = True,
) -> tuple[list[RecordedFrame], PostProcessStats]:
    """Run the full neural-network post-processing pipeline on a list of RecordedFrame objects."""
    raw_count = len(raw_frames)
    if raw_count == 0:
        empty_stats = PostProcessStats(0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, target_fps)
        return [], empty_stats

    raw_dur_ms = max(0, raw_frames[-1].timestamp_ms - raw_frames[0].timestamp_ms)
    raw_fps = round((raw_count * 1000.0) / raw_dur_ms, 2) if raw_dur_ms > 0 else 0.0

    filled, interp_f, interp_b, interp_l, interp_r = _fill_missing_modalities(raw_frames)
    trimmed, trim_start, trim_end = _trim_head_and_tail(filled, trim_idle=trim_idle)
    resampled = _resample_constant_fps(trimmed, target_fps=target_fps)
    _enforce_quaternion_continuity(resampled)

    proc_count = len(resampled)
    proc_dur_ms = (
        max(0, resampled[-1].timestamp_ms - resampled[0].timestamp_ms)
        if proc_count > 1
        else 0
    )
    stats = PostProcessStats(
        raw_frames=raw_count,
        raw_duration_ms=raw_dur_ms,
        raw_fps=raw_fps,
        trimmed_start_frames=trim_start,
        trimmed_end_frames=trim_end,
        interpolated_face_frames=interp_f,
        interpolated_body_frames=interp_b,
        interpolated_left_frames=interp_l,
        interpolated_right_frames=interp_r,
        processed_frames=proc_count,
        processed_duration_ms=proc_dur_ms,
        target_fps=target_fps,
    )
    return resampled, stats


def save_processed_session(
    out_session_dir: Path,
    frames: list[RecordedFrame],
    sentence: str,
    prompt_idx: int,
    mannequin_cfg: MannequinConfig,
    stats: PostProcessStats,
    manifest_path: Path | None = None,
) -> None:
    """Write post-processed 4 CSVs + metadata.json and optionally append to manifest.csv."""
    out_session_dir.mkdir(parents=True, exist_ok=True)
    with (
        (out_session_dir / "left_hand.csv").open("w", newline="", encoding="utf-8") as lf,
        (out_session_dir / "right_hand.csv").open("w", newline="", encoding="utf-8") as rf,
        (out_session_dir / "face.csv").open("w", newline="", encoding="utf-8") as ff,
        (out_session_dir / "body.csv").open("w", newline="", encoding="utf-8") as bf,
    ):
        lw = csv.writer(lf)
        rw = csv.writer(rf)
        fw = csv.writer(ff)
        bw = csv.writer(bf)

        lw.writerow(LEFT_HAND_HEADER)
        rw.writerow(RIGHT_HAND_HEADER)
        fw.writerow(FACE_HEADER)
        bw.writerow(BODY_HEADER)

        for rec in frames:
            lw.writerow(
                SessionCsvExporter._format_hand_row(
                    rec.frame_id,
                    rec.timestamp_ms,
                    rec.hands.get("left"),
                    LEFT_HAND_JOINT_NAMES,
                )
            )
            rw.writerow(
                SessionCsvExporter._format_hand_row(
                    rec.frame_id,
                    rec.timestamp_ms,
                    rec.hands.get("right"),
                    RIGHT_HAND_JOINT_NAMES,
                )
            )
            fw.writerow(
                SessionCsvExporter._format_face_row(
                    rec.frame_id,
                    rec.timestamp_ms,
                    rec.face_joints,
                    rec.face_blendshapes,
                )
            )
            bw.writerow(
                SessionCsvExporter._format_body_row(
                    rec.frame_id,
                    rec.timestamp_ms,
                    rec.body_joints,
                )
            )

    created_at = datetime.now().isoformat(timespec="seconds")
    avg_fps = (
        round((len(frames) * 1000.0) / stats.processed_duration_ms, 2)
        if stats.processed_duration_ms > 0
        else stats.target_fps
    )
    metadata = {
        "sample_id": out_session_dir.name,
        "prompt_index": prompt_idx,
        "sentence": sentence,
        "num_frames": len(frames),
        "duration_ms": stats.processed_duration_ms,
        "avg_fps": avg_fps,
        "created_at": created_at,
        "postprocessed": True,
        "postprocess_stats": {
            "raw_frames": stats.raw_frames,
            "raw_duration_ms": stats.raw_duration_ms,
            "raw_fps": stats.raw_fps,
            "trimmed_start_frames": stats.trimmed_start_frames,
            "trimmed_end_frames": stats.trimmed_end_frames,
            "interpolated_face_frames": stats.interpolated_face_frames,
            "interpolated_body_frames": stats.interpolated_body_frames,
            "interpolated_left_frames": stats.interpolated_left_frames,
            "interpolated_right_frames": stats.interpolated_right_frames,
            "target_fps": stats.target_fps,
        },
        "mannequin": mannequin_cfg.to_dict(),
        "files": {
            "left_hand": "left_hand.csv",
            "right_hand": "right_hand.csv",
            "face": "face.csv",
            "body": "body.csv",
        },
    }
    (out_session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if manifest_path is not None:
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            if write_header:
                writer.writerow(MANIFEST_HEADER)
            writer.writerow(
                [
                    out_session_dir.name,
                    prompt_idx,
                    sentence,
                    len(frames),
                    stats.processed_duration_ms,
                    avg_fps,
                    created_at,
                    out_session_dir.name,
                ]
            )


def postprocess_session_dir(
    src_session_dir: Path,
    dst_session_dir: Path,
    target_fps: float = DEFAULT_TARGET_FPS,
    trim_idle: bool = True,
    manifest_path: Path | None = None,
) -> PostProcessStats:
    """Load a raw session directory, run the post-processing pipeline, and write to dst_session_dir."""
    raw_frames, sentence, session_cfg = load_session(src_session_dir)
    mannequin_cfg = session_cfg or MannequinConfig()

    prompt_idx = 0
    meta_path = src_session_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prompt_idx = int(meta.get("prompt_index", 0))
        except Exception:
            pass

    proc_frames, stats = postprocess_frames(
        raw_frames=raw_frames,
        target_fps=target_fps,
        trim_idle=trim_idle,
    )
    save_processed_session(
        out_session_dir=dst_session_dir,
        frames=proc_frames,
        sentence=sentence,
        prompt_idx=prompt_idx,
        mannequin_cfg=mannequin_cfg,
        stats=stats,
        manifest_path=manifest_path,
    )
    return stats
