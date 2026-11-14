"""Dynamic header-driven loader and cleaner for Smartgloves II CSV and Face Mesh streams.

No column indices are hardcoded beyond optional device label col 0: every column
is resolved dynamically from the CSV header row using config patterns.
Includes explicit diagnostics comparing Group 1 (raw) vs Group 2 (EM-corrected) sensors.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.signal_processing import (
    Quat,
    Vec3,
    build_uniform_timestamps,
    interpolate_quat_dropouts,
    interpolate_scalar_dropouts,
    interpolate_vec3_dropouts,
    quat_angle_rad,
    quat_norm,
    resample_quats,
    resample_scalars,
    resample_vec3,
    smooth_quat_one_euro,
    smooth_vec3_one_euro,
    to_xyzw,
)


@dataclass
class SensorComparisonStats:
    sensor_name: str
    rms_position_diff_m: float
    rms_angle_diff_deg: float
    mean_emf_residual: float
    max_emf_residual: float
    group1_dropout_rate: float
    group2_dropout_rate: float


@dataclass
class CleanedHandClip:
    hand: str
    times: list[float]
    reset_frames_discarded: int
    raw_frame_count: int
    hub_orientations: list[Quat]
    # Preferred sensor orientations and positions per canonical sensor key
    # Keys: "thumb", "index", "middle", "ring", "little", "arm"
    positions: dict[str, list[Vec3]]
    orientations: dict[str, list[Quat]]
    # Kept separately for QA visual comparison (Group 1 raw vs Group 2 EM-corrected)
    group1_orientations: dict[str, list[Quat]]
    group2_orientations: dict[str, list[Quat]]
    emf_residuals: dict[str, list[float]]
    comparison_stats: list[SensorComparisonStats] = field(default_factory=list)


@dataclass
class CleanedFaceClip:
    times: list[float]
    head_rotations: list[Quat]
    blendshapes: dict[str, list[float]]


def _canonical_sensor_key(name: str) -> str:
    low = name.lower().strip()
    if "little" in low:
        return "little"
    if low.startswith("arm"):
        return "arm"
    return low


def build_header_map(headers: list[str]) -> dict[str, int]:
    """Map stripped header strings (case-preserved and lower-cased) to column indices."""
    mapping: dict[str, int] = {}
    for idx, col in enumerate(headers):
        clean = col.strip()
        if clean and clean not in mapping:
            mapping[clean] = idx
            mapping[clean.lower()] = idx
    return mapping


def _find_col(header_map: dict[str, int], candidate: str) -> int | None:
    if candidate in header_map:
        return header_map[candidate]
    low = candidate.lower()
    if low in header_map:
        return header_map[low]
    return None


def _is_reset_row(
    row: list[str],
    header_map: dict[str, int],
    sg_cfg: dict[str, Any],
    clean_cfg: dict[str, Any],
) -> bool:
    """Detect initial Smartgloves II reset frames where all sensors sit at the reset signature."""
    sig = clean_cfg.get("reset_position_signature", [-0.018835, 0.006585, 0.0])
    tol = float(clean_cfg.get("reset_position_tolerance", 1e-4))
    tpl = sg_cfg["header_patterns"]["group1"]["transform"]
    matches = 0
    checked = 0
    for sensor in sg_cfg["sensors_group1"]:
        c0 = _find_col(header_map, tpl.format(sensor=sensor, i=0))
        c1 = _find_col(header_map, tpl.format(sensor=sensor, i=1))
        c2 = _find_col(header_map, tpl.format(sensor=sensor, i=2))
        if c0 is None or c1 is None or c2 is None:
            continue
        checked += 1
        try:
            x, y, z = float(row[c0]), float(row[c1]), float(row[c2])
            if (
                abs(x - sig[0]) <= tol
                and abs(y - sig[1]) <= tol
                and abs(z - sig[2]) <= tol
            ):
                matches += 1
        except (ValueError, IndexError):
            continue
    return checked > 0 and matches == checked


def _extract_vec3(row: list[str], cols: list[int | None]) -> Vec3:
    vals = []
    for c in cols:
        if c is None or c >= len(row) or row[c] == "":
            vals.append(0.0)
        else:
            vals.append(float(row[c]))
    return (vals[0], vals[1], vals[2])


def _extract_quat(row: list[str], cols: list[int | None], order: str) -> tuple[Quat, float]:
    vals = []
    for c in cols:
        if c is None or c >= len(row) or row[c] == "":
            vals.append(0.0)
        else:
            vals.append(float(row[c]))
    q_raw = to_xyzw(vals, order=order)
    return q_raw, quat_norm(q_raw)


def _extract_float(row: list[str], col: int | None, default: float = 0.0) -> float:
    if col is None or col >= len(row) or row[col] == "":
        return default
    try:
        return float(row[col])
    except ValueError:
        return default


def load_and_clean_glove_csv(
    csv_path: str | Path,
    hand: str,
    config: dict[str, Any],
    target_times: list[float] | None = None,
) -> CleanedHandClip:
    """Load a Smartgloves II CSV, remove reset frames, compare Group 1 vs Group 2, clean & resample."""
    sg_cfg = config["smartgloves"]
    clean_cfg = config["cleaning"]
    quat_order = sg_cfg.get("quaternion_order", "xyzw")
    ts_scale = float(sg_cfg.get("local_timestamp_unit_scale_to_sec", 1e-6))
    min_q_norm = float(clean_cfg.get("min_quaternion_norm", 0.5))
    max_emf_res = float(clean_cfg.get("max_emf_residual", 0.35))
    zero_eps = float(clean_cfg.get("zero_dropout_epsilon", 1e-6))

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [r for r in reader if r]

    header_map = build_header_map(headers)
    ts_col = _find_col(header_map, sg_cfg["local_timestamp_header"])
    if ts_col is None:
        raise ValueError(
            f"Local timestamp header '{sg_cfg['local_timestamp_header']}' not found in {csv_path}"
        )

    # Discard leading reset frames until valid sensor data begins
    start_idx = 0
    while start_idx < len(rows) and _is_reset_row(rows[start_idx], header_map, sg_cfg, clean_cfg):
        start_idx += 1
    valid_rows = rows[start_idx:]
    if not valid_rows:
        raise ValueError(f"No non-reset frames found in {csv_path}")

    raw_times: list[float] = []
    t0 = _extract_float(valid_rows[0], ts_col) * ts_scale
    prev_t = -1e-6
    for row in valid_rows:
        t = _extract_float(row, ts_col) * ts_scale - t0
        if t <= prev_t:
            t = prev_t + 1.0 / float(sg_cfg.get("expected_sample_rate_hz", 83.33))
        raw_times.append(t)
        prev_t = t

    # Hub orientation
    hub_pat = sg_cfg["header_patterns"]["hub"]
    hub_q_cols = [_find_col(header_map, hub_pat["orientation"].format(i=i)) for i in range(4)]
    hub_flag_col = _find_col(header_map, hub_pat["status_flags"])

    hub_raw_q: list[Quat] = []
    hub_mask: list[bool] = []
    for row in valid_rows:
        q, n = _extract_quat(row, hub_q_cols, quat_order)
        flag = int(_extract_float(row, hub_flag_col, 0.0))
        hub_raw_q.append(q)
        hub_mask.append(n >= min_q_norm and flag == 0)

    hub_clean_q = interpolate_quat_dropouts(raw_times, hub_raw_q, hub_mask)
    hub_smooth_q = smooth_quat_one_euro(
        raw_times,
        hub_clean_q,
        min_cutoff=float(clean_cfg.get("one_euro_min_cutoff", 2.2)),
        beta=float(clean_cfg.get("one_euro_beta", 0.08)),
        d_cutoff=float(clean_cfg.get("one_euro_d_cutoff", 1.0)),
    )

    # Parse Group 1 & Group 2 per sensor
    g1_pat = sg_cfg["header_patterns"]["group1"]
    g2_pat = sg_cfg["header_patterns"]["group2"]

    g1_pos_clean: dict[str, list[Vec3]] = {}
    g1_quat_clean: dict[str, list[Quat]] = {}
    g1_dropouts: dict[str, float] = {}

    for sensor in sg_cfg["sensors_group1"]:
        key = _canonical_sensor_key(sensor)
        p_cols = [_find_col(header_map, g1_pat["transform"].format(sensor=sensor, i=i)) for i in range(3)]
        q_cols = [_find_col(header_map, g1_pat["transform"].format(sensor=sensor, i=i)) for i in range(3, 7)]
        f_col = _find_col(header_map, g1_pat["status_flags"].format(sensor=sensor))

        raw_p: list[Vec3] = []
        raw_q: list[Quat] = []
        mask: list[bool] = []
        for row in valid_rows:
            p = _extract_vec3(row, p_cols)
            q, n = _extract_quat(row, q_cols, quat_order)
            flag = int(_extract_float(row, f_col, 0.0))
            is_zero_pos = (abs(p[0]) + abs(p[1]) + abs(p[2])) < zero_eps
            ok = (not is_zero_pos) and (n >= min_q_norm) and (flag == 0)
            raw_p.append(p)
            raw_q.append(q)
            mask.append(ok)

        g1_dropouts[key] = 1.0 - (sum(1 for m in mask if m) / len(mask))
        g1_pos_clean[key] = smooth_vec3_one_euro(
            raw_times,
            interpolate_vec3_dropouts(raw_times, raw_p, mask),
            min_cutoff=float(clean_cfg.get("one_euro_min_cutoff", 2.2)),
            beta=float(clean_cfg.get("one_euro_beta", 0.08)),
        )
        g1_quat_clean[key] = smooth_quat_one_euro(
            raw_times,
            interpolate_quat_dropouts(raw_times, raw_q, mask),
            min_cutoff=float(clean_cfg.get("one_euro_min_cutoff", 2.2)),
            beta=float(clean_cfg.get("one_euro_beta", 0.08)),
        )

    g2_pos_clean: dict[str, list[Vec3]] = {}
    g2_quat_clean: dict[str, list[Quat]] = {}
    g2_emf_clean: dict[str, list[float]] = {}
    g2_dropouts: dict[str, float] = {}

    for sensor in sg_cfg["sensors_group2"]:
        if sensor == "arm_2":
            # Secondary trailing arm column in Group 2; keep primary arm unless missing
            continue
        key = _canonical_sensor_key(sensor)
        p_cols = [_find_col(header_map, g2_pat["transform"].format(sensor=sensor, i=i)) for i in range(3)]
        q_cols = [_find_col(header_map, g2_pat["transform"].format(sensor=sensor, i=i)) for i in range(3, 7)]
        e_col = _find_col(header_map, g2_pat["emf_residual"].format(sensor=sensor))
        f_col = _find_col(header_map, g2_pat["status_flags"].format(sensor=sensor))

        if any(c is None for c in p_cols + q_cols):
            continue

        raw_p = []
        raw_q = []
        raw_emf = []
        mask = []
        for row in valid_rows:
            p = _extract_vec3(row, p_cols)
            q, n = _extract_quat(row, q_cols, quat_order)
            emf = _extract_float(row, e_col, 0.0)
            flag = int(_extract_float(row, f_col, 0.0))
            is_zero_pos = (abs(p[0]) + abs(p[1]) + abs(p[2])) < zero_eps
            ok = (not is_zero_pos) and (n >= min_q_norm) and (flag == 0) and (emf <= max_emf_res)
            raw_p.append(p)
            raw_q.append(q)
            raw_emf.append(emf)
            mask.append(ok)

        g2_dropouts[key] = 1.0 - (sum(1 for m in mask if m) / len(mask))
        g2_pos_clean[key] = smooth_vec3_one_euro(
            raw_times,
            interpolate_vec3_dropouts(raw_times, raw_p, mask),
            min_cutoff=float(clean_cfg.get("one_euro_min_cutoff", 2.2)),
            beta=float(clean_cfg.get("one_euro_beta", 0.08)),
        )
        g2_quat_clean[key] = smooth_quat_one_euro(
            raw_times,
            interpolate_quat_dropouts(raw_times, raw_q, mask),
            min_cutoff=float(clean_cfg.get("one_euro_min_cutoff", 2.2)),
            beta=float(clean_cfg.get("one_euro_beta", 0.08)),
        )
        g2_emf_clean[key] = interpolate_scalar_dropouts(raw_times, raw_emf, mask)

    # Compare Group 1 vs Group 2 per sensor for QA transparency
    comparison_stats: list[SensorComparisonStats] = []
    for key in g1_pos_clean:
        if key not in g2_pos_clean:
            continue
        p1_list = g1_pos_clean[key]
        p2_list = g2_pos_clean[key]
        q1_list = g1_quat_clean[key]
        q2_list = g2_quat_clean[key]
        emf_list = g2_emf_clean.get(key, [0.0] * len(p1_list))
        sq_pos = 0.0
        sq_ang = 0.0
        n = len(p1_list)
        for p1, p2, q1, q2 in zip(p1_list, p2_list, q1_list, q2_list):
            dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
            sq_pos += dx * dx + dy * dy + dz * dz
            ang_deg = math.degrees(quat_angle_rad(q1, q2))
            sq_ang += ang_deg * ang_deg
        comparison_stats.append(
            SensorComparisonStats(
                sensor_name=key,
                rms_position_diff_m=math.sqrt(sq_pos / max(1, n)),
                rms_angle_diff_deg=math.sqrt(sq_ang / max(1, n)),
                mean_emf_residual=sum(emf_list) / max(1, len(emf_list)),
                max_emf_residual=max(emf_list) if emf_list else 0.0,
                group1_dropout_rate=g1_dropouts.get(key, 0.0),
                group2_dropout_rate=g2_dropouts.get(key, 0.0),
            )
        )

    # Resample to uniform target timestamps (e.g. 30 fps)
    fps = float(clean_cfg.get("target_fps", 30))
    dst_times = target_times if target_times is not None else build_uniform_timestamps(0.0, raw_times[-1], fps)

    preferred = sg_cfg.get("preferred_sensor_group", "group2")
    pref_pos_src = g2_pos_clean if (preferred == "group2" and g2_pos_clean) else g1_pos_clean
    pref_quat_src = g2_quat_clean if (preferred == "group2" and g2_quat_clean) else g1_quat_clean

    res_hub = resample_quats(raw_times, hub_smooth_q, dst_times)
    res_pos = {k: resample_vec3(raw_times, v, dst_times) for k, v in pref_pos_src.items()}
    res_quat = {k: resample_quats(raw_times, v, dst_times) for k, v in pref_quat_src.items()}
    res_g1_q = {k: resample_quats(raw_times, v, dst_times) for k, v in g1_quat_clean.items()}
    res_g2_q = {k: resample_quats(raw_times, v, dst_times) for k, v in g2_quat_clean.items()}
    res_emf = {k: resample_scalars(raw_times, v, dst_times) for k, v in g2_emf_clean.items()}

    return CleanedHandClip(
        hand=hand,
        times=dst_times,
        reset_frames_discarded=start_idx,
        raw_frame_count=len(rows),
        hub_orientations=res_hub,
        positions=res_pos,
        orientations=res_quat,
        group1_orientations=res_g1_q,
        group2_orientations=res_g2_q,
        emf_residuals=res_emf,
        comparison_stats=comparison_stats,
    )


def load_and_clean_face_json(
    face_path: str | Path,
    target_times: list[float],
) -> CleanedFaceClip:
    """Load synchronized Face Mesh stream (52 blendshapes + head pose) and resample to target_times."""
    data = json.loads(Path(face_path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        return CleanedFaceClip(
            times=target_times,
            head_rotations=[(0.0, 0.0, 0.0, 1.0)] * len(target_times),
            blendshapes={},
        )
    sync_offset = float(data.get("sync_offset_sec", 0.0))
    src_times = [max(0.0, float(fr["timestamp_sec"]) - sync_offset) for fr in frames]
    head_quats = [
        to_xyzw(fr.get("head_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]), "xyzw")
        for fr in frames
    ]
    blendshape_names = list(frames[0].get("blendshapes", {}).keys())
    blendshapes: dict[str, list[float]] = {}
    for name in blendshape_names:
        series = [float(fr["blendshapes"].get(name, 0.0)) for fr in frames]
        blendshapes[name] = resample_scalars(src_times, series, target_times)

    return CleanedFaceClip(
        times=target_times,
        head_rotations=resample_quats(src_times, head_quats, target_times),
        blendshapes=blendshapes,
    )
