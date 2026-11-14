"""Synthetic Smartgloves II CSV + Face Mesh dataset generator for smoke testing.

Generates realistic multi-signer, multi-clip synthetic recordings matching the
exact Smartgloves II header layout (including initial reset frame, ~83 Hz device
local microsecond timestamps, irregular global host timestamps, Group 1 raw +
Hub + Group 2 EM-corrected streams, occasional dropouts and EMF spikes) and
synchronized ~30 fps Face Mesh (52 ARKit blendshapes + head pose).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from pipeline.signal_processing import (
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_mul,
    quat_normalize,
)
from pipeline.yaml_io import load_yaml

ARKIT_52_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]

# Class-specific finger curl targets (0.0 = extended, 1.0 = fully curled fist)
# Order: thumb, index, middle, ring, little
CLASS_HANDSHAPES: dict[str, dict[str, Any]] = {
    "ciao": {"r_curls": [0.05, 0.05, 0.05, 0.05, 0.05], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "smile"},
    "grazie": {"r_curls": [0.08, 0.06, 0.06, 0.06, 0.06], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "smile"},
    "per_favore": {"r_curls": [0.10, 0.08, 0.08, 0.08, 0.08], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "polite"},
    "come_stai": {"r_curls": [0.15, 0.12, 0.12, 0.15, 0.15], "l_curls": [0.15, 0.12, 0.12, 0.15, 0.15], "face": "question"},
    "mi_chiamo": {"r_curls": [0.65, 0.06, 0.85, 0.88, 0.90], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "neutral"},
    "non_capisco": {"r_curls": [0.30, 0.12, 0.78, 0.82, 0.85], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "confused"},
    "aiuto": {"r_curls": [0.75, 0.88, 0.90, 0.90, 0.90], "l_curls": [0.08, 0.06, 0.06, 0.06, 0.06], "face": "urgent"},
    "acqua": {"r_curls": [0.55, 0.10, 0.82, 0.85, 0.88], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "mouth_open"},
    "bagno": {"r_curls": [0.50, 0.12, 0.20, 0.82, 0.85], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "question"},
    "arrivederci": {"r_curls": [0.12, 0.10, 0.10, 0.10, 0.10], "l_curls": [0.15, 0.15, 0.15, 0.15, 0.15], "face": "smile"},
}


def build_glove_headers(sg_cfg: dict[str, Any]) -> list[str]:
    headers: list[str] = [
        "",  # Unnamed column 0: device label
        sg_cfg["local_timestamp_header"],
        sg_cfg["global_timestamp_header"],
    ]
    g1_pat = sg_cfg["header_patterns"]["group1"]
    for sensor in sg_cfg["sensors_group1"]:
        for i in range(7):
            headers.append(g1_pat["transform"].format(sensor=sensor, i=i))
        for i in range(3):
            headers.append(g1_pat["angular_velocity"].format(sensor=sensor, i=i))
        for i in range(3):
            headers.append(g1_pat["proper_acceleration"].format(sensor=sensor, i=i))
        headers.append(g1_pat["status_flags"].format(sensor=sensor))

    hub_pat = sg_cfg["header_patterns"]["hub"]
    for i in range(4):
        headers.append(hub_pat["orientation"].format(i=i))
    for i in range(3):
        headers.append(hub_pat["angular_velocity"].format(i=i))
    for i in range(3):
        headers.append(hub_pat["proper_acceleration"].format(i=i))
    headers.append(hub_pat["status_flags"])

    g2_pat = sg_cfg["header_patterns"]["group2"]
    for sensor in sg_cfg["sensors_group2"]:
        for i in range(7):
            headers.append(g2_pat["transform"].format(sensor=sensor, i=i))
        headers.append(g2_pat["emf_residual"].format(sensor=sensor))
        headers.append(g2_pat["status_flags"].format(sensor=sensor))

    return headers


def _reset_row(device_label: str, sg_cfg: dict[str, Any], clean_cfg: dict[str, Any]) -> list[Any]:
    sig = clean_cfg["reset_position_signature"]
    row: list[Any] = [device_label, 0, 1700000000000000]
    for _ in sg_cfg["sensors_group1"]:
        row.extend([sig[0], sig[1], sig[2], 0.0, 0.0, 0.0, 1.0])
        row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    row.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    for _ in sg_cfg["sensors_group2"]:
        row.extend([sig[0], sig[1], sig[2], 0.0, 0.0, 0.0, 1.0, 0.0, 0])
    return row


def _gesture_envelope(u: float) -> float:
    """Smooth bell curve peaking during the active sign hold (0 <= u <= 1)."""
    if u < 0.18:
        return 0.5 - 0.5 * math.cos(math.pi * (u / 0.18))
    if u > 0.82:
        return 0.5 - 0.5 * math.cos(math.pi * ((1.0 - u) / 0.18))
    return 1.0


def generate_clip(
    out_dir: Path,
    cls_entry: dict[str, Any],
    signer_idx: int,
    clip_idx: int,
    config: dict[str, Any],
    rng: random.Random,
) -> dict[str, str]:
    sg_cfg = config["smartgloves"]
    clean_cfg = config["cleaning"]
    headers = build_glove_headers(sg_cfg)

    cls_id = cls_entry["id"]
    base_dur = float(cls_entry["anchor"].get("duration_sec", 1.5))
    tempo_scale = 0.88 + 0.08 * signer_idx + rng.uniform(-0.04, 0.04)
    duration = base_dur * tempo_scale
    hz = float(sg_cfg.get("expected_sample_rate_hz", 83.33))
    n_samples = max(30, int(round(duration * hz)))

    shape = CLASS_HANDSHAPES.get(cls_id, CLASS_HANDSHAPES["ciao"])
    clip_stem = f"{cls_id}_s{signer_idx:02d}_c{clip_idx:02d}"
    clip_dir = out_dir / cls_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    hand_paths: dict[str, str] = {}
    for hand in ("right", "left"):
        is_right = hand == "right"
        active_hand = is_right or bool(cls_entry.get("two_handed", False))
        curls = shape["r_curls"] if is_right else shape["l_curls"]
        device_label = f"SG2-{'R' if is_right else 'L'}-00{signer_idx + 1}"
        csv_path = clip_dir / f"{clip_stem}_{hand}.csv"

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            # Row 0: mandatory reset frame
            writer.writerow(_reset_row(device_label, sg_cfg, clean_cfg))

            t_us = 150000
            global_us = 1700000000000000
            for idx in range(n_samples):
                u = idx / max(1, n_samples - 1)
                env = _gesture_envelope(u)
                osc = math.sin(2.0 * math.pi * 2.0 * u) * env
                dt_us = int(1_000_000 / hz + rng.randint(-350, 350))
                t_us += max(5000, dt_us)
                global_us += dt_us + rng.randint(-1200, 3500)

                # Hub wrist orientation (smooth sign-dependent rotation)
                wrist_deg = cls_entry["anchor"].get(
                    "right_wrist_euler_deg" if is_right else "left_wrist_euler_deg",
                    [0.0, 0.0, 0.0],
                )
                rx = math.radians(wrist_deg[0]) * env + (0.14 * osc if active_hand else 0.02 * env)
                ry = math.radians(wrist_deg[1]) * env + (0.18 * osc if cls_id in ("ciao", "arrivederci", "bagno") and is_right else 0.0)
                rz = math.radians(wrist_deg[2]) * env + (0.20 * osc if cls_id == "non_capisco" and is_right else 0.0)
                q_hub = quat_from_euler_xyz(rx, ry, rz)
                q_arm = quat_from_euler_xyz(rx * 0.65, ry * 0.55, rz * 0.45)

                # Sensor local offsets & orientations for Group 1 and Group 2
                sensor_data_g1: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float], int]] = {}
                sensor_data_g2: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float], float, int]] = {}

                # Inject rare synthetic dropout or EMF spike for testing cleaning pipeline
                inject_dropout = idx == int(n_samples * 0.45) and clip_idx == 0
                inject_emf_spike = idx == int(n_samples * 0.62) and clip_idx == 0

                finger_names = ["thumb", "index", "middle", "ring", "little finger"]
                for f_idx, fname in enumerate(finger_names):
                    curl_rad = curls[f_idx] * 1.65 * env + (0.12 * osc if cls_id == "arrivederci" and is_right else 0.0)
                    # Add slight signer variation
                    curl_rad += rng.gauss(0.0, 0.015)
                    q_local_flex = quat_from_axis_angle((1.0, 0.0, 0.0), curl_rad)
                    q_finger_g2 = quat_mul(q_hub, q_local_flex)
                    # Group 1 has minor EM drift relative to Group 2
                    drift = quat_from_euler_xyz(0.03 * math.sin(u * 4.0), -0.02 * math.cos(u * 3.0), 0.015)
                    q_finger_g1 = quat_normalize(quat_mul(q_finger_g2, drift))
                    p_g2 = (0.02 * (f_idx - 2), 0.01, 0.07 - 0.03 * curls[f_idx] * env)
                    p_g1 = (p_g2[0] + 0.002 * math.sin(u * 6.0), p_g2[1] - 0.001, p_g2[2])

                    emf_res = abs(rng.gauss(0.02, 0.01))
                    flag1 = 0
                    flag2 = 0
                    if inject_dropout and fname == "index":
                        p_g1 = (0.0, 0.0, 0.0)
                        q_finger_g1 = (0.0, 0.0, 0.0, 0.0)
                        p_g2 = (0.0, 0.0, 0.0)
                        q_finger_g2 = (0.0, 0.0, 0.0, 0.0)
                        flag1 = 4
                        flag2 = 4
                    elif inject_emf_spike and fname == "middle":
                        emf_res = 0.85

                    sensor_data_g1[fname] = (p_g1, q_finger_g1, flag1)
                    canon = "little" if "little" in fname else fname
                    sensor_data_g2[canon] = (p_g2, q_finger_g2, emf_res, flag2)

                sensor_data_g1["arm"] = ((0.0, -0.05, -0.18), q_arm, 0)
                sensor_data_g2["arm"] = ((0.0, -0.05, -0.18), q_arm, abs(rng.gauss(0.015, 0.005)), 0)
                sensor_data_g2["arm_2"] = sensor_data_g2["arm"]

                row: list[Any] = [device_label, t_us, global_us]
                for sensor in sg_cfg["sensors_group1"]:
                    p, q, flag = sensor_data_g1[sensor]
                    row.extend([round(p[0], 6), round(p[1], 6), round(p[2], 6)])
                    row.extend([round(q[0], 6), round(q[1], 6), round(q[2], 6), round(q[3], 6)])
                    row.extend([round(rng.gauss(0.0, 0.05), 4) for _ in range(3)])
                    row.extend([0.0, 9.81, 0.0])
                    row.append(flag)

                row.extend([round(q_hub[0], 6), round(q_hub[1], 6), round(q_hub[2], 6), round(q_hub[3], 6)])
                row.extend([0.0, 0.0, 0.0, 0.0, 9.81, 0.0, 0])

                for sensor in sg_cfg["sensors_group2"]:
                    p, q, emf, flag = sensor_data_g2[sensor]
                    row.extend([round(p[0], 6), round(p[1], 6), round(p[2], 6)])
                    row.extend([round(q[0], 6), round(q[1], 6), round(q[2], 6), round(q[3], 6)])
                    row.extend([round(emf, 5), flag])

                writer.writerow(row)
        hand_paths[hand] = str(csv_path)

    # Synchronized Face Mesh ~30fps
    face_fps = 30.0
    n_face = max(12, int(round(duration * face_fps)))
    face_frames: list[dict[str, Any]] = []
    face_mode = shape["face"]
    for i in range(n_face):
        u = i / max(1, n_face - 1)
        env = _gesture_envelope(u)
        blend = {name: 0.0 for name in ARKIT_52_BLENDSHAPES}
        # Subtle natural blink
        if 0.48 < u < 0.54:
            blend["eyeBlinkLeft"] = 0.85
            blend["eyeBlinkRight"] = 0.85

        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        if face_mode == "smile":
            blend["mouthSmileLeft"] = round(0.75 * env, 4)
            blend["mouthSmileRight"] = round(0.75 * env, 4)
            blend["browInnerUp"] = round(0.25 * env, 4)
            pitch = 0.06 * math.sin(math.pi * u)
        elif face_mode == "polite":
            blend["mouthSmileLeft"] = round(0.45 * env, 4)
            blend["mouthSmileRight"] = round(0.45 * env, 4)
            blend["browInnerUp"] = round(0.35 * env, 4)
            pitch = 0.08 * env
        elif face_mode == "question":
            blend["browInnerUp"] = round(0.85 * env, 4)
            blend["browOuterUpLeft"] = round(0.65 * env, 4)
            blend["browOuterUpRight"] = round(0.65 * env, 4)
            blend["jawOpen"] = round(0.22 * env, 4)
            roll = 0.08 * env
            pitch = -0.06 * env
        elif face_mode == "confused":
            blend["browDownLeft"] = round(0.85 * env, 4)
            blend["browDownRight"] = round(0.85 * env, 4)
            blend["mouthFrownLeft"] = round(0.55 * env, 4)
            blend["mouthFrownRight"] = round(0.55 * env, 4)
            yaw = 0.14 * math.sin(4.0 * math.pi * u) * env
        elif face_mode == "urgent":
            blend["browInnerUp"] = round(0.75 * env, 4)
            blend["eyeWideLeft"] = round(0.65 * env, 4)
            blend["eyeWideRight"] = round(0.65 * env, 4)
            blend["jawOpen"] = round(0.45 * env, 4)
            pitch = -0.05 * env
        elif face_mode == "mouth_open":
            blend["jawOpen"] = round(0.48 * env, 4)
            blend["mouthPucker"] = round(0.40 * env, 4)
        else:
            blend["mouthSmileLeft"] = round(0.25 * env, 4)
            blend["mouthSmileRight"] = round(0.25 * env, 4)

        q_head = quat_from_euler_xyz(pitch, yaw, roll)
        face_frames.append(
            {
                "timestamp_sec": round(i / face_fps + 0.05, 4),
                "head_quaternion_xyzw": [round(c, 5) for c in q_head],
                "blendshapes": blend,
            }
        )

    face_path = clip_dir / f"{clip_stem}_face.json"
    face_path.write_text(
        json.dumps({"sync_offset_sec": 0.05, "fps": face_fps, "frames": face_frames}, indent=2),
        encoding="utf-8",
    )

    return {
        "class_id": cls_id,
        "signer": f"s{signer_idx:02d}",
        "clip_id": clip_stem,
        "right_csv": hand_paths["right"],
        "left_csv": hand_paths["left"],
        "face_json": str(face_path),
    }


def generate_synthetic_dataset(
    dictionary_path: Path,
    sensor_cfg_path: Path,
    output_dir: Path,
    clips_per_class: int = 4,
    signers: int = 4,
    seed: int = 2026,
) -> list[dict[str, str]]:
    dict_cfg = load_yaml(dictionary_path)
    sensor_cfg = load_yaml(sensor_cfg_path)
    rng = random.Random(seed)
    manifest: list[dict[str, str]] = []

    for cls_entry in dict_cfg["classes"]:
        for idx in range(clips_per_class):
            signer_idx = idx % signers
            clip_idx = idx // signers
            manifest.append(
                generate_clip(output_dir, cls_entry, signer_idx, clip_idx, sensor_cfg, rng)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Smartgloves II + Face Mesh dataset.")
    parser.add_argument("--dictionary", default="config/dictionary.yaml")
    parser.add_argument("--config", default="config/sensors_and_retargeting.yaml")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--clips-per-class", type=int, default=4)
    parser.add_argument("--signers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = generate_synthetic_dataset(
        Path(args.dictionary),
        Path(args.config),
        Path(args.output),
        clips_per_class=args.clips_per_class,
        signers=args.signers,
        seed=args.seed,
    )
    print(f"Generated {len(manifest)} synthetic LIS clips in {args.output}")


if __name__ == "__main__":
    main()
