"""Retargeting Smartgloves II + Face Mesh streams onto a VRM-compatible humanoid rig.

Computes:
1. Relative finger flexion from (q_hub^-1 * q_finger), distributed over 3 phalanges
   using configurable heuristic weights.
2. Analytical 2-bone arm IK from shoulder anchor toward class-specific torso-relative
   signing zone + dynamic motion arc, combined with forearm/wrist orientation.
3. Neutral rest pose fade-in and fade-out.
4. Face Mesh 52 ARKit blendshapes -> avatar morph targets + head rotation.
"""

from __future__ import annotations

import math
from typing import Any

from pipeline.glove_loader import CleanedFaceClip, CleanedHandClip
from pipeline.signal_processing import (
    Quat,
    Vec3,
    enforce_quaternion_continuity,
    quat_angle_rad,
    quat_conj,
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_mul,
    quat_normalize,
    quat_slerp,
)

# Neutral standing arm pose (arms relaxed gently downward along sides)
REST_POSE_BONES: dict[str, Quat] = {
    "hips": (0.0, 0.0, 0.0, 1.0),
    "spine": (0.0, 0.0, 0.0, 1.0),
    "chest": (0.0, 0.0, 0.0, 1.0),
    "neck": (0.0, 0.0, 0.0, 1.0),
    "head": (0.0, 0.0, 0.0, 1.0),
    "rightShoulder": (0.0, 0.0, 0.0, 1.0),
    "rightUpperArm": quat_from_euler_xyz(0.10, 0.05, -0.18),
    "rightLowerArm": quat_from_euler_xyz(-0.22, 0.0, 0.0),
    "rightHand": (0.0, 0.0, 0.0, 1.0),
    "leftShoulder": (0.0, 0.0, 0.0, 1.0),
    "leftUpperArm": quat_from_euler_xyz(0.10, -0.05, 0.18),
    "leftLowerArm": quat_from_euler_xyz(-0.22, 0.0, 0.0),
    "leftHand": (0.0, 0.0, 0.0, 1.0),
}

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
PHALANX_SUFFIXES = ("Proximal", "Intermediate", "Distal")

# Approximate shoulder joint positions in character torso space (meters, Y-up, +Z forward)
SHOULDER_POS = {
    "right": (-0.18, 1.42, 0.0),
    "left": (0.18, 1.42, 0.0),
}


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_len(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_norm(v: Vec3) -> Vec3:
    l = _vec_len(v)
    if l < 1e-9:
        return (0.0, -1.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def _vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _quat_from_two_vectors(v_from: Vec3, v_to: Vec3) -> Quat:
    u = _vec_norm(v_from)
    v = _vec_norm(v_to)
    dot = max(-1.0, min(1.0, _vec_dot(u, v)))
    if dot > 0.9999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.9999:
        ortho = _vec_cross((1.0, 0.0, 0.0), u)
        if _vec_len(ortho) < 1e-4:
            ortho = _vec_cross((0.0, 1.0, 0.0), u)
        return quat_from_axis_angle(_vec_norm(ortho), math.pi)
    axis = _vec_cross(u, v)
    w = 1.0 + dot
    return quat_normalize((axis[0], axis[1], axis[2], w))


def _rotate_vec_by_quat(v: Vec3, q: Quat) -> Vec3:
    qx, qy, qz, qw = q
    vx, vy, vz = v
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )


def _solve_two_bone_arm_ik(
    shoulder: Vec3,
    target_wrist: Vec3,
    upper_len: float,
    lower_len: float,
    is_right: bool,
) -> tuple[Quat, Quat, Vec3]:
    """Analytical 2-bone IK returning (upperArm_local_quat, lowerArm_local_quat, achieved_wrist).

    Rest bone direction for both arms in our humanoid rig is straight down (0, -1, 0),
    with elbow bending forward (+Z) and slightly outward.
    """
    diff = _vec_sub(target_wrist, shoulder)
    dist = _vec_len(diff)
    max_reach = (upper_len + lower_len) * 0.985
    min_reach = abs(upper_len - lower_len) + 0.05
    clamped_dist = max(min_reach, min(max_reach, dist))
    dir_sw = _vec_norm(diff)

    # Law of cosines for elbow interior angle
    cos_a = (upper_len * upper_len + clamped_dist * clamped_dist - lower_len * lower_len) / (
        2.0 * upper_len * clamped_dist
    )
    cos_a = max(-1.0, min(1.0, cos_a))
    angle_a = math.acos(cos_a)

    # Pole direction: outward & slightly backward so elbow looks natural when signing in front
    pole_hint: Vec3 = (-0.55 if is_right else 0.55, -0.35, -0.75)
    # Project pole_hint perpendicular to dir_sw
    proj = _vec_dot(pole_hint, dir_sw)
    bend_dir = _vec_norm(
        (
            pole_hint[0] - proj * dir_sw[0],
            pole_hint[1] - proj * dir_sw[1],
            pole_hint[2] - proj * dir_sw[2],
        )
    )

    elbow_pos: Vec3 = (
        shoulder[0] + upper_len * (math.cos(angle_a) * dir_sw[0] + math.sin(angle_a) * bend_dir[0]),
        shoulder[1] + upper_len * (math.cos(angle_a) * dir_sw[1] + math.sin(angle_a) * bend_dir[1]),
        shoulder[2] + upper_len * (math.cos(angle_a) * dir_sw[2] + math.sin(angle_a) * bend_dir[2]),
    )
    achieved_wrist: Vec3 = (
        shoulder[0] + clamped_dist * dir_sw[0],
        shoulder[1] + clamped_dist * dir_sw[1],
        shoulder[2] + clamped_dist * dir_sw[2],
    )

    upper_dir = _vec_norm(_vec_sub(elbow_pos, shoulder))
    lower_dir = _vec_norm(_vec_sub(achieved_wrist, elbow_pos))

    # Rest local bone direction is (0, -1, 0)
    q_upper = _quat_from_two_vectors((0.0, -1.0, 0.0), upper_dir)
    # Express lower_dir in upperArm local coordinates
    local_lower_dir = _rotate_vec_by_quat(lower_dir, quat_conj(q_upper))
    q_lower = _quat_from_two_vectors((0.0, -1.0, 0.0), local_lower_dir)

    return q_upper, q_lower, achieved_wrist


def _compute_anchor_trajectory(
    anchor_cfg: dict[str, Any],
    u: float,
    is_right: bool,
) -> Vec3:
    base = anchor_cfg["right_hand"] if is_right else anchor_cfg["left_hand"]
    amp = anchor_cfg.get("motion_amplitude", [0.0, 0.0, 0.0])
    arc = anchor_cfg.get("motion_arc", "wave_side")
    active = is_right or bool(anchor_cfg.get("two_handed_override", False))

    if not active and not is_right:
        return (float(base[0]), float(base[1]), float(base[2]))

    # Active envelope within clip
    env = math.sin(math.pi * u)
    phase = 2.0 * math.pi * 2.0 * u
    dx, dy, dz = 0.0, 0.0, 0.0
    if arc == "wave_side":
        dx = amp[0] * math.sin(phase) * env
        dy = amp[1] * math.cos(phase) * env
    elif arc == "chin_forward":
        dz = amp[2] * math.sin(math.pi * u)
        dy = amp[1] * math.sin(math.pi * u)
    elif arc == "chest_circle":
        dx = amp[0] * math.cos(phase) * env
        dy = amp[1] * math.sin(phase) * env
    elif arc == "symmetric_open":
        sign = -1.0 if is_right else 1.0
        dx = sign * amp[0] * math.sin(math.pi * u)
        dz = amp[2] * math.sin(math.pi * u)
    elif arc == "chest_to_front":
        dx = amp[0] * u
        dz = amp[2] * math.sin(math.pi * u)
    elif arc == "temple_twist":
        dx = amp[0] * math.sin(phase) * env
        dz = amp[2] * env
    elif arc == "lift_together":
        dy = amp[1] * math.sin(math.pi * u)
        dz = amp[2] * math.sin(math.pi * u)
    elif arc == "chin_tap":
        dy = amp[1] * abs(math.sin(phase)) * env
    elif arc == "wrist_shake":
        dx = amp[0] * math.sin(phase * 1.4) * env
    elif arc == "wave_farewell":
        dx = amp[0] * math.sin(phase) * env
        dy = amp[1] * math.sin(math.pi * u)

    return (float(base[0]) + dx, float(base[1]) + dy, float(base[2]) + dz)


def retarget_clip_to_avatar(
    cls_entry: dict[str, Any],
    right_clip: CleanedHandClip,
    left_clip: CleanedHandClip,
    face_clip: CleanedFaceClip,
    sensor_cfg: dict[str, Any],
    prototype_id: str,
) -> dict[str, Any]:
    """Convert cleaned glove + face streams into precomputed Three.js AnimationClip JSON."""
    ret_cfg = sensor_cfg["retargeting"]
    weights_cfg = ret_cfg["finger_curl_weights"]
    arm_len = ret_cfg["arm_lengths"]
    upper_len = float(arm_len.get("upper_arm", 0.28))
    lower_len = float(arm_len.get("lower_arm", 0.25))
    fade_frames = int(ret_cfg.get("fade_frames", 8))
    face_map: dict[str, str] = ret_cfg["face_blendshape_map"]
    axis_remap_r = quat_from_euler_xyz(
        *[math.radians(a) for a in ret_cfg["axis_remap_euler_deg"]["right"]]
    )
    axis_remap_l = quat_from_euler_xyz(
        *[math.radians(a) for a in ret_cfg["axis_remap_euler_deg"]["left"]]
    )

    times = right_clip.times
    n_frames = len(times)
    anchor_cfg = dict(cls_entry["anchor"])
    anchor_cfg["two_handed_override"] = bool(cls_entry.get("two_handed", False))

    bone_tracks: dict[str, list[Quat]] = {name: [] for name in REST_POSE_BONES}
    for side in ("right", "left"):
        for f in FINGER_NAMES:
            for ph in PHALANX_SUFFIXES:
                bone_tracks[f"{side}{f.capitalize()}{ph}"] = []

    morph_targets = sorted(set(face_map.values()))
    morph_tracks: dict[str, list[float]] = {m: [] for m in morph_targets}

    # QA telemetry per frame for side-by-side comparison
    qa_frames: list[dict[str, Any]] = []

    # Neutral frame 0 calibration for hub orientation
    r_hub0_inv = quat_conj(right_clip.hub_orientations[0])
    l_hub0_inv = quat_conj(left_clip.hub_orientations[0])

    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        # Smooth rest-pose fade envelope (0 at ends, 1 during active signing)
        fade = 1.0
        if fade_frames > 0:
            if i < fade_frames:
                fade = 0.5 - 0.5 * math.cos(math.pi * (i / fade_frames))
            elif i >= n_frames - fade_frames:
                rem = max(0, n_frames - 1 - i)
                fade = 0.5 - 0.5 * math.cos(math.pi * (rem / fade_frames))

        # Torso subtle breathing / engagement
        bone_tracks["hips"].append((0.0, 0.0, 0.0, 1.0))
        bone_tracks["spine"].append(quat_from_euler_xyz(0.02 * fade, 0.0, 0.0))
        bone_tracks["chest"].append(quat_from_euler_xyz(0.03 * fade, 0.0, 0.0))
        bone_tracks["neck"].append((0.0, 0.0, 0.0, 1.0))
        bone_tracks["rightShoulder"].append((0.0, 0.0, 0.0, 1.0))
        bone_tracks["leftShoulder"].append((0.0, 0.0, 0.0, 1.0))

        # Head from Face Mesh (blended with rest pose)
        q_head_raw = face_clip.head_rotations[i] if i < len(face_clip.head_rotations) else (0.0, 0.0, 0.0, 1.0)
        bone_tracks["head"].append(quat_slerp(REST_POSE_BONES["head"], q_head_raw, fade))

        # Arms & hands
        hand_qa: dict[str, Any] = {}
        for side, clip, hub0_inv, axis_remap in (
            ("right", right_clip, r_hub0_inv, axis_remap_r),
            ("left", left_clip, l_hub0_inv, axis_remap_l),
        ):
            is_right = side == "right"
            active_hand = is_right or bool(cls_entry.get("two_handed", False))
            target_pos = _compute_anchor_trajectory(anchor_cfg, u, is_right)
            q_upper_ik, q_lower_ik, achieved_pos = _solve_two_bone_arm_ik(
                SHOULDER_POS[side], target_pos, upper_len, lower_len, is_right
            )

            if active_hand:
                q_upper = quat_slerp(REST_POSE_BONES[f"{side}UpperArm"], q_upper_ik, fade)
                q_lower = quat_slerp(REST_POSE_BONES[f"{side}LowerArm"], q_lower_ik, fade)
            else:
                q_upper = REST_POSE_BONES[f"{side}UpperArm"]
                q_lower = REST_POSE_BONES[f"{side}LowerArm"]

            bone_tracks[f"{side}UpperArm"].append(q_upper)
            bone_tracks[f"{side}LowerArm"].append(q_lower)

            # Relative hand orientation from calibrated hub sensor
            q_hub_cal = quat_mul(axis_remap, quat_mul(hub0_inv, clip.hub_orientations[i]))
            q_hand = quat_slerp(REST_POSE_BONES[f"{side}Hand"], q_hub_cal, fade if active_hand else fade * 0.25)
            bone_tracks[f"{side}Hand"].append(q_hand)

            # Finger curls relative to hub back-of-hand orientation
            q_hub_inv = quat_conj(clip.hub_orientations[i])
            g2_curls: dict[str, float] = {}
            g1_curls: dict[str, float] = {}
            for f in FINGER_NAMES:
                q_f2 = clip.orientations.get(f, clip.hub_orientations)[i]
                q_f1 = clip.group1_orientations.get(f, clip.hub_orientations)[i]
                curl_rad = quat_angle_rad((0.0, 0.0, 0.0, 1.0), quat_mul(q_hub_inv, q_f2))
                curl_rad_g1 = quat_angle_rad((0.0, 0.0, 0.0, 1.0), quat_mul(q_hub_inv, q_f1))
                curl_clamped = max(0.0, min(2.35, curl_rad)) * (fade if active_hand else 0.35)
                g2_curls[f] = round(math.degrees(curl_clamped), 2)
                g1_curls[f] = round(math.degrees(max(0.0, min(2.35, curl_rad_g1)) * (fade if active_hand else 0.35)), 2)

                ph_weights = weights_cfg.get(f, [0.45, 0.35, 0.20])
                for p_idx, ph in enumerate(PHALANX_SUFFIXES):
                    bname = f"{side}{f.capitalize()}{ph}"
                    angle = curl_clamped * float(ph_weights[p_idx])
                    # Thumb curls inward on Z/X depending on handedness; fingers flex around +X
                    if f == "thumb":
                        axis = (-0.55, 0.25 if is_right else -0.25, -0.78 if is_right else 0.78)
                    else:
                        axis = (-1.0, 0.0, 0.0)
                    bone_tracks[bname].append(quat_from_axis_angle(axis, angle))

            hand_qa[side] = {
                "target_wrist": [round(x, 4) for x in target_pos],
                "achieved_wrist": [round(x, 4) for x in achieved_pos],
                "curls_deg_g2": g2_curls,
                "curls_deg_g1": g1_curls,
                "mean_emf_residual": round(
                    sum(clip.emf_residuals.get(f, [0.0])[i] for f in FINGER_NAMES) / len(FINGER_NAMES),
                    4,
                ),
            }

        # Face morph targets (max aggregation when multiple ARKit shapes map to one target)
        frame_morphs = {m: 0.0 for m in morph_targets}
        for arkit_name, target_name in face_map.items():
            val = face_clip.blendshapes.get(arkit_name, [0.0] * n_frames)[i] * fade
            if val > frame_morphs[target_name]:
                frame_morphs[target_name] = val
        for m in morph_targets:
            morph_tracks[m].append(round(max(0.0, min(1.0, frame_morphs[m])), 4))

        qa_frames.append(
            {
                "t": round(times[i], 4),
                "right": hand_qa["right"],
                "left": hand_qa["left"],
                "morphs": {m: morph_tracks[m][-1] for m in morph_targets},
            }
        )

    # Enforce quaternion sign continuity across every exported bone track
    serialized_tracks: dict[str, list[float]] = {}
    for bname, quats in bone_tracks.items():
        cont = enforce_quaternion_continuity(quats)
        flat: list[float] = []
        for q in cont:
            flat.extend([round(q[0], 5), round(q[1], 5), round(q[2], 5), round(q[3], 5)])
        serialized_tracks[bname] = flat

    return {
        "classId": cls_entry["id"],
        "phrase": cls_entry["phrase"],
        "category": cls_entry["category"],
        "description": cls_entry["description"],
        "prototypeClipId": prototype_id,
        "fps": int(sensor_cfg["cleaning"].get("target_fps", 30)),
        "duration": round(times[-1], 4),
        "times": [round(t, 4) for t in times],
        "boneTracks": serialized_tracks,
        "morphTracks": morph_tracks,
        "qaComparison": {
            "group1VsGroup2Right": [vars(s) for s in right_clip.comparison_stats],
            "group1VsGroup2Left": [vars(s) for s in left_clip.comparison_stats],
            "frames": qa_frames,
        },
    }
