from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from core.config import MannequinConfig
from core.csv_exporter import RecordedFrame
from core.gloves_tracker import HAND_PARENT_EDGES

VIEW_W, VIEW_H = 1280, 720
STUDIO_W = 880
SIDEBAR_W = VIEW_W - STUDIO_W
SIDEBAR_PAD = 12
CARD_W = SIDEBAR_W - 2 * SIDEBAR_PAD
TOP_CARD_H = 220

STUDIO_TORSO_BONES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("pelvis", "neck"),
    ("neck", "head_base"),
)

STUDIO_UPPER_ARM_BONES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("left_shoulder", "left_elbow", (255, 190, 60)),
    ("right_shoulder", "right_elbow", (70, 180, 255)),
)

_SMOOTHED_MANNEQUIN: dict[str, tuple[float, float, float]] = {}
EMA_ALPHA = 0.40


def reset_mannequin_smoothing(cfg: MannequinConfig | None = None) -> None:
    """Reset the EMA mannequin filter (used when scrubbing/restarting playback)."""
    _SMOOTHED_MANNEQUIN.clear()
    if cfg is not None:
        _SMOOTHED_MANNEQUIN.update(cfg.default_joints())


def resolve_mannequin_joints(
    body_joints: dict[str, tuple[float, float, float, float]],
    cfg: MannequinConfig,
) -> dict[str, tuple[float, float, float]]:
    """Return smoothed 3D upper-body joints driven by the distant body camera and scaled by MannequinConfig."""
    defaults = cfg.default_joints()
    if not body_joints:
        return defaults

    target = dict(defaults)
    ls = body_joints.get("left_shoulder")
    rs = body_joints.get("right_shoulder")
    flip_x = -1.0 if (ls and rs and ls[0] > rs[0]) else 1.0
    h_scale = cfg.height_scale

    for name in ("neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
        bj = body_joints.get(name)
        if bj is not None and bj[3] >= 0.30:
            mx, my, mz, _vis = bj
            target[name] = (
                mx * flip_x * h_scale,
                cfg.shoulder_height_m + my * h_scale,
                max(-0.35, min(0.35, mz * h_scale)),
            )

    neck_x, neck_y, neck_z = target["neck"]
    target["head_base"] = (
        neck_x,
        neck_y + cfg.neck_to_chin_y_m,
        neck_z + cfg.neck_to_chin_z_m,
    )

    for k, (tx, ty, tz) in target.items():
        px, py, pz = _SMOOTHED_MANNEQUIN.get(k, (tx, ty, tz))
        _SMOOTHED_MANNEQUIN[k] = (
            px + EMA_ALPHA * (tx - px),
            py + EMA_ALPHA * (ty - py),
            pz + EMA_ALPHA * (tz - pz),
        )
    return dict(_SMOOTHED_MANNEQUIN)


class Camera3D:
    """Perspective 3D camera for the Rokoko Studio-style viewport."""

    def __init__(
        self,
        width: int,
        height: int,
        yaw_deg: float = 0.0,
        pitch_deg: float = 12.0,
        distance: float = 1.75,
        target: tuple[float, float, float] = (0.0, 1.28, 0.05),
    ) -> None:
        self.width = width
        self.height = height
        self.focal = min(width, height) * 1.15
        self.distance = distance
        self.target = target

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        self.cos_y, self.sin_y = math.cos(yaw), math.sin(yaw)
        self.cos_p, self.sin_p = math.cos(pitch), math.sin(pitch)

    def project(self, pt: tuple[float, float, float]) -> tuple[int, int, float] | None:
        x = pt[0] - self.target[0]
        y = pt[1] - self.target[1]
        z = pt[2] - self.target[2]

        x1 = x * self.cos_y - z * self.sin_y
        z1 = x * self.sin_y + z * self.cos_y

        y2 = y * self.cos_p - z1 * self.sin_p
        z2 = y * self.sin_p + z1 * self.cos_p

        depth = self.distance - z2
        if depth <= 0.12:
            return None

        u = int(self.width * 0.5 + (x1 * self.focal) / depth)
        v = int(self.height * 0.5 - (y2 * self.focal) / depth)
        return u, v, depth

    def project_points_np(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized perspective projection for (N, 3) 3D points -> returns (u, v) arrays of valid indices."""
        dx = pts[:, 0] - self.target[0]
        dy = pts[:, 1] - self.target[1]
        dz = pts[:, 2] - self.target[2]

        x1 = dx * self.cos_y - dz * self.sin_y
        z1 = dx * self.sin_y + dz * self.cos_y

        y2 = dy * self.cos_p - z1 * self.sin_p
        z2 = dy * self.sin_p + z1 * self.cos_p

        depth = self.distance - z2
        valid = depth > 0.12
        if not np.any(valid):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        inv_d = self.focal / depth[valid]
        u = (self.width * 0.5 + x1[valid] * inv_d).astype(np.int32)
        v = (self.height * 0.5 - y2[valid] * inv_d).astype(np.int32)
        in_bounds = (u >= 1) & (u < self.width - 1) & (v >= 1) & (v < self.height - 1)
        return u[in_bounds], v[in_bounds]


def _draw_floor_grid(canvas: np.ndarray, cam: Camera3D) -> None:
    grid_y = 0.65
    extent = 1.0
    step = 0.2
    vals = np.arange(-extent, extent + 1e-4, step)
    for g in vals:
        is_axis = abs(g) < 1e-4
        col_x = (70, 70, 160) if is_axis else (52, 54, 62)
        col_z = (160, 90, 70) if is_axis else (52, 54, 62)

        p1 = cam.project((-extent, grid_y, float(g)))
        p2 = cam.project((extent, grid_y, float(g)))
        if p1 and p2:
            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), col_x, 1, cv2.LINE_AA)

        p3 = cam.project((float(g), grid_y, -extent))
        p4 = cam.project((float(g), grid_y, extent))
        if p3 and p4:
            cv2.line(canvas, (p3[0], p3[1]), (p4[0], p4[1]), col_z, 1, cv2.LINE_AA)


def _draw_face_on_mannequin_vectorized(
    canvas: np.ndarray,
    cam: Camera3D,
    face_joints: list[tuple[float, float, float]],
    head_base_3d: tuple[float, float, float],
    cfg: MannequinConfig,
) -> None:
    """Vectorized 478-landmark 3D face projection anchored by the chin (#152) on head_base_3d."""
    bx, by, bz = head_base_3d
    if not face_joints:
        pr = cam.project((bx, by + cfg.face_height_m * 0.5, bz))
        if pr is not None:
            cv2.circle(canvas, (pr[0], pr[1]), 22, (110, 115, 130), 2, cv2.LINE_AA)
        return

    pts = np.asarray(face_joints, dtype=np.float32)
    chin_idx = 152 if len(pts) > 152 else 0
    chin = pts[chin_idx]

    face_span_x = max(0.05, float(pts[:, 0].max() - pts[:, 0].min()))
    face_span_y = max(0.05, float(pts[:, 1].max() - pts[:, 1].min()))
    scale_x = cfg.face_width_m / face_span_x
    scale_y = cfg.face_height_m / face_span_y
    scale_z = cfg.face_depth_m / face_span_x

    world_pts = np.empty_like(pts)
    world_pts[:, 0] = bx - (pts[:, 0] - chin[0]) * scale_x
    world_pts[:, 1] = by - (pts[:, 1] - chin[1]) * scale_y
    world_pts[:, 2] = bz - (pts[:, 2] - chin[2]) * scale_z

    u, v = cam.project_points_np(world_pts)
    if u.size > 0:
        color = (100, 255, 190)
        canvas[v, u] = color
        canvas[v, u + 1] = color
        canvas[v + 1, u] = color
        canvas[v + 1, u + 1] = color


def _draw_forearm_and_hand_3d(
    canvas: np.ndarray,
    cam: Camera3D,
    hand_data: dict[str, Any] | None,
    side: str,
    elbow_world: tuple[float, float, float],
    bone_color: tuple[int, int, int],
    hand_rig_scale: float,
) -> None:
    """Draw the Rokoko ForeArm + Wrist + 5 Fingers anchored directly at the mannequin's elbow."""
    ex, ey, ez = elbow_world
    if not hand_data:
        default_wrist = (ex, ey - 0.03, ez + 0.25 * hand_rig_scale)
        p_el = cam.project(elbow_world)
        p_wr = cam.project(default_wrist)
        if p_el and p_wr:
            cv2.line(
                canvas,
                (p_el[0], p_el[1]),
                (p_wr[0], p_wr[1]),
                bone_color,
                4,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas, (p_wr[0], p_wr[1]), 5, (240, 240, 245), -1, cv2.LINE_AA
            )
        return

    joints = hand_data.get("joints", {})
    if not joints:
        return

    proj_pts: dict[str, tuple[int, int]] = {}
    for name, j in joints.items():
        px, py, pz = j["position"]
        world_pt = (
            ex - px * hand_rig_scale,
            ey + py * hand_rig_scale,
            ez + pz * hand_rig_scale,
        )
        pr = cam.project(world_pt)
        if pr is not None:
            proj_pts[name] = (pr[0], pr[1])

    for parent_name, child_name in HAND_PARENT_EDGES.get(side, []):
        if parent_name in proj_pts and child_name in proj_pts:
            thickness = 5 if "ForeArm" in parent_name else 2
            cv2.line(
                canvas,
                proj_pts[parent_name],
                proj_pts[child_name],
                bone_color,
                thickness,
                cv2.LINE_AA,
            )

    for name, pt in proj_pts.items():
        if "Hand" in name:
            r, c = 6, (255, 255, 255)
        elif "Tip" in name:
            r, c = 3, (0, 255, 255)
        else:
            r, c = 2, (235, 235, 235)
        cv2.circle(canvas, pt, r, c, -1, cv2.LINE_AA)


def _draw_glove_closeup_card(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    side: str,
    hand_data: dict[str, Any] | None,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (38, 40, 48), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (75, 80, 95), 1)
    cv2.putText(
        canvas,
        title,
        (x0 + 10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )

    if not hand_data:
        cv2.putText(
            canvas,
            "Waiting for glove stream...",
            (x0 + 12, y0 + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (130, 130, 140),
            1,
            cv2.LINE_AA,
        )
        return

    joints = hand_data.get("joints", {})
    wrist_key = "RightHand" if side == "right" else "LeftHand"
    wrist_pos = joints.get(wrist_key, {}).get("position", (0.0, 0.0, 0.0))
    wx, wy, wz = wrist_pos

    cx, cy = x0 + w // 2, y0 + int(h * 0.72)
    scale = min(w, h) * 3.0
    pts_2d: dict[str, tuple[int, int]] = {}
    for name, j in joints.items():
        if "ForeArm" in name:
            continue
        px, py, pz = j["position"]
        rx = -(px - wx)
        ry = (py - wy) + (pz - wz) * 0.7
        sx = int(cx + rx * scale)
        sy = int(cy - ry * scale)
        pts_2d[name] = (sx, sy)

    for p_name, c_name in HAND_PARENT_EDGES.get(side, []):
        if p_name in pts_2d and c_name in pts_2d:
            cv2.line(canvas, pts_2d[p_name], pts_2d[c_name], color, 2, cv2.LINE_AA)

    for name, pt in pts_2d.items():
        r = 4 if "Tip" in name else 3
        c = (0, 255, 255) if "Tip" in name else (235, 235, 235)
        cv2.circle(canvas, pt, r, c, -1, cv2.LINE_AA)


def _render_base_3d_scene_and_glove_cards(
    face_joints: list[tuple[float, float, float]],
    body_joints: dict[str, tuple[float, float, float, float]],
    hands_snapshot: dict[str, dict[str, Any]],
    view_yaw_deg: float,
    view_pitch_deg: float,
    mannequin_cfg: MannequinConfig,
) -> np.ndarray:
    """Shared 3D viewport + right-sidebar glove cards used by both live capture and offline playback."""
    canvas = np.full((VIEW_H, VIEW_W, 3), (30, 32, 38), dtype=np.uint8)
    studio_viewport = canvas[:, :STUDIO_W]

    cam = Camera3D(
        width=STUDIO_W,
        height=VIEW_H,
        yaw_deg=view_yaw_deg,
        pitch_deg=view_pitch_deg,
        distance=1.75 * mannequin_cfg.height_scale,
        target=(0.0, 1.28 * mannequin_cfg.height_scale, 0.05),
    )
    _draw_floor_grid(studio_viewport, cam)

    avatar_pts = resolve_mannequin_joints(body_joints, mannequin_cfg)
    proj_avatar: dict[str, tuple[int, int]] = {}
    for name, pt3d in avatar_pts.items():
        pr = cam.project(pt3d)
        if pr is not None:
            proj_avatar[name] = (pr[0], pr[1])

    for a, b in STUDIO_TORSO_BONES:
        if a in proj_avatar and b in proj_avatar:
            cv2.line(
                studio_viewport,
                proj_avatar[a],
                proj_avatar[b],
                (170, 175, 190),
                3,
                cv2.LINE_AA,
            )

    for a, b, col in STUDIO_UPPER_ARM_BONES:
        if a in proj_avatar and b in proj_avatar:
            cv2.line(
                studio_viewport,
                proj_avatar[a],
                proj_avatar[b],
                col,
                5,
                cv2.LINE_AA,
            )

    for name, pt2d in proj_avatar.items():
        r = 7 if "shoulder" in name or "elbow" in name else 4
        cv2.circle(studio_viewport, pt2d, r, (240, 240, 245), -1, cv2.LINE_AA)
        cv2.circle(studio_viewport, pt2d, r, (40, 40, 48), 1, cv2.LINE_AA)

    _draw_face_on_mannequin_vectorized(
        studio_viewport,
        cam,
        face_joints,
        avatar_pts["head_base"],
        mannequin_cfg,
    )

    _draw_forearm_and_hand_3d(
        studio_viewport,
        cam,
        hands_snapshot.get("left"),
        "left",
        avatar_pts["left_elbow"],
        (255, 200, 60),
        mannequin_cfg.hand_rig_scale,
    )
    _draw_forearm_and_hand_3d(
        studio_viewport,
        cam,
        hands_snapshot.get("right"),
        "right",
        avatar_pts["right_elbow"],
        (60, 200, 255),
        mannequin_cfg.hand_rig_scale,
    )

    cv2.line(canvas, (STUDIO_W, 0), (STUDIO_W, VIEW_H), (70, 72, 82), 2)

    glove_card_h = (VIEW_H - TOP_CARD_H - 4 * SIDEBAR_PAD) // 2
    y_left = 2 * SIDEBAR_PAD + TOP_CARD_H
    y_right = y_left + glove_card_h + SIDEBAR_PAD
    _draw_glove_closeup_card(
        canvas,
        STUDIO_W + SIDEBAR_PAD,
        y_left,
        CARD_W,
        glove_card_h,
        "LEFT SMARTGLOVE (26 JOINTS)",
        "left",
        hands_snapshot.get("left"),
        (255, 200, 60),
    )
    _draw_glove_closeup_card(
        canvas,
        STUDIO_W + SIDEBAR_PAD,
        y_right,
        CARD_W,
        glove_card_h,
        "RIGHT SMARTGLOVE (26 JOINTS)",
        "right",
        hands_snapshot.get("right"),
        (60, 200, 255),
    )
    return canvas


def render_live_studio_frame(
    face_pip_box: np.ndarray,
    body_pip_box: np.ndarray,
    face_joints: list[tuple[float, float, float]],
    body_joints: dict[str, tuple[float, float, float, float]],
    hands_snapshot: dict[str, dict[str, Any]],
    is_recording: bool,
    countdown_remaining: float,
    view_yaw_deg: float,
    view_pitch_deg: float,
    prompt_info: tuple[int, int, str, int],
    mannequin_cfg: MannequinConfig,
) -> np.ndarray:
    """Render live 3D Studio viewport + pre-rendered dual camera PiP + Teleprompter banner for dataset.py."""
    canvas = _render_base_3d_scene_and_glove_cards(
        face_joints=face_joints,
        body_joints=body_joints,
        hands_snapshot=hands_snapshot,
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
        mannequin_cfg=mannequin_cfg,
    )

    x0 = STUDIO_W + SIDEBAR_PAD
    half_w = face_pip_box.shape[1]
    canvas[SIDEBAR_PAD : SIDEBAR_PAD + TOP_CARD_H, x0 : x0 + half_w] = face_pip_box
    canvas[SIDEBAR_PAD : SIDEBAR_PAD + TOP_CARD_H, x0 + half_w : x0 + CARD_W] = (
        body_pip_box
    )
    cv2.rectangle(
        canvas,
        (x0, SIDEBAR_PAD),
        (x0 + CARD_W, SIDEBAR_PAD + TOP_CARD_H),
        (90, 95, 110),
        1,
    )

    if countdown_remaining > 0.0:
        status_text = f"CALIBRATING IN {countdown_remaining:.1f}s (HOLD HANDS FORWARD)"
        badge_col = (0, 165, 255)
    elif is_recording:
        status_text = "RECORDING TO CSV (SPACE TO SAVE & ADVANCE)"
        badge_col = (40, 40, 230)
    else:
        status_text = f"READY (H={mannequin_cfg.actor_height_m:.2f}m | SPACE TO RECORD)"
        badge_col = (50, 170, 70)

    cv2.rectangle(canvas, (16, 12), (560, 46), badge_col, -1)
    cv2.putText(
        canvas,
        status_text,
        (26, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    p_idx, p_total, sentence, takes = prompt_info
    if sentence:
        cv2.rectangle(canvas, (16, 54), (STUDIO_W - 16, 122), (22, 24, 30), -1)
        cv2.rectangle(
            canvas,
            (16, 54),
            (STUDIO_W - 16, 122),
            (40, 40, 230) if is_recording else (90, 180, 255),
            2,
        )
        meta_label = (
            f"FRASE [{p_idx + 1}/{p_total}]  |  Takes registrati per questa frase: {takes}"
        )
        cv2.putText(
            canvas,
            meta_label,
            (28, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 200, 255),
            1,
            cv2.LINE_AA,
        )
        display_sentence = sentence if len(sentence) <= 58 else sentence[:55] + "..."
        cv2.putText(
            canvas,
            f'"{display_sentence}"',
            (28, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (100, 255, 190) if not is_recording else (120, 230, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "[SPACE]: Rec/Stop(Auto-Next)  [X/Bksp]: Redo Take  [P/N]: Phrase  [C]: Calib  [A/D/W/S/R]: 3D  [Q]: Quit",
        (16, VIEW_H - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (180, 185, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_session_info_card(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    session_name: str,
    rec_frame: RecordedFrame,
    mannequin_cfg: MannequinConfig,
) -> None:
    """Draw session info and top active facial blendshapes in the top-right sidebar card."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (38, 40, 48), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (75, 80, 95), 1)

    cv2.putText(
        canvas,
        f"SESSION: {session_name[:28]}",
        (x0 + 12, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (100, 255, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"H={mannequin_cfg.actor_height_m:.2f}m | "
        f"Face: {'478' if rec_frame.face_joints else '0'} | "
        f"Body: {'5' if rec_frame.body_joints else '0'} | "
        f"Hands: {list(rec_frame.hands.keys()) or 'none'}",
        (x0 + 12, y0 + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 205, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "TOP FACIAL BLENDSHAPES:",
        (x0 + 12, y0 + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 170, 190),
        1,
        cv2.LINE_AA,
    )

    items = [
        (k, v) for k, v in rec_frame.face_blendshapes.items() if k != "_neutral"
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    for idx, (name, score) in enumerate(items[:5]):
        ry = y0 + 100 + idx * 22
        cv2.putText(
            canvas,
            f"{name[:18]:18s}",
            (x0 + 12, ry),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 230),
            1,
            cv2.LINE_AA,
        )
        bar_x = x0 + 165
        bar_w = max(10, w - 180)
        cv2.rectangle(
            canvas, (bar_x, ry - 10), (bar_x + bar_w, ry - 1), (55, 58, 68), -1
        )
        fill_w = int(bar_w * max(0.0, min(1.0, score)))
        cv2.rectangle(
            canvas, (bar_x, ry - 10), (bar_x + fill_w, ry - 1), (100, 230, 170), -1
        )


def render_playback_studio_frame(
    session_name: str,
    sentence: str,
    frames: list[RecordedFrame],
    idx: int,
    is_playing: bool,
    speed: float,
    view_yaw_deg: float,
    view_pitch_deg: float,
    mannequin_cfg: MannequinConfig,
    mode_label: str = "RAW",
) -> np.ndarray:
    """Render offline 3D playback viewport + blendshapes card + timeline bar for preview.py."""
    rec = frames[idx]
    canvas = _render_base_3d_scene_and_glove_cards(
        face_joints=rec.face_joints,
        body_joints=rec.body_joints,
        hands_snapshot=rec.hands,
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
        mannequin_cfg=mannequin_cfg,
    )

    _draw_session_info_card(
        canvas,
        STUDIO_W + SIDEBAR_PAD,
        SIDEBAR_PAD,
        CARD_W,
        TOP_CARD_H,
        session_name,
        rec,
        mannequin_cfg,
    )

    status_str = "PLAYING" if is_playing else "PAUSED"
    badge_col = (50, 170, 70) if is_playing else (0, 140, 230)
    t_sec = (rec.timestamp_ms - frames[0].timestamp_ms) / 1000.0
    total_sec = max(0.0, (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000.0)
    header_text = (
        f"{status_str} ({speed:.2g}x) | Frame {idx + 1}/{len(frames)} "
        f"| {t_sec:.2f}s / {total_sec:.2f}s"
    )
    cv2.rectangle(canvas, (16, 12), (540, 46), badge_col, -1)
    cv2.putText(
        canvas,
        header_text,
        (26, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mode_is_proc = mode_label.upper().startswith("POST")
    mode_col = (180, 70, 210) if mode_is_proc else (75, 85, 110)
    cv2.rectangle(canvas, (550, 12), (STUDIO_W - 16, 46), mode_col, -1)
    cv2.putText(
        canvas,
        f"{mode_label}  [T: Toggle]",
        (562, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if sentence:
        cv2.rectangle(canvas, (16, 54), (STUDIO_W - 16, 102), (22, 24, 30), -1)
        cv2.rectangle(canvas, (16, 54), (STUDIO_W - 16, 102), (90, 180, 255), 1)
        display_sentence = sentence if len(sentence) <= 58 else sentence[:55] + "..."
        cv2.putText(
            canvas,
            f'FRASE: "{display_sentence}"',
            (28, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (100, 255, 190),
            2,
            cv2.LINE_AA,
        )

    bar_x0, bar_x1 = 16, STUDIO_W - 16
    bar_y = VIEW_H - 44
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + 10), (55, 58, 68), -1)
    progress = idx / max(1, len(frames) - 1)
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * progress)
    cv2.rectangle(canvas, (bar_x0, bar_y), (fill_x, bar_y + 10), (80, 200, 255), -1)

    cv2.putText(
        canvas,
        "[SPACE]: Play/Pause  [T]: Toggle Raw/Post-Proc  [N/M]: Step  [+/-]: Speed  [0]: Restart  [A/D/W/S/R]: 3D  [Q]: Quit",
        (16, VIEW_H - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 185, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas
