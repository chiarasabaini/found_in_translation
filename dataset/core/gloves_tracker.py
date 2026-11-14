from __future__ import annotations

import math
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_HAND_WIDTH_M,
    DEFAULT_RGMP_HOST,
    DEFAULT_RGMP_PORT,
    EXAMPLES_DIR,
)
from core.state import SharedState

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from hand_solver.driver import Driver, DriverConfig, DriverError, spawn, wait_for_port
from rgmp_core.client import RgmpClient
from rgmp_core.layout import StreamLayout
from solver_core.finger_poser import (
    glove_state_from_sample,
    is_finite_pose,
    solve_fingers,
)
from solver_core.math import Quat, Vec3, angle_axis, q_mul, v_sub, v_transform
from solver_core.skeleton import HandSkeleton
from solver_core.skeleton_data import HAND_JOINTS
from solver_core.smartglove import (
    GloveDecodeError,
    SensorSlots,
    decode_glove_sample,
    detect_is_right,
    find_glove_layout,
    is_smartglove,
)

LEFT_HAND_JOINT_NAMES: list[str] = [j.name for j in HAND_JOINTS["left"]]
RIGHT_HAND_JOINT_NAMES: list[str] = [j.name for j in HAND_JOINTS["right"]]

HAND_PARENT_EDGES: dict[str, list[tuple[str, str]]] = {
    side: [(joints[j.parent].name, j.name) for j in joints if j.parent >= 0]
    for side, joints in HAND_JOINTS.items()
}

DEFAULT_SDK_BIN = (
    Path.home() / ".local" / "share" / "rokoko-device-sdk" / "bin" / "rokoko-sdk"
)

# Official Rokoko device->scene rebasing (+90 deg about +X): maps device +Z (gravity down) to scene -Y (down)
_SQRT_HALF = math.sqrt(0.5)
DEVICE_TO_SCENE_QUAT: Quat = (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def ensure_rokoko_driver() -> Driver | None:
    """Start rokoko-sdk --auto-stream-usb if no RGMP server is already listening."""
    if _is_port_open(DEFAULT_RGMP_HOST, DEFAULT_RGMP_PORT):
        return None

    driver_bin = shutil.which("rokoko-sdk")
    if driver_bin is None and DEFAULT_SDK_BIN.exists():
        driver_bin = str(DEFAULT_SDK_BIN)

    if driver_bin is None:
        print(
            f"[GLOVES] No RGMP server on {DEFAULT_RGMP_HOST}:{DEFAULT_RGMP_PORT} and 'rokoko-sdk' not found in PATH.",
            file=sys.stderr,
        )
        return None

    cfg = DriverConfig(path=driver_bin, quiet=True)
    try:
        driver = spawn(cfg)
        if driver is not None:
            print(f"[GLOVES] Started rokoko-sdk driver ({driver_bin})...", flush=True)
            wait_for_port(DEFAULT_RGMP_HOST, DEFAULT_RGMP_PORT, driver, timeout=10.0)
        return driver
    except DriverError as err:
        print(f"[GLOVES] Could not start rokoko-sdk driver: {err}", file=sys.stderr)
        return None


def compute_forearm_heading_deg(skeleton: HandSkeleton) -> float:
    """Compute the yaw angle (degrees about scene +Y) that aligns the forearm (elbow->wrist) with +Z."""
    elbow_dev = skeleton.nodes[0].position
    wrist_dev = skeleton.nodes[1].position
    v_dev = v_sub(wrist_dev, elbow_dev)
    vx, _vy, vz = v_transform(v_dev, DEVICE_TO_SCENE_QUAT)
    if math.hypot(vx, vz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(-vx, vz))


def serialize_hand_skeleton(
    skeleton: HandSkeleton,
    ts_us: int,
    yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """Convert solved skeleton to Y-up scene space, anchored at the elbow (ForeArm = 0,0,0)."""
    yaw_quat: Quat = angle_axis(yaw_deg, (0.0, 1.0, 0.0))
    total_rot: Quat = q_mul(yaw_quat, DEVICE_TO_SCENE_QUAT)

    elbow_pos: Vec3 = skeleton.nodes[0].position
    joints: dict[str, Any] = {}
    for node in skeleton.nodes:
        rel_dev = v_sub(node.position, elbow_pos)
        joints[node.name] = {
            "position": v_transform(rel_dev, total_rot),
            "rotation": q_mul(total_rot, node.rotation),
        }
    return {"timestamp_us": ts_us, "joints": joints}


def rgmp_glove_worker(state: SharedState) -> None:
    """Background worker that reads Smartgloves from rokoko-sdk RGMP v2 and solves 3D joints."""
    rigs: dict[int, tuple[int, dict[int, SensorSlots], bool, HandSkeleton]] = {}
    latest_heading_deg: dict[str, float] = {}
    calibrated_yaw_deg: dict[str, float] = {}

    def on_definition(definition: dict, _layout: StreamLayout) -> None:
        device_id = int(definition["device_id"])
        if not is_smartglove(definition):
            return
        try:
            found = find_glove_layout(definition)
        except GloveDecodeError:
            return
        if found is None:
            return

        sensor_group_id, slots = found
        is_right = detect_is_right(definition)
        side = "right" if is_right else "left"
        skeleton = HandSkeleton(side)
        rigs[device_id] = (sensor_group_id, slots, is_right, skeleton)
        state.register_glove(device_id, side)
        print(
            f"[GLOVES] Connected {side} Smartglove (device_id={device_id}, {len(slots)} sensors)",
            flush=True,
        )

    def on_data(
        device_id: int,
        group_id: int,
        ts_us: int,
        payload: bytes,
        _layout: StreamLayout,
    ) -> None:
        if not state.is_running:
            return

        rig = rigs.get(device_id)
        if rig is None:
            return
        sensor_group_id, slots, is_right, skeleton = rig
        if group_id != sensor_group_id:
            return
        try:
            sample = decode_glove_sample(payload, slots)
            glove_state = glove_state_from_sample(sample, is_right, DEFAULT_HAND_WIDTH_M)
            solve_fingers(glove_state, skeleton, hinges=True)
        except (GloveDecodeError, ArithmeticError):
            return
        if not is_finite_pose(skeleton):
            return

        side = "right" if is_right else "left"
        latest_heading_deg[side] = compute_forearm_heading_deg(skeleton)

        if side not in calibrated_yaw_deg:
            calibrated_yaw_deg[side] = latest_heading_deg[side]

        if state.consume_calibration_request():
            for s, h_deg in latest_heading_deg.items():
                calibrated_yaw_deg[s] = h_deg
            print(
                f"\n>>> [CALIBRATED] Forearm heading aligned for gloves: {list(calibrated_yaw_deg.keys())} <<<\n",
                flush=True,
            )

        state.update_hand(
            side,
            serialize_hand_skeleton(
                skeleton, ts_us, calibrated_yaw_deg.get(side, 0.0)
            ),
        )

    def on_disconnect(device_id: int) -> None:
        rigs.pop(device_id, None)
        side = state.remove_glove(device_id)
        if side:
            print(
                f"[GLOVES] Disconnected {side} Smartglove (device_id={device_id})",
                flush=True,
            )

    while state.is_running:
        try:
            client = RgmpClient(DEFAULT_RGMP_HOST, DEFAULT_RGMP_PORT)
            client.on_definition = on_definition
            client.on_data = on_data
            client.on_disconnect = on_disconnect
            state.set_rgmp_client(client)
            with client:
                print(
                    f"[GLOVES] Connected to rokoko-sdk RGMP server at {DEFAULT_RGMP_HOST}:{DEFAULT_RGMP_PORT}",
                    flush=True,
                )
                client.run()
        except OSError:
            if not state.is_running:
                break
            time.sleep(2.0)
        finally:
            state.set_rgmp_client(None)
