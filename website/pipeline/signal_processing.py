"""Signal processing utilities for Smartgloves II quaternions, vectors, and scalars.

Implements quaternion normalization, hemisphere sign continuity, SLERP,
dropout gap interpolation, 1-Euro adaptive low-pass filtering, and resampling.
"""

from __future__ import annotations

import math
from typing import Sequence

Quat = tuple[float, float, float, float]  # Always normalized internally as (x, y, z, w)
Vec3 = tuple[float, float, float]


def to_xyzw(raw: Sequence[float], order: str = "xyzw") -> Quat:
    """Convert raw 4-component quaternion into canonical (x, y, z, w) order."""
    if order.lower() == "wxyz":
        return (float(raw[1]), float(raw[2]), float(raw[3]), float(raw[0]))
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def quat_norm(q: Quat) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quat_normalize(q: Quat) -> Quat:
    n = quat_norm(q)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_conj(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_dot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_negate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], -q[3])


def quat_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    ax, ay, az = axis
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half = 0.5 * angle_rad
    s = math.sin(half) / length
    return quat_normalize((ax * s, ay * s, az * s, math.cos(half)))


def quat_from_euler_xyz(rx: float, ry: float, rz: float) -> Quat:
    """Intrinsic XYZ Euler angles (radians) -> (x, y, z, w) quaternion."""
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), rx)
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), ry)
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), rz)
    return quat_normalize(quat_mul(quat_mul(qx, qy), qz))


def quat_angle_rad(a: Quat, b: Quat) -> float:
    """Geodesic angular distance in radians between two unit quaternions."""
    d = abs(max(-1.0, min(1.0, quat_dot(quat_normalize(a), quat_normalize(b)))))
    return 2.0 * math.acos(d)


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Shortest-arc spherical linear interpolation between unit quaternions."""
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    dot = quat_dot(qa, qb)
    if dot < 0.0:
        qb = quat_negate(qb)
        dot = -dot
    if dot > 0.9995:
        res = (
            qa[0] + t * (qb[0] - qa[0]),
            qa[1] + t * (qb[1] - qa[1]),
            qa[2] + t * (qb[2] - qa[2]),
            qa[3] + t * (qb[3] - qa[3]),
        )
        return quat_normalize(res)
    dot = max(-1.0, min(1.0, dot))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize(
        (
            s0 * qa[0] + s1 * qb[0],
            s0 * qa[1] + s1 * qb[1],
            s0 * qa[2] + s1 * qb[2],
            s0 * qa[3] + s1 * qb[3],
        )
    )


def enforce_quaternion_continuity(quats: Sequence[Quat]) -> list[Quat]:
    """Ensure consecutive quaternions stay in the same hemisphere (no sign flips)."""
    if not quats:
        return []
    out: list[Quat] = [quat_normalize(quats[0])]
    for q in quats[1:]:
        qn = quat_normalize(q)
        if quat_dot(out[-1], qn) < 0.0:
            qn = quat_negate(qn)
        out.append(qn)
    return out


def interpolate_vec3_dropouts(
    times: Sequence[float],
    values: Sequence[Vec3],
    valid_mask: Sequence[bool],
) -> list[Vec3]:
    """Linearly interpolate masked/dropped Vec3 samples, clamping endpoints."""
    n = len(values)
    if n == 0:
        return []
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        return list(values)
    out: list[Vec3] = list(values)
    # Clamp leading and trailing invalid runs
    first_idx = valid_indices[0]
    for i in range(0, first_idx):
        out[i] = values[first_idx]
    last_idx = valid_indices[-1]
    for i in range(last_idx + 1, n):
        out[i] = values[last_idx]
    # Interpolate interior gaps
    for k in range(len(valid_indices) - 1):
        i0 = valid_indices[k]
        i1 = valid_indices[k + 1]
        if i1 == i0 + 1:
            continue
        t0, t1 = times[i0], times[i1]
        dt = max(1e-9, t1 - t0)
        v0, v1 = values[i0], values[i1]
        for j in range(i0 + 1, i1):
            alpha = (times[j] - t0) / dt
            out[j] = (
                v0[0] + alpha * (v1[0] - v0[0]),
                v0[1] + alpha * (v1[1] - v0[1]),
                v0[2] + alpha * (v1[2] - v0[2]),
            )
    return out


def interpolate_quat_dropouts(
    times: Sequence[float],
    quats: Sequence[Quat],
    valid_mask: Sequence[bool],
) -> list[Quat]:
    """SLERP across masked/dropped quaternion samples with hemisphere continuity."""
    n = len(quats)
    if n == 0:
        return []
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        return enforce_quaternion_continuity(quats)
    out: list[Quat] = [quat_normalize(q) for q in quats]
    first_idx = valid_indices[0]
    for i in range(0, first_idx):
        out[i] = out[first_idx]
    last_idx = valid_indices[-1]
    for i in range(last_idx + 1, n):
        out[i] = out[last_idx]
    for k in range(len(valid_indices) - 1):
        i0 = valid_indices[k]
        i1 = valid_indices[k + 1]
        if i1 == i0 + 1:
            continue
        t0, t1 = times[i0], times[i1]
        dt = max(1e-9, t1 - t0)
        q0, q1 = out[i0], out[i1]
        for j in range(i0 + 1, i1):
            alpha = (times[j] - t0) / dt
            out[j] = quat_slerp(q0, q1, alpha)
    return enforce_quaternion_continuity(out)


def interpolate_scalar_dropouts(
    times: Sequence[float],
    values: Sequence[float],
    valid_mask: Sequence[bool],
) -> list[float]:
    """Linearly interpolate masked scalar series."""
    n = len(values)
    if n == 0:
        return []
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        return list(values)
    out = list(values)
    for i in range(0, valid_indices[0]):
        out[i] = values[valid_indices[0]]
    for i in range(valid_indices[-1] + 1, n):
        out[i] = values[valid_indices[-1]]
    for k in range(len(valid_indices) - 1):
        i0, i1 = valid_indices[k], valid_indices[k + 1]
        if i1 == i0 + 1:
            continue
        dt = max(1e-9, times[i1] - times[i0])
        for j in range(i0 + 1, i1):
            alpha = (times[j] - times[i0]) / dt
            out[j] = values[i0] + alpha * (values[i1] - values[i0])
    return out


class OneEuroFilter:
    """1-Euro adaptive low-pass filter (Casiez et al., 2012) for low-latency smoothing."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / max(1e-6, dt))

    def filter(self, t: float, x: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._dx_prev = 0.0
            return x
        dt = max(1e-4, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def smooth_vec3_one_euro(
    times: Sequence[float],
    values: Sequence[Vec3],
    min_cutoff: float = 2.2,
    beta: float = 0.08,
    d_cutoff: float = 1.0,
) -> list[Vec3]:
    if not values:
        return []
    fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
    fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
    fz = OneEuroFilter(min_cutoff, beta, d_cutoff)
    return [(fx.filter(t, v[0]), fy.filter(t, v[1]), fz.filter(t, v[2])) for t, v in zip(times, values)]


def smooth_quat_one_euro(
    times: Sequence[float],
    quats: Sequence[Quat],
    min_cutoff: float = 2.2,
    beta: float = 0.08,
    d_cutoff: float = 1.0,
) -> list[Quat]:
    """Filter hemisphere-continuous quaternions component-wise then re-normalize."""
    if not quats:
        return []
    cont = enforce_quaternion_continuity(quats)
    fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
    fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
    fz = OneEuroFilter(min_cutoff, beta, d_cutoff)
    fw = OneEuroFilter(min_cutoff, beta, d_cutoff)
    filtered = [
        quat_normalize((fx.filter(t, q[0]), fy.filter(t, q[1]), fz.filter(t, q[2]), fw.filter(t, q[3])))
        for t, q in zip(times, cont)
    ]
    return enforce_quaternion_continuity(filtered)


def build_uniform_timestamps(t_start: float, t_end: float, fps: float) -> list[float]:
    if t_end <= t_start:
        return [t_start]
    dt = 1.0 / fps
    n_frames = max(2, int(round((t_end - t_start) * fps)) + 1)
    return [t_start + i * dt for i in range(n_frames)]


def resample_vec3(
    src_times: Sequence[float], src_vals: Sequence[Vec3], dst_times: Sequence[float]
) -> list[Vec3]:
    if not src_times:
        return [(0.0, 0.0, 0.0) for _ in dst_times]
    out: list[Vec3] = []
    idx = 0
    n = len(src_times)
    for t in dst_times:
        while idx + 1 < n and src_times[idx + 1] < t:
            idx += 1
        if t <= src_times[0]:
            out.append(src_vals[0])
        elif idx >= n - 1:
            out.append(src_vals[-1])
        else:
            t0, t1 = src_times[idx], src_times[idx + 1]
            alpha = (t - t0) / max(1e-9, t1 - t0)
            v0, v1 = src_vals[idx], src_vals[idx + 1]
            out.append(
                (
                    v0[0] + alpha * (v1[0] - v0[0]),
                    v0[1] + alpha * (v1[1] - v0[1]),
                    v0[2] + alpha * (v1[2] - v0[2]),
                )
            )
    return out


def resample_quats(
    src_times: Sequence[float], src_quats: Sequence[Quat], dst_times: Sequence[float]
) -> list[Quat]:
    if not src_times:
        return [(0.0, 0.0, 0.0, 1.0) for _ in dst_times]
    out: list[Quat] = []
    idx = 0
    n = len(src_times)
    for t in dst_times:
        while idx + 1 < n and src_times[idx + 1] < t:
            idx += 1
        if t <= src_times[0]:
            out.append(src_quats[0])
        elif idx >= n - 1:
            out.append(src_quats[-1])
        else:
            t0, t1 = src_times[idx], src_times[idx + 1]
            alpha = (t - t0) / max(1e-9, t1 - t0)
            out.append(quat_slerp(src_quats[idx], src_quats[idx + 1], alpha))
    return enforce_quaternion_continuity(out)


def resample_scalars(
    src_times: Sequence[float], src_vals: Sequence[float], dst_times: Sequence[float]
) -> list[float]:
    if not src_times:
        return [0.0 for _ in dst_times]
    out: list[float] = []
    idx = 0
    n = len(src_times)
    for t in dst_times:
        while idx + 1 < n and src_times[idx + 1] < t:
            idx += 1
        if t <= src_times[0]:
            out.append(src_vals[0])
        elif idx >= n - 1:
            out.append(src_vals[-1])
        else:
            t0, t1 = src_times[idx], src_times[idx + 1]
            alpha = (t - t0) / max(1e-9, t1 - t0)
            out.append(src_vals[idx] + alpha * (src_vals[idx + 1] - src_vals[idx]))
    return out
