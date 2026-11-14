/**
 * Dependency-free vector and quaternion helpers (x, y, z, w order).
 * Kept free of three.js so the recording pipeline runs in plain Node tests.
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

export const IDENTITY: Quat = [0, 0, 0, 1];

export function vSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vAdd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vScale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function vDot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vCross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function vLen(a: Vec3): number {
  return Math.sqrt(vDot(a, a));
}

export function vNorm(a: Vec3, fallback: Vec3 = [0, -1, 0]): Vec3 {
  const l = vLen(a);
  if (!(l > 1e-9)) return [fallback[0], fallback[1], fallback[2]];
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function vLerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function qNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(l > 1e-12)) return [0, 0, 0, 1];
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function qConj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** Hamilton product a * b (apply b first, then a). */
export function qMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function qRotate(q: Quat, v: Vec3): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}

export function qFromAxisAngle(axis: Vec3, angle: number): Quat {
  const a = vNorm(axis, [0, 1, 0]);
  const s = Math.sin(angle / 2);
  return [a[0] * s, a[1] * s, a[2] * s, Math.cos(angle / 2)];
}

/** Shortest rotation taking unit vector `from` onto unit vector `to`. */
export function qFromUnitVectors(from: Vec3, to: Vec3): Quat {
  const u = vNorm(from);
  const v = vNorm(to);
  const d = vDot(u, v);
  if (d > 0.999999) return [0, 0, 0, 1];
  if (d < -0.999999) {
    let ortho = vCross([1, 0, 0], u);
    if (vLen(ortho) < 1e-4) ortho = vCross([0, 1, 0], u);
    return qFromAxisAngle(ortho, Math.PI);
  }
  const c = vCross(u, v);
  return qNormalize([c[0], c[1], c[2], 1 + d]);
}

/** Quaternion from a rotation matrix given as three column vectors. */
export function qFromBasis(c0: Vec3, c1: Vec3, c2: Vec3): Quat {
  const m00 = c0[0], m10 = c0[1], m20 = c0[2];
  const m01 = c1[0], m11 = c1[1], m21 = c1[2];
  const m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const trace = m00 + m11 + m22;
  let q: Quat;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    q = [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  return qNormalize(q);
}

/** Orthonormal frame (forward, side, normal) built from a primary and a secondary direction. */
export function orthoFrame(primary: Vec3, secondary: Vec3): [Vec3, Vec3, Vec3] {
  const f = vNorm(primary);
  const n = vNorm(vCross(f, secondary), [0, 0, 1]);
  const s = vCross(n, f);
  return [f, s, n];
}

/** Rotation that maps frame `from` (f, s, n) onto frame `to` (f, s, n). */
export function qFrameToFrame(from: [Vec3, Vec3, Vec3], to: [Vec3, Vec3, Vec3]): Quat {
  const qFrom = qFromBasis(from[0], from[1], from[2]);
  const qTo = qFromBasis(to[0], to[1], to[2]);
  return qNormalize(qMul(qTo, qConj(qFrom)));
}

export function qDot(a: Quat, b: Quat): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

export function qSlerp(a: Quat, b: Quat, t: number): Quat {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cos = qDot(a, b);
  if (cos < 0) {
    cos = -cos;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  if (cos > 0.9995) {
    return qNormalize([
      a[0] + (bx - a[0]) * t,
      a[1] + (by - a[1]) * t,
      a[2] + (bz - a[2]) * t,
      a[3] + (bw - a[3]) * t,
    ]);
  }
  const theta = Math.acos(Math.min(1, cos));
  const sin = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sin;
  const wb = Math.sin(t * theta) / sin;
  return [a[0] * wa + bx * wb, a[1] * wa + by * wb, a[2] * wa + bz * wb, a[3] * wa + bw * wb];
}

/** Rotation angle of q in radians, in [0, pi]. */
export function qAngle(q: Quat): number {
  return 2 * Math.acos(Math.min(1, Math.abs(qNormalize(q)[3])));
}

/** Clamp the rotation angle of q to maxAngle radians, keeping its axis. */
export function qClampAngle(q: Quat, maxAngle: number): Quat {
  const n = qNormalize(q[3] < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q);
  const angle = qAngle(n);
  if (angle <= maxAngle) return n;
  return qSlerp(IDENTITY, n, maxAngle / angle);
}

export function qFromEulerXYZ(x: number, y: number, z: number): Quat {
  const qx = qFromAxisAngle([1, 0, 0], x);
  const qy = qFromAxisAngle([0, 1, 0], y);
  const qz = qFromAxisAngle([0, 0, 1], z);
  return qNormalize(qMul(qMul(qx, qy), qz));
}

export function degToRad(d: number): number {
  return (d * Math.PI) / 180;
}

export function radToDeg(r: number): number {
  return (r * 180) / Math.PI;
}

export function median(values: number[]): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
