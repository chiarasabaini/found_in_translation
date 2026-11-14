"""Prototype selection across multiple motion clips per class via DTW Medoid or DBA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from pipeline.glove_loader import CleanedHandClip
from pipeline.signal_processing import quat_angle_rad, quat_mul, quat_conj


@dataclass
class PrototypeSelectionResult:
    selected_index: int
    selected_clip_id: str
    method: str
    pairwise_distances: list[list[float]]
    mean_distance_to_others: float


def _extract_feature_sequence(right_clip: CleanedHandClip, left_clip: CleanedHandClip) -> list[list[float]]:
    """Extract frame-wise pose signature [5 right finger curls, 5 left finger curls, hub angles]."""
    fingers = ("thumb", "index", "middle", "ring", "little")
    seq: list[list[float]] = []
    n = min(len(right_clip.times), len(left_clip.times))
    for i in range(n):
        row: list[float] = []
        r_hub_inv = quat_conj(right_clip.hub_orientations[i])
        for f in fingers:
            q_f = right_clip.orientations.get(f, right_clip.hub_orientations)[i]
            row.append(quat_angle_rad((0.0, 0.0, 0.0, 1.0), quat_mul(r_hub_inv, q_f)))
        l_hub_inv = quat_conj(left_clip.hub_orientations[i])
        for f in fingers:
            q_f = left_clip.orientations.get(f, left_clip.hub_orientations)[i]
            row.append(quat_angle_rad((0.0, 0.0, 0.0, 1.0), quat_mul(l_hub_inv, q_f)))
        row.append(quat_angle_rad((0.0, 0.0, 0.0, 1.0), right_clip.hub_orientations[i]))
        row.append(quat_angle_rad((0.0, 0.0, 0.0, 1.0), left_clip.hub_orientations[i]))
        seq.append(row)
    return seq


def _frame_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def dtw_distance(
    seq_a: Sequence[Sequence[float]],
    seq_b: Sequence[Sequence[float]],
    window_ratio: float = 0.25,
) -> tuple[float, list[tuple[int, int]]]:
    """Sakoe-Chiba banded DTW distance + warping path."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0, []
    w = max(abs(n - m), int(max(n, m) * window_ratio))
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = _frame_dist(seq_a[i - 1], seq_b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack warping path
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = min(
            (dp[i - 1][j - 1], i - 1, j - 1),
            (dp[i - 1][j], i - 1, j),
            (dp[i][j - 1], i, j - 1),
            key=lambda item: item[0],
        )
        i, j = prev[1], prev[2]
    path.reverse()
    norm_dist = dp[n][m] / max(1, len(path))
    return norm_dist, path


def select_class_prototype(
    clip_ids: list[str],
    right_clips: list[CleanedHandClip],
    left_clips: list[CleanedHandClip],
    method: str = "medoid_dtw",
    window_ratio: float = 0.20,
    manual_override_id: str | None = None,
) -> PrototypeSelectionResult:
    """Select the representative prototype clip for a LIS class using DTW Medoid (or manual override)."""
    n = len(clip_ids)
    if n == 0:
        raise ValueError("Cannot select prototype from an empty clip list.")
    if n == 1:
        return PrototypeSelectionResult(0, clip_ids[0], method, [[0.0]], 0.0)

    if manual_override_id and manual_override_id in clip_ids:
        idx = clip_ids.index(manual_override_id)
        return PrototypeSelectionResult(idx, manual_override_id, "manual_override", [], 0.0)

    features = [_extract_feature_sequence(r, l) for r, l in zip(right_clips, left_clips)]
    dist_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = dtw_distance(features[i], features[j], window_ratio=window_ratio)
            dist_mat[i][j] = d
            dist_mat[j][i] = d

    totals = [sum(row) for row in dist_mat]
    best_idx = min(range(n), key=lambda k: totals[k])
    mean_d = totals[best_idx] / max(1, n - 1)
    return PrototypeSelectionResult(
        selected_index=best_idx,
        selected_clip_id=clip_ids[best_idx],
        method=method,
        pairwise_distances=dist_mat,
        mean_distance_to_others=round(mean_d, 5),
    )
