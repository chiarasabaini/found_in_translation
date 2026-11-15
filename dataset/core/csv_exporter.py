from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Any, TextIO

from core.camera_tracker import (
    BODY_JOINT_NAMES,
    FACE_BLENDSHAPE_NAMES,
    NUM_FACE_JOINTS,
)
from core.config import MannequinConfig
from core.gloves_tracker import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES


@dataclass
class RecordedFrame:
    frame_id: int
    timestamp_ms: int
    face_joints: list[tuple[float, float, float]]
    face_blendshapes: dict[str, float]
    body_joints: dict[str, tuple[float, float, float, float]]
    hands: dict[str, dict[str, Any]]


def _slugify(text: str, max_len: int = 32) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", norm.lower()).strip("_")
    return (slug[:max_len].rstrip("_")) or "sample"


def _build_hand_header(joint_names: list[str]) -> list[str]:
    header = ["frame", "timestamp_ms", "glove_timestamp_us", "hand_detected"]
    for name in joint_names:
        header.extend(
            [
                f"{name}_pos_x",
                f"{name}_pos_y",
                f"{name}_pos_z",
                f"{name}_rot_x",
                f"{name}_rot_y",
                f"{name}_rot_z",
                f"{name}_rot_w",
            ]
        )
    return header


def _build_face_header() -> list[str]:
    header = ["frame", "timestamp_ms", "face_detected"]
    for idx in range(NUM_FACE_JOINTS):
        header.extend([f"face_{idx}_x", f"face_{idx}_y", f"face_{idx}_z"])
    for bs_name in FACE_BLENDSHAPE_NAMES:
        header.append(f"bs_{bs_name}")
    return header


def _build_body_header() -> list[str]:
    header = ["frame", "timestamp_ms", "body_detected"]
    for name in BODY_JOINT_NAMES:
        header.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_vis",
            ]
        )
    return header


LEFT_HAND_HEADER = _build_hand_header(LEFT_HAND_JOINT_NAMES)
RIGHT_HAND_HEADER = _build_hand_header(RIGHT_HAND_JOINT_NAMES)
FACE_HEADER = _build_face_header()
BODY_HEADER = _build_body_header()

MANIFEST_HEADER = [
    "sample_id",
    "prompt_index",
    "sentence",
    "num_frames",
    "duration_ms",
    "avg_fps",
    "created_at",
    "session_dir",
]


def load_manifest_take_counts(output_root: Path) -> dict[str, int]:
    """Read output/manifest.csv, prune entries whose folder was deleted, and count valid takes per sentence."""
    manifest_path = output_root / "manifest.csv"
    counts: dict[str, int] = {}
    if not manifest_path.exists():
        return counts
    try:
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or MANIFEST_HEADER)
            all_rows = list(reader)

        valid_rows: list[dict[str, str]] = []
        pruned = 0
        for row in all_rows:
            rel_dir = (row.get("session_dir") or row.get("sample_id") or "").strip()
            if not rel_dir or not (output_root / rel_dir).is_dir():
                pruned += 1
                continue
            valid_rows.append(row)
            sent = row.get("sentence", "").strip()
            if sent:
                counts[sent] = counts.get(sent, 0) + 1

        if pruned > 0:
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
            print(
                f"[MANIFEST] Cleaned {pruned} missing session(s) from {manifest_path}",
                flush=True,
            )
    except OSError:
        pass
    return counts


class SessionCsvExporter:
    """Buffers frames in RAM during recording and flushes 4 CSVs + metadata.json + manifest.csv on close()."""

    def __init__(
        self, output_root: Path, mannequin_cfg: MannequinConfig | None = None
    ) -> None:
        self.output_root = output_root
        self.mannequin_cfg = mannequin_cfg or MannequinConfig()
        self.manifest_path = output_root / "manifest.csv"
        self.session_dir: Path | None = None
        self.current_prompt_idx: int = 0
        self.current_sentence: str = ""
        self.frame_count = 0
        self._first_ts_ms: int | None = None
        self._last_ts_ms: int | None = None
        self._buffer: list[
            tuple[
                int,
                int,
                list[tuple[float, float, float]],
                dict[str, float],
                dict[str, tuple[float, float, float, float]],
                dict[str, dict[str, Any]],
            ]
        ] = []
        self._last_saved_info: tuple[Path, str, int] | None = None

    @property
    def is_open(self) -> bool:
        return self.session_dir is not None

    def start_session(self, prompt_idx: int, sentence: str) -> Path:
        self.close()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(sentence)
        sample_id = f"{slug}_{stamp}"
        self.session_dir = self.output_root / sample_id

        self.current_prompt_idx = prompt_idx
        self.current_sentence = sentence
        self.frame_count = 0
        self._first_ts_ms = None
        self._last_ts_ms = None
        self._buffer.clear()

        print(
            f'[CSV] Recording [{prompt_idx + 1}] "{sentence}" -> {self.session_dir.name}',
            flush=True,
        )
        return self.session_dir

    def write_frame(
        self,
        prompt_idx: int,
        sentence: str,
        timestamp_ms: int,
        face_joints: list[tuple[float, float, float]],
        face_blendshapes: dict[str, float],
        body_joints: dict[str, tuple[float, float, float, float]],
        hands_snapshot: dict[str, dict[str, Any]],
    ) -> int:
        if not self.is_open:
            self.start_session(prompt_idx, sentence)

        if self._first_ts_ms is None:
            self._first_ts_ms = timestamp_ms
        self._last_ts_ms = timestamp_ms

        self.frame_count += 1
        frame_id = self.frame_count
        self._buffer.append(
            (
                frame_id,
                timestamp_ms,
                face_joints,
                face_blendshapes,
                body_joints,
                hands_snapshot,
            )
        )
        return frame_id

    @staticmethod
    def _format_hand_row(
        frame_id: int,
        timestamp_ms: int,
        hand_data: dict[str, Any] | None,
        joint_names: list[str],
    ) -> list[Any]:
        if not hand_data:
            row: list[Any] = [frame_id, timestamp_ms, "", 0]
            row.extend([""] * (len(joint_names) * 7))
            return row

        ts_us = hand_data.get("timestamp_us", "")
        joints = hand_data.get("joints", {})
        row = [frame_id, timestamp_ms, ts_us, 1]
        for name in joint_names:
            j = joints.get(name)
            if j is None:
                row.extend(["", "", "", "", "", "", ""])
            else:
                px, py, pz = j["position"]
                rx, ry, rz, rw = j["rotation"]
                row.extend(
                    [
                        round(px, 5),
                        round(py, 5),
                        round(pz, 5),
                        round(rx, 5),
                        round(ry, 5),
                        round(rz, 5),
                        round(rw, 5),
                    ]
                )
        return row

    @staticmethod
    def _format_face_row(
        frame_id: int,
        timestamp_ms: int,
        face_joints: list[tuple[float, float, float]],
        face_blendshapes: dict[str, float],
    ) -> list[Any]:
        face_detected = 1 if len(face_joints) == NUM_FACE_JOINTS else 0
        row: list[Any] = [frame_id, timestamp_ms, face_detected]

        if face_detected:
            for fx, fy, fz in face_joints:
                row.extend([round(fx, 5), round(fy, 5), round(fz, 5)])
        else:
            row.extend([""] * (NUM_FACE_JOINTS * 3))

        if face_blendshapes:
            for bs_name in FACE_BLENDSHAPE_NAMES:
                row.append(round(face_blendshapes.get(bs_name, 0.0), 4))
        else:
            row.extend([""] * len(FACE_BLENDSHAPE_NAMES))

        return row

    @staticmethod
    def _format_body_row(
        frame_id: int,
        timestamp_ms: int,
        body_joints: dict[str, tuple[float, float, float, float]],
    ) -> list[Any]:
        body_detected = 1 if body_joints else 0
        row: list[Any] = [frame_id, timestamp_ms, body_detected]
        for name in BODY_JOINT_NAMES:
            bj = body_joints.get(name)
            if bj is None:
                row.extend(["", "", "", ""])
            else:
                bx, by, bz, bvis = bj
                row.extend([round(bx, 5), round(by, 5), round(bz, 5), round(bvis, 4)])
        return row

    def close(self) -> str | None:
        """Flush buffered frames to 4 CSVs, write metadata.json and manifest.csv, and return saved sentence."""
        if self.session_dir is None:
            return None

        saved_dir = self.session_dir
        saved_frames = len(self._buffer)
        prompt_idx = self.current_prompt_idx
        sentence = self.current_sentence
        duration_ms = max(0, (self._last_ts_ms or 0) - (self._first_ts_ms or 0))
        avg_fps = (
            round((saved_frames * 1000.0) / duration_ms, 2)
            if duration_ms > 0
            else 0.0
        )
        created_at = datetime.now().isoformat(timespec="seconds")

        frames_to_flush = list(self._buffer)
        self._buffer.clear()
        self.session_dir = None

        if saved_frames == 0:
            return None

        saved_dir.mkdir(parents=True, exist_ok=True)
        with (
            (saved_dir / "left_hand.csv").open("w", newline="", encoding="utf-8") as lf,
            (saved_dir / "right_hand.csv").open("w", newline="", encoding="utf-8") as rf,
            (saved_dir / "face.csv").open("w", newline="", encoding="utf-8") as ff,
            (saved_dir / "body.csv").open("w", newline="", encoding="utf-8") as bf,
        ):
            lw = csv.writer(lf)
            rw = csv.writer(rf)
            fw = csv.writer(ff)
            bw = csv.writer(bf)

            lw.writerow(LEFT_HAND_HEADER)
            rw.writerow(RIGHT_HAND_HEADER)
            fw.writerow(FACE_HEADER)
            bw.writerow(BODY_HEADER)

            for (
                frame_id,
                ts_ms,
                face_joints,
                face_blendshapes,
                body_joints,
                hands_snapshot,
            ) in frames_to_flush:
                lw.writerow(
                    self._format_hand_row(
                        frame_id,
                        ts_ms,
                        hands_snapshot.get("left"),
                        LEFT_HAND_JOINT_NAMES,
                    )
                )
                rw.writerow(
                    self._format_hand_row(
                        frame_id,
                        ts_ms,
                        hands_snapshot.get("right"),
                        RIGHT_HAND_JOINT_NAMES,
                    )
                )
                fw.writerow(
                    self._format_face_row(
                        frame_id, ts_ms, face_joints, face_blendshapes
                    )
                )
                bw.writerow(self._format_body_row(frame_id, ts_ms, body_joints))

        metadata = {
            "sample_id": saved_dir.name,
            "prompt_index": prompt_idx,
            "sentence": sentence,
            "num_frames": saved_frames,
            "duration_ms": duration_ms,
            "avg_fps": avg_fps,
            "created_at": created_at,
            "mannequin": self.mannequin_cfg.to_dict(),
            "files": {
                "left_hand": "left_hand.csv",
                "right_hand": "right_hand.csv",
                "face": "face.csv",
                "body": "body.csv",
            },
        }
        (saved_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        write_header = not self.manifest_path.exists()
        with self.manifest_path.open("a", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            if write_header:
                writer.writerow(MANIFEST_HEADER)
            writer.writerow(
                [
                    saved_dir.name,
                    prompt_idx,
                    sentence,
                    saved_frames,
                    duration_ms,
                    avg_fps,
                    created_at,
                    saved_dir.name,
                ]
            )

        self._last_saved_info = (saved_dir, sentence, prompt_idx)
        print(
            f'[CSV] Saved {saved_frames} frames ({duration_ms}ms, {avg_fps:.1f} FPS) for "{sentence}" -> {saved_dir.name}',
            flush=True,
        )
        return sentence

    def discard_active_or_last(self) -> tuple[bool, int, str | None]:
        """Delete the currently active recording or the most recently saved take."""
        if self.session_dir is not None:
            aborted_dir = self.session_dir
            sentence = self.current_sentence
            prompt_idx = self.current_prompt_idx
            self._buffer.clear()
            self.session_dir = None
            shutil.rmtree(aborted_dir, ignore_errors=True)
            print(
                f'\n>>> [DISCARDED] Aborted in-progress recording for "{sentence}" <<<\n',
                flush=True,
            )
            return True, prompt_idx, sentence

        if self._last_saved_info is not None:
            last_dir, sentence, prompt_idx = self._last_saved_info
            self._last_saved_info = None
            shutil.rmtree(last_dir, ignore_errors=True)
            self._remove_from_manifest(last_dir.name)
            print(
                f'\n>>> [DISCARDED] Deleted last take "{last_dir.name}" — ready to redo "{sentence}" <<<\n',
                flush=True,
            )
            return False, prompt_idx, sentence

        print("\n>>> [DISCARD] Nothing to discard <<<\n", flush=True)
        return False, 0, None

    def _remove_from_manifest(self, sample_id: str) -> None:
        if not self.manifest_path.exists():
            return
        try:
            with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if not rows:
                return
            header, body = rows[0], rows[1:]
            kept = [r for r in body if r and r[0] != sample_id]
            with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(kept)
        except OSError:
            pass


def _parse_hand_row(
    row: dict[str, str], joint_names: list[str]
) -> dict[str, Any] | None:
    if not row or row.get("hand_detected", "0") != "1":
        return None

    joints: dict[str, Any] = {}
    for name in joint_names:
        px = row.get(f"{name}_pos_x", "")
        py = row.get(f"{name}_pos_y", "")
        pz = row.get(f"{name}_pos_z", "")
        rx = row.get(f"{name}_rot_x", "")
        ry = row.get(f"{name}_rot_y", "")
        rz = row.get(f"{name}_rot_z", "")
        rw = row.get(f"{name}_rot_w", "")
        if px == "" or py == "" or pz == "":
            continue
        joints[name] = {
            "position": (float(px), float(py), float(pz)),
            "rotation": (
                float(rx or 0.0),
                float(ry or 0.0),
                float(rz or 0.0),
                float(rw or 1.0),
            ),
        }

    if not joints:
        return None

    ts_us_str = row.get("glove_timestamp_us", "0")
    return {
        "timestamp_us": int(ts_us_str) if ts_us_str.isdigit() else 0,
        "joints": joints,
    }


def _parse_face_row(
    row: dict[str, str],
) -> tuple[list[tuple[float, float, float]], dict[str, float]]:
    if not row or row.get("face_detected", "0") != "1":
        return [], {}

    face_joints: list[tuple[float, float, float]] = []
    for idx in range(NUM_FACE_JOINTS):
        fx = row.get(f"face_{idx}_x", "")
        fy = row.get(f"face_{idx}_y", "")
        fz = row.get(f"face_{idx}_z", "")
        if fx == "" or fy == "" or fz == "":
            return [], {}
        face_joints.append((float(fx), float(fy), float(fz)))

    blendshapes: dict[str, float] = {}
    for bs_name in FACE_BLENDSHAPE_NAMES:
        val_str = row.get(f"bs_{bs_name}", "")
        if val_str != "":
            blendshapes[bs_name] = float(val_str)

    return face_joints, blendshapes


def _parse_body_row(
    row: dict[str, str],
) -> dict[str, tuple[float, float, float, float]]:
    if not row or row.get("body_detected", "0") != "1":
        return {}

    body_joints: dict[str, tuple[float, float, float, float]] = {}
    for name in BODY_JOINT_NAMES:
        bx = row.get(f"{name}_x", "")
        by = row.get(f"{name}_y", "")
        bz = row.get(f"{name}_z", "")
        bvis = row.get(f"{name}_vis", "")
        if bx == "" or by == "" or bz == "":
            continue
        body_joints[name] = (
            float(bx),
            float(by),
            float(bz),
            float(bvis) if bvis != "" else 1.0,
        )
    return body_joints


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_session(
    session_dir: Path,
) -> tuple[list[RecordedFrame], str, MannequinConfig | None]:
    """Load left_hand.csv, right_hand.csv, face.csv, body.csv, and metadata.json from a session directory."""
    left_rows = _read_csv_rows(session_dir / "left_hand.csv")
    right_rows = _read_csv_rows(session_dir / "right_hand.csv")
    face_rows = _read_csv_rows(session_dir / "face.csv")
    body_rows = _read_csv_rows(session_dir / "body.csv")

    total = max(len(left_rows), len(right_rows), len(face_rows), len(body_rows))
    if total == 0:
        raise FileNotFoundError(
            f"No valid CSV rows found in '{session_dir}'. Expected left_hand.csv, right_hand.csv, face.csv."
        )

    sentence = ""
    session_cfg: MannequinConfig | None = None
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sentence = str(meta.get("sentence", ""))
            if isinstance(meta.get("mannequin"), dict):
                session_cfg = MannequinConfig.from_dict(meta["mannequin"])
        except Exception:
            pass

    frames: list[RecordedFrame] = []
    for i in range(total):
        l_row = left_rows[i] if i < len(left_rows) else {}
        r_row = right_rows[i] if i < len(right_rows) else {}
        f_row = face_rows[i] if i < len(face_rows) else {}
        b_row = body_rows[i] if i < len(body_rows) else {}

        ref_row = f_row or l_row or r_row or b_row
        frame_id = int(ref_row.get("frame", i + 1))
        ts_ms = int(float(ref_row.get("timestamp_ms", i * 33)))

        face_joints, face_blendshapes = _parse_face_row(f_row)
        body_joints = _parse_body_row(b_row)

        hands: dict[str, dict[str, Any]] = {}
        left_hand = _parse_hand_row(l_row, LEFT_HAND_JOINT_NAMES)
        if left_hand is not None:
            hands["left"] = left_hand
        right_hand = _parse_hand_row(r_row, RIGHT_HAND_JOINT_NAMES)
        if right_hand is not None:
            hands["right"] = right_hand

        frames.append(
            RecordedFrame(
                frame_id=frame_id,
                timestamp_ms=ts_ms,
                face_joints=face_joints,
                face_blendshapes=face_blendshapes,
                body_joints=body_joints,
                hands=hands,
            )
        )
    return frames, sentence, session_cfg


def find_latest_session(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    if output_root.is_dir() and (output_root / "face.csv").exists():
        return output_root
    candidates: list[Path] = []

    def _walk(curr: Path) -> None:
        if (curr / "face.csv").exists():
            candidates.append(curr)
            return
        for child in curr.iterdir():
            if child.is_dir():
                _walk(child)

    _walk(output_root)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)
