from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.config import DEFAULT_OUTPUT_DIR
from core.postprocessor import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_TARGET_FPS,
    postprocess_session_dir,
)


def _discover_sessions(input_path: Path) -> list[Path]:
    if (input_path / "face.csv").exists():
        return [input_path]
    if not input_path.exists():
        return []
    found: list[Path] = []
    for child in sorted(input_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "face.csv").exists():
            found.append(child)
        else:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and (sub / "face.csv").exists():
                    found.append(sub)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process recorded sign-language CSV sessions for neural network training (gap interpolation, static calibration-pose trimming, canonical face normalization, quaternion continuity, and constant-FPS resampling)."
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Optional single session directory or subfolder to process (default: process all sessions in --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory containing raw recorded session folders (default: ./output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Destination directory for post-processed sessions and manifest.csv (default: ./output_processed)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help=f"Target constant frame rate in Hz for LERP+SLERP resampling (default: {DEFAULT_TARGET_FPS:.1f})",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable automatic trimming of static start/end calibration poses",
    )
    args = parser.parse_args()

    target_source = args.session_dir if args.session_dir is not None else args.input_dir
    sessions = _discover_sessions(target_source)
    if not sessions:
        print(
            f"Error: no valid session folders with face.csv found in '{target_source}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    if args.session_dir is None and manifest_path.exists():
        manifest_path.unlink()

    print(
        f"[POSTPROCESS] Processing {len(sessions)} session(s) -> {args.output_dir.resolve()} "
        f"(target_fps={args.fps:.1f} Hz, trim_static_pose={not args.no_trim})"
    )

    processed_ok = 0
    for idx, src_dir in enumerate(sessions, start=1):
        dst_dir = args.output_dir / src_dir.name
        try:
            stats = postprocess_session_dir(
                src_session_dir=src_dir,
                dst_session_dir=dst_dir,
                target_fps=args.fps,
                trim_idle=not args.no_trim,
                manifest_path=manifest_path,
            )
            processed_ok += 1
            print(
                f"  [{idx:03d}/{len(sessions):03d}] {src_dir.name}: "
                f"raw={stats.raw_frames}f ({stats.raw_duration_ms}ms @ {stats.raw_fps:.1f}fps) "
                f"| trimmed=-{stats.trimmed_start_frames}start/-{stats.trimmed_end_frames}end "
                f"| filled=(F:{stats.interpolated_face_frames},B:{stats.interpolated_body_frames},"
                f"L:{stats.interpolated_left_frames},R:{stats.interpolated_right_frames}) "
                f"-> proc={stats.processed_frames}f ({stats.processed_duration_ms}ms @ {stats.target_fps:.1f}fps)"
            )
        except Exception as exc:
            print(f"  [ERROR] Failed to process '{src_dir.name}': {exc}", file=sys.stderr)

    print(
        f"\n[DONE] Successfully post-processed {processed_ok}/{len(sessions)} session(s) into '{args.output_dir}'."
    )
    print("       Test in 3D viewer with: python preview.py --processed")


if __name__ == "__main__":
    main()
