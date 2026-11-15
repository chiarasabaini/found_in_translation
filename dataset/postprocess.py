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

DEFAULT_RAW_DIR = Path("./output/row")


def _resolve_input_root(input_path: Path) -> Path:
    if input_path == DEFAULT_OUTPUT_DIR and (input_path / "row").is_dir():
        return input_path / "row"
    return input_path


def _discover_sessions(input_path: Path, exclude_dir: Path | None = None) -> list[Path]:
    if (input_path / "face.csv").exists():
        return [input_path]
    if not input_path.exists():
        return []

    exclude_resolved = exclude_dir.resolve() if exclude_dir and exclude_dir.exists() else None
    found: list[Path] = []

    def _walk(curr: Path) -> None:
        if exclude_resolved is not None and curr.resolve() == exclude_resolved:
            return
        if (curr / "face.csv").exists():
            found.append(curr)
            return
        for child in sorted(curr.iterdir()):
            if child.is_dir():
                _walk(child)

    _walk(input_path)
    return found


def main() -> None:
    default_in = DEFAULT_RAW_DIR if DEFAULT_RAW_DIR.exists() else DEFAULT_OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Post-process recorded sign-language CSV sessions for neural network training, preserving the folder structure from output/row into output/processed."
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
        default=default_in,
        help="Root directory containing raw recorded session folders (default: ./output/row)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Destination directory for post-processed sessions (default: ./output/processed)",
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

    base_root = _resolve_input_root(args.input_dir)
    target_source = (
        _resolve_input_root(args.session_dir)
        if args.session_dir is not None
        else base_root
    )
    sessions = _discover_sessions(target_source, exclude_dir=args.output_dir)
    if not sessions:
        print(
            f"Error: no valid session folders with face.csv found in '{target_source}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleared_manifests: set[Path] = set()

    print(
        f"[POSTPROCESS] Processing {len(sessions)} session(s) from '{target_source}' -> '{args.output_dir}' "
        f"(target_fps={args.fps:.1f} Hz, trim_static_pose={not args.no_trim})"
    )

    processed_ok = 0
    for idx, src_dir in enumerate(sessions, start=1):
        try:
            rel_path = src_dir.resolve().relative_to(base_root.resolve())
        except ValueError:
            rel_path = Path(src_dir.name)

        dst_dir = args.output_dir / rel_path
        manifest_path = dst_dir.parent / "manifest.csv"
        if manifest_path not in cleared_manifests:
            cleared_manifests.add(manifest_path)
            if manifest_path.exists():
                manifest_path.unlink()

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
                f"  [{idx:03d}/{len(sessions):03d}] {rel_path}: "
                f"raw={stats.raw_frames}f ({stats.raw_duration_ms}ms @ {stats.raw_fps:.1f}fps) "
                f"| trimmed=-{stats.trimmed_start_frames}start/-{stats.trimmed_end_frames}end "
                f"| filled=(F:{stats.interpolated_face_frames},B:{stats.interpolated_body_frames},"
                f"L:{stats.interpolated_left_frames},R:{stats.interpolated_right_frames}) "
                f"-> proc={stats.processed_frames}f ({stats.processed_duration_ms}ms @ {stats.target_fps:.1f}fps)"
            )
        except Exception as exc:
            print(f"  [ERROR] Failed to process '{rel_path}': {exc}", file=sys.stderr)

    print(
        f"\n[DONE] Successfully post-processed {processed_ok}/{len(sessions)} session(s) into '{args.output_dir}'."
    )
    print("       Test in 3D viewer with: python preview.py --processed")


if __name__ == "__main__":
    main()
