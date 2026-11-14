from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

from core.config import (
    DEFAULT_OUTPUT_DIR,
    MannequinConfig,
    load_mannequin_config,
)
from core.csv_exporter import find_latest_session, load_session
from core.postprocessor import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_TARGET_FPS,
    postprocess_frames,
)
from core.renderer_3d import (
    render_playback_studio_frame,
    reset_mannequin_smoothing,
)


def _is_already_postprocessed(session_dir: Path) -> bool:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return bool(meta.get("postprocessed", False))
    except Exception:
        return False


def run_player(
    session_dir: Path,
    loop: bool,
    override_cfg: MannequinConfig | None = None,
    start_processed: bool = False,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> None:
    loaded_frames, sentence, session_cfg = load_session(session_dir)
    mannequin_cfg = override_cfg or session_cfg or load_mannequin_config()
    session_name = session_dir.name
    reset_mannequin_smoothing(mannequin_cfg)

    already_proc = _is_already_postprocessed(session_dir)
    if already_proc:
        raw_sibling = DEFAULT_OUTPUT_DIR / session_name
        if (raw_sibling / "face.csv").exists():
            raw_frames, _, _ = load_session(raw_sibling)
        else:
            raw_frames = loaded_frames
        proc_frames = loaded_frames
        stats_summary = f"{len(proc_frames)} frames (loaded from {session_dir.parent.name})"
    else:
        raw_frames = loaded_frames
        proc_frames, stats = postprocess_frames(
            raw_frames=raw_frames,
            target_fps=target_fps,
            trim_idle=True,
        )
        stats_summary = (
            f"raw={stats.raw_frames}f ({stats.raw_fps:.1f}fps) -> "
            f"proc={stats.processed_frames}f ({stats.target_fps:.0f}fps, "
            f"-{stats.trimmed_start_frames}start/-{stats.trimmed_end_frames}end)"
        )

    use_processed = start_processed or already_proc

    print(
        f"[PREVIEW] Loaded '{session_name}' (H={mannequin_cfg.actor_height_m:.2f}m)"
        + (f' — "{sentence}"' if sentence else "")
    )
    print(f"[POSTPROC] {stats_summary}")
    print("Controls:")
    print("  [SPACE]          : Play / Pause")
    print("  [t]              : Toggle live between RAW and POST-PROCESSED playback")
    print("  [n] / [m]        : Previous / Next frame (also Left / Right arrow keys)")
    print("  [+] / [-]        : Increase / Decrease playback speed")
    print("  [0]              : Restart from Frame 1")
    print("  [a/d/w/s/r]      : Orbit / Tilt / Reset 3D Studio camera")
    print("  [q] / [ESC]      : Quit\n")

    idx = 0
    is_playing = True
    speed = 1.0
    view_yaw_deg = 0.0
    view_pitch_deg = 12.0
    last_step_time = time.monotonic()

    try:
        while True:
            frames = proc_frames if use_processed else raw_frames
            mode_label = (
                f"POST-PROC {target_fps:.0f}FPS"
                if use_processed
                else f"RAW ({len(raw_frames)}f)"
            )
            if idx >= len(frames):
                idx = max(0, len(frames) - 1)

            now = time.monotonic()
            if is_playing and len(frames) > 1:
                next_idx = idx + 1
                if next_idx < len(frames):
                    dt_ms = max(
                        10, frames[next_idx].timestamp_ms - frames[idx].timestamp_ms
                    )
                else:
                    dt_ms = 33
                wait_sec = (dt_ms / 1000.0) / max(0.1, speed)
                if now - last_step_time >= wait_sec:
                    last_step_time = now
                    if idx + 1 < len(frames):
                        idx += 1
                    elif loop:
                        idx = 0
                        reset_mannequin_smoothing(mannequin_cfg)
                    else:
                        is_playing = False

            view = render_playback_studio_frame(
                session_name=session_name,
                sentence=sentence,
                frames=frames,
                idx=idx,
                is_playing=is_playing,
                speed=speed,
                view_yaw_deg=view_yaw_deg,
                view_pitch_deg=view_pitch_deg,
                mannequin_cfg=mannequin_cfg,
                mode_label=mode_label,
            )
            cv2.imshow("Rokoko Studio 3D Playback - Dataset Preview", view)

            key = cv2.waitKey(10) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                is_playing = not is_playing
                last_step_time = time.monotonic()
                if is_playing and idx >= len(frames) - 1:
                    idx = 0
                    reset_mannequin_smoothing(mannequin_cfg)
            elif key in (ord("t"), ord("T")):
                old_progress = idx / max(1, len(frames) - 1)
                use_processed = not use_processed
                new_frames = proc_frames if use_processed else raw_frames
                idx = int(round(old_progress * max(0, len(new_frames) - 1)))
                reset_mannequin_smoothing(mannequin_cfg)
                last_step_time = time.monotonic()
                print(
                    f"[PREVIEW] Switched to {'POST-PROCESSED' if use_processed else 'RAW'} "
                    f"({len(new_frames)} frames)",
                    flush=True,
                )
            elif key in (ord("n"), ord("N"), 81, 2):
                is_playing = False
                idx = max(0, idx - 1)
                reset_mannequin_smoothing(mannequin_cfg)
            elif key in (ord("m"), ord("M"), 83, 3):
                is_playing = False
                idx = min(len(frames) - 1, idx + 1)
                reset_mannequin_smoothing(mannequin_cfg)
            elif key == ord("0"):
                idx = 0
                reset_mannequin_smoothing(mannequin_cfg)
                last_step_time = time.monotonic()
            elif key in (ord("+"), ord("=")):
                speed = min(4.0, round(speed + 0.25, 2))
            elif key in (ord("-"), ord("_")):
                speed = max(0.25, round(speed - 0.25, 2))
            elif key in (ord("a"), ord("A")):
                view_yaw_deg = (view_yaw_deg - 8.0) % 360.0
            elif key in (ord("d"), ord("D")):
                view_yaw_deg = (view_yaw_deg + 8.0) % 360.0
            elif key in (ord("w"), ord("W")):
                view_pitch_deg = max(-60.0, min(60.0, view_pitch_deg + 5.0))
            elif key in (ord("s"), ord("S")):
                view_pitch_deg = max(-60.0, min(60.0, view_pitch_deg - 5.0))
            elif key in (ord("r"), ord("R")):
                view_yaw_deg = 0.0
                view_pitch_deg = 12.0
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D Playback Viewer for recorded or post-processed sign-language CSV sessions."
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the recorded session folder (default: latest folder inside ./output or ./output_processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory searched when session_dir is omitted (default: ./output, or ./output_processed with --processed)",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Start playback in POST-PROCESSED mode (searches ./output_processed first, or post-processes on the fly)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help=f"Target constant FPS when previewing on-the-fly post-processing (default: {DEFAULT_TARGET_FPS:.1f})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file overriding the session's mannequin dimensions",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Optional actor height in meters overriding the session's mannequin dimensions",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop at the last frame instead of looping playback",
    )
    args = parser.parse_args()

    target_dir = args.session_dir
    if target_dir is None:
        if args.output_dir is not None:
            search_dir = args.output_dir
        elif args.processed and DEFAULT_PROCESSED_DIR.exists():
            search_dir = DEFAULT_PROCESSED_DIR
        else:
            search_dir = DEFAULT_OUTPUT_DIR

        target_dir = find_latest_session(search_dir)
        if target_dir is None and args.processed and search_dir != DEFAULT_OUTPUT_DIR:
            target_dir = find_latest_session(DEFAULT_OUTPUT_DIR)

        if target_dir is None:
            print(
                f"Error: no recorded sessions found in '{search_dir}'. "
                "Record a session with dataset.py first or pass a session folder path.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not target_dir.exists():
        print(f"Error: session folder '{target_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    override_cfg = None
    if args.config is not None or args.height is not None:
        override_cfg = load_mannequin_config(
            config_path=args.config, height_m=args.height
        )

    run_player(
        session_dir=target_dir,
        loop=not args.no_loop,
        override_cfg=override_cfg,
        start_processed=args.processed,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()
