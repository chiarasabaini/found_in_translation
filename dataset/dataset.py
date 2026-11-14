from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import cv2

from core.camera_tracker import BodyTrackerWorker, FaceTrackerWorker
from core.config import (
    BODY_CAMERA_INDEX,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_RGMP_HOST,
    DEFAULT_RGMP_PORT,
    FACE_CAMERA_INDEX,
    MannequinConfig,
    load_mannequin_config,
)
from core.csv_exporter import SessionCsvExporter, load_manifest_take_counts
from core.gloves_tracker import ensure_rokoko_driver, rgmp_glove_worker
from core.renderer_3d import render_live_studio_frame, reset_mannequin_smoothing
from core.state import SharedState, keyboard_listener, load_prompts_file


def run_capture(
    output_dir: Path,
    prompts: list[str],
    mannequin_cfg: MannequinConfig,
) -> None:
    reset_mannequin_smoothing(mannequin_cfg)
    initial_takes = load_manifest_take_counts(output_dir)
    state = SharedState(prompts=prompts, initial_take_counts=initial_takes)
    exporter = SessionCsvExporter(output_root=output_dir, mannequin_cfg=mannequin_cfg)

    listener_thread = threading.Thread(
        target=keyboard_listener,
        args=(state,),
        daemon=True,
    )
    listener_thread.start()

    spawned_driver = ensure_rokoko_driver()
    glove_thread = threading.Thread(
        target=rgmp_glove_worker,
        args=(state,),
        daemon=True,
    )
    glove_thread.start()

    face_worker = FaceTrackerWorker(FACE_CAMERA_INDEX)
    if not face_worker.is_opened:
        print(
            f"Error: could not open face camera index {FACE_CAMERA_INDEX}",
            file=sys.stderr,
        )
        state.stop()
        if spawned_driver is not None:
            spawned_driver.kill()
        return

    body_worker = BodyTrackerWorker(BODY_CAMERA_INDEX)
    if not body_worker.is_opened:
        print(
            f"Error: could not open body camera index {BODY_CAMERA_INDEX}",
            file=sys.stderr,
        )
        face_worker.release()
        state.stop()
        if spawned_driver is not None:
            spawned_driver.kill()
        return

    print(
        f"[FACE CAMERA] Camera #{FACE_CAMERA_INDEX} ({face_worker.width}x{face_worker.height}, rotated 90° CW, async thread) -> 478 Face Landmarks + 52 Blendshapes"
    )
    print(
        f"[BODY CAMERA] Camera #{BODY_CAMERA_INDEX} ({body_worker.width}x{body_worker.height}, 0°, async thread) -> Neck, Shoulders & Elbows (5 metric 3D joints)"
    )
    print(
        f"[SMARTGLOVES] RGMP v2 on {DEFAULT_RGMP_HOST}:{DEFAULT_RGMP_PORT} -> 26 3D Joints per Hand + Mandatory 3s Calibration"
    )
    print(
        f"[MANNEQUIN]   Actor height={mannequin_cfg.actor_height_m:.2f}m (face={mannequin_cfg.face_height_m:.3f}m, neck_to_chin={mannequin_cfg.neck_to_chin_y_m:.3f}m, hand_scale={mannequin_cfg.hand_rig_scale:.3f})"
    )

    p_idx, p_total, curr_sentence, curr_takes = state.get_prompt_info()
    print(f"[OUTPUT]      {output_dir.resolve()} ({p_total} prompts loaded)")
    print(
        f'[TELEPROMPTER] Starting at [{p_idx + 1}/{p_total}] "{curr_sentence}" (takes={curr_takes})\n'
    )
    print("Controls:")
    print(
        "  [SPACE]            : Start (3s calibration countdown) / Stop recording & auto-advance"
    )
    print("  [x] / [Backspace]  : Discard & redo current or last recorded take")
    print("  [p] / [n] (or [/]) : Previous / Next phrase in prompts.txt")
    print("  [c]                : Re-calibrate gloves neutral forward orientation")
    print("  [a/d/w/s/r]        : Orbit / Tilt / Reset 3D Studio camera")
    print("  [q] / [ESC]        : Quit\n")

    start_monotonic = time.monotonic()
    last_ts_ms = 0
    last_status_log = 0.0
    last_rec_face_seq = -1
    last_rec_body_seq = -1

    try:
        while state.is_running:
            ready_f, face_seq, face_joints, face_blendshapes, face_pip = (
                face_worker.snapshot()
            )
            ready_b, body_seq, body_joints, body_pip = body_worker.snapshot()
            if not ready_f or not ready_b:
                time.sleep(0.005)
                continue

            now_ms = int((time.monotonic() - start_monotonic) * 1000)
            if now_ms <= last_ts_ms:
                now_ms = last_ts_ms + 1
            last_ts_ms = now_ms

            _, countdown_rem = state.tick_countdown()

            if state.consume_discard_request():
                was_active, disc_idx, disc_sent = exporter.discard_active_or_last()
                if disc_sent is not None:
                    state.on_take_discarded(
                        disc_idx, disc_sent, was_already_saved=not was_active
                    )
                    p_i, p_tot, s_txt, s_tks = state.get_prompt_info()
                    print(
                        f'>>> Ready to re-record [{p_i + 1}/{p_tot}] "{s_txt}" (takes: {s_tks}) <<<',
                        flush=True,
                    )

            hands_snapshot = state.snapshot_hands()
            prompt_info = state.get_prompt_info()
            yaw_deg, pitch_deg = state.get_view_angles()

            studio_view = render_live_studio_frame(
                face_pip_box=face_pip,
                body_pip_box=body_pip,
                face_joints=face_joints,
                body_joints=body_joints,
                hands_snapshot=hands_snapshot,
                is_recording=state.is_recording,
                countdown_remaining=countdown_rem,
                view_yaw_deg=yaw_deg,
                view_pitch_deg=pitch_deg,
                prompt_info=prompt_info,
                mannequin_cfg=mannequin_cfg,
            )
            cv2.imshow(
                "Rokoko Studio 3D Preview - Sign Language Capture", studio_view
            )
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                state.handle_key(chr(key) if key < 128 else "")

            if not state.is_recording:
                last_rec_face_seq = -1
                last_rec_body_seq = -1
                if exporter.is_open:
                    saved_sentence = exporter.close()
                    if saved_sentence:
                        next_idx, tot_p, next_sent, next_tks = state.on_take_saved(
                            saved_sentence
                        )
                        print(
                            f'>>> Next phrase [{next_idx + 1}/{tot_p}]: "{next_sent}" (takes: {next_tks}) <<<\n',
                            flush=True,
                        )
                now = time.monotonic()
                if now - last_status_log >= 2.0:
                    last_status_log = now
                    if countdown_rem > 0.0:
                        print(
                            f"[CALIBRATING] Hold hands forward... {countdown_rem:.1f}s remaining",
                            flush=True,
                        )
                    else:
                        p_i, p_tot, s_txt, s_tks = state.get_prompt_info()
                        print(
                            f'[IDLE | {p_i + 1}/{p_tot} "{s_txt}" (takes:{s_tks})] '
                            f"face: {len(face_joints)} | "
                            f"body: {len(body_joints)} | "
                            f"gloves: {list(hands_snapshot.keys()) or 'none'} "
                            "(SPACE: rec, x: redo, p/n: phrase)",
                            flush=True,
                        )
                continue

            if face_seq == last_rec_face_seq and body_seq == last_rec_body_seq:
                continue
            last_rec_face_seq = face_seq
            last_rec_body_seq = body_seq

            p_idx_cur, p_tot_cur, sentence_cur, _tks_cur = prompt_info
            frame_id = exporter.write_frame(
                prompt_idx=p_idx_cur,
                sentence=sentence_cur,
                timestamp_ms=now_ms,
                face_joints=face_joints,
                face_blendshapes=face_blendshapes,
                body_joints=body_joints,
                hands_snapshot=hands_snapshot,
            )
            if frame_id == 1 or frame_id % 30 == 0:
                print(
                    f"[REC #{frame_id:04d} @ {now_ms}ms | {p_idx_cur + 1}/{p_tot_cur}] "
                    f"face={len(face_joints)} body={len(body_joints)} "
                    f"left={'yes' if 'left' in hands_snapshot else 'no'} "
                    f"right={'yes' if 'right' in hands_snapshot else 'no'}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        state.stop()
        exporter.close()
        face_worker.release()
        body_worker.release()
        cv2.destroyAllWindows()
        if spawned_driver is not None:
            spawned_driver.kill()
        listener_thread.join(timeout=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign language dataset recorder with Teleprompter (Face Cam #0 + Body Cam #1 + Rokoko Smartgloves -> 4 CSVs + metadata.json + manifest.csv)."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to text file with one Italian sentence per line for the Teleprompter (default: ./prompts.txt)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional single sentence override (ignores --prompts when set)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory where session subfolders and manifest.csv are saved (default: ./output)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON file with mannequin lengths/proportions (default: ./config.json if present)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Actor height in meters (e.g. 1.75): deduces all mannequin lengths automatically like Rokoko Studio",
    )
    args = parser.parse_args()

    mannequin_cfg = load_mannequin_config(config_path=args.config, height_m=args.height)
    prompts = load_prompts_file(args.prompts, args.label)
    run_capture(
        output_dir=args.output_dir,
        prompts=prompts,
        mannequin_cfg=mannequin_cfg,
    )


if __name__ == "__main__":
    main()
