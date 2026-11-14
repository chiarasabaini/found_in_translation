from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

from core.config import CALIBRATION_COUNTDOWN_SEC


def load_prompts_file(path: Path, fallback_label: str | None = None) -> list[str]:
    """Load non-empty, non-comment sentences from prompts.txt."""
    if fallback_label:
        return [fallback_label.strip()]
    if path.exists():
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if lines:
            return lines
    return ["Session Test"]


class SharedState:
    """Thread-safe state shared across keyboard, glove, and camera threads."""

    def __init__(
        self,
        prompts: list[str],
        initial_take_counts: dict[str, int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.prompts = prompts if prompts else ["Sessione"]
        self.take_counts: dict[str, int] = dict(initial_take_counts or {})
        self.prompt_idx = self._find_first_unrecorded_index()
        self.is_recording = False
        self.is_running = True
        self.calibrate_requested = False
        self.discard_requested = False
        self._countdown_end_time: float | None = None
        self.hands: dict[str, dict[str, Any]] = {}
        self.connected_gloves: dict[int, str] = {}
        self._rgmp_client: Any = None
        self.view_yaw_deg: float = 0.0
        self.view_pitch_deg: float = 12.0

    def _find_first_unrecorded_index(self) -> int:
        for i, sentence in enumerate(self.prompts):
            if self.take_counts.get(sentence, 0) == 0:
                return i
        return 0

    def set_rgmp_client(self, client: Any) -> None:
        with self._lock:
            self._rgmp_client = client

    def get_prompt_info(self) -> tuple[int, int, str, int]:
        """Return (prompt_idx, total_prompts, current_sentence, takes_for_sentence)."""
        with self._lock:
            sentence = self.prompts[self.prompt_idx]
            takes = self.take_counts.get(sentence, 0)
            return self.prompt_idx, len(self.prompts), sentence, takes

    def step_prompt(self, delta: int) -> tuple[int, int, str, int]:
        with self._lock:
            if not self.is_recording and self._countdown_end_time is None:
                self.prompt_idx = (self.prompt_idx + delta) % len(self.prompts)
            sentence = self.prompts[self.prompt_idx]
            takes = self.take_counts.get(sentence, 0)
            return self.prompt_idx, len(self.prompts), sentence, takes

    def on_take_saved(self, sentence: str) -> tuple[int, int, str, int]:
        """Increment take counter for the saved sentence and advance to the next prompt."""
        with self._lock:
            self.take_counts[sentence] = self.take_counts.get(sentence, 0) + 1
            if self.prompt_idx + 1 < len(self.prompts):
                self.prompt_idx += 1
            next_sentence = self.prompts[self.prompt_idx]
            next_takes = self.take_counts.get(next_sentence, 0)
            return self.prompt_idx, len(self.prompts), next_sentence, next_takes

    def on_take_discarded(
        self, prompt_idx: int, sentence: str, was_already_saved: bool
    ) -> None:
        """Restore teleprompter to the discarded prompt and decrement take counter if already saved."""
        with self._lock:
            if was_already_saved:
                cur = self.take_counts.get(sentence, 0)
                if cur > 0:
                    self.take_counts[sentence] = cur - 1
            if 0 <= prompt_idx < len(self.prompts):
                self.prompt_idx = prompt_idx

    def toggle_recording(self) -> str:
        """Toggle recording state; always starts with the mandatory 3-second calibration countdown."""
        with self._lock:
            if self.is_recording or self._countdown_end_time is not None:
                self.is_recording = False
                self._countdown_end_time = None
                return "STOPPED"

            self._countdown_end_time = time.monotonic() + CALIBRATION_COUNTDOWN_SEC
            return "COUNTDOWN"

    def tick_countdown(self) -> tuple[bool, float]:
        """Check if mandatory calibration countdown is active or just finished."""
        with self._lock:
            if self._countdown_end_time is None:
                return False, 0.0
            remaining = self._countdown_end_time - time.monotonic()
            if remaining > 0.0:
                if remaining <= 0.15:
                    self.calibrate_requested = True
                return True, remaining
            self._countdown_end_time = None
            self.calibrate_requested = True
            self.is_recording = True
        print(
            "\n>>> [CALIBRATED] Calibration complete -> Recording STARTED <<<\n",
            flush=True,
        )
        return False, 0.0

    def request_calibration(self) -> None:
        with self._lock:
            self.calibrate_requested = True

    def consume_calibration_request(self) -> bool:
        with self._lock:
            if self.calibrate_requested:
                self.calibrate_requested = False
                return True
            return False

    def request_discard_take(self) -> None:
        with self._lock:
            self.is_recording = False
            self._countdown_end_time = None
            self.discard_requested = True

    def consume_discard_request(self) -> bool:
        with self._lock:
            if self.discard_requested:
                self.discard_requested = False
                return True
            return False

    def adjust_view(
        self, dyaw: float = 0.0, dpitch: float = 0.0, reset: bool = False
    ) -> None:
        with self._lock:
            if reset:
                self.view_yaw_deg = 0.0
                self.view_pitch_deg = 12.0
            else:
                self.view_yaw_deg = (self.view_yaw_deg + dyaw) % 360.0
                self.view_pitch_deg = max(
                    -60.0, min(60.0, self.view_pitch_deg + dpitch)
                )

    def get_view_angles(self) -> tuple[float, float]:
        with self._lock:
            return self.view_yaw_deg, self.view_pitch_deg

    def handle_key(self, ch: str) -> None:
        """Unified key handler shared between terminal stdin and OpenCV window."""
        if ch == " ":
            status = self.toggle_recording()
            idx, total, sent, _ = self.get_prompt_info()
            if status == "COUNTDOWN":
                print(
                    f'\n>>> [{idx + 1}/{total}] "{sent}" — Hold STRAIGHT POSE, calibrating in {CALIBRATION_COUNTDOWN_SEC:.0f}s... <<<\n',
                    flush=True,
                )
            else:
                print("\n>>> Recording STOPPED <<<\n", flush=True)
        elif ch in ("x", "X", "\x7f", "\x08"):
            self.request_discard_take()
        elif ch in ("n", "N", "]"):
            idx, total, sent, takes = self.step_prompt(+1)
            print(
                f'\n>>> Prompt [{idx + 1}/{total}]: "{sent}" (takes: {takes}) <<<\n',
                flush=True,
            )
        elif ch in ("p", "P", "["):
            idx, total, sent, takes = self.step_prompt(-1)
            print(
                f'\n>>> Prompt [{idx + 1}/{total}]: "{sent}" (takes: {takes}) <<<\n',
                flush=True,
            )
        elif ch in ("c", "C"):
            self.request_calibration()
        elif ch in ("a", "A"):
            self.adjust_view(dyaw=-8.0)
        elif ch in ("d", "D"):
            self.adjust_view(dyaw=8.0)
        elif ch in ("w", "W"):
            self.adjust_view(dpitch=5.0)
        elif ch in ("s", "S"):
            self.adjust_view(dpitch=-5.0)
        elif ch in ("r", "R"):
            self.adjust_view(reset=True)
        elif ch in ("q", "Q", "\x1b", "\x03"):
            print("\n>>> Exiting... <<<", flush=True)
            self.stop()

    def stop(self) -> None:
        with self._lock:
            self.is_running = False
            self.is_recording = False
            self._countdown_end_time = None
            client = self._rgmp_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def update_hand(self, side: str, hand_data: dict[str, Any]) -> None:
        with self._lock:
            self.hands[side] = hand_data

    def register_glove(self, device_id: int, side: str) -> None:
        with self._lock:
            self.connected_gloves[device_id] = side

    def remove_glove(self, device_id: int) -> str | None:
        with self._lock:
            side = self.connected_gloves.pop(device_id, None)
            if side:
                self.hands.pop(side, None)
            return side

    def snapshot_hands(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self.hands)


def keyboard_listener(state: SharedState) -> None:
    """Listen for single keypresses on stdin and forward them to SharedState.handle_key()."""
    if not sys.stdin.isatty():
        while state.is_running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip()
                state.handle_key(cmd[0] if cmd else " ")
            except EOFError:
                break
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while state.is_running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            ch = sys.stdin.read(1)
            state.handle_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
