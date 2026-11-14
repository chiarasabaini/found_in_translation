from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EXAMPLES_DIR = BASE_DIR / "examples"
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_PROMPTS_PATH = (
    CONFIG_DIR / "prompts.txt"
    if (CONFIG_DIR / "prompts.txt").exists()
    else BASE_DIR / "prompts.txt"
)
DEFAULT_CONFIG_PATH = (
    CONFIG_DIR / "config.json"
    if (CONFIG_DIR / "config.json").exists()
    else BASE_DIR / "config.json"
)

# Fixed hardware configuration (2 cameras + Rokoko Smartgloves)
FACE_CAMERA_INDEX = 0
BODY_CAMERA_INDEX = 1

DEFAULT_HAND_WIDTH_M = 0.0  # 0.0 uses solver_core default (0.0565m)
DEFAULT_RGMP_HOST = "127.0.0.1"
DEFAULT_RGMP_PORT = 12276

CALIBRATION_COUNTDOWN_SEC = 3.0
REFERENCE_HEIGHT_M = 1.75


@dataclass
class MannequinConfig:
    """Anatomical dimensions and scale factors for the 3D mannequin."""

    actor_height_m: float = 1.75
    shoulder_height_m: float = 1.35
    pelvis_height_m: float = 0.88
    shoulder_half_width_m: float = 0.19
    hip_half_width_m: float = 0.11
    elbow_x_m: float = 0.24
    elbow_y_m: float = 1.09
    neck_to_chin_y_m: float = 0.115
    neck_to_chin_z_m: float = 0.04
    face_height_m: float = 0.25
    face_width_m: float = 0.195
    face_depth_m: float = 0.16
    hand_rig_scale: float = 0.82

    @property
    def height_scale(self) -> float:
        return self.actor_height_m / REFERENCE_HEIGHT_M

    def default_joints(self) -> dict[str, tuple[float, float, float]]:
        """Return neutral 3D joint positions for the mannequin."""
        return {
            "head_base": (
                0.0,
                self.shoulder_height_m + self.neck_to_chin_y_m,
                self.neck_to_chin_z_m,
            ),
            "neck": (0.0, self.shoulder_height_m, 0.0),
            "pelvis": (0.0, self.pelvis_height_m, 0.0),
            "left_hip": (-self.hip_half_width_m, self.pelvis_height_m, 0.0),
            "right_hip": (self.hip_half_width_m, self.pelvis_height_m, 0.0),
            "left_shoulder": (-self.shoulder_half_width_m, self.shoulder_height_m, 0.0),
            "right_shoulder": (self.shoulder_half_width_m, self.shoulder_height_m, 0.0),
            "left_elbow": (-self.elbow_x_m, self.elbow_y_m, 0.0),
            "right_elbow": (self.elbow_x_m, self.elbow_y_m, 0.0),
        }

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_height(cls, height_m: float) -> MannequinConfig:
        """Deduce all body segment dimensions from total actor height (Rokoko Studio anthropometric scaling)."""
        h = max(1.20, min(2.30, float(height_m)))
        s = h / REFERENCE_HEIGHT_M
        return cls(
            actor_height_m=round(h, 4),
            shoulder_height_m=round(1.35 * s, 4),
            pelvis_height_m=round(0.88 * s, 4),
            shoulder_half_width_m=round(0.19 * s, 4),
            hip_half_width_m=round(0.11 * s, 4),
            elbow_x_m=round(0.24 * s, 4),
            elbow_y_m=round(1.09 * s, 4),
            neck_to_chin_y_m=round(0.115 * s, 4),
            neck_to_chin_z_m=round(0.04 * s, 4),
            face_height_m=round(0.25 * s, 4),
            face_width_m=round(0.195 * s, 4),
            face_depth_m=round(0.16 * s, 4),
            hand_rig_scale=round(0.82 * s, 4),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MannequinConfig:
        base_height = float(data.get("actor_height_m", REFERENCE_HEIGHT_M))
        cfg = cls.from_height(base_height)
        for field_name in cfg.__dataclass_fields__:
            if field_name in data and data[field_name] is not None:
                setattr(cfg, field_name, float(data[field_name]))
        return cfg


def load_mannequin_config(
    config_path: Path | None = None,
    height_m: float | None = None,
) -> MannequinConfig:
    """Load mannequin dimensions from JSON config or deduce them from actor height in meters."""
    if height_m is not None and config_path is None:
        return MannequinConfig.from_height(height_m)

    target_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if target_path.exists():
        raw = json.loads(target_path.read_text(encoding="utf-8"))
        if height_m is not None:
            # If both --config and --height are passed, scale from height first, then apply explicit overrides
            return MannequinConfig.from_height(height_m)
        return MannequinConfig.from_dict(raw)

    if height_m is not None:
        return MannequinConfig.from_height(height_m)

    return MannequinConfig()
