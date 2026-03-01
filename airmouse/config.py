"""
Centralized configuration management for AirMouse application.
"""

from dataclasses import dataclass, field
from pathlib import Path

from airmouse import paths


@dataclass
class UiConfig:
    """UI and application configuration."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    mouse_sensitivity: float = 1.0
    scroll_step: int = 3
    camera_roi_margin: float = 0.08
    camera_index: int = 0

    click_cooldown: float = 0.5
    click_hold_threshold_s: float = 2.5

    log_enabled: bool = True
    log_interval_s: float = 1.0

    theme: str = "dark"

    compute_device: str = "cpu"

    window_width: int = 1320
    window_height: int = 840

    def to_dict(self) -> dict:
        return {
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "mouse_sensitivity": self.mouse_sensitivity,
            "scroll_step": self.scroll_step,
            "camera_roi_margin": self.camera_roi_margin,
            "camera_index": self.camera_index,
            "click_cooldown": self.click_cooldown,
            "click_hold_threshold_s": self.click_hold_threshold_s,
            "log_enabled": self.log_enabled,
            "log_interval_s": self.log_interval_s,
            "theme": self.theme,
            "compute_device": self.compute_device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UiConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass
class PathConfig:
    """Path configuration for the application."""

    project_root: Path = field(default_factory=lambda: paths.PROJECT_ROOT)
    weights_dir: Path = field(default_factory=lambda: paths.WEIGHTS_DIR)
    logs_dir: Path = field(default_factory=lambda: paths.LOGS_DIR)

    def __post_init__(self) -> None:
        paths.ensure_runtime_dirs()

    @property
    def model_weights(self) -> Path:
        return paths.model_weights_path()

    @property
    def app_log(self) -> Path:
        return paths.app_log_path()


class AppConfig:
    """Global application configuration holder."""

    _instance: "AppConfig | None" = None

    def __init__(self):
        self.ui = UiConfig()
        self.paths = PathConfig()

    @classmethod
    def get_instance(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def get_config() -> AppConfig:
    return AppConfig.get_instance()
