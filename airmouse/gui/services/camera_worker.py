"""
Camera worker thread for gesture recognition and mouse control.
Separated from UI code for better maintainability.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from airmouse.actions import MouseKeyboardController
from airmouse.config import get_config
from airmouse.gestures.registry import GESTURES as gestures
from airmouse.log import setup_logger
from airmouse.vision.inference.openvino_runtime import OpenVinoGestureRuntime
from airmouse.vision.preprocess import preprocess


class CameraWorker(QThread):
    """Worker thread for camera input processing and gesture recognition."""

    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)
    event_ready = pyqtSignal(str, str, bool)
    failed = pyqtSignal(str)

    def __init__(self, config_obj, model_path: Path) -> None:
        super().__init__()
        self.config = config_obj
        self.model_path = model_path
        self.running = False

        self.mouse_actions = MouseKeyboardController(
            mouse_sensitivity=config_obj.mouse_sensitivity,
            camera_roi_margin=config_obj.camera_roi_margin,
            scroll_step=config_obj.scroll_step,
            click_cooldown=config_obj.click_cooldown,
            click_hold_threshold_s=config_obj.click_hold_threshold_s,
        )

        self.frame_count = 0
        self.started_at = 0.0
        self.fps_series: list[float] = []
        self.conf_series: list[float] = []
        self.lat_series_ms: list[float] = []
        self.infer_series_ms: list[float] = []

        self.device = "cpu"
        self.ov: OpenVinoGestureRuntime | None = None

        paths = get_config().paths
        self.logger = setup_logger("airmouse.desktop", paths.app_log)
        self._last_event_log_mono = 0.0

    def _predict_openvino(self, data: np.ndarray) -> tuple[int, float, float]:
        if self.ov is None:
            raise RuntimeError("OpenVINO runtime is not initialized")
        pred = self.ov.predict(np.asarray(data, dtype=np.float32))
        return pred.gesture_id, pred.confidence, float(pred.inference_ms)

    def _predict(self, data: np.ndarray) -> tuple[int, float, float]:
        return self._predict_openvino(data)

    def _load_openvino_model(self) -> bool:
        paths = get_config().paths
        ov_path = paths.model_weights
        if not ov_path.exists():
            self.failed.emit(
                f"Не найдена модель: {ov_path}. "
            )
            return False

        try:
            self.ov = OpenVinoGestureRuntime(model_path=ov_path, device="CPU")
            self.ov.load()
            return True
        except Exception as exc:
            self.failed.emit(f"Ошибка инициализации OpenVINO: {exc}")
            return False

    def _setup_inference_engine(self) -> bool:
        # The app runs only with OpenVINO runtime.
        self.device = "cpu"
        success = self._load_openvino_model()

        if success:
            self.logger.info("session_start device=%s", self.device.upper())

        return success

    def _setup_camera(self) -> cv2.VideoCapture | None:
        index = int(getattr(self.config, "camera_index", 0))
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            self.failed.emit(f"Не удалось открыть камеру (индекс {index})")
            return None

        return cap

    def _update_metrics(self, elapsed: float, fps: float, confidence: float, latency_ms: float, infer_ms: float) -> dict:
        self.frame_count += 1
        self.fps_series.append(fps)
        self.conf_series.append(confidence)
        self.lat_series_ms.append(latency_ms)
        self.infer_series_ms.append(infer_ms)

        self.fps_series = self.fps_series[-200:]
        self.conf_series = self.conf_series[-200:]
        self.lat_series_ms = self.lat_series_ms[-200:]
        self.infer_series_ms = self.infer_series_ms[-200:]

        return {
            "duration": time.time() - self.started_at,
            "frames": self.frame_count,
            "fps": fps,
            "avg_confidence": float(np.mean(self.conf_series)) if self.conf_series else 0.0,
            "avg_latency_ms": float(np.mean(self.lat_series_ms)) if self.lat_series_ms else 0.0,
            "avg_inference_ms": float(np.mean(self.infer_series_ms)) if self.infer_series_ms else 0.0,
            "device": self.device.upper(),
            "fps_series": self.fps_series,
            "confidence_series": self.conf_series,
            "latency_series_ms": self.lat_series_ms,
            "inference_series_ms": self.infer_series_ms,
        }

    def run(self) -> None:
        if not self._setup_inference_engine():
            return

        self.mouse_actions.reset()

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        mp_draw = mp.solutions.drawing_utils

        cap = self._setup_camera()
        if cap is None:
            return

        self.running = True
        self.started_at = time.time()

        try:
            while self.running:
                step_started = time.time()
                success, image = cap.read()

                if not success:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                confidence = 0.0
                infer_ms = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(rgb, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                        data = preprocess(hand_landmarks)
                        gesture_id, confidence, infer_ms = self._predict(data)

                        ok = self.mouse_actions.handle_gesture(hand_landmarks, gesture_id)

                        self.event_ready.emit(
                            time.strftime("%H:%M:%S"),
                            gestures.get(gesture_id, str(gesture_id)),
                            ok,
                        )

                        if getattr(self.config, "log_enabled", True):
                            now_mono = time.monotonic()
                            log_every = float(getattr(self.config, "log_interval_s", 1.0))
                            if log_every <= 0.0 or now_mono - self._last_event_log_mono >= log_every:
                                self._last_event_log_mono = now_mono
                                self.logger.info(
                                    "event gesture=%s ok=%s inference_ms=%.3f conf=%.4f",
                                    gestures.get(gesture_id, str(gesture_id)),
                                    ok,
                                    infer_ms,
                                    confidence,
                                )

                elapsed = max(1e-6, time.time() - step_started)
                fps = 1.0 / elapsed
                latency_ms = elapsed * 1000.0

                stats = self._update_metrics(elapsed, fps, confidence, latency_ms, infer_ms)
                self.stats_ready.emit(stats)

                self.frame_ready.emit(rgb)

                time.sleep(0.005)

        except Exception as exc:
            # In frozen builds, uncaught worker exceptions may fail silently for users.
            self.logger.exception("camera_worker_crash: %s", exc)
            self.failed.emit(f"Ошибка в потоке камеры: {exc}")
        finally:
            cap.release()
            hands.close()
            self.logger.info("session_stop frames=%s", self.frame_count)

    def stop(self) -> None:
        self.running = False
        self.wait(2000)
