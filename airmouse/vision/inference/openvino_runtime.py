from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openvino import Core


@dataclass
class OvPrediction:
    gesture_id: int
    confidence: float
    inference_ms: float


class OpenVinoGestureRuntime:
    """OpenVINO inference runtime (IR .xml or ONNX)."""

    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        self.model_path = model_path
        self.device = device
        self._compiled: Any | None = None
        self._input_key: Any | None = None
        self._output_key: Any | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")

        core = Core()
        ov_model = core.read_model(model=str(self.model_path))
        compiled = core.compile_model(ov_model, self.device)
        self._compiled = compiled
        self._input_key = compiled.inputs[0]
        self._output_key = compiled.outputs[0]

    def predict(self, x63: np.ndarray) -> OvPrediction:
        if self._compiled is None:
            self.load()

        x = np.asarray(x63, dtype=np.float32).reshape(1, 63)
        started = time.perf_counter()
        result = self._compiled({self._input_key: x})
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        logits = result[self._output_key]
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        gesture_id = cls + 1
        return OvPrediction(gesture_id=gesture_id, confidence=conf, inference_ms=float(elapsed_ms))
