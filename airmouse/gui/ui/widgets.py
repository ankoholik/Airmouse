"""
Reusable custom widgets to reduce code duplication.
Each widget encapsulates related UI and logic.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from airmouse.vision.cameras import probe_cameras


class SettingsGroup(QGroupBox):
    """A reusable group box for settings sections."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.form = QFormLayout()
        self.setLayout(self.form)
    
    def add_field(self, label: str, widget: QWidget) -> None:
        """Add a labeled field to the group."""
        self.form.addRow(label, widget)


class DetectionSettings(SettingsGroup):
    """Detection confidence settings group."""
    
    def __init__(self, det_conf: float = 0.5, track_conf: float = 0.5, parent=None):
        super().__init__("Параметры распознавания", parent)
        
        self.det_conf_spin = QDoubleSpinBox()
        self.det_conf_spin.setRange(0.1, 0.99)
        self.det_conf_spin.setSingleStep(0.05)
        self.det_conf_spin.setValue(det_conf)
        
        self.track_conf_spin = QDoubleSpinBox()
        self.track_conf_spin.setRange(0.1, 0.99)
        self.track_conf_spin.setSingleStep(0.05)
        self.track_conf_spin.setValue(track_conf)
        
        self.add_field("Detection confidence", self.det_conf_spin)
        self.add_field("Tracking confidence", self.track_conf_spin)


class MouseSettings(SettingsGroup):
    """Mouse control settings group."""
    
    def __init__(
        self,
        sensitivity: float = 1.0,
        roi_margin: float = 0.08,
        scroll_step: int = 3,
        click_cooldown: float = 0.5,
        click_hold_threshold_s: float = 2.5,
        parent=None
    ):
        super().__init__("Управление мышью", parent)
        
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.2, 2.5)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setValue(sensitivity)
        
        self.roi_margin_spin = QDoubleSpinBox()
        self.roi_margin_spin.setRange(0.0, 0.35)
        self.roi_margin_spin.setSingleStep(0.02)
        self.roi_margin_spin.setValue(roi_margin)
        
        self.scroll_step_spin = QSpinBox()
        self.scroll_step_spin.setRange(1, 20)
        self.scroll_step_spin.setValue(scroll_step)

        self.click_cooldown_spin = QDoubleSpinBox()
        self.click_cooldown_spin.setRange(0.0, 5.0)
        self.click_cooldown_spin.setSingleStep(0.05)
        self.click_cooldown_spin.setValue(float(click_cooldown))

        self.click_hold_threshold_spin = QDoubleSpinBox()
        self.click_hold_threshold_spin.setRange(0.1, 10.0)
        self.click_hold_threshold_spin.setSingleStep(0.1)
        self.click_hold_threshold_spin.setValue(float(click_hold_threshold_s))

        self.add_field("Чувствительность", self.sensitivity_spin)
        self.add_field("Зона камеры (margin)", self.roi_margin_spin)
        self.add_field("Шаг скролла", self.scroll_step_spin)
        self.add_field("Кулдаун клика (сек)", self.click_cooldown_spin)
        self.add_field("Удержание для зажатия (сек)", self.click_hold_threshold_spin)


class LogsSettings(SettingsGroup):
    """Logging configuration group."""

    def __init__(self, enabled: bool = True, interval_s: float = 1.0, parent=None):
        super().__init__("Логи", parent)

        self.enable_on_btn = QPushButton("Вкл")
        self.enable_off_btn = QPushButton("Выкл")

        for btn in (self.enable_on_btn, self.enable_off_btn):
            btn.setCheckable(True)
            btn.setMinimumWidth(70)

        def _apply_toggle_styles() -> None:
            if self.enable_on_btn.isChecked():
                self.enable_on_btn.setStyleSheet(
                    "background-color: #2563eb; color: #ffffff; border-radius: 8px; padding: 4px 10px;"
                )
            else:
                self.enable_on_btn.setStyleSheet(
                    "background-color: transparent; color: #2563eb; "
                    "border: 1px solid #2563eb; border-radius: 8px; padding: 4px 10px;"
                )

            if self.enable_off_btn.isChecked():
                self.enable_off_btn.setStyleSheet(
                    "background-color: #ef4444; color: #ffffff; border-radius: 8px; padding: 4px 10px;"
                )
            else:
                self.enable_off_btn.setStyleSheet(
                    "background-color: transparent; color: #ef4444; "
                    "border: 1px solid #ef4444; border-radius: 8px; padding: 4px 10px;"
                )

        def _set_enabled(value: bool) -> None:
            self.enable_on_btn.setChecked(value)
            self.enable_off_btn.setChecked(not value)
            _apply_toggle_styles()

        self.enable_on_btn.clicked.connect(lambda: _set_enabled(True))
        self.enable_off_btn.clicked.connect(lambda: _set_enabled(False))

        _set_enabled(bool(enabled))

        self.log_interval_spin = QDoubleSpinBox()
        self.log_interval_spin.setRange(0.0, 10.0)
        self.log_interval_spin.setSingleStep(0.25)
        self.log_interval_spin.setValue(float(interval_s))

        row = QWidget(parent)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.enable_on_btn)
        row_layout.addWidget(self.enable_off_btn)

        self.add_field("Логи", row)
        self.add_field("Интервал логов (сек)", self.log_interval_spin)

    def is_enabled(self) -> bool:
        return self.enable_on_btn.isChecked()


class AppearanceSettings(SettingsGroup):
    """Theme and appearance settings group."""
    
    def __init__(self, current_theme: str = "light", parent=None):
        super().__init__("Внешний вид", parent)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["light", "dark"])
        self.theme_combo.setCurrentText(current_theme)
        
        self.add_field("Тема", self.theme_combo)


class ComputeSettings(SettingsGroup):
    """Computation device settings group."""
    
    def __init__(self, current_device: str = "cpu", parent=None):
        super().__init__("Вычисления", parent)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "gpu"])
        self.device_combo.setCurrentText(current_device)
        
        self.add_field("Устройство", self.device_combo)


class CameraSettings(SettingsGroup):
    """Web camera selection."""

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__("Камера", parent)

        row = QWidget(parent)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_combo = QComboBox()
        self.refresh_btn = QPushButton("Обновить")
        self.refresh_btn.clicked.connect(self.populate_cameras)

        row_layout.addWidget(self.camera_combo, 1)
        row_layout.addWidget(self.refresh_btn)

        self.add_field("Устройство", row)
        self._pending_index = camera_index

    def populate_cameras(self) -> None:
        if self.camera_combo.count():
            self._pending_index = self.selected_index()

        self.camera_combo.clear()
        cameras = probe_cameras()
        if not cameras:
            self.camera_combo.addItem("Камера не найдена", 0)
            return

        for index, label in cameras:
            self.camera_combo.addItem(label, index)

        target = self._pending_index
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == target:
                self.camera_combo.setCurrentIndex(i)
                return
        self.camera_combo.setCurrentIndex(0)

    def selected_index(self) -> int:
        data = self.camera_combo.currentData()
        if data is None:
            return 0
        return int(data)


class ControlButtonsPanel(QWidget):
    """Unified start/stop buttons panel."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_btn = QPushButton("Запуск")
        self.stop_btn = QPushButton("Остановка")
        
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)


class StatsTable(QTableWidget):
    """Statistics display table."""
    
    STATS_LABELS = [
        "Duration",
        "Frames",
        "FPS",
        "Avg confidence",
        "Avg latency (ms)",
        "Avg inference (ms)",
    ]
    
    def __init__(self, parent=None):
        super().__init__(len(self.STATS_LABELS), 2, parent)
        self.setHorizontalHeaderLabels(["Параметр", "Значение"])
        
        for idx, label in enumerate(self.STATS_LABELS):
            self.setItem(idx, 0, QTableWidgetItem(label))
            self.setItem(idx, 1, QTableWidgetItem("-"))

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
    
    def update_stats(self, stats: dict) -> None:
        """Update all statistics from a stats dictionary."""
        values = [
            f"{float(stats.get('duration', 0.0)):.1f}",
            str(stats.get("frames", 0)),
            f"{float(stats.get('fps', 0.0)):.2f}",
            f"{float(stats.get('avg_confidence', 0.0)):.3f}",
            f"{float(stats.get('avg_latency_ms', 0.0)):.2f}",
            f"{float(stats.get('avg_inference_ms', 0.0)):.3f}",
        ]
        for i, value in enumerate(values):
            self.setItem(i, 1, QTableWidgetItem(value))


class EventsTable(QTableWidget):
    """Gesture events log table."""
    
    MAX_ROWS = 120
    
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Время", "Жест", "OK"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
    
    def add_event(self, timestamp: str, gesture: str, ok: bool) -> None:
        """Add a new event row."""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(timestamp))
        self.setItem(row, 1, QTableWidgetItem(gesture))
        self.setItem(row, 2, QTableWidgetItem("yes" if ok else "no"))
        
        # Keep only last MAX_ROWS events
        while self.rowCount() > self.MAX_ROWS:
            self.removeRow(0)


class GesturesTable(QTableWidget):
    """Gestures reference table."""
    
    def __init__(self, gestures_dict: dict, parent=None):
        super().__init__(len(gestures_dict), 2, parent)
        self.setHorizontalHeaderLabels(["ID", "Жест"])
        
        for idx, (gesture_id, gesture_name) in enumerate(gestures_dict.items()):
            self.setItem(idx, 0, QTableWidgetItem(str(gesture_id)))
            self.setItem(idx, 1, QTableWidgetItem(gesture_name))

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
