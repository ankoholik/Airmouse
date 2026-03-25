"""
Main application window for AirMouse gesture recognition system.
Orchestrates all UI components and coordinates between worker thread and UI.
"""

import sys
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
import cv2
import numpy as np

from airmouse.config import get_config
from airmouse.gestures.registry import GESTURES as gestures
from airmouse.gui.services.camera_worker import CameraWorker
from airmouse.gui.themes.palette import get_palette
from airmouse.gui.ui.styles import generate_stylesheet, button_style_active
from airmouse.gui.ui.tabs import StatsTab, ChartsTab, GesturesTab, SettingsTab, ReportsTab
from airmouse.gui.ui.widgets import (
    AppearanceSettings,
    CameraSettings,
    ComputeSettings,
    ControlButtonsPanel,
    DetectionSettings,
    MouseSettings,
    LogsSettings,
)


class MainWindow(QMainWindow):
    """Main application window."""
    
    WINDOW_TITLE = "AirMouse"
    DEFAULT_THEME = "light"
    
    def __init__(self) -> None:
        super().__init__()
        
        # Load configuration
        self.app_config = get_config()
        self.config = self.app_config.ui
        self.paths = self.app_config.paths
        
        # Initialize window
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(self.config.window_width, self.config.window_height)
        
        # State
        self.worker: CameraWorker | None = None
        self.video_stream_active = False
        self.stats: dict[str, object] = {}
        self.current_palette = get_palette(self.config.theme)
        
        # Build UI
        self._build_ui()
        self._apply_theme(self.config.theme)
        
        # Setup timer for chart updates
        self.chart_timer = QTimer(self)
        self.chart_timer.timeout.connect(self._refresh_charts)
        self.chart_timer.start(600)
    
    def _build_ui(self) -> None:
        """Construct the main UI layout."""
        root = QWidget()
        self.setCentralWidget(root)
        
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ===== LEFT SIDE: VIDEO + CONTROLS =====
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Видео не запущено")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, 1)
        
        # Control buttons
        self.controls = ControlButtonsPanel()
        self.start_btn = self.controls.start_btn
        self.stop_btn = self.controls.stop_btn
        
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

        left_layout.addWidget(self.controls, 0, alignment=Qt.AlignBottom)
        
        main_layout.addLayout(left_layout, 2)
        
        # ===== RIGHT SIDE: TABS =====
        self.tabs = QTabWidget()
        self._build_tabs()
        main_layout.addWidget(self.tabs, 1)
    
    def _build_tabs(self) -> None:
        """Build tab widgets."""
        # Stats tab
        self.stats_tab = StatsTab()
        self.tabs.addTab(self.stats_tab, "Статистика")
        
        # Charts tab
        self.charts_tab = ChartsTab()
        self.tabs.addTab(self.charts_tab, "Графики")
        
        # Gestures tab
        self.gestures_tab = GesturesTab(gestures)
        self.tabs.addTab(self.gestures_tab, "Жесты")
        
        # Settings tab
        self._build_settings_tab()
        
        # Reports tab
        self.reports_tab = ReportsTab()
        self.reports_tab.pdf_btn.clicked.connect(self._on_export_pdf)
        self.reports_tab.xlsx_btn.clicked.connect(self._on_export_xlsx)
        self.tabs.addTab(self.reports_tab, "Отчеты")
    
    def _build_settings_tab(self) -> None:
        """Build settings tab with all configuration groups."""
        # Create individual setting groups
        self.detection_settings = DetectionSettings(
            det_conf=self.config.min_detection_confidence,
            track_conf=self.config.min_tracking_confidence
        )
        
        self.mouse_settings = MouseSettings(
            sensitivity=self.config.mouse_sensitivity,
            roi_margin=self.config.camera_roi_margin,
            scroll_step=self.config.scroll_step,
            click_cooldown=getattr(self.config, "click_cooldown", 0.5),
            click_hold_threshold_s=getattr(self.config, "click_hold_threshold_s", 2.5),
        )

        self.logs_settings = LogsSettings(
            enabled=getattr(self.config, "log_enabled", True),
            interval_s=getattr(self.config, "log_interval_s", 1.0),
        )
        # Apply logging toggle to stats UI immediately (no restart needed)
        self.stats_tab.set_events_enabled(self.logs_settings.is_enabled())
        self.stats_tab.set_events_interval_s(self.logs_settings.log_interval_spin.value())
        self.logs_settings.enable_on_btn.clicked.connect(lambda: self._on_logs_toggle(True))
        self.logs_settings.enable_off_btn.clicked.connect(lambda: self._on_logs_toggle(False))
        self.logs_settings.log_interval_spin.valueChanged.connect(self._on_logs_interval_changed)

        self.appearance_settings = AppearanceSettings(self.config.theme)
        self.appearance_settings.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        
        self.compute_settings = ComputeSettings(self.config.compute_device)

        self.camera_settings = CameraSettings(self.config.camera_index)
        self.camera_settings.populate_cameras()
        
        # Create settings tab with all groups
        self.settings_tab = SettingsTab(
            self.detection_settings,
            self.mouse_settings,
            self.logs_settings,
            self.camera_settings,
            self.appearance_settings,
            self.compute_settings,
        )
        
        self.settings_tab.apply_btn.clicked.connect(self._on_apply_settings)
        self.tabs.addTab(self.settings_tab, "Настройки")

    def _on_logs_toggle(self, enabled: bool) -> None:
        """Handle runtime log toggle (affects UI log table)."""
        self.stats_tab.set_events_enabled(bool(enabled))

    def _on_logs_interval_changed(self, interval_s: float) -> None:
        """Handle runtime log interval changes (affects UI log table)."""
        self.stats_tab.set_events_interval_s(interval_s)
    
    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme selection change."""
        self._apply_theme(theme_name)
    
    def _apply_theme(self, theme_name: str) -> None:
        """Apply theme to entire application."""
        self.config.theme = theme_name
        self.current_palette = get_palette(theme_name)
        
        # Generate and apply stylesheet
        stylesheet = generate_stylesheet(self.current_palette)
        self.setStyleSheet(stylesheet)
        
        # Update chart colors
        if hasattr(self, 'charts_tab'):
            self.charts_tab.set_palette(self.current_palette)
        
        # Update button styles
        self._update_control_buttons(self.video_stream_active)
    
    def _update_control_buttons(self, is_running: bool) -> None:
        """Update start/stop button styles based on state."""
        self.start_btn.setStyleSheet(
            button_style_active(self.current_palette, base_active=not is_running)
        )
        self.stop_btn.setStyleSheet(
            button_style_active(self.current_palette, base_active=is_running)
        )
    
    def _on_apply_settings(self) -> None:
        """Handle settings application."""
        self.config.min_detection_confidence = self.detection_settings.det_conf_spin.value()
        self.config.min_tracking_confidence = self.detection_settings.track_conf_spin.value()
        self.config.mouse_sensitivity = self.mouse_settings.sensitivity_spin.value()
        self.config.camera_roi_margin = self.mouse_settings.roi_margin_spin.value()
        self.config.scroll_step = self.mouse_settings.scroll_step_spin.value()
        self.config.click_cooldown = self.mouse_settings.click_cooldown_spin.value()
        self.config.click_hold_threshold_s = self.mouse_settings.click_hold_threshold_spin.value()
        self.config.log_enabled = self.logs_settings.is_enabled()
        self.config.log_interval_s = self.logs_settings.log_interval_spin.value()
        self.config.compute_device = self.compute_settings.device_combo.currentText()
        self.config.camera_index = self.camera_settings.selected_index()
        
        QMessageBox.information(
            self,
            "Готово",
            "Новые настройки применятся при следующем запуске потока."
        )
    
    def _on_start(self) -> None:
        """Handle start button click."""
        if self.worker and self.worker.isRunning():
            return
        try:
            self.worker = CameraWorker(self.config, model_path=self.paths.model_weights)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить поток камеры: {exc}")
            return
        
        # Connect signals
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.stats_ready.connect(self._on_stats_ready)
        self.worker.event_ready.connect(self._on_event_ready)
        self.worker.failed.connect(self._on_worker_failed)
        
        self.worker.start()
        
        self.video_stream_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._update_control_buttons(is_running=True)
    
    def _on_stop(self) -> None:
        """Handle stop button click."""
        self.video_stream_active = False
        
        if self.worker:
            self.worker.stop()
            self.worker = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._update_control_buttons(is_running=False)
        
        self.video_label.clear()
        self.video_label.setText("Видео не запущено")
    
    def _on_frame_ready(self, rgb: np.ndarray) -> None:
        """Handle new frame from camera worker."""
        if not self.video_stream_active:
            return
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w, ch = bgr.shape
        
        qimg = QImage(bgr.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        
        self.video_label.setPixmap(pix)
    
    def _on_stats_ready(self, stats: dict) -> None:
        """Handle statistics update."""
        self.stats = stats
        self.stats_tab.update_stats(stats)
    
    def _on_event_ready(self, timestamp: str, gesture: str, ok: bool) -> None:
        """Handle gesture event."""
        if not self.logs_settings.is_enabled():
            return
        self.stats_tab.add_event(timestamp, gesture, ok)
    
    def _on_worker_failed(self, message: str) -> None:
        """Handle worker error."""
        QMessageBox.critical(self, "Ошибка", message)
        self._on_stop()
    
    def _refresh_charts(self) -> None:
        """Refresh chart display."""
        if self.stats:
            self.charts_tab.refresh(self.stats)
    
    def _on_export_pdf(self) -> None:
        """Export report as PDF."""
        if not self.stats:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала запустите сессию и соберите статистику."
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PDF",
            str(Path.cwd() / "mainproject_report.pdf"),
            "PDF (*.pdf)",
        )
        
        if not file_path:
            return
        
        try:
            self.reports_tab.export_pdf(self.stats, Path(file_path))
            QMessageBox.information(self, "Готово", f"PDF сохранен: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения PDF: {exc}")
    
    def _on_export_xlsx(self) -> None:
        """Export report as Excel."""
        if not self.stats:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала запустите сессию и соберите статистику."
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить XLSX",
            str(Path.cwd() / "mainproject_report.xlsx"),
            "Excel (*.xlsx)",
        )
        
        if not file_path:
            return
        
        try:
            self.reports_tab.export_xlsx(self.stats, Path(file_path))
            QMessageBox.information(self, "Готово", f"XLSX сохранен: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения XLSX: {exc}")
    
    def closeEvent(self, event) -> None:  # noqa: N802
        """Handle window close event."""
        self._on_stop()
        super().closeEvent(event)


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
