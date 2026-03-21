"""
Tab widgets for the main interface.
Each tab handles a specific category of functionality.
"""

import time
from pathlib import Path

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
)
from pyqtgraph import PlotWidget

from airmouse.reports import export_pdf as write_pdf_report
from airmouse.reports import export_xlsx as write_xlsx_report

from .widgets import StatsTable, EventsTable, GesturesTable


class StatsTab(QWidget):
    """Statistics display tab with live updates."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.stats_table = StatsTable()
        self.events_table = EventsTable()
        self._events_enabled = True
        self._events_min_interval_s = 0.0
        self._last_event_ts = 0.0
        
        layout.addWidget(self.stats_table)
        layout.addWidget(self.events_table)
    
    def update_stats(self, stats: dict) -> None:
        """Update statistics display."""
        self.stats_table.update_stats(stats)
    
    def add_event(self, timestamp: str, gesture: str, ok: bool) -> None:
        """Add gesture event to the log."""
        if not self._events_enabled:
            return

        now = time.monotonic()
        if self._events_min_interval_s > 0 and (now - self._last_event_ts) < self._events_min_interval_s:
            return
        self._last_event_ts = now

        self.events_table.add_event(timestamp, gesture, ok)

    def set_events_enabled(self, enabled: bool) -> None:
        """Enable/disable events table updates (table stays visible)."""
        enabled = bool(enabled)
        self._events_enabled = enabled

    def set_events_interval_s(self, interval_s: float) -> None:
        """Set minimum interval between UI log rows (seconds)."""
        try:
            interval_s = float(interval_s)
        except (TypeError, ValueError):
            interval_s = 0.0
        self._events_min_interval_s = max(0.0, interval_s)


class ChartsTab(QWidget):
    """Real-time performance charts tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Configure pyqtgraph for light background
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        
        # FPS Chart
        self.fps_plot = PlotWidget(title="FPS")
        self.fps_line = self.fps_plot.plot([], pen="b")
        self.fps_plot.setLabel("left", "FPS")
        self.fps_plot.setLabel("bottom", "Samples")
        
        # Confidence Chart
        self.conf_plot = PlotWidget(title="Confidence")
        self.conf_line = self.conf_plot.plot([], pen="g")
        self.conf_plot.setLabel("left", "Confidence")
        self.conf_plot.setLabel("bottom", "Samples")
        
        # Latency Chart
        self.lat_plot = PlotWidget(title="Latency (ms)")
        self.lat_line = self.lat_plot.plot([], pen="#ef4444")
        self.lat_plot.setLabel("left", "ms")
        self.lat_plot.setLabel("bottom", "Samples")
        
        layout.addWidget(self.fps_plot)
        layout.addWidget(self.conf_plot)
        layout.addWidget(self.lat_plot)
    
    def refresh(self, stats: dict) -> None:
        """Refresh all charts with latest data."""
        if not stats:
            return
        self.fps_line.setData(stats.get("fps_series", []))
        self.conf_line.setData(stats.get("confidence_series", []))
        self.lat_line.setData(stats.get("latency_series_ms", []))
    
    def set_palette(self, palette) -> None:
        """Update chart colors based on palette."""
        self.fps_line.setPen(palette.chart_fps)
        self.conf_line.setPen(palette.chart_confidence)
        self.lat_line.setPen(palette.chart_latency)

        for plot in (self.fps_plot, self.conf_plot, self.lat_plot):
            plot.setBackground(palette.bg_secondary)
            for axis in ("left", "bottom"):
                ax = plot.getAxis(axis)
                ax.setPen(palette.text_primary)
                ax.setTextPen(palette.text_primary)


class GesturesTab(QWidget):
    """Gesture classes reference tab."""
    
    def __init__(self, gestures_dict: dict, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Классы вашей модели:"))
        
        self.table = GesturesTable(gestures_dict)
        layout.addWidget(self.table)


class SettingsTab(QWidget):
    """Settings and configuration tab."""
    
    def __init__(
        self,
        detection_settings,
        mouse_settings,
        logs_settings,
        camera_settings,
        appearance_settings,
        compute_settings,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        layout.addWidget(detection_settings)
        layout.addWidget(mouse_settings)
        layout.addWidget(logs_settings)
        layout.addWidget(camera_settings)
        layout.addWidget(appearance_settings)
        layout.addWidget(compute_settings)
        
        self.apply_btn = QPushButton("Применить")
        layout.addWidget(self.apply_btn)
        layout.addStretch()
    
    def get_settings(self) -> dict:
        """Retrieve current settings from all groups."""
        # This will be called by parent window
        # Settings groups are accessible as attributes
        return {}


class ReportsTab(QWidget):
    """Export and reporting tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.pdf_btn = QPushButton("Экспорт PDF")
        self.xlsx_btn = QPushButton("Экспорт XLSX")
        
        layout.addWidget(self.pdf_btn)
        layout.addWidget(self.xlsx_btn)
        info = QLabel("Отчет содержит метрики, диаграммы и динамические ряды.")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignHCenter)
        layout.addWidget(info)
        layout.addStretch()
    
    def export_pdf(self, stats: dict, output_path: Path) -> None:
        write_pdf_report(stats, output_path)

    def export_xlsx(self, stats: dict, output_path: Path) -> None:
        write_xlsx_report(stats, output_path)
