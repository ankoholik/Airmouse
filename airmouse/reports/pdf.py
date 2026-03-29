"""PDF report export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def _save_report_chart(stats: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(stats.get("fps_series", []) or [0.0], color="#2563eb")
    axes[0].set_title("FPS")

    axes[1].plot(stats.get("confidence_series", []) or [0.0], color="#10b981")
    axes[1].set_title("Confidence")

    axes[2].plot(stats.get("latency_series_ms", []) or [0.0], color="#ef4444")
    axes[2].set_title("Latency (ms)")
    axes[2].set_xlabel("Frame window")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def export_pdf(stats: dict, output_path: Path) -> None:
    if not stats:
        raise ValueError("No statistics available for export")

    chart_path = output_path.with_suffix(".chart.png")
    _save_report_chart(stats, chart_path)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_w, page_h = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(24, page_h - 40, "AirMouse Gesture Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(24, page_h - 56, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = page_h - 86
    for key in ("duration", "frames", "fps", "avg_confidence", "avg_latency_ms", "device"):
        value = stats.get(key, "-")
        pdf.drawString(24, y, f"{key}: {value}")
        y -= 14

    pdf.drawImage(
        str(chart_path),
        24,
        y - 220,
        width=540,
        height=220,
        preserveAspectRatio=True,
        mask="auto",
    )

    pdf.showPage()
    pdf.save()

    if chart_path.exists():
        chart_path.unlink()
