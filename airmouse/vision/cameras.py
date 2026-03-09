"""Webcam discovery on Windows (DirectShow + PnP names)."""

from __future__ import annotations

import re
import subprocess

import cv2
from pygrabber.dshow_graph import FilterGraph


def _is_same_device_name(a: str, b: str) -> bool:
    return a.lower().strip() == b.lower().strip()


def _merge_unique_names(primary: list[str], *extra_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for source in (primary, *extra_lists):
        for name in source:
            cleaned = name.strip()
            if not cleaned:
                continue
            if any(_is_same_device_name(cleaned, existing) for existing in merged):
                continue
            merged.append(cleaned)
    return merged


def _dshow_names() -> list[str]:
    """Video capture device names via DirectShow (same order as OpenCV CAP_DSHOW)."""
    return [name.strip() for name in FilterGraph().get_input_devices() if name.strip()]


def _powershell_camera_names() -> list[str]:
    """PnP / USB video device names when DShow list is incomplete."""
    script = (
        "$names = @(); "
        "$names += Get-CimInstance Win32_PnPEntity | Where-Object { "
        "$_.PNPClass -eq 'Camera' -and $_.Status -eq 'OK' } | ForEach-Object { $_.Name }; "
        "$names += Get-CimInstance Win32_PnPEntity | Where-Object { "
        "$_.Service -eq 'usbvideo' -and $_.Status -eq 'OK' } | ForEach-Object { $_.Name }; "
        "$names | Sort-Object -Unique | ForEach-Object { Write-Output $_ }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return []

    names: list[str] = []
    for line in re.split(r"[\r\n]+", result.stdout):
        cleaned = line.strip()
        if cleaned:
            names.append(cleaned)
    return names


def _collect_camera_names() -> list[str]:
    """Device names in DirectShow order (pygrabber + PowerShell supplement)."""
    return _merge_unique_names(_dshow_names(), _powershell_camera_names())


def _try_open_capture(index: int) -> bool:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    cap.release()
    return ok


def probe_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """
    Return available cameras as (opencv_index, display_name).
    The i-th opened device gets the i-th name from the merged DirectShow list.
    """
    device_names = _collect_camera_names()

    opened_indices: list[int] = []
    for index in range(max_index):
        if _try_open_capture(index):
            opened_indices.append(index)

    found: list[tuple[int, str]] = []
    for order, index in enumerate(opened_indices):
        if order < len(device_names):
            label = device_names[order]
        elif index < len(device_names):
            label = device_names[index]
        else:
            label = f"Камера {index}"
        found.append((index, label))

    return found
