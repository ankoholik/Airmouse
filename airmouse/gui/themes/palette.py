"""
Centralized color palette management for AirMouse application.
Supports light and dark themes with easy customization.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ColorScheme:
    """A complete color scheme for the application."""
    
    # Primary colors
    primary: str
    primary_hover: str
    primary_text: str
    
    # Secondary colors
    secondary: str
    secondary_hover: str
    secondary_text: str
    
    # Backgrounds
    bg_main: str
    bg_secondary: str
    bg_tertiary: str
    
    # Borders
    border_light: str
    border_dark: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    
    # Status colors
    success: str
    warning: str
    error: str
    info: str
    
    # Chart colors
    chart_fps: str
    chart_confidence: str
    chart_latency: str
    
    # Accent colors
    accent_blue: str
    accent_green: str
    accent_red: str


# ============================================================================
# LIGHT THEME
# ============================================================================

LIGHT_PALETTE = ColorScheme(
    # Primary - Blue
    primary="#2563eb",
    primary_hover="#1d4ed8",
    primary_text="#ffffff",
    
    # Secondary - Slate
    secondary="#64748b",
    secondary_hover="#475569",
    secondary_text="#ffffff",
    
    # Backgrounds
    bg_main="#f8fafc",
    bg_secondary="#ffffff",
    bg_tertiary="#f1f5f9",
    
    # Borders
    border_light="#e2e8f0",
    border_dark="#94a3b8",
    
    # Text
    text_primary="#0f172a",
    text_secondary="#475569",
    text_disabled="#cbd5e1",
    
    # Status
    success="#10b981",
    warning="#f59e0b",
    error="#ef4444",
    info="#3b82f6",
    
    # Charts
    chart_fps="#2563eb",
    chart_confidence="#10b981",
    chart_latency="#ef4444",
    
    # Accents
    accent_blue="#2563eb",
    accent_green="#10b981",
    accent_red="#ef4444",
)


# ============================================================================
# DARK THEME
# ============================================================================

DARK_PALETTE = ColorScheme(
    # Primary - Blue
    primary="#2563eb",
    primary_hover="#3b82f6",
    primary_text="#ffffff",
    
    # Secondary - Slate
    secondary="#64748b",
    secondary_hover="#94a3b8",
    secondary_text="#ffffff",
    
    # Backgrounds
    bg_main="#0f172a",
    bg_secondary="#111827",
    bg_tertiary="#1f2937",
    
    # Borders
    border_light="#334155",
    border_dark="#64748b",
    
    # Text
    text_primary="#f8fafc",
    text_secondary="#cbd5e1",
    text_disabled="#64748b",
    
    # Status
    success="#10b981",
    warning="#f59e0b",
    error="#ef4444",
    info="#3b82f6",
    
    # Charts
    chart_fps="#3b82f6",
    chart_confidence="#34d399",
    chart_latency="#f87171",
    
    # Accents
    accent_blue="#3b82f6",
    accent_green="#34d399",
    accent_red="#f87171",
)


# ============================================================================
# THEME REGISTRY
# ============================================================================

THEMES = {
    "light": LIGHT_PALETTE,
    "dark": DARK_PALETTE,
}


def get_palette(theme: Literal["light", "dark"]) -> ColorScheme:
    """Get color palette for the specified theme."""
    return THEMES.get(theme, LIGHT_PALETTE)


def list_available_themes() -> list[str]:
    """Return list of available theme names."""
    return list(THEMES.keys())
