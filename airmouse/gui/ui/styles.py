"""
Dynamic stylesheet generation based on color palette.
Centralizes all QSS styling logic for easy maintenance.
"""

from ..themes.palette import ColorScheme


def generate_stylesheet(palette: ColorScheme) -> str:
    """Generate complete QSS stylesheet from a color palette."""
    
    return f"""
    /* ====== MAIN WIDGET ====== */
    QWidget {{
        background: {palette.bg_main};
        color: {palette.text_primary};
        font-size: 13px;
        font-family: "Segoe UI";
    }}
    
    /* ====== GROUP BOX ====== */
    QGroupBox {{
        border: 1px solid {palette.border_dark};
        border-radius: 10px;
        margin-top: 8px;
        padding-top: 10px;
        padding-left: 15px;
        padding-right: 15px;
        padding-bottom: 10px;
        color: {palette.text_primary};
        font-weight: 500;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }}
    
    /* ====== BUTTONS ====== */
    QPushButton {{
        background: {palette.primary};
        color: {palette.primary_text};
        border: none;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 500;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background: {palette.primary_hover};
    }}
    
    QPushButton:pressed {{
        background: {palette.primary_hover};
        padding: 9px 11px 7px 13px;
    }}
    
    QPushButton:disabled {{
        background: {palette.border_dark};
        color: {palette.text_disabled};
    }}
    
    /* ====== TABS ====== */
    QTabWidget::pane {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        border: 0px;
    }}
    
    QTabBar::tab {{
        border: 0px;
        background: transparent;
        padding: 8px 12px;
        margin-right: 6px;
        border-radius: 8px;
        color: {palette.text_secondary};
    }}
    
    QTabBar::tab:selected {{
        background: {palette.primary};
        color: {palette.primary_text};
        border-radius: 8px;
    }}
    
    QTabBar::tab:hover:!selected {{
        background: {palette.bg_tertiary};
    }}
    
    /* ====== TABLE ====== */
    QTableWidget {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        border: 1px solid {palette.border_light};
        border-radius: 8px;
        gridline-color: {palette.border_light};
    }}
    
    QTableWidget::item {{
        padding: 4px;
        border: none;
    }}
    
    QTableWidget::item:selected {{
        background: {palette.primary};
        color: {palette.primary_text};
    }}
    
    QHeaderView::section {{
        background: {palette.bg_tertiary};
        color: {palette.text_primary};
        padding: 4px;
        border: none;
        border-right: 1px solid {palette.border_light};
        font-weight: bold;
    }}
    
    /* ====== SPINBOX & COMBO ====== */
    QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        border: 1px solid {palette.border_dark};
        border-radius: 8px;
        padding: 5px;
    }}
    
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 2px solid {palette.primary};
    }}
    
    QComboBox::drop-down {{
        border: none;
        padding-right: 5px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        width: 0px;
    }}
    
    QAbstractItemView {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        selection-background-color: {palette.primary};
        border: 1px solid {palette.border_light};
    }}
    
    /* ====== SCROLLBAR ====== */
    QScrollBar:vertical {{
        background: {palette.bg_main};
        width: 12px;
        border: none;
    }}
    
    QScrollBar::handle:vertical {{
        background: {palette.border_dark};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {palette.text_secondary};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}
    
    /* ====== LABEL ====== */
    QLabel {{
        color: {palette.text_primary};
    }}
    
    /* ====== MENU ====== */
    QMenuBar {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        border-bottom: 1px solid {palette.border_light};
    }}
    
    QMenuBar::item:selected {{
        background: {palette.primary};
        color: {palette.primary_text};
    }}
    
    QMenu {{
        background: {palette.bg_secondary};
        color: {palette.text_primary};
        border: 1px solid {palette.border_light};
    }}
    
    QMenu::item:selected {{
        background: {palette.primary};
        color: {palette.primary_text};
    }}
    """


def button_style_active(palette: ColorScheme, base_active: bool = True) -> str:
    """Generate style for start/stop buttons based on active state."""
    if base_active:
        bg = palette.primary
        bg_hover = palette.primary_hover
        fg = palette.primary_text
    else:
        bg = palette.border_dark
        bg_hover = palette.secondary_hover
        fg = palette.primary_text

    # Note: this style is applied directly to a button instance via setStyleSheet(),
    # so we must include :hover here (global QPushButton:hover won't apply).
    return f"""
    QPushButton {{
        background: {bg};
        color: {fg};
        border: none;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {bg_hover};
    }}
    QPushButton:pressed {{
        background: {bg_hover};
        padding: 9px 11px 7px 13px;
    }}
    """
