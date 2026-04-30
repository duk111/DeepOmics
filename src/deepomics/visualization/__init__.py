from __future__ import annotations

from .context import VisualizationContext
from .registry import FIGURE_REGISTRY, STATIC_FIGURE_SPECS, FigureSpec, iter_figure_specs
from .reports import generate_html_report, generate_markdown_report, generate_report_plots

__all__ = [
    "FIGURE_REGISTRY",
    "STATIC_FIGURE_SPECS",
    "FigureSpec",
    "VisualizationContext",
    "generate_html_report",
    "generate_markdown_report",
    "generate_report_plots",
    "iter_figure_specs",
]
