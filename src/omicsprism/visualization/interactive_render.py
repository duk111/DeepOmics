from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..utils import safe_mkdir
from .interactive_assets import _interactive_html_template
from .interactive_model import InteractiveReportModel


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Index, pd.Series)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def _json_script_payload(data: Any) -> str:
    return _json_dumps(data).replace("</", "<\\/")


def render_interactive_report_html_from_model(model: InteractiveReportModel, project_name: str) -> str:
    html_text = _interactive_html_template()
    html_text = html_text.replace("__PROJECT_NAME__", html.escape(str(project_name)))
    html_text = html_text.replace("__PAYLOAD__", _json_script_payload(model.to_dict()))
    return html_text


def render_interactive_report_html(
    engine: Any,
    cfg: Any,
    model_builder: Callable[[Any, Any], InteractiveReportModel],
) -> str:
    model = model_builder(engine, cfg)
    return render_interactive_report_html_from_model(model, str(cfg.project_name))


def generate_interactive_visual_report(
    engine: Any,
    cfg: Any,
    report_path: str | Path,
    model_builder: Callable[[Any, Any], InteractiveReportModel],
) -> None:
    output_path = Path(report_path)
    safe_mkdir(output_path.parent)
    output_path.write_text(render_interactive_report_html(engine, cfg, model_builder), encoding="utf-8")


__all__ = [
    "_json_default",
    "_json_dumps",
    "_json_script_payload",
    "generate_interactive_visual_report",
    "render_interactive_report_html",
    "render_interactive_report_html_from_model",
]
