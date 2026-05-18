from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_ui(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Launch the local Streamlit UI."""
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "The local Web UI requires Streamlit. Install it with: "
            "pip install 'omicsprism[ui]'"
        ) from exc

    app_path = Path(__file__).with_name("local_ui_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(int(port)),
    ]
    raise SystemExit(subprocess.call(command))


def main() -> None:
    launch_ui()


__all__ = ["launch_ui", "main"]
