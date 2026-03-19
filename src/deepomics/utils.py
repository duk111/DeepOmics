from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, Optional


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved directory path.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with UTF-8 encoding.

    Args:
        data: Serializable mapping.
        path: Output file path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def get_logger(
    name: str = "DeepOmics",
    log_file: Optional[str | Path] = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """Build a standardized logger.

    Args:
        name: Logger name.
        log_file: Optional file handler path.
        level: Logging level string or integer.

    Returns:
        Configured logger instance.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else int(level)
    logger = logging.getLogger(name)

    if not getattr(logger, "_deepomics_configured", False):
        logger.setLevel(resolved_level)
        logger.propagate = False

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger._deepomics_configured = True  # type: ignore[attr-defined]
    else:
        logger.setLevel(resolved_level)
        for handler in logger.handlers:
            handler.setLevel(resolved_level)

    if log_file is not None:
        log_path = Path(log_file)
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == log_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    return logger


@contextmanager
def log_step(logger: logging.Logger, step_name: str) -> Iterator[None]:
    """Measure elapsed time for a logical step.

    Args:
        logger: Logger instance.
        step_name: Human-readable step name.
    """
    start = perf_counter()
    logger.info("Started: %s", step_name)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("Finished: %s (%.2fs)", step_name, elapsed)
