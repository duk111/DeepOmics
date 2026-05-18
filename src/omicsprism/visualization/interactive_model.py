from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControlSpec:
    id: str
    type: str
    label: str
    default: Any
    options: list[dict[str, Any]] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class InteractiveViewSpec:
    id: str
    title: str
    kind: str
    schema_id: str
    enabled: bool = True
    description: str = ""
    data_key: str = ""


@dataclass(frozen=True)
class InteractiveReportModel:
    meta: dict[str, Any]
    figures: tuple[dict[str, Any], ...]
    views: tuple[InteractiveViewSpec, ...]
    schemas: dict[str, Any]
    datasets: dict[str, Any]
    initial_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ControlSpec",
    "InteractiveReportModel",
    "InteractiveViewSpec",
]
