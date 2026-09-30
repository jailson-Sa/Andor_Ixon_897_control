"""Simple scientific file output helpers for the first control milestone."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_frame_npy(path: str | Path, frame: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frame)
    return path


def save_frames_npz(path: str | Path, frames: np.ndarray, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames, metadata=json.dumps(metadata or {}, indent=2))
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def save_metadata_json(path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return path
