"""HDF5 writers for Andor image stacks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def write_hdf5_stack(
    path: str | Path,
    frames: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    processed_frames: np.ndarray | None = None,
    photon_counted_frames: np.ndarray | None = None,
    compression: str | None = "gzip",
) -> Path:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("HDF5 support requires h5py. Install with: py -3.11 -m pip install h5py") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("raw", data=np.asarray(frames), compression=compression)
        if processed_frames is not None:
            h5.create_dataset("processed", data=np.asarray(processed_frames), compression=compression)
        if photon_counted_frames is not None:
            h5.create_dataset("photon_counted", data=np.asarray(photon_counted_frames), compression=compression)
        meta = metadata or {}
        h5.attrs["metadata_json"] = json.dumps(meta, indent=2, default=_json_default)
        meta_group = h5.create_group("metadata")
        meta_group.attrs["json"] = json.dumps(meta, indent=2, default=_json_default)
    return path


def write_npz_stack(
    path: str | Path,
    frames: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    processed_frames: np.ndarray | None = None,
    photon_counted_frames: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"raw": np.asarray(frames)}
    if processed_frames is not None:
        payload["processed"] = np.asarray(processed_frames)
    if photon_counted_frames is not None:
        payload["photon_counted"] = np.asarray(photon_counted_frames)
    payload["metadata_json"] = np.array([json.dumps(metadata or {}, indent=2, default=_json_default)], dtype=object)
    np.savez_compressed(path, **payload)
    return path
