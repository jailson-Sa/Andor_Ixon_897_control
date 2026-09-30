"""Session and run organization for scientific camera acquisition."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_slug(text: str, fallback: str = "session") -> str:
    cleaned = _SAFE.sub("_", text.strip()).strip("_")
    return cleaned or fallback


@dataclass(slots=True)
class ExperimentSession:
    root: Path
    experiment_name: str
    sample_name: str = ""
    operator: str = ""
    notes: str = ""
    created_unix_s: float = 0.0

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        *,
        experiment_name: str,
        sample_name: str = "",
        operator: str = "",
        notes: str = "",
    ) -> "ExperimentSession":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = safe_slug(experiment_name, "andor_session")
        root = Path(base_dir) / f"{stamp}_{name}"
        session = cls(
            root=root,
            experiment_name=experiment_name,
            sample_name=sample_name,
            operator=operator,
            notes=notes,
            created_unix_s=time.time(),
        )
        session.ensure_directories()
        session.write_metadata()
        session.log("Session created")
        return session

    def ensure_directories(self) -> None:
        for rel in ["raw", "processed", "photon_counted", "calibration", "metadata", "logs", "exports"]:
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        data["created_local_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_unix_s or time.time()))
        return data

    def write_metadata(self) -> None:
        self.ensure_directories()
        path = self.root / "metadata" / "session_metadata.json"
        path.write_text(json.dumps(self.to_metadata(), indent=2), encoding="utf-8")

    def log(self, message: str) -> None:
        self.ensure_directories()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with (self.root / "logs" / "session.log").open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")

    def next_path(self, subdir: str, stem: str, suffix: str) -> Path:
        self.ensure_directories()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return self.root / subdir / f"{stamp}_{safe_slug(stem)}{suffix}"
