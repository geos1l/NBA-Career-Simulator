from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


@dataclass
class LoadedModel:
    version: str
    artifact: Dict[str, Any]
    metadata: Dict[str, Any]


class ModelStore:
    def __init__(self, root: str = "artifacts/models/simulator") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_pointer = self.root / "latest.json"

    def _version_dir(self, version: str) -> Path:
        return self.root / version

    def save(self, artifact: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        version = metadata.get("version") or datetime.now(tz=timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.joblib"
        meta_path = version_dir / "metadata.json"

        joblib.dump(artifact, model_path)
        metadata = {**metadata, "version": version, "model_path": str(model_path)}
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.latest_pointer.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
        return metadata

    def load(self, version: Optional[str] = None) -> LoadedModel:
        selected_version = version
        if selected_version is None:
            if not self.latest_pointer.exists():
                raise FileNotFoundError("No trained model found. Train a model first.")
            latest = json.loads(self.latest_pointer.read_text(encoding="utf-8"))
            selected_version = latest["version"]

        version_dir = self._version_dir(selected_version)
        model_path = version_dir / "model.joblib"
        meta_path = version_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Model version '{selected_version}' is incomplete.")

        artifact = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return LoadedModel(version=selected_version, artifact=artifact, metadata=metadata)

