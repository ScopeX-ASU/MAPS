"""Versioned local artifact management for MAPS optimization runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
import yaml

from core.env_args import ENV_ARGS, get_env_arg

RUN_SCHEMA = "maps.optimization-run/v1"
_DIRECTORIES = {
    "checkpoints": "checkpoints",
    "config": "config",
    "design": "design",
    "geometry": "geometry",
    "layout": "layout",
    "logs": "logs",
    "metadata": "metadata",
    "metrics": "metrics",
    "plots": "plots",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    """Convert common scientific/configuration objects to portable metadata."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", repr(value))
        return f"{module}.{name}".lstrip(".")
    return repr(value)


def _validate_segment(value: str, field: str) -> str:
    segment = str(value).strip()
    if not segment or segment in {".", ".."} or Path(segment).name != segment:
        raise ValueError(f"{field} must be one non-empty path segment, got {value!r}")
    return segment


class OptimizationRunManager:
    """Own the filesystem contract and provenance for one optimization run.

    Args:
        root: Root directory containing experiment folders.
        experiment: Stable experiment/category name.
        run_id: Optional exact run identifier. A UTC timestamp, optional tag,
            and random suffix are generated when omitted.
        tag: Optional human-readable label included in generated run IDs and
            stored separately in the manifest.
        resume: Reopen an existing run directory instead of requiring a new one.
        resume_from: Checkpoint inside an existing managed run. When supplied
            during resume, its parent run determines ``run_id``.

    Attributes:
        run_dir: Absolute path to the managed run directory.
        run_id: Run identifier stored in the manifest.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        experiment: str,
        run_id: str | None = None,
        *,
        tag: str | None = None,
        resume: bool = False,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.experiment = _validate_segment(experiment, "experiment")
        self.tag = None if tag is None else _validate_segment(tag, "tag")
        if resume_from is not None:
            if not resume:
                raise ValueError("resume_from requires resume=True")
            checkpoint = Path(resume_from).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint does not exist: {checkpoint}"
                )
            if checkpoint.parent.name != "checkpoints":
                raise ValueError(
                    "Managed resume checkpoint must be inside a checkpoints directory"
                )
            inferred_run_dir = checkpoint.parent.parent
            expected_experiment_dir = self.root / self.experiment
            if inferred_run_dir.parent != expected_experiment_dir:
                raise ValueError(
                    f"Checkpoint run {inferred_run_dir} is outside configured "
                    f"experiment directory {expected_experiment_dir}"
                )
            inferred_run_id = inferred_run_dir.name
            if run_id is not None and run_id != inferred_run_id:
                raise ValueError(
                    f"Configured run_id {run_id!r} does not match checkpoint run "
                    f"{inferred_run_id!r}"
                )
            run_id = inferred_run_id
        if resume and run_id is None:
            raise ValueError("resume requires run_id or resume_from")
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            tag_part = "" if self.tag is None else f"_{self.tag}"
            run_id = f"{timestamp}{tag_part}_{uuid.uuid4().hex[:8]}"
        self.run_id = _validate_segment(run_id, "run_id")
        self.run_dir = self.root / self.experiment / self.run_id
        self.manifest_path = self.run_dir / "manifest.yaml"

        if self.manifest_path.exists() and not resume:
            raise FileExistsError(
                f"Run already exists: {self.run_dir}. Set resume=True to reopen it."
            )
        if resume and not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume run without manifest: {self.manifest_path}"
            )

        for directory in _DIRECTORIES.values():
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)

        if resume:
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                self.manifest = yaml.safe_load(stream) or {}
            if self.manifest.get("schema_version") != RUN_SCHEMA:
                raise ValueError(
                    f"Unsupported run schema: {self.manifest.get('schema_version')!r}"
                )
            manifest_tag = self.manifest.get("tag")
            if self.tag is not None and self.tag != manifest_tag:
                raise ValueError(
                    f"Configured tag {self.tag!r} does not match resumed run tag "
                    f"{manifest_tag!r}"
                )
            self.tag = manifest_tag
            self.manifest["status"] = "running"
            self.manifest["resumed_at"] = _utc_now()
        else:
            self.manifest = {
                "schema_version": RUN_SCHEMA,
                "run_id": self.run_id,
                "experiment": self.experiment,
                "tag": self.tag,
                "status": "running",
                "created_at": _utc_now(),
                "artifacts": {},
            }
        self._write_yaml_atomic(self.manifest_path, self.manifest)
        self._write_environment_metadata()
        self._log_handler: logging.Handler | None = None
        self._attached_logger: logging.Logger | None = None
        self.log("run initialized" if not resume else "run resumed")

    def path(self, category: str, filename: str | os.PathLike[str]) -> str:
        """Return an absolute path inside a managed artifact category."""
        if category not in _DIRECTORIES:
            raise KeyError(f"Unknown run artifact category: {category!r}")
        filename_path = Path(filename)
        if filename_path.is_absolute() or ".." in filename_path.parts:
            raise ValueError("Managed artifact filenames must be relative to the run")
        destination = self.run_dir / _DIRECTORIES[category] / filename_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        return str(destination)

    def bind_simulation_config(self, sim_cfg: Any) -> str:
        """Route simulation-generated plots into this managed run.

        Args:
            sim_cfg: Mutable simulation configuration exposing ``plot_root``
                either as a mapping key or an attribute.

        Returns:
            Absolute managed plots directory assigned to ``sim_cfg``.

        Raises:
            TypeError: If ``sim_cfg`` cannot accept a ``plot_root`` value.
        """
        plot_root = self.path("plots", "")
        if isinstance(sim_cfg, MutableMapping):
            sim_cfg["plot_root"] = plot_root
        elif hasattr(sim_cfg, "plot_root"):
            sim_cfg.plot_root = plot_root
        else:
            raise TypeError(
                "sim_cfg must be a mutable mapping or expose a plot_root attribute"
            )
        return plot_root

    def write_resolved_config(self, config: Mapping[str, Any]) -> str:
        """Write the effective run configuration as portable YAML."""
        path = Path(self.path("config", "resolved.yaml"))
        self._write_yaml_atomic(path, _plain(config))
        self.register_artifact("resolved_config", path, checksum=True)
        return str(path)

    def append_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Append one structured metric event to ``history.jsonl``."""
        event = {"timestamp": _utc_now(), **_plain(metrics)}
        path = Path(self.path("metrics", "history.jsonl"))
        line = json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def log(self, message: str) -> None:
        """Append a timestamped human-readable run log entry."""
        path = Path(self.path("logs", "run.log"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{_utc_now()} {message.rstrip()}\n")

    def attach_logger(self, target_logger: logging.Logger) -> logging.Handler:
        """Mirror a Python logger into this run's human-readable log.

        Args:
            target_logger: Logger that already emits MAPS messages to the
                terminal.

        Returns:
            The run-scoped file handler. Repeated calls for this manager return
            the existing handler instead of duplicating messages.
        """
        if self._log_handler is not None:
            return self._log_handler
        path = self.path("logs", "run.log")
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        target_logger.addHandler(handler)
        self._log_handler = handler
        self._attached_logger = target_logger
        return handler

    def detach_logger(self) -> None:
        """Flush, close, and remove this run's logger handler."""
        if self._log_handler is None or self._attached_logger is None:
            return
        self._log_handler.flush()
        self._attached_logger.removeHandler(self._log_handler)
        self._log_handler.close()
        self._log_handler = None
        self._attached_logger = None

    def register_artifact(
        self,
        name: str,
        path: str | os.PathLike[str],
        *,
        checksum: bool = False,
    ) -> None:
        """Register an existing artifact and optional SHA-256 digest."""
        artifact = Path(path).resolve()
        try:
            relative_path = artifact.relative_to(self.run_dir)
        except ValueError as error:
            raise ValueError(f"Artifact is outside managed run: {artifact}") from error
        record: dict[str, Any] = {
            "path": str(relative_path),
            "size_bytes": artifact.stat().st_size,
            "updated_at": _utc_now(),
        }
        if checksum:
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        self.manifest.setdefault("artifacts", {})[name] = record
        self._write_yaml_atomic(self.manifest_path, self.manifest)

    def finish(self, status: str, **summary: Any) -> None:
        """Finalize the manifest and write the machine-readable run summary."""
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError(f"Invalid terminal run status: {status!r}")
        finished_at = _utc_now()
        summary_data = {"status": status, "finished_at": finished_at, **_plain(summary)}
        summary_path = Path(self.path("metrics", "summary.json"))
        self._write_json_atomic(summary_path, summary_data)
        self.manifest["status"] = status
        self.manifest["finished_at"] = finished_at
        self.manifest["summary"] = str(summary_path.relative_to(self.run_dir))
        self._write_yaml_atomic(self.manifest_path, self.manifest)
        self.log(f"run {status}")
        self.detach_logger()

    def _write_environment_metadata(self) -> None:
        maps_env_args = {}
        for name, spec in ENV_ARGS.items():
            explicitly_set = name in os.environ
            raw_value = os.environ.get(name, spec.default)
            maps_env_args[name] = {
                "source": "environment" if explicitly_set else "default",
                "raw": raw_value,
                "backend": get_env_arg(name, context="backend"),
                "device": get_env_arg(name, context="device"),
            }
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "maps_env_args": maps_env_args,
        }
        self._write_json_atomic(
            Path(self.path("metadata", "environment.json")), environment
        )
        source = {"commit": None, "dirty": None}
        try:
            source["commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            source["dirty"] = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self._write_json_atomic(Path(self.path("metadata", "source.json")), source)

    @staticmethod
    def _write_yaml_atomic(path: Path, value: Any) -> None:
        OptimizationRunManager._write_text_atomic(
            path, yaml.safe_dump(value, sort_keys=False)
        )

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        OptimizationRunManager._write_text_atomic(
            path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
