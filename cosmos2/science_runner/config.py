"""Schema + loader helpers for cosmos2 science runner configurations."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from cosmos2.fits.joint import load_joint_config as load_joint_config_file
from cosmos2.fits.registry import FIT_REGISTRY
from cosmos2.models.pbuf.fits import PBUF_FIT_REGISTRY

from cosmos2.science_runner.utils import load_json_or_yaml
from cosmos2.science_runner.jackknife import JackknifeConfig


@dataclass(frozen=True)
class ScienceRunOutputConfig:
    base_dir: Path
    generate_plots: bool
    generate_reports: bool
    report_formats: list[str]
    save_space: bool


@dataclass(frozen=True)
class ScienceRunPredictionsConfig:
    enabled: bool
    modules: list[str]
    module_configs: dict[str, dict[str, Any]]

    def get_module_config(self, name: str) -> dict[str, Any]:
        return self.module_configs.get(name.strip().lower(), {})


@dataclass
class ScienceRunConfig:
    path: Path
    raw: dict[str, Any]

    run_name: str
    description: str | None
    models: list[str]
    mode: str
    auto_mode: str | None
    fits_config: Path | None
    fits_override: list[str]
    engine: str
    engine_settings: dict[str, Any]
    parameter_bounds_file: Path | None
    parameter_bounds_inline: dict[str, Any]
    joint_config_inline: dict[str, Any] | None
    fixed_parameters: dict[str, float]
    initial_parameters: dict[str, float]
    profile_likelihood: dict[str, Any] | None
    jackknife: JackknifeConfig | None
    predictions: ScienceRunPredictionsConfig
    output: ScienceRunOutputConfig
    interactive: bool

    joint_config_path: Path | None = field(default=None, init=False, repr=False)
    fits_list: list[str] = field(default_factory=list, init=False)
    fit_weights: dict[str, float] = field(default_factory=dict, init=False)
    _joint_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _temp_joint_path: Path | None = field(default=None, init=False, repr=False)
    _parameter_bounds_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _parameter_bounds_cache: tuple[dict[str, tuple[float, float]], dict[str, dict[str, tuple[float, float]]]] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_path(cls, path: Path | str) -> "ScienceRunConfig":
        resolved = Path(path).expanduser().resolve()
        payload = load_json_or_yaml(resolved)
        if not isinstance(payload, dict):
            raise ValueError(f"Science config at '{resolved}' must contain an object at the top level.")

        run_name = str(payload.get("run_name") or resolved.stem)
        description = payload.get("description")
        models = list(cls._normalize_list(payload.get("models") or payload.get("model") or ["lcdm"]))
        mode = str(payload.get("mode", "fit")).strip().lower()
        auto_mode_raw = payload.get("auto_mode")
        auto_mode = None
        if isinstance(auto_mode_raw, str):
            normalized = auto_mode_raw.strip().lower()
            if normalized:
                auto_mode = normalized
        joint_config_source = payload.get("joint_config")
        fits_config_path = None
        joint_config_inline: dict[str, Any] | None = None
        if isinstance(joint_config_source, dict):
            joint_config_inline = dict(joint_config_source)
        elif joint_config_source is not None:
            fits_config_path = cls._resolve_path(joint_config_source, resolved)
        else:
            fits_config_path = cls._resolve_path(payload.get("fits_config") or payload.get("joint_config_path"), resolved)
        fits_config = fits_config_path
        fits_override = cls._normalize_list(payload.get("fits_override") or [])
        if not fits_config and not fits_override and joint_config_inline is None:
            raise ValueError("Science config must define 'fits_config' or provide a non-empty 'fits_override'.")

        engine = str(payload.get("engine", "basin")).strip().lower()
        engine_settings = dict(payload.get("engine_settings") or {})
        workers = engine_settings.get("workers") or engine_settings.get("threads") or engine_settings.get("n_threads")
        if workers is not None:
            try:
                workers_int = int(workers)
                if workers_int < 1:
                    raise ValueError
                engine_settings["workers"] = workers_int
            except Exception:
                raise ValueError(f"engine_settings.workers must be a positive integer (got {workers!r})")
        parameter_bounds_file = cls._resolve_path(payload.get("parameter_bounds_file"), resolved)
        parameter_bounds_inline = cls._normalize_bounds_source(
            payload.get("parameter_bounds") or payload.get("parameter_bounds_inline") or {}
        )
        fixed_parameters = cls._coerce_params(payload.get("fixed_parameters") or {})
        initial_parameters = cls._coerce_params(payload.get("initial_parameters") or {})
        profile_likelihood = payload.get("profile_likelihood")
        jackknife_data = payload.get("jackknife")
        jackknife = JackknifeConfig.from_dict(jackknife_data) if jackknife_data else None
        predictions_payload = payload.get("predictions") or {}
        predictions = cls._parse_predictions_config(predictions_payload)
        interactive = bool(payload.get("interactive", False))

        output_payload = payload.get("output") or {}
        base_dir = cls._resolve_path(output_payload.get("base_dir") or "data/science_runs", resolved)
        generate_plots = bool(output_payload.get("generate_plots", True))
        generate_reports = bool(output_payload.get("generate_reports", True))
        report_formats = list(cls._normalize_list(output_payload.get("report_formats") or ["json", "html", "pdf"]))
        save_space = bool(output_payload.get("save_space", False))
        output_config = ScienceRunOutputConfig(
            base_dir=base_dir,
            generate_plots=generate_plots,
            generate_reports=generate_reports,
            report_formats=report_formats,
            save_space=save_space,
        )

        return cls(
            path=resolved,
            raw=payload,
            run_name=run_name,
            description=description,
            models=models,
            mode=mode,
            fits_config=fits_config,
            fits_override=fits_override,
            engine=engine,
            engine_settings=engine_settings,
            parameter_bounds_file=parameter_bounds_file,
            parameter_bounds_inline=parameter_bounds_inline,
            joint_config_inline=joint_config_inline,
            fixed_parameters=fixed_parameters,
            initial_parameters=initial_parameters,
            profile_likelihood=profile_likelihood,
            jackknife=jackknife,
            predictions=predictions,
            auto_mode=auto_mode,
            output=output_config,
            interactive=interactive,
        )

    def __post_init__(self) -> None:
        self.joint_config_path = self.fits_config
        self._joint_payload = self._load_base_joint_payload()
        self.fit_weights = self._normalize_weights(self._joint_payload.get("fit_weights"))
        base_fits = self._normalize_list(self._joint_payload.get("fits"))
        overrides = self._normalize_list(self.fits_override)
        self.fits_list = overrides if overrides else base_fits
        if not self.fits_list:
            raise ValueError("Science config must define 'fits_config' or provide a non-empty 'fits_override'.")
        self._flush_joint_payload()
        self._validate_fits()

    @staticmethod
    def _resolve_path(value: str | None, reference: Path) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.expanduser().resolve()
        # For data/science_runs, resolve relative to project root, not config file location
        if str(candidate).startswith("data/science_runs"):
            return candidate.expanduser().resolve()
        return (reference.parent / candidate).expanduser().resolve()

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        normalized: list[str] = []
        if value is None:
            return normalized
        if isinstance(value, str):
            entry = value.strip().lower()
            if entry:
                normalized.append(entry)
            return normalized
        if isinstance(value, Sequence):
            for entry in value:
                if not isinstance(entry, str):
                    continue
                candidate = entry.strip().lower()
                if candidate and candidate not in normalized:
                    normalized.append(candidate)
        return normalized

    @staticmethod
    def _normalize_bounds_source(payload: Any) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        if not isinstance(payload, Mapping):
            return normalized
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            normalized[key] = value
        return normalized

    @staticmethod
    def _parse_predictions_config(payload: Any) -> ScienceRunPredictionsConfig:
        if not isinstance(payload, Mapping):
            payload = {}
        enabled = bool(payload.get("enabled", False))
        modules = list(ScienceRunConfig._normalize_list(payload.get("modules") or []))
        module_configs: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            normalized = key.strip().lower()
            if normalized in {"enabled", "modules"} or not normalized:
                continue
            if isinstance(value, Mapping):
                module_configs[normalized] = dict(value)
            else:
                module_configs[normalized] = {"value": value}
        return ScienceRunPredictionsConfig(
            enabled=enabled,
            modules=modules,
            module_configs=module_configs,
        )

    @staticmethod
    def _normalize_interval(value: Any) -> tuple[float, float] | None:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            return None
        try:
            lower = float(value[0])
            upper = float(value[1])
        except (TypeError, ValueError):
            return None
        return (lower, upper)

    @staticmethod
    def _normalize_parameter_map(payload: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        normalized: dict[str, tuple[float, float]] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            interval = ScienceRunConfig._normalize_interval(value)
            if interval is None:
                continue
            normalized[key] = interval
        return normalized

    @staticmethod
    def _split_parameter_bounds(
        payload: Mapping[str, Any]
    ) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, tuple[float, float]]]]:
        global_bounds: dict[str, tuple[float, float]] = {}
        per_model_bounds: dict[str, dict[str, tuple[float, float]]] = {}

        def _ingest_model_bounds(name: str, source: Mapping[str, Any]) -> None:
            normalized = ScienceRunConfig._normalize_parameter_map(source)
            if normalized:
                per_model_bounds[name.strip().lower()] = normalized

        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            lower_key = key.strip().lower()
            if lower_key == "global" and isinstance(value, Mapping):
                normalized = ScienceRunConfig._normalize_parameter_map(value)
                if normalized:
                    global_bounds.update(normalized)
                continue
            if lower_key == "models" and isinstance(value, Mapping):
                for model_key, model_value in value.items():
                    if isinstance(model_key, str) and isinstance(model_value, Mapping):
                        _ingest_model_bounds(model_key, model_value)
                continue

            if isinstance(value, Mapping):
                _ingest_model_bounds(key, value)
                continue
            interval = ScienceRunConfig._normalize_interval(value)
            if interval is not None:
                global_bounds[key.strip()] = interval
        return global_bounds, per_model_bounds

    def _load_parameter_bounds_payload(self) -> dict[str, Any]:
        if self._parameter_bounds_payload is None:
            combined: dict[str, Any] = {}
            if self.parameter_bounds_file:
                payload = load_json_or_yaml(self.parameter_bounds_file)
                if isinstance(payload, Mapping):
                    source = payload.get("parameters") or payload
                    if isinstance(source, Mapping):
                        for key, value in source.items():
                            if isinstance(key, str):
                                combined[key] = value
            for key, value in self.parameter_bounds_inline.items():
                combined[key] = value
            self._parameter_bounds_payload = combined
        return self._parameter_bounds_payload

    def _ensure_parameter_bounds(
        self,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, tuple[float, float]]]]:
        if self._parameter_bounds_cache is None:
            payload = self._load_parameter_bounds_payload()
            self._parameter_bounds_cache = self._split_parameter_bounds(payload)
        return self._parameter_bounds_cache

    @property
    def parameter_bounds_payload(self) -> dict[str, Any]:
        global_bounds, per_model_bounds = self._ensure_parameter_bounds()
        structured: dict[str, Any] = {
            "global": {
                key: [interval[0], interval[1]]
                for key, interval in sorted(global_bounds.items())
            },
            "models": {
                model: {
                    param: [bounds[0], bounds[1]]
                    for param, bounds in sorted(bounds_dict.items())
                }
                for model, bounds_dict in sorted(per_model_bounds.items())
            },
        }
        return structured

    def parameter_bounds_for_model(self, model_name: str) -> dict[str, tuple[float, float]]:
        global_bounds, per_model_bounds = self._ensure_parameter_bounds()
        result = dict(global_bounds)
        candidate = per_model_bounds.get(model_name.strip().lower())
        if candidate:
            result.update(candidate)
        return result

    @staticmethod
    def _coerce_params(payload: Mapping[str, Any]) -> dict[str, float]:
        coalesced: dict[str, float] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            try:
                coalesced[key] = float(value)
            except (TypeError, ValueError):
                continue
        return coalesced

    @staticmethod
    def _normalize_weights(payload: Any) -> dict[str, float]:
        normalized: dict[str, float] = {}
        if not isinstance(payload, Mapping):
            return normalized
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            candidate = key.strip().lower()
            if not candidate:
                continue
            try:
                normalized[candidate] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    @property
    def fits_used(self) -> list[str]:
        return list(self.fits_list)

    @property
    def jackknife_enabled(self) -> bool:
        return bool(self.jackknife and self.jackknife.enabled)

    @property
    def joint_config_payload(self) -> dict[str, Any]:
        self._flush_joint_payload()
        return dict(self._joint_payload or {})

    def _load_base_joint_payload(self) -> dict[str, Any]:
        if self.joint_config_inline:
            return dict(self.joint_config_inline)
        if not self.joint_config_path:
            return {}
        return load_joint_config_file(self.joint_config_path)

    def _flush_joint_payload(self) -> None:
        payload = dict(self._joint_payload) if self._joint_payload is not None else {}
        payload["fits"] = list(self.fits_list)
        if self.fit_weights:
            payload["fit_weights"] = dict(self.fit_weights)
        else:
            payload.pop("fit_weights", None)
        self._joint_payload = payload

    def _validate_fits(self) -> None:
        allowed = set(FIT_REGISTRY) | set(PBUF_FIT_REGISTRY)
        for fit_name in self.fits_list:
            if fit_name not in allowed:
                raise ValueError(f"Unknown fit '{fit_name}' in joint config.")

    def set_fits(self, fits: Sequence[str]) -> None:
        normalized = [entry for entry in self._normalize_list(fits) if entry]
        if not normalized:
            raise ValueError("At least one fit must be specified.")
        self.fits_list = normalized
        self._flush_joint_payload()
        self._validate_fits()

    def set_models(self, models: Sequence[str]) -> None:
        normalized = [entry for entry in self._normalize_list(models) if entry]
        if not normalized:
            raise ValueError("At least one model must be specified.")
        self.models = normalized

    def get_joint_config_path(self) -> Path:
        payload = self.joint_config_payload
        if not payload.get("fits"):
            raise ValueError("Joint config does not define any fits.")
        payload_path = self._temp_joint_path
        if payload_path is None:
            handle = tempfile.NamedTemporaryFile(prefix="science_joint_", suffix=".json", delete=False)
            handle.write(json.dumps(payload, indent=2).encode("utf-8"))
            handle.close()
            payload_path = Path(handle.name)
            self._temp_joint_path = payload_path
        else:
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload_path

    def cleanup(self) -> None:
        if self._temp_joint_path and self._temp_joint_path.exists():
            try:
                self._temp_joint_path.unlink()
            except OSError:
                pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "description": self.description,
            "models": self.models,
            "mode": self.mode,
            "auto_mode": self.auto_mode,
            "fits_config": str(self.fits_config) if self.fits_config else None,
            "joint_config": dict(self.joint_config_inline) if self.joint_config_inline is not None else None,
            "joint_config_path": str(self.joint_config_path) if self.joint_config_path else None,
            "fits_list": list(self.fits_list),
            "fit_weights": dict(self.fit_weights),
            "fits_override": self.fits_override,
            "engine": self.engine,
            "engine_settings": self.engine_settings,
            "parameter_bounds_file": str(self.parameter_bounds_file) if self.parameter_bounds_file else None,
            "parameter_bounds_inline": self.parameter_bounds_inline,
            "fixed_parameters": self.fixed_parameters,
            "initial_parameters": self.initial_parameters,
            "profile_likelihood": self.profile_likelihood,
            "jackknife": self.jackknife.to_dict() if self.jackknife else None,
            "predictions": {
                "enabled": self.predictions.enabled,
                "modules": list(self.predictions.modules),
                **{
                    key: dict(value)
                    for key, value in self.predictions.module_configs.items()
                },
            },
            "output": {
                "base_dir": str(self.output.base_dir),
                "generate_plots": self.output.generate_plots,
                "generate_reports": self.output.generate_reports,
                "report_formats": self.output.report_formats,
                "save_space": self.output.save_space,
            },
            "interactive": self.interactive,
        }


__all__ = ["ScienceRunConfig", "ScienceRunOutputConfig", "ScienceRunPredictionsConfig"]
