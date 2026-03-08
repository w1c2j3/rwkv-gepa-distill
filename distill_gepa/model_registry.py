from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .teacher_client import TeacherClient, TeacherConfig, api_base_looks_local, parse_api_protocol


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model config is missing non-empty string field {key!r}")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Model config field {key!r} must be a string when provided")
    stripped = value.strip()
    return stripped or None


def _resolve_api_key(payload: dict[str, Any], api_base: str | None) -> str | None:
    direct = _optional_string(payload, "api_key")
    if direct:
        return direct

    env_name = _optional_string(payload, "api_key_env")
    if env_name:
        import os

        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    if api_base_looks_local(api_base):
        return "local"
    return None


def _resolve_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Model config field {key!r} must be an integer")
    if value < 0:
        raise ValueError(f"Model config field {key!r} must be non-negative")
    return value


def _resolve_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Model config field {key!r} must be numeric")
    if value < 0:
        raise ValueError(f"Model config field {key!r} must be non-negative")
    return float(value)


def _resolve_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Model config field {key!r} must be a boolean")
    return value


@dataclass(frozen=True)
class ModelEndpointConfig:
    name: str
    model: str
    api_base: str | None
    api_key: str | None
    api_protocol: str
    timeout_seconds: float
    num_retries: int
    prefer_stream: bool
    max_tokens: int | None
    force_mock: bool

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        defaults: dict[str, Any] | None = None,
    ) -> "ModelEndpointConfig":
        defaults = defaults or {}
        merged = {**defaults, **payload}
        model_name = _require_string(merged, "model")
        api_base = _optional_string(merged, "base_url") or _optional_string(merged, "api_base")
        return cls(
            name=_require_string(merged, "name"),
            model=model_name,
            api_base=api_base,
            api_key=_resolve_api_key(merged, api_base),
            api_protocol=parse_api_protocol(_optional_string(merged, "api_protocol"), model_name=model_name),
            timeout_seconds=_resolve_float(merged, "timeout_seconds", 30.0),
            num_retries=_resolve_int(merged, "num_retries", 0),
            prefer_stream=_resolve_bool(merged, "prefer_stream", False),
            max_tokens=_resolve_int(merged, "max_tokens", 256),
            force_mock=_resolve_bool(merged, "mock", False),
        )

    def to_teacher_client(self) -> TeacherClient:
        return TeacherClient(
            TeacherConfig(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
                api_protocol=self.api_protocol,
                timeout_seconds=self.timeout_seconds,
                num_retries=self.num_retries,
                prefer_stream=self.prefer_stream,
                max_tokens=self.max_tokens,
                force_mock=self.force_mock,
            )
        )


@dataclass(frozen=True)
class PipelineModelConfig:
    base_models: tuple[ModelEndpointConfig, ...]
    gepa_reflection_model: ModelEndpointConfig
    rewrite_model: ModelEndpointConfig

    def base_model(self, name: str) -> ModelEndpointConfig:
        for item in self.base_models:
            if item.name == name:
                return item
        raise KeyError(f"Unknown base model {name!r}")


def load_pipeline_model_config(path: Path) -> PipelineModelConfig:
    load_dotenv()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pipeline config: {path}. Create it from config/world_pipeline.example.yaml."
        )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")

    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: 'defaults' must be an object")

    base_models = payload.get("base_models")
    if not isinstance(base_models, list) or not base_models:
        raise ValueError(f"{path}: 'base_models' must be a non-empty list")

    parsed_base_models = tuple(
        ModelEndpointConfig.from_dict(item, defaults=defaults) for item in base_models if isinstance(item, dict)
    )
    if len(parsed_base_models) != len(base_models):
        raise ValueError(f"{path}: every item in 'base_models' must be an object")

    gepa_payload = payload.get("gepa_reflection_model")
    if not isinstance(gepa_payload, dict):
        raise ValueError(f"{path}: 'gepa_reflection_model' must be an object")

    rewrite_payload = payload.get("rewrite_model")
    if not isinstance(rewrite_payload, dict):
        raise ValueError(f"{path}: 'rewrite_model' must be an object")

    return PipelineModelConfig(
        base_models=parsed_base_models,
        gepa_reflection_model=ModelEndpointConfig.from_dict(gepa_payload, defaults=defaults),
        rewrite_model=ModelEndpointConfig.from_dict(rewrite_payload, defaults=defaults),
    )
