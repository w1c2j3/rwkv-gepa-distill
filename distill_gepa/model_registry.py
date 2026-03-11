from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .teacher_client import (
    TeacherClient,
    TeacherConfig,
    api_base_looks_local,
    parse_api_protocol,
    parse_bool_env,
    parse_float_env,
    parse_int_env,
)


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


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_api_key(payload: dict[str, Any], api_base: str | None) -> str | None:
    direct = _optional_string(payload, "api_key")
    if direct:
        return direct

    env_name = _optional_string(payload, "api_key_env")
    if env_name:
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


def _parse_env_bool(raw_value: str | None) -> bool | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment boolean value {raw_value!r} is invalid")


def _parse_env_int(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment integer value {raw_value!r} is invalid") from exc


def _parse_env_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment float value {raw_value!r} is invalid") from exc


def _env_with_shared(prefix: str, key: str, *, shared_prefix: str | None = None) -> str | None:
    direct = _optional_env(f"{prefix}_{key}")
    if direct is not None:
        return direct
    if shared_prefix is not None:
        return _optional_env(f"{shared_prefix}_SHARED_{key}")
    return None


def _model_payload_from_env_prefix(
    env_prefix: str,
    *,
    shared_prefix: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    name = _env_with_shared(env_prefix, "NAME", shared_prefix=shared_prefix) or _env_with_shared(
        env_prefix, "MODEL", shared_prefix=shared_prefix
    )
    model = _env_with_shared(env_prefix, "MODEL", shared_prefix=shared_prefix) or name
    if name is not None:
        payload["name"] = name
    if model is not None:
        payload["model"] = model

    base_url = _env_with_shared(env_prefix, "BASE_URL", shared_prefix=shared_prefix) or _env_with_shared(
        env_prefix, "API_BASE", shared_prefix=shared_prefix
    )
    if base_url is not None:
        payload["base_url"] = base_url

    api_key = _env_with_shared(env_prefix, "API_KEY", shared_prefix=shared_prefix)
    if api_key is not None:
        payload["api_key"] = api_key

    api_protocol = _env_with_shared(env_prefix, "API_PROTOCOL", shared_prefix=shared_prefix)
    if api_protocol is not None:
        payload["api_protocol"] = api_protocol

    timeout_seconds = _parse_env_float(_env_with_shared(env_prefix, "TIMEOUT_SECONDS", shared_prefix=shared_prefix))
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds

    num_retries = _parse_env_int(_env_with_shared(env_prefix, "NUM_RETRIES", shared_prefix=shared_prefix))
    if num_retries is not None:
        payload["num_retries"] = num_retries

    max_tokens = _parse_env_int(_env_with_shared(env_prefix, "MAX_TOKENS", shared_prefix=shared_prefix))
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    prefer_stream = _parse_env_bool(_env_with_shared(env_prefix, "PREFER_STREAM", shared_prefix=shared_prefix))
    if prefer_stream is not None:
        payload["prefer_stream"] = prefer_stream

    force_mock = _parse_env_bool(_env_with_shared(env_prefix, "MOCK", shared_prefix=shared_prefix))
    if force_mock is not None:
        payload["mock"] = force_mock

    return payload


def _endpoint_from_env_spec(payload: dict[str, Any], *, defaults: dict[str, Any]) -> "ModelEndpointConfig":
    env_prefix = _optional_string(payload, "from_env_prefix") or _optional_string(payload, "env_prefix")
    if env_prefix is None:
        raise ValueError("Endpoint env spec must contain 'from_env_prefix'")
    model_payload = _model_payload_from_env_prefix(env_prefix)
    if not model_payload:
        raise ValueError(f"No environment values found for endpoint prefix {env_prefix!r}")
    return ModelEndpointConfig.from_dict(model_payload, defaults=defaults)


def _base_models_from_env_spec(payload: dict[str, Any], *, defaults: dict[str, Any]) -> tuple["ModelEndpointConfig", ...]:
    collection_prefix = _optional_string(payload, "from_env_collection") or _optional_string(payload, "from_env_prefix")
    if collection_prefix is None:
        raise ValueError("Base-model env spec must contain 'from_env_collection'")

    ids_raw = _optional_env(f"{collection_prefix}_IDS") or _optional_env(f"{collection_prefix}_INDEXES")
    if ids_raw is None:
        raise ValueError(
            f"Missing environment variable {collection_prefix}_IDS for answer-model configuration"
        )
    identifiers = [item.strip() for item in ids_raw.split(",") if item.strip()]
    if not identifiers:
        raise ValueError(f"Environment variable {collection_prefix}_IDS must contain at least one identifier")

    endpoints: list[ModelEndpointConfig] = []
    for identifier in identifiers:
        env_prefix = f"{collection_prefix}_{identifier}"
        model_payload = _model_payload_from_env_prefix(env_prefix, shared_prefix=collection_prefix)
        if not model_payload:
            raise ValueError(f"No environment values found for answer-model prefix {env_prefix!r}")
        endpoints.append(ModelEndpointConfig.from_dict(model_payload, defaults=defaults))
    return tuple(endpoints)


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
    variant_generator_model: ModelEndpointConfig
    prompt_optimizer_model: ModelEndpointConfig

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
    defaults = dict(defaults)
    defaults["timeout_seconds"] = parse_float_env(
        "OPENAI_TIMEOUT",
        float(defaults.get("timeout_seconds", 30.0)),
    )
    defaults["num_retries"] = parse_int_env(
        "OPENAI_MAX_RETRIES",
        int(defaults.get("num_retries", 0)),
    )
    defaults["max_tokens"] = parse_int_env(
        "OPENAI_MAX_TOKENS",
        int(defaults.get("max_tokens", 256)),
    )
    defaults["prefer_stream"] = parse_bool_env(
        "OPENAI_PREFER_STREAM",
        bool(defaults.get("prefer_stream", False)),
    )

    base_models = payload.get("base_models")
    if isinstance(base_models, dict):
        parsed_base_models = _base_models_from_env_spec(base_models, defaults=defaults)
    elif isinstance(base_models, list) and base_models:
        parsed_base_models = tuple(
            ModelEndpointConfig.from_dict(item, defaults=defaults) for item in base_models if isinstance(item, dict)
        )
        if len(parsed_base_models) != len(base_models):
            raise ValueError(f"{path}: every item in 'base_models' must be an object")
    else:
        raise ValueError(f"{path}: 'base_models' must be a non-empty list or an env-backed object")

    generator_payload = payload.get("variant_generator_model")
    if not isinstance(generator_payload, dict):
        raise ValueError(f"{path}: 'variant_generator_model' must be an object")

    optimizer_payload = payload.get("prompt_optimizer_model")
    if not isinstance(optimizer_payload, dict):
        raise ValueError(f"{path}: 'prompt_optimizer_model' must be an object")

    return PipelineModelConfig(
        base_models=parsed_base_models,
        variant_generator_model=(
            _endpoint_from_env_spec(generator_payload, defaults=defaults)
            if ("from_env_prefix" in generator_payload or "env_prefix" in generator_payload)
            else ModelEndpointConfig.from_dict(generator_payload, defaults=defaults)
        ),
        prompt_optimizer_model=(
            _endpoint_from_env_spec(optimizer_payload, defaults=defaults)
            if ("from_env_prefix" in optimizer_payload or "env_prefix" in optimizer_payload)
            else ModelEndpointConfig.from_dict(optimizer_payload, defaults=defaults)
        ),
    )
