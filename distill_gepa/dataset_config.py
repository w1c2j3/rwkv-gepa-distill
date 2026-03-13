from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class HuggingFaceSourceConfig:
    repo_id: str
    config_name: str | None
    revision: str | None


@dataclass(frozen=True)
class LocalJsonlSourceConfig:
    path: str


@dataclass(frozen=True)
class DatasetSourceConfig:
    name: str
    provider: str
    adapter: str
    output_file: str
    enabled: bool
    merge_into_world: bool
    huggingface: HuggingFaceSourceConfig | None
    local_jsonl: LocalJsonlSourceConfig | None


@dataclass(frozen=True)
class DatasetFolderConfig:
    dataset_name: str
    config_path: Path
    root_dir: Path
    merge_output_file: str
    sources: tuple[DatasetSourceConfig, ...]

    @property
    def merge_output_path(self) -> Path:
        return self.root_dir / self.merge_output_file


def _require_str(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing non-empty string field {key!r}")
    return value.strip()


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Field {key!r} must be a string when provided")
    stripped = value.strip()
    return stripped or None


def _parse_source(payload: dict[str, Any], *, config_path: Path, index: int) -> DatasetSourceConfig:
    context = f"{config_path}:sources[{index}]"
    provider = _require_str(payload, "provider", context=context)
    adapter = _require_str(payload, "adapter", context=context)
    name = _require_str(payload, "name", context=context)
    output_file = _optional_str(payload, "output_file") or f"{name}_tasks.jsonl"
    enabled = payload.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{context}: 'enabled' must be boolean")
    merge_into_world = payload.get("merge_into_world", True)
    if not isinstance(merge_into_world, bool):
        raise ValueError(f"{context}: 'merge_into_world' must be boolean")

    huggingface = None
    if provider == "huggingface":
        hf_payload = payload.get("huggingface")
        if not isinstance(hf_payload, dict):
            raise ValueError(f"{context}: huggingface provider requires a [sources.huggingface] table")
        huggingface = HuggingFaceSourceConfig(
            repo_id=_require_str(hf_payload, "repo_id", context=f"{context}.huggingface"),
            config_name=_optional_str(hf_payload, "config_name"),
            revision=_optional_str(hf_payload, "revision"),
        )

    local_jsonl = None
    if provider == "local_jsonl":
        local_payload = payload.get("local_jsonl")
        if not isinstance(local_payload, dict):
            raise ValueError(f"{context}: local_jsonl provider requires a [sources.local_jsonl] table")
        local_jsonl = LocalJsonlSourceConfig(
            path=_require_str(local_payload, "path", context=f"{context}.local_jsonl"),
        )

    return DatasetSourceConfig(
        name=name,
        provider=provider,
        adapter=adapter,
        output_file=output_file,
        enabled=enabled,
        merge_into_world=merge_into_world,
        huggingface=huggingface,
        local_jsonl=local_jsonl,
    )


def load_dataset_config(config_path: Path, data_root: Path | None = None) -> DatasetFolderConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a TOML table")

    dataset_name = _optional_str(payload, "name") or config_path.stem
    merge_output_file = _optional_str(payload, "merge_output_file") or "tasks.jsonl"
    dataset_root = data_root or (Path("data") / dataset_name)
    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, list) or not sources_payload:
        raise ValueError(f"{config_path}: expected at least one [[sources]] table")

    sources = tuple(
        _parse_source(source_payload, config_path=config_path, index=index)
        for index, source_payload in enumerate(sources_payload)
        if isinstance(source_payload, dict)
    )
    if len(sources) != len(sources_payload):
        raise ValueError(f"{config_path}: each [[sources]] entry must be a TOML table")

    return DatasetFolderConfig(
        dataset_name=dataset_name,
        config_path=config_path,
        root_dir=dataset_root,
        merge_output_file=merge_output_file,
        sources=sources,
    )


def load_dataset_folder_config(dataset_root: Path) -> DatasetFolderConfig:
    return load_dataset_config(dataset_root / "dataset.toml", dataset_root)
