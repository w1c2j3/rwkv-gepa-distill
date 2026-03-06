from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

from .teacher_client import (
    TeacherClient,
    collect_stream_content,
    env_first,
    litellm_provider_kwargs,
    normalize_api_base,
)


def resolve_reflection_runtime(teacher: TeacherClient) -> tuple[str | None, str | None, str | None]:
    load_dotenv()

    model_name = env_first("GEPA_REFLECTION_MODEL", "GEPA_MODEL")
    if not model_name and teacher.mode == "api":
        model_name = teacher.config.model

    api_key = env_first("GEPA_API_KEY", "TEACHER_API_KEY")
    api_base = normalize_api_base(
        env_first("GEPA_API_BASE", "GEPA_BASE_URL", "TEACHER_API_BASE", "TEACHER_BASE_URL")
    )
    return model_name, api_key, api_base


def make_streaming_litellm_lm(
    *,
    model_name: str,
    api_key: str | None,
    api_base: str | None,
    timeout_seconds: float,
    num_retries: int,
) -> Any:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError("litellm is not installed. Run `bash scripts/bootstrap.sh` first.") from exc

    def lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        stream = completion(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=0,
            timeout=timeout_seconds,
            stream=True,
            num_retries=num_retries,
            **litellm_provider_kwargs(model_name, api_base),
            messages=messages,
        )

        return collect_stream_content(stream)

    return lm
