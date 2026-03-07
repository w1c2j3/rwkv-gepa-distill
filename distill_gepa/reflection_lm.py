from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

from .teacher_client import (
    TeacherClient,
    extract_chat_completion_text,
    extract_response_text,
    env_first,
    normalize_api_base,
    parse_api_protocol,
    serialize_messages_for_chat_completions,
    serialize_messages_for_responses,
)


def resolve_reflection_runtime(
    teacher: TeacherClient,
) -> tuple[str | None, str | None, str | None, str]:
    load_dotenv()

    model_name = env_first("GEPA_REFLECTION_MODEL", "GEPA_MODEL")
    if not model_name and teacher.mode == "api":
        model_name = teacher.config.model

    api_key = env_first("GEPA_API_KEY", "TEACHER_API_KEY", "OPENAI_API_KEY")
    api_base = normalize_api_base(
        env_first(
            "GEPA_API_BASE",
            "GEPA_BASE_URL",
            "TEACHER_API_BASE",
            "TEACHER_BASE_URL",
            "OPENAI_BASE_URL",
        )
    )
    protocol = parse_api_protocol(
        env_first("GEPA_API_PROTOCOL"),
        model_name=model_name,
    )
    return model_name, api_key, api_base, protocol


def make_openai_lm(
    *,
    model_name: str,
    api_key: str | None,
    api_base: str | None,
    api_protocol: str,
    timeout_seconds: float,
    num_retries: int,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Run `bash scripts/bootstrap.sh` first.") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout_seconds,
        max_retries=num_retries,
    )

    def lm(prompt: str | list[dict[str, Any]]) -> str:
        if api_protocol == "chat_completions":
            messages = serialize_messages_for_chat_completions(prompt)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            return extract_chat_completion_text(response)

        instructions, user_input = serialize_messages_for_responses(prompt)
        request_kwargs: dict[str, Any] = {"model": model_name, "input": user_input, "temperature": 0}
        if instructions is not None:
            request_kwargs["instructions"] = instructions
        response = client.responses.create(**request_kwargs)
        return extract_response_text(response)

    return lm
