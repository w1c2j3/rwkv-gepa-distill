from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import orjson
from dotenv import load_dotenv


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def normalize_api_base(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def parse_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number") from exc


def parse_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"Environment variable {name} must be non-negative")
    return value


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean value")


def collect_stream_content(stream: Iterable[Any]) -> str:
    chunks: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        if not choices:
            continue

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")

        if isinstance(content, str):
            chunks.append(content)
            continue

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)

    merged = "".join(chunks).strip()
    if not merged:
        raise RuntimeError("Teacher API returned an empty streaming response")
    return merged


def litellm_provider_kwargs(model_name: str | None, api_base: str | None) -> dict[str, Any]:
    if api_base and model_name and "/" not in model_name:
        return {"custom_llm_provider": "openai"}
    return {}


@dataclass(frozen=True)
class TeacherConfig:
    api_base: str | None
    api_key: str | None
    model: str | None
    timeout_seconds: float = 30.0
    num_retries: int = 0
    prefer_stream: bool = True

    @property
    def use_mock(self) -> bool:
        return not (self.model and (self.api_key or self.api_base))

    @classmethod
    def from_env(cls) -> "TeacherConfig":
        load_dotenv()
        return cls(
            api_base=normalize_api_base(env_first("TEACHER_API_BASE", "TEACHER_BASE_URL")),
            api_key=env_first("TEACHER_API_KEY"),
            model=env_first("TEACHER_MODEL"),
            timeout_seconds=parse_float_env("OPENAI_TIMEOUT", 30.0),
            num_retries=parse_int_env("OPENAI_MAX_RETRIES", 0),
            prefer_stream=parse_bool_env("TEACHER_PREFER_STREAM", True),
        )


@dataclass(frozen=True)
class TeacherResponse:
    content: str
    source: str
    model_name: str


class TeacherClient:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "TeacherClient":
        return cls(TeacherConfig.from_env())

    @property
    def mode(self) -> str:
        return "mock" if self.config.use_mock else "api"

    def generate(
        self,
        system_prompt: str,
        instruction: str,
        expected_keywords: Sequence[str],
    ) -> TeacherResponse:
        if not system_prompt.strip():
            raise ValueError("System prompt must be non-empty")
        if not instruction.strip():
            raise ValueError("Instruction must be non-empty")
        if not expected_keywords:
            raise ValueError("expected_keywords must be non-empty")

        if self.config.use_mock:
            return self._generate_mock(system_prompt, instruction, expected_keywords)
        user_message = self._build_user_message(instruction, expected_keywords)
        return self.generate_from_user_message(system_prompt, user_message)

    def generate_from_user_message(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if not system_prompt.strip():
            raise ValueError("System prompt must be non-empty")
        if not user_message.strip():
            raise ValueError("User message must be non-empty")

        if self.config.use_mock:
            return self._generate_mock_from_user_message(system_prompt, user_message)
        return self._generate_api(system_prompt, user_message)

    def _generate_api(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if self.config.prefer_stream:
            try:
                return self._generate_api_streaming(system_prompt, user_message)
            except Exception as exc:
                fallback_error = exc
                try:
                    return self._generate_api_non_streaming(system_prompt, user_message)
                except Exception:
                    raise fallback_error

        return self._generate_api_non_streaming(system_prompt, user_message)

    def _generate_api_non_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is not installed. Run `bash scripts/bootstrap.sh` first."
            ) from exc

        response = completion(
            model=self.config.model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=0,
            timeout=self.config.timeout_seconds,
            num_retries=self.config.num_retries,
            **litellm_provider_kwargs(self.config.model, self.config.api_base),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        content = self._extract_message_content(response)
        return TeacherResponse(
            content=content,
            source="teacher_api",
            model_name=self.config.model or "unknown",
        )

    def _generate_api_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is not installed. Run `bash scripts/bootstrap.sh` first."
            ) from exc

        stream = completion(
            model=self.config.model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=0,
            timeout=self.config.timeout_seconds,
            stream=True,
            num_retries=self.config.num_retries,
            **litellm_provider_kwargs(self.config.model, self.config.api_base),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = collect_stream_content(stream)
        return TeacherResponse(
            content=content,
            source="teacher_api",
            model_name=self.config.model or "unknown",
        )

    def _generate_mock(
        self,
        system_prompt: str,
        instruction: str,
        expected_keywords: Sequence[str],
    ) -> TeacherResponse:
        prompt_lower = system_prompt.lower()
        wants_json = "json" in prompt_lower and "answer" in prompt_lower
        wants_keywords = "keyword" in prompt_lower and any(
            token in prompt_lower for token in ("include", "required", "must", "every", "all")
        )
        wants_concise = any(
            token in prompt_lower for token in ("concise", "brief", "short", "1-2", "one or two")
        )

        keyword_limit = len(expected_keywords) if wants_keywords else max(1, len(expected_keywords) - 1)
        keywords_used = [keyword.strip() for keyword in expected_keywords[:keyword_limit]]

        answer = (
            f"{instruction.strip()} "
            f"Key points: {', '.join(keywords_used)}."
        ).strip()
        if not wants_concise:
            answer = f"{answer} " + "Extra filler sentence. " * 12
            answer = answer.strip()

        if wants_json:
            content = orjson.dumps(
                {
                    "answer": answer,
                    "keywords_used": keywords_used,
                }
            ).decode("utf-8")
        else:
            content = answer

        return TeacherResponse(
            content=content,
            source="teacher_api",
            model_name="mock/offline",
        )

    def _generate_mock_from_user_message(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        prompt_lower = system_prompt.lower()
        wants_json = "json" in prompt_lower and "answer" in prompt_lower
        wants_mcq = "answer_letter" in prompt_lower or "multiple-choice" in prompt_lower or "multiple choice" in prompt_lower

        if wants_json and wants_mcq:
            content = orjson.dumps(
                {
                    "answer_letter": "A",
                    "answer_index": 0,
                    "answer_text": "Mock answer",
                    "reasoning": "Offline mock response.",
                }
            ).decode("utf-8")
        elif wants_json:
            content = orjson.dumps(
                {
                    "answer": user_message.strip(),
                    "keywords_used": [],
                }
            ).decode("utf-8")
        else:
            content = user_message.strip()

        return TeacherResponse(
            content=content,
            source="teacher_api",
            model_name="mock/offline",
        )

    @staticmethod
    def _build_user_message(instruction: str, expected_keywords: Sequence[str]) -> str:
        keyword_list = ", ".join(expected_keywords)
        return (
            f"Instruction:\n{instruction.strip()}\n\n"
            f"Required keywords:\n{keyword_list}\n"
        )

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise RuntimeError("Teacher API returned no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            raise RuntimeError("Teacher API returned a choice without a message")

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is None:
            raise RuntimeError("Teacher API returned an empty message content")

        if isinstance(content, list):
            text_chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
            content = "\n".join(text_chunks)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Teacher API returned non-text content")

        return content.strip()
