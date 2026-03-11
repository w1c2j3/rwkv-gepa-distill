from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import orjson
from dotenv import load_dotenv

from .constants import ANSWER_LABELS


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
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def api_base_looks_local(url: str | None) -> bool:
    normalized = normalize_api_base(url)
    if not normalized:
        return False
    return any(token in normalized for token in ("127.0.0.1", "localhost", "0.0.0.0"))


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


def infer_api_protocol(model_name: str | None) -> str:
    normalized = (model_name or "").strip().lower()
    if normalized.startswith("gpt-5"):
        return "responses"
    return "chat_completions"


def parse_api_protocol(value: str | None, *, model_name: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return infer_api_protocol(model_name)
    if normalized in {"responses", "response"}:
        return "responses"
    if normalized in {"chat", "chat_completions", "chat-completions"}:
        return "chat_completions"
    raise ValueError("API protocol must be one of: responses, chat_completions")


def _dict_get(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _extract_text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []

    chunks: list[str] = []
    for item in content:
        item_type = _dict_get(item, "type")
        if item_type not in {"text", "output_text", "input_text"}:
            continue
        text_value = _dict_get(item, "text")
        if isinstance(text_value, str) and text_value:
            chunks.append(text_value)
            continue
        nested_text = _dict_get(item, "value")
        if isinstance(nested_text, str) and nested_text:
            chunks.append(nested_text)
    return chunks


def extract_response_text(response: Any) -> str:
    output_text = _dict_get(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = _dict_get(response, "output")
    if isinstance(outputs, list):
        text_chunks: list[str] = []
        for output_item in outputs:
            if _dict_get(output_item, "type") != "message":
                continue
            content = _dict_get(output_item, "content")
            text_chunks.extend(_extract_text_blocks(content))
        merged = "".join(text_chunks).strip()
        if merged:
            return merged

    raise RuntimeError("Teacher API returned an empty response")


def extract_chat_completion_text(response: Any) -> str:
    choices = _dict_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Teacher API returned no chat completion choices")
    first_choice = choices[0]
    message = _dict_get(first_choice, "message")
    if message is None:
        raise RuntimeError("Teacher API returned a chat completion without a message")

    content = _dict_get(message, "content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    text_chunks = _extract_text_blocks(content)
    merged = "".join(text_chunks).strip()
    if merged:
        return merged

    raise RuntimeError("Teacher API returned empty chat completion content")


def serialize_messages_for_responses(
    prompt: str | list[dict[str, Any]],
) -> tuple[str | None, str]:
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("Prompt must be non-empty")
        return None, prompt.strip()

    instructions_parts: list[str] = []
    conversation_parts: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            raise TypeError("Response prompt messages must be dictionaries")
        role = message.get("role")
        content = message.get("content")
        content_text = "\n".join(part for part in _extract_text_blocks(content) if part.strip())
        if not content_text and isinstance(content, str):
            content_text = content.strip()
        if not content_text:
            continue

        if role == "system":
            instructions_parts.append(content_text)
            continue

        normalized_role = str(role or "user").lower()
        if normalized_role not in {"user", "assistant", "developer"}:
            normalized_role = "user"
        conversation_parts.append(f"{normalized_role.upper()}:\n{content_text}")

    instructions = "\n\n".join(instructions_parts).strip() or None
    user_input = "\n\n".join(conversation_parts).strip()
    if not user_input:
        fallback_text = instructions or ""
        instructions = None
        if fallback_text:
            user_input = fallback_text
    if not user_input:
        raise ValueError("Response prompt must contain at least one non-empty message")
    return instructions, user_input


def serialize_messages_for_chat_completions(
    prompt: str | list[dict[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("Prompt must be non-empty")
        return [{"role": "user", "content": prompt.strip()}]

    messages: list[dict[str, str]] = []
    for message in prompt:
        if not isinstance(message, dict):
            raise TypeError("Chat prompt messages must be dictionaries")
        role = str(message.get("role") or "user").lower()
        if role not in {"system", "user", "assistant", "developer"}:
            role = "user"
        content = message.get("content")
        content_text = "\n".join(part for part in _extract_text_blocks(content) if part.strip())
        if not content_text and isinstance(content, str):
            content_text = content.strip()
        if not content_text:
            continue
        messages.append({"role": role, "content": content_text})

    if not messages:
        raise ValueError("Chat prompt must contain at least one non-empty message")
    return messages


@dataclass(frozen=True)
class TeacherConfig:
    api_base: str | None
    api_key: str | None
    model: str | None
    api_protocol: str
    timeout_seconds: float = 30.0
    num_retries: int = 0
    prefer_stream: bool = True
    max_tokens: int | None = 256
    force_mock: bool = False

    @property
    def use_mock(self) -> bool:
        return self.force_mock or not bool(self.model)

    @classmethod
    def from_env(cls) -> "TeacherConfig":
        load_dotenv()
        model_name = env_first("TEACHER_MODEL")
        api_base = normalize_api_base(
            env_first("TEACHER_API_BASE", "TEACHER_BASE_URL", "OPENAI_BASE_URL")
        )
        api_key = env_first("TEACHER_API_KEY", "OPENAI_API_KEY")
        if not api_key and api_base_looks_local(api_base):
            api_key = "local"
        return cls(
            api_base=api_base,
            api_key=api_key,
            model=model_name,
            api_protocol=parse_api_protocol(
                env_first("TEACHER_API_PROTOCOL"),
                model_name=model_name,
            ),
            timeout_seconds=parse_float_env("OPENAI_TIMEOUT", 30.0),
            num_retries=parse_int_env("OPENAI_MAX_RETRIES", 0),
            prefer_stream=parse_bool_env("TEACHER_PREFER_STREAM", False),
            max_tokens=parse_int_env("OPENAI_MAX_TOKENS", 256),
            force_mock=False,
        )


@dataclass(frozen=True)
class TeacherResponse:
    content: str
    source: str
    model_name: str


class TeacherClient:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._async_client: Any | None = None

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

    def _openai_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Run `bash scripts/bootstrap.sh` first."
            ) from exc

        self._client = OpenAI(
            api_key=self.config.api_key or "local",
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.num_retries,
        )
        return self._client

    def _openai_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Run `bash scripts/bootstrap.sh` first."
            ) from exc

        self._async_client = AsyncOpenAI(
            api_key=self.config.api_key or "local",
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.num_retries,
        )
        return self._async_client

    def _generate_api(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if self.config.api_protocol == "responses" and self.config.prefer_stream:
            try:
                return self._generate_api_streaming(system_prompt, user_message)
            except Exception as exc:
                fallback_error = exc
                try:
                    return self._generate_api_non_streaming(system_prompt, user_message)
                except Exception:
                    raise fallback_error

        return self._generate_api_non_streaming(system_prompt, user_message)

    async def _generate_api_async(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        return await self._generate_api_non_streaming_async(system_prompt, user_message)

    def _generate_api_non_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if self.config.api_protocol == "chat_completions":
            response = self._openai_client().chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=self.config.max_tokens,
            )
            return TeacherResponse(
                content=extract_chat_completion_text(response),
                source="teacher_api",
                model_name=self.config.model or "unknown",
            )

        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_message,
            "temperature": 0,
            "max_output_tokens": self.config.max_tokens,
        }
        response = self._openai_client().responses.create(**request_kwargs)
        return TeacherResponse(
            content=extract_response_text(response),
            source="teacher_api",
            model_name=self.config.model or "unknown",
        )

    def _generate_api_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if self.config.api_protocol != "responses":
            return self._generate_api_non_streaming(system_prompt, user_message)
        client = self._openai_client()
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_message,
            "temperature": 0,
            "max_output_tokens": self.config.max_tokens,
        }
        with client.responses.stream(**request_kwargs) as stream:
            final_response = stream.get_final_response()

        return TeacherResponse(
            content=extract_response_text(final_response),
            source="teacher_api",
            model_name=self.config.model or "unknown",
        )

    async def _generate_api_non_streaming_async(
        self,
        system_prompt: str,
        user_message: str,
    ) -> TeacherResponse:
        if self.config.api_protocol == "chat_completions":
            response = await self._openai_async_client().chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=self.config.max_tokens,
            )
            return TeacherResponse(
                content=extract_chat_completion_text(response),
                source="teacher_api",
                model_name=self.config.model or "unknown",
            )

        response = await self._openai_async_client().responses.create(
            model=self.config.model,
            instructions=system_prompt,
            input=user_message,
            temperature=0,
            max_output_tokens=self.config.max_tokens,
        )
        return TeacherResponse(
            content=extract_response_text(response),
            source="teacher_api",
            model_name=self.config.model or "unknown",
        )

    async def generate_from_user_message_async(
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
        return await self._generate_api_async(system_prompt, user_message)

    async def aclose(self) -> None:
        if self._async_client is not None:
            close_method = getattr(self._async_client, "close", None)
            if close_method is not None:
                close_result = close_method()
                if hasattr(close_result, "__await__"):
                    await close_result

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
        wants_rewrite = "simple_questions" in prompt_lower
        open_qa_prompt = "question type: open_qa" in user_message.lower()
        model_name = (self.config.model or "").lower()

        option_matches = re.findall(r"^[A-Z]\.\s*(.+)$", user_message, re.MULTILINE)
        if "wrong" in model_name:
            selected_index = 1 if len(option_matches) > 1 else 0
        elif "flip" in model_name:
            digest = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
            selected_index = int(digest[:8], 16) % max(1, len(option_matches) or 2)
            if len(option_matches) > 1:
                selected_index = min(selected_index, 1)
            else:
                selected_index = 0
        else:
            selected_index = 0
        selected_letter = ANSWER_LABELS[selected_index] if selected_index < len(ANSWER_LABELS) else "A"
        selected_text = option_matches[selected_index] if selected_index < len(option_matches) else "Mock answer"
        open_answer = "Wrong answer" if "wrong" in model_name else "Mock answer"

        if wants_rewrite:
            content = orjson.dumps(
                {
                    "simple_questions": [
                        "Simplified mock question one?",
                        "Simplified mock question two?",
                    ]
                }
            ).decode("utf-8")
        elif wants_json and open_qa_prompt:
            content = orjson.dumps(
                {
                    "final_answer": open_answer,
                    "reasoning": "<think>Offline mock response.</think>",
                }
            ).decode("utf-8")
        elif wants_json and wants_mcq:
            content = orjson.dumps(
                {
                    "final_answer": selected_text,
                    "answer_letter": selected_letter,
                    "answer_index": selected_index,
                    "answer_text": selected_text,
                    "reasoning": "<think>Offline mock response.</think>",
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
