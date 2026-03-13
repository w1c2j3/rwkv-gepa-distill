from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model_registry import ModelEndpointConfig
from .request_cache import RequestCache, build_request_cache_key


@dataclass(frozen=True)
class GenerationResult:
    content: str
    model_name: str
    attempt_count: int
    cache_hit: bool
    errors: list[str]


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


class AsyncRequestRunner:
    def __init__(
        self,
        *,
        cache_path: Path | None,
        default_max_concurrency: int,
        per_model_concurrency: dict[str, int] | None = None,
    ) -> None:
        if default_max_concurrency <= 0:
            raise ValueError("default_max_concurrency must be positive")
        self._cache = RequestCache(cache_path) if cache_path is not None else None
        self._default_max_concurrency = default_max_concurrency
        self._per_model_concurrency = dict(per_model_concurrency or {})
        self._clients: dict[str, object] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _client(self, endpoint: ModelEndpointConfig):
        client = self._clients.get(endpoint.name)
        if client is None:
            client = endpoint.to_teacher_client()
            self._clients[endpoint.name] = client
        return client

    def _semaphore(self, endpoint: ModelEndpointConfig) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(endpoint.name)
        if semaphore is None:
            limit = self._per_model_concurrency.get(endpoint.name, self._default_max_concurrency)
            semaphore = asyncio.Semaphore(max(1, limit))
            self._semaphores[endpoint.name] = semaphore
        return semaphore

    def _cache_key(
        self,
        *,
        endpoint: ModelEndpointConfig,
        system_prompt: str,
        user_message: str,
    ) -> str:
        return build_request_cache_key(
            api_base=endpoint.api_base,
            api_protocol=endpoint.api_protocol,
            model_name=endpoint.model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=endpoint.max_tokens,
        )

    async def generate(
        self,
        *,
        endpoint: ModelEndpointConfig,
        system_prompt: str,
        user_message: str,
        attempts: int,
        validator: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        if attempts <= 0:
            raise ValueError("attempts must be positive")

        cache_key = self._cache_key(endpoint=endpoint, system_prompt=system_prompt, user_message=user_message)
        if use_cache and self._cache is not None:
            cached_response = self._cache.get(cache_key)
            if cached_response is not None:
                try:
                    if validator is not None:
                        validator(cached_response)
                    return GenerationResult(
                        content=cached_response,
                        model_name=endpoint.name,
                        attempt_count=0,
                        cache_hit=True,
                        errors=[],
                    )
                except Exception:
                    self._cache.delete(cache_key)

        errors: list[str] = []
        async with self._semaphore(endpoint):
            client = self._client(endpoint)
            for attempt_index in range(attempts):
                try:
                    response = await client.generate_from_user_message_async(system_prompt, user_message)
                    content = response.content
                    if validator is not None:
                        validator(content)
                    if use_cache and self._cache is not None:
                        self._cache.set(cache_key, content)
                    return GenerationResult(
                        content=content,
                        model_name=endpoint.name,
                        attempt_count=attempt_index + 1,
                        cache_hit=False,
                        errors=errors,
                    )
                except Exception as exc:
                    errors.append(_format_error(exc))

        raise RuntimeError(
            f"All generation attempts failed for model {endpoint.name!r}: {' | '.join(errors)}"
        )

    async def aclose(self) -> None:
        for client in self._clients.values():
            close_method = getattr(client, "aclose", None)
            if close_method is not None:
                await close_method()
        if self._cache is not None:
            self._cache.close()
