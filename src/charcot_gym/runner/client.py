"""OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class ModelConfig:
    """Request settings for a chat-completions provider."""

    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 60.0
    retries: int = 3
    retry_backoff_s: float = 0.5
    price_table: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


class ChatCompletionError(RuntimeError):
    """A sanitized provider failure safe to classify in run artifacts."""

    def __init__(self, kind: str, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status


class ChatTimeoutError(TimeoutError):
    """A bounded chat-completions request or episode exceeded its deadline."""


class ChatClient(Protocol):
    """Minimal interface consumed by the episode loop."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Return one parsed chat-completions response."""


class OpenAIChatClient:
    """Synchronous adapter using bounded asynchronous HTTP requests."""

    def __init__(
        self, config: ModelConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.config = config
        self.transport = transport

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one bounded chat-completions request and return its JSON response."""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ChatCompletionError(
                "configuration_error",
                f"missing API key environment variable: {self.config.api_key_env}",
            )
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request_timeout = self.config.timeout
        if timeout_s is not None:
            request_timeout = min(request_timeout, timeout_s)
        if request_timeout <= 0:
            raise ChatTimeoutError("episode deadline exceeded")
        try:
            return asyncio.run(self._complete(payload, api_key, request_timeout))
        except TimeoutError as error:
            raise ChatTimeoutError("chat-completions request timed out") from error

    async def _complete(
        self, payload: dict[str, Any], api_key: str, timeout_s: float
    ) -> dict[str, Any]:
        timeout = httpx.Timeout(
            timeout_s,
            connect=timeout_s,
            read=timeout_s,
            write=timeout_s,
            pool=timeout_s,
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            async with asyncio.timeout(timeout_s):
                async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
                    for attempt in range(self.config.retries + 1):
                        try:
                            response = await client.post(
                                f"{self.config.base_url.rstrip('/')}/chat/completions",
                                json=payload,
                                headers=headers,
                            )
                            response.raise_for_status()
                            try:
                                data = response.json()
                            except ValueError as error:
                                raise ChatCompletionError(
                                    "invalid_response", "invalid chat-completions response"
                                ) from error
                            if not isinstance(data, dict):
                                raise ChatCompletionError(
                                    "invalid_response", "invalid chat-completions response"
                                )
                            return data
                        except httpx.HTTPStatusError as error:
                            status = error.response.status_code
                            retryable = status == 429 or 500 <= status < 600
                            if not retryable or attempt >= self.config.retries:
                                raise ChatCompletionError(
                                    "http_error",
                                    f"chat-completions HTTP {status}",
                                    http_status=status,
                                ) from error
                        except httpx.TimeoutException as error:
                            if attempt >= self.config.retries:
                                raise ChatTimeoutError(
                                    "chat-completions request timed out"
                                ) from error
                        except httpx.RequestError as error:
                            if attempt >= self.config.retries:
                                raise ChatCompletionError(
                                    "request_error", "chat-completions request failed"
                                ) from error
                        await asyncio.sleep(self.config.retry_backoff_s * (2**attempt))
        except TimeoutError as error:
            raise ChatTimeoutError("chat-completions request timed out") from error
        raise ChatCompletionError("request_error", "chat-completions request failed")
