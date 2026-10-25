"""OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


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


class ChatClient(Protocol):
    """Minimal interface consumed by the episode loop."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return one parsed chat-completions response."""


class OpenAIChatClient:
    """Synchronous OpenAI-compatible HTTP client with bounded retries."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Send a chat-completions request and return its JSON response."""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {self.config.api_key_env}")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read())
                    if not isinstance(data, dict):
                        raise ValueError("invalid chat-completions response")
                    return data
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt >= self.config.retries:
                    raise RuntimeError(f"chat-completions HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt >= self.config.retries:
                    raise RuntimeError("chat-completions request failed") from error
            time.sleep(self.config.retry_backoff_s * (2**attempt))
        raise RuntimeError("chat-completions request failed")
