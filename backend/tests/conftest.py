"""Shared test setup.

Tests never read a developer's real API key or provider choices, and never
reach a real LLM: providers get a mock HTTP transport or are replaced by
`FakeLLM`.
"""

from __future__ import annotations

import threading
import time
from typing import Sequence

import pytest
from pydantic import SecretStr

from app.config import settings
from app.llm import ImageInput, LLMProvider


@pytest.fixture(autouse=True)
def hermetic_llm_settings(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(""))
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_base_url", "https://gemini.test/v1beta")
    monkeypatch.setattr(settings, "gemini_thinking_level", "low")
    monkeypatch.setattr(settings, "ollama_model", "llava-phi3")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama.test")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_fallback", "")
    monkeypatch.setattr(settings, "caption_parallel", 0)
    monkeypatch.setattr(settings, "caption_max_tokens", 450)
    for role in ("vision", "text", "copywriting"):
        monkeypatch.setattr(settings, f"{role}_provider", None)


class FakeLLM(LLMProvider):
    """A provider that returns a canned reply or raises a canned error."""

    name = "fake"
    supports_vision = True

    def __init__(
        self,
        reply: str = "",
        error: Exception | None = None,
        delay: float = 0.0,
        parallel_requests: int = 1,
    ):
        super().__init__("fake-1")
        self.reply = reply
        self.error = error
        self.delay = delay
        self.parallel_requests = parallel_requests
        self.calls: list[dict] = []
        self.max_active = 0  # most calls ever in flight at once
        self._active = 0
        self._lock = threading.Lock()

    @classmethod
    def default_model(cls) -> str:
        return "fake-1"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[ImageInput] = (),
        max_tokens: int | None = None,
    ) -> str:
        with self._lock:
            self.calls.append(
                {"prompt": prompt, "system": system, "images": list(images), "max_tokens": max_tokens}
            )
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return self.reply
        finally:
            with self._lock:
                self._active -= 1
