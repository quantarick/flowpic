"""Ollama provider: local models over the Ollama HTTP API."""

from __future__ import annotations

import logging
from typing import Sequence

import httpx

from app.config import settings
from app.llm.base import ImageInput, LLMError, LLMProvider, LLMTimeout, ModelInfo

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_vision = True

    # Models that only answer on /api/generate, not /api/chat.
    GENERATE_ONLY = {"moondream", "qwen2.5vl"}

    # Models known to accept image input. Ollama's tag list has no
    # capability field, so this is an allowlist.
    VISION_CAPABLE = {
        "moondream",
        "llava",
        "llava-phi3",
        "llava-llama3",
        "bakllava",
        "gemma3",
        "qwen2.5vl",
        "qwen2-vl",
        "qwen3-vl",
        "minicpm-v",
        "llama3.2-vision",
        "granite3.2-vision",
        "deepseek-ocr",
        "smolvlm",
    }

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout if timeout is not None else settings.ollama_timeout
        self._transport = transport

    @classmethod
    def default_model(cls) -> str:
        return settings.ollama_model

    @staticmethod
    def _base_name(model: str) -> str:
        return model.split(":")[0].lower()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[ImageInput] = (),
        max_tokens: int | None = None,
    ) -> str:
        encoded = [image.b64() for image in images]
        # Text-only prompts and generate-only models use /api/generate;
        # everything else uses /api/chat.
        use_generate = not encoded or self._base_name(self.model) in self.GENERATE_ONLY

        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                if use_generate:
                    text = self._generate(client, prompt, system, encoded, max_tokens)
                else:
                    text = self._chat(client, prompt, system, encoded, max_tokens)
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"Ollama timeout after {self._timeout}s") from e
        except httpx.HTTPStatusError as e:
            # Ollama explains failures in the body, e.g. {"error": "model not found"}.
            detail = e.response.text[:200].strip()
            raise LLMError(f"Ollama HTTP {e.response.status_code}: {detail}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        except ValueError as e:
            raise LLMError("Ollama returned a body that is not JSON") from e

        if not text:
            raise LLMError(f"Ollama model '{self.model}' returned no text")
        return text

    def _generate(
        self,
        client: httpx.Client,
        prompt: str,
        system: str | None,
        images: list[str],
        max_tokens: int | None,
    ) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if images:
            payload["images"] = images
        if system:
            payload["system"] = system
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        resp = client.post(f"{self._base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def _chat(
        self,
        client: httpx.Client,
        prompt: str,
        system: str | None,
        images: list[str],
        max_tokens: int | None,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images})
        payload: dict = {"model": self.model, "messages": messages, "stream": False}
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        resp = client.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return ((resp.json().get("message") or {}).get("content") or "").strip()

    def list_models(self) -> list[ModelInfo]:
        """Pulled models that accept images. Raises LLMError if Ollama is unreachable."""
        try:
            with httpx.Client(timeout=10, transport=self._transport) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                tags = resp.json().get("models", [])
        except httpx.HTTPError as e:
            raise LLMError(f"Failed to query Ollama models: {e}") from e

        return [
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                parameter_size=(m.get("details") or {}).get("parameter_size"),
            )
            for m in tags
            if self._base_name(m.get("name", "")) in self.VISION_CAPABLE
        ]
