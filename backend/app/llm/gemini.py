"""Gemini provider: Google's Interactions API over plain HTTP.

Request and response shapes follow the Interactions API reference:
POST {base}/interactions with `model`, `input`, optional `system_instruction`
and `generation_config`. The reply text lives in
steps[type=model_output].content[type=text].text.

Key handling: the API key is held as a SecretStr, travels only in the
`x-goog-api-key` request header (never in a URL, where request logging would
record it), and is scrubbed from every error message this module raises.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Sequence

import httpx
from pydantic import SecretStr

from app.config import settings
from app.llm.base import (
    ImageInput,
    LLMError,
    LLMNotConfigured,
    LLMProvider,
    LLMQuotaExceeded,
    LLMTimeout,
)
from app.llm.images import normalize_for_upload

logger = logging.getLogger(__name__)

# The API revision this module was written against.
API_REVISION = "2026-05-20"

_RETRY_STATUS = {429, 500, 502, 503, 504}
# The longest wait worth sitting through. A per-minute limit clears within
# this; a server that asks for longer is reporting a spent daily quota.
_MAX_RETRY_DELAY = 65.0


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_vision = True
    parallel_requests = 4

    # Longer image side sent to the API. Inline requests are capped at 20 MB.
    MAX_IMAGE_SIDE = 2048
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        thinking_level: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(model)
        key = api_key if api_key is not None else settings.gemini_api_key
        self._api_key = key if isinstance(key, SecretStr) else SecretStr(key)
        self._base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self._timeout = timeout if timeout is not None else settings.gemini_timeout
        self._thinking_level = (
            thinking_level if thinking_level is not None else settings.gemini_thinking_level
        )
        self._transport = transport
        self._sleep = sleep
        # One HTTP client per provider, so a batch of captions reuses the TLS
        # connection instead of opening a new one for every picture.
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def _http(self) -> httpx.Client:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self._timeout, transport=self._transport)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def default_model(cls) -> str:
        return settings.gemini_model

    @classmethod
    def is_configured(cls) -> bool:
        return bool(settings.gemini_api_key.get_secret_value())

    def ensure_configured(self) -> None:
        if not self._api_key.get_secret_value():
            raise LLMNotConfigured("FLOWPIC_GEMINI_API_KEY not configured")

    def _redact(self, text: str) -> str:
        """Remove the API key from text that will be logged or raised."""
        key = self._api_key.get_secret_value()
        return text.replace(key, "***") if key else text

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[ImageInput] = (),
        max_tokens: int | None = None,
    ) -> str:
        self.ensure_configured()

        if images:
            parts: list[dict] = [{"type": "text", "text": prompt}]
            for image in images:
                prepared = normalize_for_upload(image, self.MAX_IMAGE_SIDE)
                parts.append(
                    {"type": "image", "data": prepared.b64(), "mime_type": prepared.mime_type}
                )
            model_input: str | list[dict] = parts
        else:
            model_input = prompt

        # store=False keeps the call stateless: Google does not keep the
        # prompt or the photos for later retrieval.
        body: dict = {"model": self.model, "input": model_input, "store": False}
        if system:
            body["system_instruction"] = system
        if self._thinking_level:
            body["generation_config"] = {"thinking_level": self._thinking_level}
        # max_tokens is deliberately not forwarded. Gemini counts thinking
        # tokens against max_output_tokens, so a cap sized for the visible
        # reply can cut the model off before it writes anything.

        return self._extract_text(self._post("/interactions", body))

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self._base_url}{path}"
        headers = {
            "x-goog-api-key": self._api_key.get_secret_value(),
            "Api-Revision": API_REVISION,
        }

        client = self._http()
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                resp = client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as e:
                raise LLMTimeout(f"Gemini timeout after {self._timeout}s") from e
            except httpx.TransportError as e:
                # A dropped or refused connection is worth another try.
                if attempt < self.MAX_ATTEMPTS:
                    delay = float(2 ** attempt)
                    logger.warning(
                        "Gemini connection error (%s), retrying in %.0fs (attempt %d of %d)",
                        type(e).__name__, delay, attempt, self.MAX_ATTEMPTS,
                    )
                    self._sleep(delay)
                    continue
                raise LLMError(self._redact(f"Gemini request failed: {e}")) from e
            except httpx.HTTPError as e:
                raise LLMError(self._redact(f"Gemini request failed: {e}")) from e

            if resp.status_code == 429 and self._quota_is_spent(resp):
                # Waiting will not help, and neither will the next photo's call.
                raise LLMQuotaExceeded(self._redact(f"Gemini HTTP 429: {self._error_message(resp)}"))

            if resp.status_code in _RETRY_STATUS and attempt < self.MAX_ATTEMPTS:
                delay = self._retry_delay(resp, attempt)
                logger.warning(
                    "Gemini HTTP %d, retrying in %.0fs (attempt %d of %d)",
                    resp.status_code, delay, attempt, self.MAX_ATTEMPTS,
                )
                self._sleep(delay)
                continue

            if resp.status_code >= 400:
                self._raise_for_status(resp)

            try:
                data = resp.json()
            except ValueError as e:
                raise LLMError("Gemini returned a body that is not JSON") from e
            if not isinstance(data, dict):
                raise LLMError("Gemini returned an unexpected JSON shape")
            return data

        raise LLMError("Gemini request failed after retries")  # unreachable

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        """Seconds the server asked us to wait, if it said."""
        try:
            return max(0.0, float(resp.headers.get("retry-after", "")))
        except ValueError:
            return None

    @classmethod
    def _retry_delay(cls, resp: httpx.Response, attempt: int) -> float:
        asked = cls._retry_after(resp)
        delay = asked if asked is not None else float(2 ** attempt)
        return min(delay, _MAX_RETRY_DELAY)

    @classmethod
    def _quota_is_spent(cls, resp: httpx.Response) -> bool:
        """True when a 429 means "come back much later", not "slow down a little"."""
        asked = cls._retry_after(resp)
        if asked is not None:
            return asked > _MAX_RETRY_DELAY
        return "per day" in cls._error_message(resp).lower()

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        message = resp.text[:300]
        try:
            error = resp.json().get("error") or {}
            message = error.get("message") or message
        except (ValueError, AttributeError):
            pass
        return message

    def _raise_for_status(self, resp: httpx.Response) -> None:
        message = self._error_message(resp)

        summary = self._redact(f"Gemini HTTP {resp.status_code}: {message}")
        # A bad key or an unknown model fails the same way on every call,
        # so these stop the job instead of failing one image at a time.
        bad_key = resp.status_code in (401, 403) or "api key" in message.lower()
        if bad_key or resp.status_code == 404:
            raise LLMNotConfigured(summary)
        raise LLMError(summary)

    def _extract_text(self, data: dict) -> str:
        chunks: list[str] = []
        for step in data.get("steps") or []:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            step_text = "".join(
                item.get("text") or ""
                for item in step.get("content") or []
                if isinstance(item, dict) and item.get("type") == "text"
            )
            if step_text:
                chunks.append(step_text)
        text = "\n".join(chunks).strip()

        status = data.get("status")
        if text:
            if status != "completed":
                logger.warning("Gemini interaction ended with status '%s'; using partial text", status)
            return text

        detail = data.get("errors") or "no text in the response"
        raise LLMError(self._redact(f"Gemini interaction {status or 'returned'}: {detail}"))
