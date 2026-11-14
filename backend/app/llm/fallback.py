"""A provider that falls back to a second one when the first cannot answer.

Used to keep a job going on the local model when the hosted one is out of
quota, has no key, or fails on a request.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from app.llm.base import ImageInput, LLMError, LLMNotConfigured, LLMProvider

logger = logging.getLogger(__name__)


class FallbackProvider(LLMProvider):
    """Tries `primary`; when that fails, asks `fallback`.

    Two kinds of failure are handled differently:

    - The primary cannot be used at all (no key, rejected key, spent quota).
      It is not tried again by this instance: every later call goes straight
      to the fallback. One instance serves one job, so the next job tries the
      primary afresh.
    - The primary failed on one request (timeout, server error after its own
      retries). Only that request goes to the fallback.

    If the fallback fails too, the caller gets an error that names both.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        # No super().__init__: this wrapper has no model of its own.
        self.primary = primary
        self.fallback = fallback
        self._primary_down: LLMNotConfigured | None = None
        self._fallback_has_worked = False
        self._lock = threading.Lock()
        # Callers may run several requests at once for the primary. The
        # fallback gets no more than it says it can take.
        self._fallback_slots = threading.Semaphore(max(1, fallback.parallel_requests))

    # --- what callers read ---

    @classmethod
    def default_model(cls) -> str:
        return ""

    @property
    def using_fallback(self) -> bool:
        """True once the primary has been given up on."""
        return self._primary_down is not None

    @property
    def _active(self) -> LLMProvider:
        return self.fallback if self.using_fallback else self.primary

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._active.name

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._active.model

    @property
    def supports_vision(self) -> bool:  # type: ignore[override]
        return self.primary.supports_vision and self.fallback.supports_vision

    @property
    def parallel_requests(self) -> int:  # type: ignore[override]
        return self._active.parallel_requests

    @property
    def note(self) -> str:
        """A short line for progress messages, empty while the primary is in use."""
        if not self.using_fallback:
            return ""
        return f"{self.primary.name} unavailable, using local {self.fallback.model}"

    def list_models(self):
        return self.primary.list_models()

    def ensure_configured(self) -> None:
        """Passes if either provider can make a call."""
        if not self.using_fallback:
            try:
                self.primary.ensure_configured()
                return
            except LLMNotConfigured as e:
                self._give_up_on_primary(e)
        self.fallback.ensure_configured()

    # --- the call ---

    def _give_up_on_primary(self, error: LLMNotConfigured) -> None:
        with self._lock:
            first = self._primary_down is None
            if first:
                self._primary_down = error
        if first:
            logger.warning(
                "%s cannot be used (%s). Falling back to %s (%s) for the rest of this job.",
                self.primary.name, error, self.fallback.name, self.fallback.model,
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[ImageInput] = (),
        max_tokens: int | None = None,
    ) -> str:
        primary_error: LLMError | None = self._primary_down

        if primary_error is None:
            try:
                return self.primary.generate(prompt, system=system, images=images, max_tokens=max_tokens)
            except LLMNotConfigured as e:
                self._give_up_on_primary(e)
                primary_error = e
            except LLMError as e:
                logger.warning(
                    "%s failed on one request (%s). Trying %s for it.",
                    self.primary.name, e, self.fallback.name,
                )
                primary_error = e

        try:
            with self._fallback_slots:
                text = self.fallback.generate(prompt, system=system, images=images, max_tokens=max_tokens)
        except LLMError as fallback_error:
            if self.using_fallback and self._fallback_has_worked:
                # The fallback works in general and failed on this one request.
                raise
            # Nothing has worked. If the primary is unusable, keep that error
            # type so a spent quota or a missing key still stops the job. A
            # one-off failure stays a plain error, which skips this request.
            kind = type(primary_error) if isinstance(primary_error, LLMNotConfigured) else LLMError
            raise kind(
                f"{primary_error}; fallback {self.fallback.name} also failed: {fallback_error}"
            ) from fallback_error

        self._fallback_has_worked = True
        return text
