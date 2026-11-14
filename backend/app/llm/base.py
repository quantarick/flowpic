"""Provider-neutral LLM interface.

Every model call in FlowPic goes through `LLMProvider.generate`: one prompt,
optional system text and images in, one string out. Services depend on this
interface and never on a vendor SDK or HTTP shape.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Sequence


class LLMError(RuntimeError):
    """A provider call failed: network, HTTP status, or no usable text."""


class LLMTimeout(LLMError):
    """The provider did not answer within its timeout."""


class LLMNotConfigured(LLMError, ValueError):
    """The selected provider cannot be used at all.

    Raised for a missing or rejected API key, an unknown provider name, or a
    provider that cannot do what the role needs. Retrying will not help, so
    callers should stop the whole job rather than skip one item.

    Also a ValueError, so routers that map ValueError to HTTP 400 keep
    doing so.
    """


class LLMQuotaExceeded(LLMNotConfigured):
    """The provider's quota is used up and will not come back within a minute.

    A daily limit is the usual case. Every further call would be refused the
    same way, so this stops the job like any other LLMNotConfigured.
    """


_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class ImageInput:
    """Encoded image bytes plus their MIME type."""

    data: bytes
    mime_type: str = "image/jpeg"

    @classmethod
    def from_path(cls, path: Path) -> "ImageInput":
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
        return cls(data=path.read_bytes(), mime_type=mime)

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model, as shown in the frontend dropdown."""

    name: str
    size: int | None = None
    parameter_size: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LLMProvider(ABC):
    """One LLM backend. Subclasses are cheap to build and do no I/O in __init__."""

    name: ClassVar[str]
    supports_vision: ClassVar[bool] = False
    # How many calls are worth making at once. A local model serves one at a
    # time on one GPU; a hosted API answers several in parallel.
    parallel_requests: ClassVar[int] = 1

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.default_model()

    @classmethod
    @abstractmethod
    def default_model(cls) -> str:
        """Model used when the caller does not name one. Read from settings."""

    @classmethod
    def is_configured(cls) -> bool:
        """Whether settings hold what this provider needs (an API key, usually)."""
        return True

    def ensure_configured(self) -> None:
        """Raise LLMNotConfigured if this instance cannot make a call."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[ImageInput] = (),
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text reply, stripped.

        Raises LLMTimeout, LLMNotConfigured, or LLMError. Never returns an
        empty string: an empty reply is an LLMError.
        """

    def list_models(self) -> list[ModelInfo]:
        """Models a user may pick for vision work. Default: the configured one."""
        return [ModelInfo(name=self.default_model())]
