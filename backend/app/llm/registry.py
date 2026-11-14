"""Which provider serves which role, and how to build it."""

from __future__ import annotations

import logging
from enum import Enum

from app.config import settings
from app.llm.base import LLMError, LLMNotConfigured, LLMProvider, ModelInfo
from app.llm.fallback import FallbackProvider
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """The three kinds of LLM work FlowPic does."""

    VISION = "vision"            # image captioning, crop review
    TEXT = "text"                # lyric emotion
    COPYWRITING = "copywriting"  # post copy, XHS style analysis


PROVIDERS: dict[str, type[LLMProvider]] = {
    GeminiProvider.name: GeminiProvider,
    OllamaProvider.name: OllamaProvider,
}


def provider_class(name: str) -> type[LLMProvider]:
    key = (name or "").strip().lower()
    try:
        return PROVIDERS[key]
    except KeyError:
        choices = ", ".join(sorted(PROVIDERS))
        raise LLMNotConfigured(f"Unknown LLM provider '{name}'. Choose one of: {choices}") from None


def provider_name_for(role: Role) -> str:
    """A role's provider: its own setting if present, else FLOWPIC_LLM_PROVIDER."""
    explicit = getattr(settings, f"{role.value}_provider")
    return (explicit or settings.llm_provider).strip().lower()


def get_llm(
    role: Role,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Build the provider for a role.

    `provider` and `model` are per-project overrides. A model given without a
    provider is treated as an Ollama model, because projects saved before
    providers existed stored a bare Ollama model name.

    When `FLOWPIC_LLM_FALLBACK` names a different provider, the result tries
    the chosen provider first and falls back to that one.
    """
    if model and not provider and role is Role.VISION:
        provider = OllamaProvider.name

    cls = provider_class(provider or provider_name_for(role))
    if role is Role.VISION and not cls.supports_vision:
        raise LLMNotConfigured(f"The {cls.name} provider cannot read images")
    llm = cls(model=model or None)

    fallback_name = (settings.llm_fallback or "").strip().lower()
    if not fallback_name or fallback_name == cls.name:
        return llm
    fallback_cls = provider_class(fallback_name)
    if role is Role.VISION and not fallback_cls.supports_vision:
        return llm
    return FallbackProvider(llm, fallback_cls())


def vision_catalog() -> dict:
    """Providers and models a user may pick for vision work, plus the server default.

    Holds names and a configured flag only. No key ever appears here.
    """
    default_cls = provider_class(provider_name_for(Role.VISION))

    providers = []
    for cls in PROVIDERS.values():
        if not cls.supports_vision:
            continue

        configured = cls.is_configured()
        models: list[ModelInfo] = []
        if configured:
            try:
                models = cls().list_models()
            except LLMError as e:
                logger.warning("Could not list %s models: %s", cls.name, e)

        # Always offer the configured default, even if it is not pulled yet
        # or the provider has no key. The UI marks unconfigured providers.
        default_base = cls.default_model().split(":")[0]
        if not any(m.name.split(":")[0] == default_base for m in models):
            models.insert(0, ModelInfo(name=cls.default_model()))

        providers.append({
            "name": cls.name,
            "configured": configured,
            "models": [m.to_dict() for m in models],
        })

    return {
        "default": {"provider": default_cls.name, "model": default_cls.default_model()},
        "providers": providers,
    }
