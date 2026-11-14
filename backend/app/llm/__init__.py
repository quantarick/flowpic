"""LLM access for FlowPic.

Services ask for a provider by role and call `generate`:

    from app.llm import Role, get_llm

    text = get_llm(Role.TEXT).generate("Describe the mood of these lyrics: ...")

Adding a provider means one new module that subclasses `LLMProvider` and one
entry in `registry.PROVIDERS`. No service changes.
"""

from app.llm.base import (
    ImageInput,
    LLMError,
    LLMNotConfigured,
    LLMProvider,
    LLMQuotaExceeded,
    LLMTimeout,
    ModelInfo,
)
from app.llm.fallback import FallbackProvider
from app.llm.parsing import parse_json_object
from app.llm.registry import (
    PROVIDERS,
    Role,
    get_llm,
    provider_class,
    provider_name_for,
    vision_catalog,
)

__all__ = [
    "FallbackProvider",
    "ImageInput",
    "LLMError",
    "LLMNotConfigured",
    "LLMProvider",
    "LLMQuotaExceeded",
    "LLMTimeout",
    "ModelInfo",
    "PROVIDERS",
    "Role",
    "get_llm",
    "parse_json_object",
    "provider_class",
    "provider_name_for",
    "vision_catalog",
]
