"""Provider selection, the model catalog, and settings behaviour."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings, settings
from app.llm import LLMError, LLMNotConfigured, ModelInfo, Role, get_llm, provider_name_for, vision_catalog
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider


@pytest.mark.parametrize("role", list(Role))
def test_every_role_defaults_to_gemini(role):
    assert provider_name_for(role) == "gemini"
    assert isinstance(get_llm(role), GeminiProvider)


def test_llm_provider_switches_every_role(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "Ollama ")
    assert [provider_name_for(r) for r in Role] == ["ollama", "ollama", "ollama"]


def test_role_setting_wins_over_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "ollama")

    assert isinstance(get_llm(Role.VISION), OllamaProvider)
    assert isinstance(get_llm(Role.TEXT), GeminiProvider)
    assert isinstance(get_llm(Role.COPYWRITING), GeminiProvider)


def test_unknown_provider_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    with pytest.raises(LLMNotConfigured, match="Unknown LLM provider 'anthropic'"):
        get_llm(Role.COPYWRITING)


def test_default_models_come_from_settings():
    assert get_llm(Role.VISION).model == "gemini-test"
    assert get_llm(Role.VISION, provider="ollama").model == "llava-phi3"


def test_project_override_with_provider_and_model():
    llm = get_llm(Role.VISION, provider="ollama", model="moondream:latest")
    assert isinstance(llm, OllamaProvider) and llm.model == "moondream:latest"


def test_bare_model_from_an_old_project_means_ollama():
    # Projects saved before providers existed stored only an Ollama model name.
    llm = get_llm(Role.VISION, model="llava-phi3:latest")
    assert isinstance(llm, OllamaProvider) and llm.model == "llava-phi3:latest"


def test_catalog_without_a_gemini_key(monkeypatch):
    monkeypatch.setattr(
        OllamaProvider, "list_models", lambda self: [ModelInfo("llava-phi3:latest", 29, "4B")]
    )

    catalog = vision_catalog()

    assert catalog["default"] == {"provider": "gemini", "model": "gemini-test"}
    by_name = {p["name"]: p for p in catalog["providers"]}
    assert by_name["gemini"] == {
        "name": "gemini",
        "configured": False,
        "models": [{"name": "gemini-test", "size": None, "parameter_size": None}],
    }
    assert by_name["ollama"]["configured"] is True
    # The pulled "llava-phi3:latest" already covers the default "llava-phi3".
    assert [m["name"] for m in by_name["ollama"]["models"]] == ["llava-phi3:latest"]


def test_catalog_survives_ollama_being_down(monkeypatch):
    def down(self):
        raise LLMError("connection refused")

    monkeypatch.setattr(OllamaProvider, "list_models", down)

    by_name = {p["name"]: p for p in vision_catalog()["providers"]}
    assert [m["name"] for m in by_name["ollama"]["models"]] == ["llava-phi3"]


def test_models_route_never_returns_the_key(monkeypatch):
    from app.routers import llm as llm_router

    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("super-secret-value"))
    monkeypatch.setattr(OllamaProvider, "list_models", lambda self: [])

    app = FastAPI()
    app.include_router(llm_router.router, prefix="/api/llm")
    response = TestClient(app).get("/api/llm/models")

    assert response.status_code == 200
    assert "super-secret-value" not in response.text
    by_name = {p["name"]: p for p in response.json()["providers"]}
    assert by_name["gemini"]["configured"] is True
    assert set(json.loads(response.text)) == {"default", "providers"}


def test_settings_skip_retired_variables(monkeypatch):
    # A setting removed from the code but still present in someone's .env must
    # not stop startup, and must not be loaded.
    monkeypatch.setenv("FLOWPIC_ANTHROPIC_API_KEY", "placeholder-not-a-real-key")
    monkeypatch.setenv("FLOWPIC_GEMINI_MODEL", "gemini-from-env")

    loaded = Settings(_env_file=None)

    assert loaded.gemini_model == "gemini-from-env"
    assert not hasattr(loaded, "anthropic_api_key")
    assert "placeholder-not-a-real-key" not in repr(loaded)
