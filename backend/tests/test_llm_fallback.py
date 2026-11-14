"""Falling back to a second provider when the first cannot answer."""

from __future__ import annotations

import httpx
import pytest

from app.config import settings
from app.llm import (
    FallbackProvider,
    ImageInput,
    LLMError,
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTimeout,
    Role,
    get_llm,
)
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from tests.conftest import FakeLLM

QUOTA = LLMQuotaExceeded("Gemini HTTP 429: limit: 20 requests per day on Free Tier")


def pair(primary: FakeLLM, fallback: FakeLLM) -> FallbackProvider:
    primary.name, fallback.name = "hosted", "local"
    return FallbackProvider(primary, fallback)


# --- when the primary works ---

def test_primary_answers_and_the_fallback_is_never_called():
    primary, fallback = FakeLLM("from hosted", parallel_requests=4), FakeLLM("from local")
    llm = pair(primary, fallback)

    assert llm.generate("hi", system="s", max_tokens=9) == "from hosted"
    assert fallback.calls == []
    assert (llm.name, llm.model, llm.parallel_requests, llm.using_fallback, llm.note) == ("hosted", "fake-1", 4, False, "")
    assert primary.calls[0]["system"] == "s" and primary.calls[0]["max_tokens"] == 9


# --- when the primary cannot be used at all ---

@pytest.mark.parametrize("error", [QUOTA, LLMNotConfigured("FLOWPIC_GEMINI_API_KEY not configured")])
def test_unusable_primary_is_tried_once_then_left_alone(error):
    primary, fallback = FakeLLM(error=error, parallel_requests=4), FakeLLM("from local")
    llm = pair(primary, fallback)
    image = ImageInput(b"x")

    assert llm.generate("one", images=[image], max_tokens=450) == "from local"
    assert llm.generate("two") == "from local"
    assert llm.generate("three") == "from local"

    assert len(primary.calls) == 1           # not asked again after it said no
    assert len(fallback.calls) == 3
    assert fallback.calls[0]["images"] == [image] and fallback.calls[0]["max_tokens"] == 450
    assert llm.using_fallback is True
    assert (llm.name, llm.parallel_requests) == ("local", 1)
    assert llm.note == "hosted unavailable, using local fake-1"


def test_spent_quota_with_no_working_fallback_still_stops_the_job():
    llm = pair(FakeLLM(error=QUOTA), FakeLLM(error=LLMError("Ollama request failed: connection refused")))

    with pytest.raises(LLMQuotaExceeded) as err:
        llm.generate("x")

    assert "20 requests per day" in str(err.value)
    assert "fallback local also failed: Ollama request failed" in str(err.value)


def test_a_bad_request_on_a_working_fallback_only_skips_that_request():
    primary, fallback = FakeLLM(error=QUOTA), FakeLLM("fine")
    llm = pair(primary, fallback)
    assert llm.generate("first") == "fine"

    fallback.error = LLMTimeout("local model too slow")
    with pytest.raises(LLMTimeout):          # not a job-stopping error
        llm.generate("second")


# --- when the primary fails on one request ---

def test_one_failed_request_goes_to_the_fallback_and_the_primary_stays_in_use():
    primary, fallback = FakeLLM(error=LLMTimeout("hosted timed out")), FakeLLM("from local")
    llm = pair(primary, fallback)

    assert llm.generate("one") == "from local"
    assert llm.using_fallback is False

    primary.error, primary.reply = None, "from hosted"
    assert llm.generate("two") == "from hosted"
    assert len(primary.calls) == 2 and len(fallback.calls) == 1


def test_one_failed_request_with_a_failing_fallback_is_a_plain_error():
    llm = pair(FakeLLM(error=LLMTimeout("hosted timed out")), FakeLLM(error=LLMError("local down")))

    with pytest.raises(LLMError) as err:
        llm.generate("x")

    assert not isinstance(err.value, (LLMTimeout, LLMNotConfigured))
    assert "hosted timed out" in str(err.value) and "local down" in str(err.value)


# --- the local model is not flooded ---

def test_fallback_gets_one_request_at_a_time_even_when_callers_run_four():
    from concurrent.futures import ThreadPoolExecutor

    primary = FakeLLM(error=QUOTA, parallel_requests=4)
    fallback = FakeLLM("ok", delay=0.05, parallel_requests=1)
    llm = pair(primary, fallback)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(llm.generate, [f"p{i}" for i in range(8)])) == ["ok"] * 8

    assert fallback.max_active == 1 and len(fallback.calls) == 8


# --- readiness checks ---

def test_ensure_configured_passes_when_only_the_fallback_is_ready():
    class NoKey(FakeLLM):
        def ensure_configured(self):
            raise LLMNotConfigured("no key")

    llm = pair(NoKey(), FakeLLM("local"))
    llm.ensure_configured()

    assert llm.using_fallback is True
    assert llm.generate("x") == "local"


# --- how get_llm builds it ---

def test_get_llm_wraps_the_hosted_provider_when_a_fallback_is_set(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback", "ollama")

    for role in Role:
        llm = get_llm(role)
        assert isinstance(llm, FallbackProvider)
        assert isinstance(llm.primary, GeminiProvider) and isinstance(llm.fallback, OllamaProvider)
        assert llm.fallback.model == "llava-phi3"


def test_no_wrapper_when_the_fallback_is_the_chosen_provider_or_is_off(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback", "ollama")
    assert isinstance(get_llm(Role.VISION, provider="ollama"), OllamaProvider)
    assert isinstance(get_llm(Role.VISION, model="moondream"), OllamaProvider)   # a project's own Ollama model

    monkeypatch.setattr(settings, "llm_fallback", "")
    assert isinstance(get_llm(Role.VISION), GeminiProvider)


def test_unknown_fallback_name_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback", "anthropic")
    with pytest.raises(LLMNotConfigured, match="Unknown LLM provider 'anthropic'"):
        get_llm(Role.TEXT)


def test_fallback_is_on_by_default():
    from app.config import Settings

    assert Settings(_env_file=None).llm_fallback == "ollama"


# --- end to end with the real providers: Gemini says the quota is spent, Ollama answers ---

def test_real_gemini_quota_error_falls_back_to_real_ollama_provider():
    daily = (
        "Rate limit exceeded for model gemini-test (limit: 20 requests per day on Free Tier). "
        "Please retry in 22h18m59s or upgrade your tier."
    )
    gemini_requests, ollama_requests = [], []

    def gemini_api(request: httpx.Request) -> httpx.Response:
        gemini_requests.append(request)
        return httpx.Response(429, headers={"retry-after": "80339"}, json={"error": {"message": daily}})

    def ollama_api(request: httpx.Request) -> httpx.Response:
        ollama_requests.append(request)
        return httpx.Response(200, json={"message": {"content": "CAPTION: A red barn."}})

    sleeps: list[float] = []
    llm = FallbackProvider(
        GeminiProvider(api_key="test-key", transport=httpx.MockTransport(gemini_api), sleep=sleeps.append),
        OllamaProvider(transport=httpx.MockTransport(ollama_api)),
    )

    image = ImageInput(b"raw-bytes", "image/jpeg")
    assert llm.generate("Describe", images=[image]) == "CAPTION: A red barn."
    assert llm.generate("Describe", images=[image]) == "CAPTION: A red barn."

    assert len(gemini_requests) == 1 and sleeps == []      # refused once, no waiting, not asked again
    assert len(ollama_requests) == 2
    assert str(ollama_requests[0].url) == "http://ollama.test/api/chat"
    assert llm.note == "gemini unavailable, using local llava-phi3"


# --- through the captioner: a spent quota no longer stops the batch ---

CAPTION = "CAPTION: A red barn.\nELEMENT: architecture\nSUBJECT: 40%, 60%\nBOUNDS: 10%, 20%, 90%, 95%"


def test_captioning_carries_on_with_the_local_model_when_the_quota_runs_out(tmp_path):
    import cv2
    import numpy as np

    from app.services.image_captioner import ImageCaptioner

    images = tmp_path / "images"
    images.mkdir()
    paths = []
    for i in range(6):
        path = images / f"photo{i}.jpg"
        cv2.imwrite(str(path), np.full((240, 320, 3), 60 + 20 * i, np.uint8))
        paths.append(path)

    primary = FakeLLM(error=QUOTA, parallel_requests=4)
    fallback = FakeLLM(CAPTION, delay=0.02, parallel_requests=1)
    captioner = ImageCaptioner(llm=pair(primary, fallback))
    assert captioner.provider_note == ""

    captions = captioner.caption_images(paths)

    assert [c.caption for c in captions] == ["A red barn."] * 6
    assert len(fallback.calls) == 6 and fallback.max_active == 1
    assert 1 <= len(primary.calls) <= 4          # only the photos already in flight asked the hosted model
    assert captioner.provider_note == "hosted unavailable, using local fake-1"
    assert len(list((tmp_path / "captions").glob("*.json"))) == 6
