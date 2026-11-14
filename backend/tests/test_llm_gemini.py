"""GeminiProvider against a mock HTTP transport. No real network."""

from __future__ import annotations

import base64
import json

import cv2
import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from app.config import settings
from app.llm import ImageInput, LLMError, LLMNotConfigured, LLMQuotaExceeded, LLMTimeout
from app.llm.gemini import API_REVISION, GeminiProvider

KEY = "test-key-123"


def reply(text: str = "hello", status: str = "completed") -> dict:
    steps = [{"type": "thought", "summary": [{"type": "text", "text": "private reasoning"}]}]
    if text:
        steps.append({"type": "model_output", "content": [{"type": "text", "text": text}]})
    return {"id": "i1", "object": "interaction", "status": status, "steps": steps}


def make(handler, **kwargs):
    """Build a provider wired to `handler`; returns it with the recorded sleeps."""
    sleeps: list[float] = []
    provider = GeminiProvider(
        api_key=KEY, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kwargs
    )
    return provider, sleeps


def capture(response: httpx.Response):
    """A handler that records every request and always answers `response`."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return handler, seen


def test_text_request_shape():
    handler, seen = capture(httpx.Response(200, json=reply("  hi there \n")))
    provider, _ = make(handler)

    assert provider.generate("Say hi", system="Be brief", max_tokens=50) == "hi there"

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://gemini.test/v1beta/interactions"
    assert request.headers["x-goog-api-key"] == KEY
    assert request.headers["api-revision"] == API_REVISION
    assert json.loads(request.content) == {
        "model": "gemini-test",
        "input": "Say hi",
        "store": False,
        "system_instruction": "Be brief",
        "generation_config": {"thinking_level": "low"},
    }


def test_max_tokens_is_not_forwarded():
    # Gemini counts thinking tokens against max_output_tokens.
    handler, seen = capture(httpx.Response(200, json=reply()))
    make(handler)[0].generate("x", max_tokens=10)
    assert "max_output_tokens" not in seen[0].content.decode()


def test_empty_thinking_level_omits_generation_config():
    handler, seen = capture(httpx.Response(200, json=reply()))
    make(handler, thinking_level="")[0].generate("x")
    assert "generation_config" not in json.loads(seen[0].content)


def test_image_is_shrunk_and_sent_inline():
    wide = np.full((1000, 3000, 3), 128, np.uint8)
    ok, png = cv2.imencode(".png", wide)
    assert ok

    handler, seen = capture(httpx.Response(200, json=reply("a caption")))
    provider, _ = make(handler)
    provider.generate("Caption this", images=[ImageInput(png.tobytes(), "image/png")])

    parts = json.loads(seen[0].content)["input"]
    assert parts[0] == {"type": "text", "text": "Caption this"}
    assert parts[1]["type"] == "image"
    assert parts[1]["mime_type"] == "image/jpeg"

    sent = cv2.imdecode(np.frombuffer(base64.b64decode(parts[1]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert sent.shape[:2] == (683, 2048)


def test_only_model_output_text_is_returned():
    handler, _ = capture(httpx.Response(200, json=reply("the answer")))
    assert make(handler)[0].generate("x") == "the answer"


def test_incomplete_with_text_returns_partial():
    handler, _ = capture(httpx.Response(200, json=reply("partial", status="incomplete")))
    assert make(handler)[0].generate("x") == "partial"


@pytest.mark.parametrize("status", ["incomplete", "failed", "completed"])
def test_no_text_is_an_error(status):
    handler, _ = capture(httpx.Response(200, json=reply("", status=status)))
    with pytest.raises(LLMError):
        make(handler)[0].generate("x")


def test_retries_on_rate_limit_then_succeeds():
    responses = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(503, headers={"retry-after": "7"}, text="busy"),
        httpx.Response(200, json=reply("ok")),
    ]
    provider, sleeps = make(lambda request: responses.pop(0))

    assert provider.generate("x") == "ok"
    assert sleeps == [2.0, 7.0]


def test_gives_up_after_three_attempts():
    handler, seen = capture(httpx.Response(429, json={"error": {"message": "quota"}}))
    provider, sleeps = make(handler)

    with pytest.raises(LLMError) as err:
        provider.generate("x")

    assert not isinstance(err.value, LLMNotConfigured)
    assert len(seen) == 3 and len(sleeps) == 2
    assert "429" in str(err.value) and "quota" in str(err.value)


@pytest.mark.parametrize(
    "status, message",
    [
        (401, "unauthenticated"),
        (403, "permission denied"),
        (400, "API key not valid. Please pass a valid API key."),
        (404, "model not found"),
    ],
)
def test_bad_key_or_model_is_a_configuration_error(status, message):
    handler, seen = capture(httpx.Response(status, json={"error": {"message": message}}))
    with pytest.raises(LLMNotConfigured):
        make(handler)[0].generate("x")
    assert len(seen) == 1  # not retried


def test_other_client_errors_are_plain_errors():
    handler, _ = capture(httpx.Response(400, json={"error": {"message": "bad input"}}))
    with pytest.raises(LLMError) as err:
        make(handler)[0].generate("x")
    assert not isinstance(err.value, LLMNotConfigured)


def test_missing_key_fails_before_any_request():
    handler, seen = capture(httpx.Response(200, json=reply()))
    provider = GeminiProvider(api_key="", transport=httpx.MockTransport(handler))

    with pytest.raises(LLMNotConfigured, match="FLOWPIC_GEMINI_API_KEY"):
        provider.generate("x")
    assert seen == []


def test_timeout_maps_to_llm_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMTimeout):
        make(handler)[0].generate("x")


def test_key_never_appears_in_url_or_errors():
    echo = {"error": {"message": f"rejected credential {KEY} for this project"}}
    handler, seen = capture(httpx.Response(400, json=echo))

    with pytest.raises(LLMError) as err:
        make(handler)[0].generate("x")

    assert KEY not in str(seen[0].url)
    assert KEY not in str(err.value)
    assert "***" in str(err.value)


def test_settings_never_print_the_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("super-secret-value"))

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())
    assert "super-secret-value" not in settings.model_dump_json()
    assert GeminiProvider.is_configured() is True


def test_connection_is_reused_across_calls():
    handler, seen = capture(httpx.Response(200, json=reply()))
    provider, _ = make(handler)

    provider.generate("one")
    first_client = provider._http()
    provider.generate("two")

    assert provider._http() is first_client and len(seen) == 2
    provider.close()
    assert provider._client is None


def test_dropped_connection_is_retried():
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=reply("ok"))

    provider, sleeps = make(handler)
    assert provider.generate("x") == "ok"
    assert sleeps == [2.0]


DAILY_LIMIT = (
    "Rate limit exceeded for model gemini-test (limit: 20 requests per day on Free Tier). "
    "Please retry in 5m19s or upgrade your tier."
)


def test_spent_daily_quota_stops_at_once():
    # The server asks for a five-minute wait. Sitting through 30 seconds twice
    # per photo, as the first version did, only wastes a minute each time.
    handler, seen = capture(
        httpx.Response(429, headers={"retry-after": "319"}, json={"error": {"message": DAILY_LIMIT}})
    )
    provider, sleeps = make(handler)

    with pytest.raises(LLMQuotaExceeded, match="20 requests per day"):
        provider.generate("x")

    assert len(seen) == 1 and sleeps == []


def test_spent_quota_stops_the_job_like_a_missing_key():
    assert issubclass(LLMQuotaExceeded, LLMNotConfigured)


def test_daily_limit_is_recognised_without_a_retry_after_header():
    handler, seen = capture(httpx.Response(429, json={"error": {"message": DAILY_LIMIT}}))
    provider, sleeps = make(handler)

    with pytest.raises(LLMQuotaExceeded):
        provider.generate("x")
    assert len(seen) == 1 and sleeps == []


def test_per_minute_limit_waits_as_long_as_asked_then_carries_on():
    responses = [
        httpx.Response(429, headers={"retry-after": "45"}, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json=reply("ok")),
    ]
    provider, sleeps = make(lambda request: responses.pop(0))

    assert provider.generate("x") == "ok"
    assert sleeps == [45.0]


def test_hosted_provider_takes_several_calls_at_once():
    assert GeminiProvider.parallel_requests == 4
