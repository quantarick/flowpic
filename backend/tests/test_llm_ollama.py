"""OllamaProvider against a mock HTTP transport. No real network."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm import ImageInput, LLMError, LLMTimeout
from app.llm.ollama import OllamaProvider

IMAGE = ImageInput(b"raw-bytes", "image/jpeg")
IMAGE_B64 = "cmF3LWJ5dGVz"


def capture(response: httpx.Response):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return httpx.MockTransport(handler), seen


def test_vision_uses_chat_and_sends_original_bytes():
    transport, seen = capture(httpx.Response(200, json={"message": {"content": " a dog \n"}}))
    provider = OllamaProvider(transport=transport)

    assert provider.generate("Describe", images=[IMAGE]) == "a dog"

    assert str(seen[0].url) == "http://ollama.test/api/chat"
    assert json.loads(seen[0].content) == {
        "model": "llava-phi3",
        "messages": [{"role": "user", "content": "Describe", "images": [IMAGE_B64]}],
        "stream": False,
    }


@pytest.mark.parametrize("model", ["moondream", "moondream:latest", "qwen2.5vl:7b"])
def test_generate_only_models_use_generate(model):
    transport, seen = capture(httpx.Response(200, json={"response": "a cat"}))
    provider = OllamaProvider(model, transport=transport)

    assert provider.generate("Describe", images=[IMAGE]) == "a cat"

    assert str(seen[0].url) == "http://ollama.test/api/generate"
    assert json.loads(seen[0].content) == {
        "model": model,
        "prompt": "Describe",
        "stream": False,
        "images": [IMAGE_B64],
    }


def test_text_only_uses_generate_without_images():
    transport, seen = capture(httpx.Response(200, json={"response": '{"theme": "x"}'}))
    provider = OllamaProvider(transport=transport)

    provider.generate("Analyze these lyrics")

    assert str(seen[0].url) == "http://ollama.test/api/generate"
    assert json.loads(seen[0].content) == {
        "model": "llava-phi3",
        "prompt": "Analyze these lyrics",
        "stream": False,
    }


def test_system_and_max_tokens_are_passed_through():
    transport, seen = capture(httpx.Response(200, json={"response": "ok"}))
    OllamaProvider(transport=transport).generate("x", system="Be brief", max_tokens=64)

    body = json.loads(seen[0].content)
    assert body["system"] == "Be brief"
    assert body["options"] == {"num_predict": 64}


def test_http_error_maps_to_llm_error():
    transport, _ = capture(httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        OllamaProvider(transport=transport).generate("x")


def test_empty_reply_is_an_error():
    transport, _ = capture(httpx.Response(200, json={"response": "   "}))
    with pytest.raises(LLMError):
        OllamaProvider(transport=transport).generate("x")


def test_timeout_maps_to_llm_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMTimeout):
        OllamaProvider(transport=httpx.MockTransport(handler)).generate("x")


def test_list_models_keeps_only_vision_capable():
    tags = {
        "models": [
            {"name": "llava-phi3:latest", "size": 2900000000, "details": {"parameter_size": "4B"}},
            {"name": "qwen2.5:7b", "size": 4700000000, "details": {"parameter_size": "7.6B"}},
            {"name": "moondream:latest", "size": 1700000000},
        ]
    }
    transport, seen = capture(httpx.Response(200, json=tags))

    models = OllamaProvider(transport=transport).list_models()

    assert str(seen[0].url) == "http://ollama.test/api/tags"
    assert [m.to_dict() for m in models] == [
        {"name": "llava-phi3:latest", "size": 2900000000, "parameter_size": "4B"},
        {"name": "moondream:latest", "size": 1700000000, "parameter_size": None},
    ]
