from typing import Any, AsyncIterator, Optional

import pytest

# Skip these tests unless google-adk is installed
pytest.importorskip("google.adk", reason="google-adk extra not installed")

from google.adk.models.llm_request import LlmRequest  # type: ignore
from google.genai import types as genai_types  # type: ignore

from portkey_ai.integrations.adk import PortkeyAdk


class _FakeDelta:
    def __init__(
        self, content: Optional[str] = None, reasoning_content: Optional[str] = None
    ):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeMessage:
    def __init__(
        self, content: Optional[str] = None, reasoning_content: Optional[str] = None
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = None


class _FakeChoice:
    def __init__(
        self,
        delta: Optional[_FakeDelta] = None,
        message: Optional[_FakeMessage] = None,
        finish_reason: Optional[str] = None,
    ):
        self.delta = delta
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, message_text: str, reasoning_content: Optional[str] = None):
        self.choices = [
            _FakeChoice(
                message=_FakeMessage(message_text, reasoning_content=reasoning_content),
                finish_reason="stop",
            )
        ]
        self.usage = type(
            "Usage", (), {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        )()


def _build_request(model: str, text: str = "Hello") -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=text)],
            )
        ],
    )


@pytest.mark.asyncio
async def test_non_streaming_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = PortkeyAdk(model="@openai/gpt-4o-mini", api_key="test")

    async def fake_create(**kwargs: Any) -> _FakeResponse:
        assert not kwargs.get("stream"), "Non-streaming path should not request stream"
        return _FakeResponse(message_text="Hello world!")

    # Patch the underlying client create method
    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)  # type: ignore[attr-defined]

    req = _build_request(model="@openai/gpt-4o-mini", text="test")
    outputs: list[str] = []
    async for resp in llm.generate_content_async(req, stream=False):
        assert not getattr(resp, "partial", False)
        assert resp.content and resp.content.parts
        for p in resp.content.parts:
            text = getattr(p, "text", None)
            if text:
                outputs.append(text)
    assert "".join(outputs).strip() == "Hello world!"


@pytest.mark.asyncio
async def test_streaming_accumulates_and_final(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = PortkeyAdk(model="@openai/gpt-4o-mini", api_key="test")

    part1 = type("Chunk", (), {"choices": [_FakeChoice(delta=_FakeDelta("Hello "))]})()
    part2 = type("Chunk", (), {"choices": [_FakeChoice(delta=_FakeDelta("world!"))]})()
    part3 = type(
        "Chunk",
        (),
        {"choices": [_FakeChoice(message=_FakeMessage(None), finish_reason="stop")]},
    )()

    async def fake_stream_gen() -> AsyncIterator[Any]:
        yield part1
        yield part2
        yield part3

    async def fake_create(**kwargs: Any) -> AsyncIterator[Any] | _FakeResponse:
        if kwargs.get("stream"):
            return fake_stream_gen()
        return _FakeResponse(message_text="unused")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)  # type: ignore[attr-defined]

    req = _build_request(model="@openai/gpt-4o-mini", text="test")
    partial_text = []
    final_text = []

    async for resp in llm.generate_content_async(req, stream=True):
        assert resp.content and resp.content.parts
        text_parts = [
            p.text for p in resp.content.parts if getattr(p, "text", None) is not None
        ]
        if getattr(resp, "partial", False):
            partial_text.extend([t for t in text_parts if t])
        else:
            final_text.extend([t for t in text_parts if t])

    assert "".join(partial_text) == "Hello world!"
    assert "".join(final_text) == "Hello world!"


@pytest.mark.asyncio
async def test_non_streaming_with_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = PortkeyAdk(model="@openai/gpt-4o-mini", api_key="test")

    async def fake_create(**kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            message_text="The answer is 42.",
            reasoning_content="Let me think about this step by step...",
        )

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)  # type: ignore[attr-defined]

    req = _build_request(model="@openai/gpt-4o-mini", text="test")
    thought_parts: list[str] = []
    text_parts: list[str] = []

    async for resp in llm.generate_content_async(req, stream=False):
        assert resp.content and resp.content.parts
        for p in resp.content.parts:
            text = getattr(p, "text", None)
            if getattr(p, "thought", False) and text:
                thought_parts.append(text)
            elif text:
                text_parts.append(text)

    assert "".join(thought_parts) == "Let me think about this step by step..."
    assert "".join(text_parts) == "The answer is 42."


@pytest.mark.asyncio
async def test_streaming_with_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = PortkeyAdk(model="@openai/gpt-4o-mini", api_key="test")

    part1 = type(
        "Chunk",
        (),
        {"choices": [_FakeChoice(delta=_FakeDelta(reasoning_content="Thinking..."))]},
    )()
    part2 = type(
        "Chunk", (), {"choices": [_FakeChoice(delta=_FakeDelta(content="Answer: 42"))]}
    )()
    part3 = type(
        "Chunk",
        (),
        {"choices": [_FakeChoice(message=_FakeMessage(None), finish_reason="stop")]},
    )()

    async def fake_stream_gen() -> AsyncIterator[Any]:
        yield part1
        yield part2
        yield part3

    async def fake_create(**kwargs: Any) -> AsyncIterator[Any] | _FakeResponse:
        if kwargs.get("stream"):
            return fake_stream_gen()
        return _FakeResponse(message_text="unused")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)  # type: ignore[attr-defined]

    req = _build_request(model="@openai/gpt-4o-mini", text="test")
    partial_thoughts: list[str] = []
    partial_text: list[str] = []
    final_thoughts: list[str] = []
    final_text: list[str] = []

    async for resp in llm.generate_content_async(req, stream=True):
        assert resp.content and resp.content.parts
        for p in resp.content.parts:
            text = getattr(p, "text", None)
            if getattr(p, "thought", False) and text:
                if getattr(resp, "partial", False):
                    partial_thoughts.append(text)
                else:
                    final_thoughts.append(text)
            elif text:
                if getattr(resp, "partial", False):
                    partial_text.append(text)
                else:
                    final_text.append(text)

    assert "".join(partial_thoughts) == "Thinking..."
    assert "".join(partial_text) == "Answer: 42"
    assert "".join(final_thoughts) == "Thinking..."
    assert "".join(final_text) == "Answer: 42"


@pytest.mark.asyncio
async def test_non_streaming_without_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = PortkeyAdk(model="@openai/gpt-4o-mini", api_key="test")

    async def fake_create(**kwargs: Any) -> _FakeResponse:
        return _FakeResponse(message_text="Simple response")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)  # type: ignore[attr-defined]

    req = _build_request(model="@openai/gpt-4o-mini", text="test")
    thought_parts: list[str] = []
    text_parts: list[str] = []

    async for resp in llm.generate_content_async(req, stream=False):
        assert resp.content and resp.content.parts
        for p in resp.content.parts:
            text = getattr(p, "text", None)
            if getattr(p, "thought", False) and text:
                thought_parts.append(text)
            elif text:
                text_parts.append(text)

    assert thought_parts == []
    assert "".join(text_parts) == "Simple response"
