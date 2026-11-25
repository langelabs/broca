"""Exercise Gemma's text and multimodal inference entry point."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from llama_cpp import llama_cpp as native
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from pydantic import ValidationError

from broca.inference.text import types, _utils
from broca.inference.text._pipeline import pipeline


@pytest.mark.parametrize("with_media", [False, True])
def test_prepare_messages_preserves_conversation_order(with_media: bool) -> None:
    """Keep prior turns intact and put inline media in the current turn.

    :param with_media: Whether to submit an image and an MP3 attachment.
    """
    attachments: list[types.TextAttachment] = (
        [
            types.ImageAttachment(type="image", format="jpeg", data="AQID"),
            types.AudioAttachment(type="input_audio", format="mp3", data="BAUG"),
        ]
        if with_media
        else []
    )
    messages = _utils.prepare_messages(
        "Current question",
        system_prompt="Instructions",
        history=[
            types.TextMessage(role="user", content="Previous question"),
            types.TextMessage(role="assistant", content="Previous answer"),
        ],
        attachments=attachments,
    )

    assert messages[:3] == [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    if with_media:
        assert messages[3:] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AQID"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:audio/mpeg;base64,BAUG"},
                    },
                    {"type": "text", "text": "Current question"},
                ],
            }
        ]
    else:
        assert messages[3:] == [{"role": "user", "content": "Current question"}]


@pytest.mark.parametrize(
    ("generated", "finish_reason", "expected_text", "expected_reasoning"),
    [
        ("  Plain answer", "stop", "  Plain answer", None),
        ("<|channel>thought Reason <channel|>  Answer", "stop", "Answer", "Reason"),
        ("<|channel>thought Unfinished ", "length", "", "Unfinished"),
    ],
)
def test_build_text_result_separates_gemma_reasoning(
    generated: str,
    finish_reason: str,
    expected_text: str,
    expected_reasoning: str | None,
) -> None:
    """Expose the generated answer and reasoning in separate fields.

    :param generated: Raw text from either Gemma completion path.
    :param finish_reason: Native reason for ending generation.
    :param expected_text: Answer expected in the result.
    :param expected_reasoning: Reasoning expected in the result.
    """
    result = _utils.build_text_result(
        generated,
        finish_reason=finish_reason,
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )

    assert result.text == expected_text
    assert result.reasoning == expected_reasoning
    assert result.finish_reason == finish_reason
    assert result.usage == types.TokenUsage(prompt_tokens=3, completion_tokens=2)


def test_gemma_passes_images_and_audio_to_the_multimodal_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate typed attachments and expose the generated chat result.

    :param monkeypatch: Replace the loaded native runtime with a call recorder.
    """
    handler = Mock(
        spec=Gemma4ChatHandler,
        return_value={
            "choices": [
                {"message": {"content": "Media answer"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 2},
        },
    )
    runtime = Mock(chat_handler=handler)
    monkeypatch.setattr(pipeline, "_runtime", runtime)

    result = asyncio.run(
        pipeline.infer(
            "Describe both files.",
            attachments=[
                types.ImageAttachment(type="image", format="png", data="AQID"),
                types.AudioAttachment(type="input_audio", format="wav", data="BAUG"),
            ],
            options=types.TextOptions(
                enable_thinking=True, max_tokens=2, stop=["END"]
            ),
        )
    )

    assert result.text == "Media answer"
    assert result.finish_reason == "length"
    assert result.usage == types.TokenUsage(prompt_tokens=120, completion_tokens=2)
    assert handler.call_args.kwargs["messages"][-1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}},
        {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,BAUG"}},
        {"type": "text", "text": "Describe both files."},
    ]
    assert handler.call_args.kwargs["enable_thinking"] is True
    assert handler.call_args.kwargs["max_tokens"] == 2
    assert "END" in handler.call_args.kwargs["stop"]


def test_single_inference_uses_one_batch_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the single-request options through the same internal hook.

    :param monkeypatch: Capture calls at the private batch boundary.
    """
    expected = types.TextResult(text="Answer", finish_reason="stop")
    batched = AsyncMock(return_value=[expected])
    monkeypatch.setattr(pipeline, "_infer_batched", batched)

    inference = asyncio.run(
        pipeline._infer(
            "Question", system_prompt="Instructions", options=types.TextOptions(seed=0)
        )
    )
    assert inference == expected
    batched.assert_awaited_once()
    (requests,) = batched.call_args.args
    (request,) = requests
    assert request.prompt == "Question"
    assert request.system_prompt == "Instructions"
    assert request.options.seed == 0


@pytest.mark.parametrize(
    "decode_status", [0, 1, -2], ids=["success", "cache_full", "failure"]
)
def test_batch_decodes_distinct_sequences_together_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch, decode_status: int
) -> None:
    """Exercise prompt chunking, output-row routing, slot reuse, and recovery.

    :param monkeypatch: Simulate model outputs at the native decoding boundary.
    :param decode_status: Status returned on the third native decode.
    """
    batches: list[list[tuple[int, int, int, bool]]] = []
    runtime = Mock(metadata={"tokenizer.chat_template": "{{ messages[-1].content }}"})
    runtime.tokenize.side_effect = lambda prompt, **_: {
        b"Long prompt": [11, 12, 13],
        b"Short": [20],
        b"Next": [30],
    }[prompt]
    runtime.detokenize.side_effect = lambda tokens, **_: {
        (101,): b"Long",
        (102,): b"Short",
        (103,): b"Next",
    }.get(tuple(tokens), b"<eos>")
    monkeypatch.setattr(pipeline, "_runtime", runtime)
    monkeypatch.setattr(native, "llama_get_memory", Mock(return_value=object()))
    clear = Mock()
    monkeypatch.setattr(native, "llama_memory_clear", clear)
    remove = Mock(return_value=True)
    monkeypatch.setattr(native, "llama_memory_seq_rm", remove)
    monkeypatch.setattr(native, "llama_n_ctx", Mock(return_value=32))
    monkeypatch.setattr(native, "llama_n_ctx_seq", Mock(return_value=32))
    monkeypatch.setattr(native, "llama_n_batch", Mock(return_value=3))
    monkeypatch.setattr(native, "llama_n_seq_max", Mock(return_value=2))
    monkeypatch.setattr(native, "llama_model_get_vocab", Mock(return_value=object()))
    monkeypatch.setattr(
        native, "llama_vocab_is_eog", Mock(side_effect=lambda _, token: token == 1)
    )

    def decode(_context: object, batch: native.llama_batch) -> int:
        """Record the tokens, sequence positions and output masks sent to llama.cpp.

        :param _context: Fake model context.
        :param batch: Native token rows filled by the scheduler.
        :returns: The configured status for the third decode or success.
        """
        batches.append(
            [
                (
                    batch.seq_id[i][0],
                    batch.pos[i],
                    batch.token[i],
                    bool(batch.logits[i]),
                )
                for i in range(batch.n_tokens)
            ]
        )
        return decode_status if len(batches) == 3 else 0

    def sample(_sampler: object, _context: object, row: int) -> int:
        """Return a token specific to the requested output row.

        :param _sampler: Request's sampler.
        :param _context: Fake model context.
        :param row: Index into the most recently decoded batch.
        :returns: A generated token or end-of-generation marker.
        """
        _, _, token, output = batches[-1][row]
        assert output
        return {13: 101, 20: 102, 30: 103}.get(token, 1)

    monkeypatch.setattr(native, "llama_decode", decode)
    monkeypatch.setattr(native, "llama_sampler_sample", sample)

    async def collect() -> list[types.TextResult]:
        """Submit options, a keyword prompt, and a prepared request together.

        :returns: The caller's results in submission order.
        """
        return await asyncio.gather(
            pipeline.infer("Long prompt", options=types.TextOptions(temperature=0)),
            pipeline.infer(prompt="Short"),
            pipeline.infer(types.TextRequest("Next")),
        )

    if decode_status < 0:
        with pytest.raises(RuntimeError, match="status -2"):
            asyncio.run(collect())
        assert clear.call_count == 2
        assert asyncio.run(pipeline.infer("Next")).text == "Next"
        return

    results = asyncio.run(collect())

    assert [result.text for result in results] == ["Long", "Short", "Next"]
    assert [result.usage for result in results] == [
        types.TokenUsage(prompt_tokens=count, completion_tokens=1)
        for count in [3, 1, 1]
    ]
    assert any(len({seq for seq, *_ in batch}) == 2 for batch in batches)
    assert [row for batch in batches for row in batch if row[2] == 30][0][1] == 0
    assert remove.call_count == 3
    if decode_status == 1:
        assert clear.call_count == 3
        assert sum(row[2] == 20 for batch in batches for row in batch) == 1
        assert sum(row[2] == 11 for batch in batches for row in batch) == 2


def test_gemma_attachments_require_inline_base64_data() -> None:
    """Reject external URLs and malformed payloads before they reach the decoder."""
    with pytest.raises(ValidationError, match="base64"):
        types.ImageAttachment(type="image", format="png", data="https://example.com")
