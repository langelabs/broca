"""Prepare Gemma conversation messages and normalize generated text."""

from collections.abc import Sequence

from llama_cpp import llama_cpp as native
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPart,
    CompletionUsage,
)

from ._types import (
    ImageAttachment,
    TextAttachment,
    TextMessage,
    TextOptions,
    TextRequest,
    TextResult,
    TokenUsage,
)


def prepare_request(
    prompt: str | TextRequest,
    *,
    system_prompt: str | None = None,
    history: Sequence[TextMessage] | None = None,
    options: TextOptions | None = None,
    attachments: Sequence[TextAttachment] = (),
) -> TextRequest:
    """Normalize a text call into one request carrying its own settings.

    :param prompt: User prompt or a request that carries its own settings.
    :param system_prompt: Optional instructions for a string prompt.
    :param history: Previous turns for a string prompt.
    :param options: Optional generation overrides for a string prompt.
    :param attachments: Inline images and audio for a string prompt.
    :returns: The assembled request.
    :raises ValueError: If a prepared request is combined with separate settings.
    """
    if isinstance(prompt, TextRequest):
        if (
            system_prompt is not None
            or history is not None
            or options is not None
            or attachments
        ):
            raise ValueError("A text request already contains its generation settings.")
        return prompt
    return TextRequest(prompt, system_prompt, history, options, attachments)


def add_batch_token(
    batch: native.llama_batch, *, token: int, position: int, slot: int, output: bool
) -> int:
    """Write a token row for one sequence into an allocated native batch.

    :param batch: Native batch with room for one more row.
    :param token: Model token ID.
    :param position: Token position within this sequence.
    :param slot: Native sequence ID.
    :param output: Whether decoding should produce logits for this row.
    :returns: Row index in the native batch.
    """
    row = batch.n_tokens
    batch.token[row] = token
    batch.pos[row] = position
    batch.n_seq_id[row] = 1
    batch.seq_id[row][0] = slot
    batch.logits[row] = int(output)
    batch.n_tokens += 1
    return row


def prepare_messages(
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: Sequence[TextMessage] | None = None,
    attachments: Sequence[TextAttachment] = (),
) -> list[ChatCompletionRequestMessage]:
    """Build an ordered conversation with inline media on the current turn.

    :param prompt: Current user prompt.
    :param system_prompt: Optional instructions prepended to the conversation.
    :param history: Prior turns in chronological order.
    :param attachments: Images and audio accompanying the current user prompt.
    :returns: Messages ready for the chat template or multimodal handler.
    """
    messages: list[ChatCompletionRequestMessage] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    for message in history or ():
        if message.role == "assistant":
            messages.append({"role": "assistant", "content": message.content})
        else:
            messages.append({"role": "user", "content": message.content})
    if not attachments:
        messages.append({"role": "user", "content": prompt})
        return messages

    content: list[ChatCompletionRequestMessageContentPart] = []
    for attachment in attachments:
        if isinstance(attachment, ImageAttachment):
            media_type = f"image/{attachment.format}"
        else:
            media_type = "audio/mpeg" if attachment.format == "mp3" else "audio/wav"
        # The MTMD handler expects image_url parts for both images and audio.
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{attachment.data}"},
            }
        )
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


def build_text_result(
    text: str, *, finish_reason: str | None, usage: CompletionUsage
) -> TextResult:
    """Split Gemma's leading thought channel from its answer.

    :param text: Generated text from the completion backend.
    :param finish_reason: Backend reason for ending generation.
    :param usage: Token counts reported by the backend.
    :returns: The answer, optional reasoning, normalized finish reason, and usage.
    """
    reasoning = None
    if text.startswith("<|channel>thought"):
        thought, _, answer = text.removeprefix("<|channel>thought").partition(
            "<channel|>"
        )
        reasoning = thought.strip()
        text = answer.lstrip()
    return TextResult(
        text=text,
        reasoning=reasoning,
        finish_reason="length" if finish_reason == "length" else "stop",
        usage=TokenUsage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        ),
    )
