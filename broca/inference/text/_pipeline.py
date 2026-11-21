"""Generate responses with the configured Gemma chat model."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from secrets import randbelow
from typing import cast

from llama_cpp import llama_cpp as native
from llama_cpp.llama_chat_format import Gemma4ChatHandler, Jinja2ChatFormatter

from .._pipelines import LlamaPipeline
from .._pipelines.__base import InferenceCall
from ._batch import BatchLlama, CompletionOptions, TextSequence
from ._model import _model
from ._types import TextAttachment, TextMessage, TextOptions, TextRequest, TextResult
from ._utils import (
    add_batch_token,
    build_text_result,
    prepare_messages,
    prepare_request,
)


class TextPipeline(LlamaPipeline[str | TextRequest, TextResult]):
    """Generate chat responses using the configured Gemma runtime."""

    _runtime_class = BatchLlama

    async def _infer(
        self,
        prompt: str | TextRequest,
        *,
        system_prompt: str | None = None,
        history: list[TextMessage] | None = None,
        options: TextOptions | None = None,
        attachments: Sequence[TextAttachment] = (),
    ) -> TextResult:
        """Generate one response through the batch scheduler.

        :param prompt: User prompt or a request that carries its own settings.
        :param system_prompt: Optional instructions for a string prompt.
        :param history: Previous turns in chronological order for a string prompt.
        :param options: Optional generation overrides for a string prompt.
        :param attachments: Inline images and audio for a string prompt.
        :returns: The result of the single request.
        :raises ValueError: If a prepared request is combined with separate settings.
        """
        request = prepare_request(
            prompt,
            system_prompt=system_prompt,
            history=history,
            options=options,
            attachments=attachments,
        )
        (result,) = await self._infer_batched([request])
        return result

    async def _infer_batched(
        self, inputs: Sequence[str | TextRequest | InferenceCall]
    ) -> list[TextResult]:
        """Decode requests with separate sampler and sequence state in shared batches.

        The inference worker owns the compute lock for the duration of this hook.

        :param inputs: Conversation turns, plain prompts, or queued call arguments.
        :returns: One completion per input, in input order.
        :raises ValueError: If a prompt is blank or cannot fit in the model context.
        :raises RuntimeError: If the loaded backend cannot complete the batch.
        """
        if not inputs:
            return []

        runtime = self.runtime
        config = self.model
        defaults = config.generation_options or {}
        waiting: deque[TextSequence] = deque()
        results: dict[int, TextResult] = {}
        formatter: Jinja2ChatFormatter | None = None

        for index, item in enumerate(inputs):
            request = (
                prepare_request(*item.args, **item.kwargs)
                if isinstance(item, InferenceCall)
                else prepare_request(item)
            )
            if not request.prompt.strip():
                raise ValueError("Text prompt must not be blank.")
            options = request.options or TextOptions()
            thinking = (
                config.enable_thinking
                if options.enable_thinking is None
                else options.enable_thinking
            )
            settings = CompletionOptions(
                max_tokens=options.max_tokens
                or cast(int | None, defaults.get("max_tokens")),
                temperature=(
                    options.temperature
                    if options.temperature is not None
                    else cast(float, defaults.get("temperature", 1.0))
                ),
                top_p=options.top_p or cast(float, defaults.get("top_p", 0.95)),
                top_k=options.top_k
                if options.top_k is not None
                else cast(int, defaults.get("top_k", 64)),
                stop=list(options.stop or ()),
                seed=options.seed if options.seed is not None else config.seed,
            )
            messages = prepare_messages(
                request.prompt,
                system_prompt=request.system_prompt,
                history=request.history,
                attachments=request.attachments,
            )
            if request.attachments:
                handler = runtime.chat_handler
                if not isinstance(handler, Gemma4ChatHandler):
                    raise RuntimeError(
                        "The Gemma runtime has no multimodal chat handler."
                    )
                response = handler(
                    llama=runtime,
                    messages=messages,
                    enable_thinking=bool(thinking),
                    **settings,
                )
                if not isinstance(response, dict):
                    raise RuntimeError("Expected a complete Gemma chat response.")
                choice = response["choices"][0]
                results[index] = build_text_result(
                    choice["message"]["content"] or "",
                    finish_reason=choice["finish_reason"],
                    usage=response["usage"],
                )
                continue

            if formatter is None:
                template = config.chat_template or runtime.metadata.get(
                    "tokenizer.chat_template"
                )
                if not isinstance(template, str) or not template:
                    raise RuntimeError("The Gemma checkpoint has no chat template.")
                formatter = Jinja2ChatFormatter(
                    template=template,
                    bos_token=runtime.detokenize(
                        [runtime.token_bos()], special=True
                    ).decode(),
                    eos_token=runtime.detokenize(
                        [runtime.token_eos()], special=True
                    ).decode(),
                )

            formatted = formatter(messages=messages, enable_thinking=bool(thinking))
            tokens = runtime.tokenize(
                formatted.prompt.encode(),
                add_bos=not formatted.added_special,
                special=True,
            )
            if not tokens:
                raise ValueError("The Gemma chat template produced an empty prompt.")
            seed = settings["seed"]
            seed = native.LLAMA_DEFAULT_SEED if seed is None else seed & 0xFFFFFFFF
            if seed == native.LLAMA_DEFAULT_SEED:
                seed = randbelow(native.LLAMA_DEFAULT_SEED)
            waiting.append(
                TextSequence(
                    index=index,
                    prompt=tokens,
                    options=settings,
                    stop=tuple(
                        stop.encode()
                        for stop in [
                            *filter(None, formatted.stop or ()),
                            *settings["stop"],
                        ]
                    ),
                    seed=seed,
                )
            )

        if not waiting:
            return [results[index] for index in range(len(inputs))]

        ctx = runtime.ctx
        memory = native.llama_get_memory(ctx)
        vocab = native.llama_model_get_vocab(runtime.model)
        if memory is None or vocab is None:
            raise RuntimeError(
                "The Gemma runtime lacks decoder memory or a vocabulary."
            )
        context_size = min(native.llama_n_ctx(ctx), native.llama_n_ctx_seq(ctx))
        capacity = native.llama_n_batch(ctx)
        slot_count = min(native.llama_n_seq_max(ctx), capacity, len(waiting))
        if slot_count < 1:
            raise RuntimeError("The Gemma runtime has no decoder slots.")
        stop_ids = frozenset(config.stop_token_ids or ())
        for sequence in waiting:
            remaining = context_size - len(sequence.prompt)
            if remaining < 1:
                raise ValueError(
                    f"Text prompt at index {sequence.index} exceeds the context window."
                )
            limit = sequence.options["max_tokens"]
            sequence.max_tokens = (
                min(limit, remaining) if limit is not None and limit > 0 else remaining
            )

        runtime.reset()
        native.llama_memory_clear(memory, False)
        batch = native.llama_batch_init(capacity, 0, 1)
        active: dict[int, tuple[TextSequence, native.llama_sampler_p]] = {}
        try:
            while waiting or active:
                for slot in range(slot_count):
                    if not waiting:
                        break
                    if slot in active:
                        continue
                    sequence = waiting.popleft()
                    sampler = native.llama_sampler_chain_init(
                        native.llama_sampler_chain_default_params()
                    )
                    if not sampler:
                        raise RuntimeError("Could not allocate a text sampler.")
                    active[slot] = (sequence, sampler)
                    settings = sequence.options
                    if settings["temperature"] == 0:
                        native.llama_sampler_chain_add(
                            sampler, native.llama_sampler_init_greedy()
                        )
                    else:
                        for stage in (
                            native.llama_sampler_init_top_k(settings["top_k"]),
                            native.llama_sampler_init_top_p(settings["top_p"], 1),
                            native.llama_sampler_init_min_p(0.05, 1),
                            native.llama_sampler_init_temp(settings["temperature"]),
                            native.llama_sampler_init_dist(sequence.seed),
                        ):
                            native.llama_sampler_chain_add(sampler, stage)

                batch.n_tokens = 0
                outputs: dict[int, int] = {}
                prefilling: list[tuple[int, TextSequence]] = []
                for slot, (sequence, _) in active.items():
                    if sequence.position < len(sequence.prompt):
                        prefilling.append((slot, sequence))
                        continue
                    outputs[slot] = add_batch_token(
                        batch,
                        token=sequence.generated[-1],
                        position=sequence.position,
                        slot=slot,
                        output=True,
                    )
                    sequence.position += 1

                for offset, (slot, sequence) in enumerate(prefilling):
                    available = (capacity - batch.n_tokens) // (
                        len(prefilling) - offset
                    )
                    end = min(sequence.position + available, len(sequence.prompt))
                    for position in range(sequence.position, end):
                        output = position == len(sequence.prompt) - 1
                        row = add_batch_token(
                            batch,
                            token=sequence.prompt[position],
                            position=position,
                            slot=slot,
                            output=output,
                        )
                        if output:
                            outputs[slot] = row
                    sequence.position = end

                status = native.llama_decode(ctx, batch)
                if status == 1 and len(active) > 1:
                    # Competing sequences filled the shared cache. Replay unfinished
                    # requests with their original seeds and the full context each.
                    for sequence, sampler in reversed(active.values()):
                        native.llama_sampler_free(sampler)
                        sequence.position = 0
                        sequence.generated.clear()
                        waiting.appendleft(sequence)
                    active.clear()
                    native.llama_memory_clear(memory, False)
                    slot_count = 1
                    continue
                if status != 0:
                    raise RuntimeError(
                        f"Gemma batch decode failed with status {status}."
                    )

                finished: list[int] = []
                for slot, row in outputs.items():
                    sequence, sampler = active[slot]
                    token = native.llama_sampler_sample(sampler, ctx, row)
                    stopped = token in stop_ids or native.llama_vocab_is_eog(
                        vocab, token
                    )
                    if not stopped:
                        sequence.generated.append(token)
                    text = runtime.detokenize(
                        sequence.generated, prev_tokens=sequence.prompt
                    )
                    for stop in sequence.stop:
                        end = text.find(stop)
                        if end >= 0:
                            text = text[:end]
                            stopped = True
                            break
                    generated_count = len(sequence.generated)
                    if not stopped and generated_count < sequence.max_tokens:
                        continue

                    prompt_count = len(sequence.prompt)
                    results[sequence.index] = build_text_result(
                        text.decode(errors="ignore"),
                        finish_reason="stop" if stopped else "length",
                        usage={
                            "prompt_tokens": prompt_count,
                            "completion_tokens": generated_count,
                            "total_tokens": prompt_count + generated_count,
                        },
                    )
                    finished.append(slot)

                for slot in finished:
                    if not native.llama_memory_seq_rm(memory, slot, -1, -1):
                        raise RuntimeError(
                            "Could not release a Gemma sequence from decoder memory."
                        )
                    _, sampler = active.pop(slot)
                    native.llama_sampler_free(sampler)
        finally:
            for _, sampler in active.values():
                native.llama_sampler_free(sampler)
            native.llama_batch_free(batch)
            native.llama_memory_clear(memory, False)

        return [results[index] for index in range(len(inputs))]


pipeline = TextPipeline(_model)
