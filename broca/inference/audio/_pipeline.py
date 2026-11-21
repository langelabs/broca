"""Generate encoded speech with the configured Bark model."""

from contextlib import nullcontext
from io import BytesIO

import av
import torch

from .._pipelines import BarkPipeline
from ._model import _model
from ._types import AudioOptions, AudioResult, AudioVoice


class AudioPipeline(BarkPipeline[str, AudioResult]):
    """Synthesize speech using the configured Bark runtime."""

    async def _infer(
        self,
        text: str,
        *,
        voice: AudioVoice | None = None,
        options: AudioOptions | None = None,
    ) -> AudioResult:
        """Synthesize speech with optional voice conditioning.

        :param text: Nonblank text to synthesize.
        :param voice: Optional Bark voice preset.
        :param options: Optional overrides of configured generation defaults.
        :returns: Encoded audio with format and playback metadata.
        :raises ValueError: If the text is blank or the preset is not built in.
        """
        runtime = self.runtime
        if not text.strip():
            raise ValueError("Speech text must not be blank.")
        options = options or AudioOptions()
        preset = voice.preset if voice else None
        if preset is not None:
            presets = runtime.processor.speaker_embeddings or {}
            if preset == "repo_or_path" or preset not in presets:
                raise ValueError("Select a built-in Bark voice preset.")

        device = runtime.model.device
        inputs = runtime.processor(text, voice_preset=preset, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        generation = {}
        if options.semantic_max_new_tokens is not None:
            generation["semantic_max_new_tokens"] = options.semantic_max_new_tokens
        if options.temperature == 0:
            generation["semantic_do_sample"] = False
        elif options.temperature is not None:
            generation["semantic_temperature"] = options.temperature

        seed = options.seed if options.seed is not None else self.model.seed
        rng = (
            torch.random.fork_rng(
                devices=[] if device.type == "cpu" else [device],
                device_type=device.type,
            )
            if seed is not None
            else nullcontext()
        )
        with rng, torch.inference_mode():
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if device.type == "mps":
                    torch.mps.manual_seed(seed)
                elif device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
            audio = runtime.model.generate(**inputs, **generation)

        samples = audio.detach().cpu().float().reshape(1, -1)
        sample_rate = int(runtime.model.generation_config.to_dict()["sample_rate"])
        pcm = (samples.clamp(-1, 1) * 32767).round().short().numpy()
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = sample_rate
        output = BytesIO()
        with av.open(output, "w", format=options.output_format) as container:
            codec = "flac" if options.output_format == "flac" else "pcm_s16le"
            stream = container.add_stream(codec, rate=sample_rate, layout="mono")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        return AudioResult(
            data=output.getvalue(),
            media_type="audio/flac" if options.output_format == "flac" else "audio/wav",
            sample_rate=sample_rate,
            channels=1,
            duration_seconds=frame.samples / sample_rate,
            seed=seed,
        )


pipeline = AudioPipeline(_model)
