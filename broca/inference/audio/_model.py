"""Bark Small text-to-speech model configuration."""

from ...ai_model import AiModel

# https://huggingface.co/suno/bark-small
_model: AiModel = AiModel(
    id="suno/bark-small",
    name="Bark Small",
    created_at=1_689_688_246,
    type="audio",
    family="bark",
    model_engine="transformers",
    model_format="pytorch",
    revision="1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
    allow_patterns=[
        "*.json",
        "pytorch_model.bin",
        "vocab.txt",
        "speaker_embeddings/*.npy",
        "README.md",
    ],
    estimated_weights_bytes=1_676_663_913,
    dtype="float32",
    context_window=256,
    runtime_options={"sample_rate": 24000},
)
