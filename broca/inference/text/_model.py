"""Gemma 4 E4B instruction model and multimodal projector configuration."""

from ...ai_model import AiModel

# https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
# Keep the embedded GGUF chat template and the repository's BF16 projector.
_model: AiModel = AiModel(
    id="unsloth/gemma-4-E4B-it-GGUF",
    name="Gemma 4 E4B IT",
    created_at=1_775_054_524,
    type="text",
    family="gemma4",
    model_engine="llama.cpp",
    model_format="gguf",
    quantization="Q4_K_M",
    file="gemma-4-E4B-it-Q4_K_M.gguf",
    clip="mmproj-BF16.gguf",
    revision="bfc15c382204943c3a8fff0c750b94ae2364d7a3",
    estimated_weights_bytes=4_977_171_584,
    projector_bytes=991_552_320,
    context_window=131072,
    enable_thinking=False,
    stop_token_ids=[1, 106],
    gpu_layers=-1,
    use_mmap=True,
    runtime_options={
        "chat_handler": "llama_cpp.llama_chat_format.Gemma4ChatHandler",
        "verbose": False,
    },
    generation_options={
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    },
)
