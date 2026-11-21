"""Bekko typed Choice, Noul, and Score decision model configuration."""

from ...ai_model import AiModel

# https://huggingface.co/hotchpotch/bekko-system-one-v0-400m
# Load the pinned custom class and call predict(); encode() is not the typed API.
_model: AiModel = AiModel(
    id="hotchpotch/bekko-system-one-v0-400m",
    name="Bekko System One v0 400M",
    created_at=1_790_683_659,
    type="decision",
    family="modernbert",
    languages=["en"],
    model_engine="sentence-transformers",
    model_format="safetensors",
    revision="1960df5602bd93cc8d336fdebd9fb68a30926e13",
    ignore_patterns=["onnx_browser/*"],
    estimated_weights_bytes=1_579_159_388,
    dtype="float32",
    context_window=7999,
    trust_remote_code=True,
    runtime_options={
        "model_class": "inference_v0.BekkoSentenceTransformer",
        "attn_implementation": "sdpa",
    },
    generation_options={
        "batch_size": 128,
        "token_budget": 64000,
        "context_length": 7999,
        "query_length": 7997,
        "document_length": 3800,
        "prefix_layout": "instruction_state",
        "show_progress_bar": False,
    },
)
