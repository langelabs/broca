"""FLUX 3 Image Klein distilled GGUF model and pipeline configuration."""

from ...ai_model import AiModel

# https://huggingface.co/unsloth/flux-3-image-klein-GGUF
# The primary snapshot supplies the modular pipeline, Qwen3-VL encoder, and VAE.
# Only the transformer is replaced by the selected Unsloth GGUF checkpoint.
_model: AiModel = AiModel(
    id="black-forest-labs/flux-3-image-klein",
    name="FLUX 3 Image Klein",
    created_at=1_791_889_120,
    type="image",
    family="flux3",
    model_engine="diffusers",
    model_format="gguf",
    quantization="Q4_K_M",
    weights_repo="unsloth/flux-3-image-klein-GGUF",
    file="flux-3-image-klein-Q4_K_M.gguf",
    revision="09561231f44e41381c461caf53d5452005a60ac9",
    weights_revision="12ef13ce920d53f45fe805093910c899668b9e76",
    allow_patterns=[
        "modular_model_index.json",
        "transformer/config.json",
        "text_encoder/*",
        "vae/*",
        "README.md",
        "LICENSE.md",
        "NOTICE.md",
    ],
    estimated_weights_bytes=3_666_908_064,
    # Published BF16 text encoder shards and the Diffusers VAE, without duplicates.
    supporting_weights_bytes=9_211_932_900,
    dtype="bfloat16",
    runtime_options={
        "pipeline_class": "Flux3DistilledModularPipeline",
        "transformer_class": "Flux3Transformer2DModel",
        "transformer_subfolder": "transformer",
        "quantization_config": {"compute_dtype": "bfloat16"},
    },
    # The distilled pipeline uses 16 flow-map steps without classifier-free guidance.
    generation_options={
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 16,
        "output": "images",
    },
)
