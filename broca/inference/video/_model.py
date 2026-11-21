"""Wan 2.2 TI2V 5B GGUF transformer and companion pipeline configuration."""

from ...ai_model import AiModel

# https://huggingface.co/unsloth/Wan2.2-TI2V-5B-GGUF
# Download the Diffusers tokenizer, UMT5 encoder, scheduler, and FP32 VAE;
# its unquantized transformer is replaced by the selected Unsloth GGUF.
_model: AiModel = AiModel(
    id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    name="Wan 2.2 TI2V 5B",
    created_at=1_753_693_409,
    type="video",
    family="wan",
    parameter_count_billions=5.0,
    model_engine="diffusers",
    model_format="gguf",
    quantization="Q4_K_M",
    weights_repo="unsloth/Wan2.2-TI2V-5B-GGUF",
    file="Wan2.2-TI2V-5B-Q4_K_M.gguf",
    revision="b8fff7315c768468a5333511427288870b2e9635",
    weights_revision="b5c2a3816e7056e57e200f6be726fb36ce523b49",
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/config.json",
        "vae/*",
        "README.md",
    ],
    estimated_weights_bytes=3_433_116_000,
    supporting_weights_bytes=14_180_629_016,
    dtype="bfloat16",
    cpu_offload=True,
    vae_tiling=True,
    runtime_options={
        "pipeline_class": "WanPipeline",
        "transformer_class": "WanTransformer3DModel",
        "transformer_subfolder": "transformer",
        "quantization_config": {"compute_dtype": "bfloat16"},
        "vae_dtype": "float32",
        "fps": 24,
        "frame_count_multiple": 4,
        "frame_count_remainder": 1,
        "supported_resolutions": [[1280, 704], [704, 1280]],
    },
    generation_options={
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    },
)
