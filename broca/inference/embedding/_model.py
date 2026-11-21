"""EmbeddingGemma 2 multimodal embedding model configuration."""

from ...ai_model import AiModel

# https://huggingface.co/unsloth/embeddinggemma-2-GGUF
# Use the repository's default quantization and retain the vision/audio projector.
_model: AiModel = AiModel(
    id="unsloth/embeddinggemma-2-GGUF",
    name="EmbeddingGemma 2",
    created_at=1_791_284_330,
    type="embedding",
    family="embeddinggemma2",
    parameter_count_billions=0.74,
    model_engine="llama.cpp",
    model_format="gguf",
    quantization="UD-Q4_K_XL",
    file="embeddinggemma-2-UD-Q4_K_XL.gguf",
    clip="mmproj-BF16.gguf",
    revision="031f0d4b35536f69ab3509d4893c923264fcf253",
    estimated_weights_bytes=175_673_856,
    projector_bytes=982_074_880,
    # FP16 compute can produce invalid embeddings; FP32 is portable.
    dtype="float32",
    context_window=8192,
    gpu_layers=-1,
    use_mmap=True,
    runtime_options={
        "embedding": True,
        "pooling_type": "mean",
        "embedding_dimensions": 768,
        "normalize_embeddings": True,
        "query_prefix": "task: search result | query: ",
        "document_template": "title: {title} | text: {content}",
        "default_document_title": "none",
    },
)
