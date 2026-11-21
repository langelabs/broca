"""Configuration, local layout, and memory estimates for portable AI models."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from broca.ai_model import AiModel


def test_only_repository_id_is_required() -> None:
    """Keep minimal configs slim and leave engine defaults unspecified.

    :returns: None.
    """
    model = AiModel(id="google/gemma-4")

    assert model.model_dump(exclude_none=True) == {"id": "google/gemma-4"}
    assert model.local_path == Path.home() / "ai/google/gemma-4"
    assert {
        name for name, field in AiModel.model_fields.items() if field.is_required()
    } == {"id"}


def test_reference_pack_quantisation_spelling_is_supported() -> None:
    """Accept catalog quantisation values and serialize their canonical spelling.

    :returns: None.
    """
    model = AiModel(id="lab/model", quantisation="Q4_K_M")

    assert model.quantization == "Q4_K_M"
    assert model.model_dump(exclude_none=True)["quantization"] == "Q4_K_M"


def test_configuration_preserves_runtime_and_modality_options() -> None:
    """Round-trip explicit settings without choosing an inference backend.

    :returns: None.
    """
    options = {
        "id": "lab/multimodal",
        "name": "Multimodal",
        "type": "chat",
        "model_engine": "custom-runtime",
        "model_format": "safetensors",
        "parameter_count_billions": 0.3,
        "quantization": "int8",
        "context_window": 8192,
        "enable_thinking": False,
        "device": "cuda:1",
        "dtype": "bfloat16",
        "gpu_layers": -1,
        "main_gpu": 1,
        "tensor_split": [0.5, 0.5],
        "batch_size": 128,
        "physical_batch_size": 64,
        "cpu_threads": 4,
        "cpu_batch_threads": 8,
        "use_mmap": True,
        "use_mlock": False,
        "flash_attention": True,
        "offload_kqv": True,
        "cache_enabled": False,
        "cache_type": "ram",
        "cache_size": 4096,
        "seed": 42,
        "kv_bits": 4,
        "kv_quant_scheme": "uniform",
        "kv_group_size": 64,
        "kv_max_size": 4096,
        "chat_template": "{{ messages }}",
        "stop": ["<end>"],
        "stop_token_ids": [2],
        "cpu_offload": True,
        "vae_tiling": True,
        "trust_remote_code": False,
        "runtime_options": {"attention_backend": "sdpa"},
        "generation_options": {"width": 512, "num_frames": 9, "temperature": 0.2},
    }

    model = AiModel.model_validate(options)

    assert model.model_dump(exclude_none=True) == options
    assert AiModel.model_validate_json(model.model_dump_json()) == model


@pytest.mark.parametrize(
    "model_type", ["chat", "text", "decision", "image", "video", "audio", "embedding"]
)
def test_supported_model_types(model_type: str) -> None:
    """Accept every planned inference modality.

    :param model_type: Modality from the public model contract.
    :returns: None.
    """
    assert (
        AiModel.model_validate({"id": "lab/model", "type": model_type}).type
        == model_type
    )


@pytest.mark.parametrize(
    "repository",
    [
        "model",
        "../model",
        "/lab/model",
        "lab/../../model",
        "lab\\model",
        "lab/model..bad",
        "lab/model.git",
        "lab/-bad",
        "lab/model/extra",
        "lab/model\n",
    ],
)
@pytest.mark.parametrize("field", ["id", "weights_repo"])
def test_rejects_unsafe_repository_identifiers(field: str, repository: str) -> None:
    """Require a safe lab/model repository layout before any filesystem access.

    :param field: Repository field to validate.
    :param repository: Invalid repository identifier.
    :returns: None.
    """
    with pytest.raises(ValidationError):
        AiModel.model_validate({"id": "lab/model", field: repository})


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "../outside",
        "/outside",
        "nested/../../outside",
        "nested\\file",
        "C:/outside",
        "bad\x00file",
        ".cache/huggingface/token",
    ],
)
@pytest.mark.parametrize("field", ["file", "clip"])
def test_rejects_unsafe_asset_paths(field: str, filename: str) -> None:
    """Prevent checkpoint paths from escaping or overwriting Hub metadata.

    :param field: Asset field to validate.
    :param filename: Unsafe repository-relative path.
    :returns: None.
    """
    with pytest.raises(ValidationError):
        AiModel.model_validate({"id": "lab/model", field: filename})


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_window", 0),
        ("gpu_layers", -2),
        ("batch_size", 0),
        ("cpu_threads", -1),
        ("tensor_split", [-0.1]),
        ("estimated_weights_bytes", -1),
        ("max_workers", 0),
        ("kv_bits", 0),
        ("parameter_count_billions", float("inf")),
        ("tensor_split", [float("nan")]),
    ],
)
def test_rejects_invalid_numeric_settings(field: str, value: object) -> None:
    """Reject invalid sizes while retaining valid zero and false overrides.

    :param field: Numeric configuration field.
    :param value: Out-of-range value.
    :returns: None.
    """
    with pytest.raises(ValidationError):
        AiModel.model_validate({"id": "lab/model", field: value})


def test_rejects_unknown_fields_and_unsafe_mutation() -> None:
    """Detect misspelled config fields and retain validated path identities.

    :returns: None.
    """
    with pytest.raises(ValidationError):
        AiModel(id="lab/model", unknown_setting=True)
    with pytest.raises(ValidationError):
        AiModel(id="lab/model").id = "../outside"


def test_checkpoint_uses_its_own_weights_repository() -> None:
    """Keep separate quantized checkpoints in their owning repository.

    :returns: None.
    """
    model = AiModel(
        id="lab/pipeline", weights_repo="quantized/weights", file="nested/model.gguf"
    )

    assert (
        model.checkpoint_path == Path.home() / "ai/quantized/weights/nested/model.gguf"
    )
    assert (
        AiModel(id="lab/pipeline", file="model.gguf").checkpoint_path
        == Path.home() / "ai/lab/pipeline/model.gguf"
    )
    with pytest.raises(ValueError, match="file"):
        AiModel(id="lab/pipeline").checkpoint_path
    with pytest.raises(ValidationError):
        AiModel(id="lab/pipeline", weights_repo="quantized/weights")


def test_get_file_path_resolves_assets_in_their_own_repository() -> None:
    """Resolve a checkpoint, projector, and supporting assets from the same model."""
    model = AiModel(
        id="lab/pipeline",
        weights_repo="quantized/weights",
        file="transformer/model.gguf",
        clip="projector.gguf",
    )

    assert model.get_file_path(model.file) == model.checkpoint_path
    assert model.get_file_path(model.clip) == model.local_path / "projector.gguf"
    assert (
        model.get_file_path("vae/config.json") == model.local_path / "vae/config.json"
    )
    assert (
        model.get_file_path(model.file, weights=False) == model.local_path / model.file
    )


def test_get_file_path_rejects_unsafe_paths_and_symlinks(tmp_path: Path) -> None:
    """Reject paths that could leave the selected repository or touch Hub metadata.

    :param tmp_path: Temporary directory for the repository and symlink target.
    """
    model = AiModel(id="lab/model")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Path, "home", Mock(return_value=tmp_path))
        model.local_path.mkdir(parents=True)
        link = model.local_path / "linked"
        link.symlink_to(
            model.local_path.parent / "another-model", target_is_directory=True
        )
        for filename in (
            None,
            "../outside",
            ".cache/config.json",
            "linked/config.json",
        ):
            with pytest.raises(ValueError):
                model.get_file_path(filename)


def test_memory_estimate_prefers_local_checkpoint_and_projector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use local weight sizes and round the VRAM estimate up to whole bytes.

    :param tmp_path: Temporary model home.
    :param monkeypatch: Fixture for substituting the user's home.
    :returns: None.
    """
    monkeypatch.setattr(Path, "home", Mock(return_value=tmp_path))
    model = AiModel(
        id="lab/model",
        file="model.gguf",
        clip="projector.gguf",
        estimated_weights_bytes=101,
        supporting_weights_bytes=20,
        projector_bytes=10,
        runtime_bytes=50,
    )

    assert model.size == 208
    assert model.estimated_vram_bytes == 208
    model.local_path.mkdir(parents=True)
    model.checkpoint_path.write_bytes(b"w" * 200)
    (model.local_path / "projector.gguf").write_bytes(b"p" * 30)
    assert model.size == 350
    assert model.estimated_vram_bytes == 350
    unknown = AiModel(id="lab/unknown")
    assert unknown.size is None
    assert unknown.estimated_vram_bytes is None
