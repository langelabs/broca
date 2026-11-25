"""Tests for the public Broca package boundaries."""

import importlib

import pytest

from broca import AiModel
from broca.inference import get_pipeline


def test_ai_domain_exports_model() -> None:
    """Expose model configuration at the package root.

    :returns: None.
    """
    assert AiModel.__name__ == "AiModel"


@pytest.mark.parametrize(
    ("modality", "class_name"),
    [
        ("audio", "AudioPipeline"),
        ("decision", "DecisionPipeline"),
        ("embedding", "EmbeddingPipeline"),
        ("images", "ImagePipeline"),
        ("text", "TextPipeline"),
        ("video", "VideoPipeline"),
    ],
)
def test_inference_module_exports_configurable_pipeline(
    modality: str, class_name: str
) -> None:
    """Expose the pipeline class and preserve the catalog's shared instance.

    :param modality: Public inference subpackage.
    :param class_name: Public pipeline class name.
    """
    module = importlib.import_module(f"broca.inference.{modality}")
    pipeline_class = getattr(module, class_name, None)
    assert pipeline_class is not None

    shared_pipeline = module.pipeline
    assert isinstance(shared_pipeline, pipeline_class)
    assert shared_pipeline is get_pipeline(shared_pipeline.model)
    assert module.infer == shared_pipeline.infer
    assert pipeline_class().model is shared_pipeline.model

    custom_model = AiModel(id="example/custom-model", device="cpu")
    custom_pipeline = pipeline_class(model=custom_model)
    assert custom_pipeline.model is custom_model
    assert custom_pipeline.get_device() == "cpu"
    assert not custom_pipeline.is_loaded()


@pytest.mark.parametrize(
    "module_name",
    [
        "broca.contracts",
        "broca.servers",
        "broca.plugin",
    ],
)
def test_legacy_domain_imports_are_removed(module_name: str) -> None:
    """Reject removed contract, server, and plugin import paths.

    :param module_name: Removed module path.
    :returns: None.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
