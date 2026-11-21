# Broca

Broca is a local AI inference server and Python library for NVIDIA and AMD GPUs
on Linux and Apple Silicon on macOS. It manages model downloads and serves text,
embeddings, decisions, speech, images, and video from your own machine.

Use Broca when you want to prototype against a local API or keep inference
inputs on hardware you control. GPU acceleration depends on the model and
installed runtime. Use CUDA for NVIDIA, ROCm for supported AMD GPUs, and
Metal or MPS for Apple Silicon.
[llama.cpp accelerator builds](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[PyTorch ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html),
[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Quick start

With Python 3.11 or newer and `uv` installed, run this from the repository root.

```bash
uv run --locked --extra ai uvicorn broca:broca_app --host 127.0.0.1 --port 9001
```

Open the dashboard at `http://localhost:9001/`, download a model, then load and
test it. Applications can call the `/v1` API. Interactive API docs are at
`http://localhost:9001/docs`. To embed Broca in an existing FastAPI app, import
`broca_router` from `broca`.

## Default models

| Modality | Model | Hugging Face repository | Runtime |
| --- | --- | --- | --- |
| Audio | Bark Small | `suno/bark-small` | Transformers |
| Decision | Bekko System One v0 400M | `hotchpotch/bekko-system-one-v0-400m` | Sentence Transformers |
| Embedding | EmbeddingGemma 2 | `unsloth/embeddinggemma-2-GGUF` | llama.cpp |
| Image | FLUX 3 Image Klein | `black-forest-labs/flux-3-image-klein` | Diffusers |
| Text | Gemma 4 E4B IT | `unsloth/gemma-4-E4B-it-GGUF` | llama.cpp |
| Video | Wan 2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Diffusers |

The image and video defaults use GGUF weights from
`unsloth/flux-3-image-klein-GGUF` and `unsloth/Wan2.2-TI2V-5B-GGUF`,
respectively. The defaults are defined in the
[inference catalog](broca/inference/__init__.py).

## Manage model assets in Python

Only a namespaced Hugging Face repository ID is required. All other fields are
optional, keeping engine defaults available and individual configurations small.

```python
import asyncio

from broca import AiModel

model = AiModel(id="google/embeddinggemma-300m")
asyncio.run(model.download())
print(model.local_path)  # ~/ai/google/embeddinggemma-300m
```

For a single GGUF checkpoint and an optional multimodal projector:

```python
model = AiModel(
    id="lmstudio-community/gemma-4-E4B-it-GGUF",
    type="chat",
    file="gemma-4-E4B-it-Q6_K.gguf",
    clip="mmproj-gemma-4-E4B-it-BF16.gguf",
    quantization="Q6_K",
    context_window=16384,
)
```

Use `weights_repo` with `file` for a quantized image/video transformer held in a
separate repository. The primary repository supplies pipeline assets while its
`transformer/*.safetensors`, `transformer/*.bin`, and `transformer/*.gguf` weights
are skipped. Each repository is stored at `~/ai/{lab}/{model}`. Set `revision`
and `weights_revision` independently to pin pipeline and checkpoint versions.
`allow_patterns` and `ignore_patterns` filter snapshots; explicit `file` and
`clip` selections are always downloaded.

From async code:

```python
await model.download()            # True on success; transfer errors propagate
await model.is_downloaded()       # Verify persisted completion and asset metadata
await model.download_progress()   # 0 missing/failed, active fraction, 1 complete
```

Matching configurations share a process lock, and concurrent callers wait for
one transfer. Different selections sharing a repository are serialized so their
files cannot be overwritten by simultaneous transfers. Completion survives
restarts, and missing or changed assets cause
the next download to retry. A partial directory alone does not establish snapshot
completeness. State manifests live in `~/ai/.downloads`. Cancelling an active
download waits for native transfer I/O to finish before releasing the lock.

Fields cover metadata, artifact selection, memory estimates, device/precision,
context and chat formatting, GPU allocation, batching and CPU threads, attention,
offloading, and prompt/KV caches. `runtime_options` and `generation_options`
accept additional backend-specific JSON settings. Runtime fields describe loader
configuration; `AiModel` currently manages assets without loading an inference
pipeline. Every field is documented in the model's Pydantic schema.

Use the Hugging Face credential store or `HF_TOKEN` for gated models. Model
configs and download manifests do not contain authentication tokens.

## Inference pipelines

Install the optional backends with `pip install '.[ai]'`. Each
pipeline takes an `AiModel` for its model family. Download the model assets
first, then call `load()` to initialize the backend and `unload()` to release
it. Both operations are synchronous. Call `infer()` asynchronously to run
the configured modality. Use `get_device()` to inspect the selected device and
`runtime` to access the loaded backend.

```python
import asyncio

from broca.inference.embedding._pipeline import pipeline

asyncio.run(pipeline.model.download())
pipeline.load()
try:
    print(asyncio.run(pipeline.infer(["A short example"])))
finally:
    pipeline.unload()
```

Bark Small uses the Transformers runtime included in the `ai` extra.
`AudioPipeline` exposes its Bark model and voice preset processor through `runtime`.

## Development

```bash
uv run ruff check .
uv run --extra ai pytest -q
node --test tests/*.mjs
uv build
```

The benchmarks use an OpenAI-compatible server on `http://127.0.0.1:8080/v1`.
With a compatible server running, use
`uv run --group benchmarks python tests/benchmarks/gemma_4_llama_cpp.py` or
`uv run --group benchmarks python tests/benchmarks/gemma_4_mlx_vlm.py`.
