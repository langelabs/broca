FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM ubuntu:24.04 AS build
WORKDIR /build
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates cmake curl ninja-build python3.12-dev python3.12-venv \
    && curl --fail --silent --show-error --location \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        --output /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        cuda-cudart-dev-13-2=13.2.86-1 cuda-nvcc-13-2=13.2.86-1 libcublas-dev-13-2=13.4.1.3-1 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/broca \
    UV_PYTHON_DOWNLOADS=never \
    CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
    CMAKE_BUILD_PARALLEL_LEVEL=2
# Disabling CUDA VMM allows this backend to load on hosts without libcuda.so.
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;120-real"
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY broca/ broca/
RUN uv --no-cache sync --python /usr/bin/python3.12 --locked --no-default-groups --no-editable

FROM ubuntu:24.04 AS runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 libstdc++6 python3.12 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 broca \
    && install -d -o broca -g broca /home/broca/ai

COPY --from=build /opt/broca /opt/broca
# llama.cpp and PyTorch share the CUDA libraries installed from uv.lock.
RUN printf '%s\n' /opt/broca/lib/python3.12/site-packages/nvidia/cu13/lib \
        > /etc/ld.so.conf.d/broca-cuda.conf \
    && ldconfig

ENV HOME=/home/broca \
    PATH="/opt/broca/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER broca
WORKDIR /home/broca
VOLUME ["/home/broca/ai"]
EXPOSE 9001
CMD ["python", "-m", "broca"]
