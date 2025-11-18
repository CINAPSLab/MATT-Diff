FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 x11-apps vim \
    ca-certificates gnupg lsb-release wget software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# venv
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \        
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace:/workspace/src \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.9" \
    PYTORCH_CUDA_ALLOC_CONF=

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt


CMD ["/bin/bash"]