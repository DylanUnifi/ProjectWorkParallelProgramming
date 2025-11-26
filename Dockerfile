ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WANDB_MODE=online

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    libgomp1 libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Torch CUDA 12.4
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_SPEC="torch==2.3.1 torchvision==0.18.1"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} ${TORCH_SPEC}

# CuPy CUDA 12.x (compatible avec 12.4 → 13.0 drivers)
RUN pip install --no-cache-dir cupy-cuda12x==13.6.0

# PennyLane + Lightning GPU
RUN pip install --no-cache-dir pennylane pennylane-lightning pennylane-lightning[gpu]

# Install rest of deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["bash"]
