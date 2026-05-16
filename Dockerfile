# Full DEVEL CUDA image (required for CuPy & nvrtc kernels)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------
# Base system + Python
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git libgomp1 libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip

COPY requirements.txt .

# ----------------------------------------------------
# PyTorch NIGHTLY for CUDA 13.0 (Blackwell support)
# ----------------------------------------------------
RUN pip install --no-cache-dir --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# ----------------------------------------------------
# CuPy (requires DEVEL image → OK)
# Match the CUDA version of the base image (12.4)
# ----------------------------------------------------
RUN pip install --no-cache-dir cupy-cuda124

# ----------------------------------------------------
# PennyLane + Lightning GPU backend
# ----------------------------------------------------
RUN pip install --no-cache-dir \
    pennylane \
    pennylane-lightning[gpu]

# ----------------------------------------------------
# Project dependencies
# ----------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["bash"]
