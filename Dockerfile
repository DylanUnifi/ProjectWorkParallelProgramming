# CPU-only base image for classical training and CPU fallbacks
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------
# Base system + Python
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git libgomp1 libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip

COPY requirements.txt .

# ----------------------------------------------------
# PyTorch CPU wheels
# ----------------------------------------------------
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ----------------------------------------------------
# PennyLane + Lightning CPU backend
# ----------------------------------------------------
RUN pip install --no-cache-dir \
    pennylane \
    pennylane-lightning

# ----------------------------------------------------
# Project dependencies
# ----------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["bash"]
