# Base image with Python 3.12 for CPU workloads
FROM python:3.12-slim

# Prevent Python from writing .pyc files and ensure logs are flushed
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WANDB_MODE=online

WORKDIR /app

# System dependencies for scientific Python stacks and image utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command opens a shell for interactive use (VS Code, Jupyter, bash)
CMD ["bash"]
