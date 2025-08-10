# Minimal dev image for TRT YOLOv8 accelerator
# Note: adjust CUDA/TRT versions to your host/target
FROM nvcr.io/nvidia/tensorrt:24.03-py3

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim build-essential cmake libopencv-dev python3-opencv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

# Project
COPY . /workspace

# Build placeholder targets
RUN cmake -S . -B build && cmake --build build -j

CMD ["bash"]
