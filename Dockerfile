# Dockerfile
# HayStack: Zero-Shot to Few-Shot Classifier
# Supports: ResNet, DINO, DINOv2, CLIP, ViT

FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e ".[clip]"

# Download example data (optional)
RUN mkdir -p data/coco && \
    wget -q http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip && \
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations.zip && \
    unzip -q data/coco/val2017.zip -d data/coco/ && \
    unzip -q data/coco/annotations.zip -d data/coco/ && \
    rm data/coco/*.zip

# Set environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "scripts/run_benchmark.py", "--run_all", "--device", "cuda"]