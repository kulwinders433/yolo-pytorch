# Use official PyTorch image with Python 3.10 and CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies (minimal set for OpenCV and image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Pin NumPy to <2.0 for compatibility with PyTorch 2.1.0
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    ultralytics>=8.0.0 \
    onnx>=1.14.0 \
    onnxsim>=0.4.0 \
    onnxruntime-gpu>=1.16.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0

# Create non-root user for security (production best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /workspace

# Switch to non-root user
USER appuser

# Default command
CMD ["/bin/bash"]

