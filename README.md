# YOLO PyTorch to ONNX Project

A clean, production-ready setup for running YOLO models with PyTorch, converting them to ONNX, and performing inference.

## Project Structure

```
yolo-test/
├── docker/                 # Docker configuration files
│   └── Dockerfile
├── models/                 # Model files
│   ├── pytorch/           # PyTorch models (.pt)
│   └── onnx/              # ONNX models (.onnx)
├── data/                   # Input data
│   ├── images/
│   │   ├── input/         # Input images
│   │   └── test/          # Test images
│   └── videos/
│       └── input/         # Input videos
├── scripts/                # Python scripts
│   └── utils/             # Utility functions
├── outputs/                # Results and exports
│   ├── predictions/
│   │   ├── pytorch/       # PyTorch inference results
│   │   └── onnx/          # ONNX inference results
│   ├── exports/           # Exported models/logs
│   └── logs/              # Training/inference logs
├── notebooks/              # Jupyter notebooks
├── configs/               # Configuration files
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (recommended)

### Build and Run

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Start the container:**
   ```bash
   docker-compose up -d
   ```

3. **Enter the container:**
   ```bash
   docker-compose exec yolo bash
   ```

4. **Run your scripts:**
   ```bash
   # Inside container
   python scripts/inference_pytorch.py
   python scripts/convert_to_onnx.py
   python scripts/inference_onnx.py
   ```

## Usage

### Download Models

Models are automatically downloaded on first use, or place them in `models/pytorch/`:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

### Place Input Data

- Images: `data/images/input/`
- Videos: `data/videos/input/`
- Test data: `data/images/test/`

### View Results

- PyTorch predictions: `outputs/predictions/pytorch/`
- ONNX predictions: `outputs/predictions/onnx/`
- Exported models: `outputs/exports/`

## Development

### Without Docker

If you prefer to run locally:

```bash
pip install -r requirements.txt
```

Note: You'll need to install CUDA and cuDNN separately for GPU support.

### Project Structure Details

- **docker/**: Docker configuration
- **models/**: Model storage (see `models/README.md`)
- **data/**: Input data (see `data/README.md`)
- **scripts/**: Python scripts for inference and conversion
- **outputs/**: All generated outputs and results
- **configs/**: YAML configuration files
- **notebooks/**: Jupyter notebooks for experimentation

## Dependencies

See `requirements.txt` for Python dependencies. The Dockerfile includes all system dependencies.

## License

[Add your license here]
