# Models Directory

This directory contains YOLO models in different formats.

## Structure

- `pytorch/` - PyTorch model files (`.pt` format)
  - Download YOLO models here (e.g., `yolov8n.pt`, `yolov8s.pt`, etc.)
  
- `onnx/` - ONNX converted model files (`.onnx` format)
  - Converted models will be saved here after running the conversion script

## Downloading Models

YOLO models are automatically downloaded by Ultralytics on first use, or you can download them manually:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically if not present
```

