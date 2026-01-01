#!/usr/bin/env python3
"""
YOLO PyTorch to ONNX Conversion Script

Converts a YOLO PyTorch model (.pt) to ONNX format (.onnx) using Ultralytics.
Includes recommended settings and handles common pitfalls.

Usage:
    python scripts/convert_to_onnx.py --model models/pytorch/yolo11n.pt
    python scripts/convert_to_onnx.py --model yolov8s.pt --imgsz 640 --opset 17
    python scripts/convert_to_onnx.py --model model.pt --simplify --dynamic

Common Pitfalls and Solutions:
1. Opset Version:
   - Too low (<11): May not support all YOLO operations (e.g., GridSample)
   - Too high (>17): May have compatibility issues with older runtimes
   - Recommended: Opset 17 (default) - supports all YOLO ops and most runtimes

2. Input Shape:
   - Fixed size (default): Faster inference, but requires resizing input
   - Dynamic size (--dynamic): Flexible but may be slower and less compatible
   - Recommended: Use fixed size (640x640) unless you need variable input sizes

3. Model Simplification:
   - Always use --simplify: Reduces model size, improves compatibility
   - Removes redundant operations and optimizes the graph
   - May take longer but results in better ONNX models

4. FP16 vs FP32:
   - FP16 (--half): Smaller model, faster inference, ~2% accuracy loss
   - FP32 (default): Better accuracy, larger model
   - Recommended: Use FP32 unless you need the speed/size benefits

5. Runtime Compatibility:
   - ONNX Runtime: Works with opset 11-17
   - TensorRT: May require specific opset versions
   - OpenVINO: Prefers opset 11
   - Test your target runtime after conversion

6. Common Errors:
   - "GridSample not supported": Use opset >= 11
   - "Shape inference failed": Try --simplify or check input shape
   - "Export failed": Ensure model is fully trained and PyTorch version is compatible
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO PyTorch model to ONNX format'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to YOLO PyTorch model file (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for ONNX model. Default: models/onnx/'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (height, width). Default: 640'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version. Recommended: 17 (supports all YOLO ops). Default: 17'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model using onnxsim (recommended)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Export with dynamic batch size and image dimensions'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Export FP16 model (faster inference, slightly less accurate)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4096,
        help='TensorRT workspace size in MB (for TensorRT export). Default: 4096'
    )
    
    args = parser.parse_args()
    
    # Validate input model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Set up output path
    if args.output is None:
        output_dir = Path('models/onnx')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename from input model name
        model_name = Path(args.model).stem
        args.output = str(output_dir / f"{model_name}.onnx")
    else:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate opset version
    if args.opset < 11:
        print("Warning: Opset version < 11 may not support all YOLO operations")
    elif args.opset > 17:
        print("Warning: Opset version > 17 may have compatibility issues with some runtimes")
    
    # Load model
    print(f"Loading YOLO model: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully. Model type: {model.task}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Prepare export arguments
    export_kwargs = {
        'format': 'onnx',
        'imgsz': args.imgsz,
        'opset': args.opset,
        'simplify': args.simplify,
        'half': args.half,
    }
    
    # Handle dynamic shapes
    if args.dynamic:
        export_kwargs['dynamic'] = True
        print("Exporting with dynamic batch and image dimensions")
    else:
        print(f"Exporting with fixed input size: {args.imgsz}x{args.imgsz}")
    
    # Display export settings
    print(f"\nExport settings:")
    print(f"  Input size: {args.imgsz}x{args.imgsz}")
    print(f"  Opset version: {args.opset}")
    print(f"  Simplify: {args.simplify}")
    print(f"  Dynamic shapes: {args.dynamic}")
    print(f"  FP16 (half precision): {args.half}")
    print(f"  Output: {args.output}")
    
    # Export to ONNX
    print(f"\nConverting to ONNX...")
    try:
        # Export model (returns path to exported file)
        exported_path = model.export(**export_kwargs)
        
        if not exported_path:
            print("Error: Export returned None or empty path")
            return 1
        
        exported_path = Path(exported_path)
        
        # Move to desired output location if different
        if str(exported_path) != args.output:
            import shutil
            if exported_path.exists():
                shutil.move(str(exported_path), args.output)
                print(f"Moved ONNX model to: {args.output}")
            else:
                print(f"Warning: Exported file not found at {exported_path}")
                # Try to find it in the model directory
                model_dir = Path(args.model).parent
                model_name = Path(args.model).stem
                default_onnx_path = model_dir / f"{model_name}.onnx"
                if default_onnx_path.exists():
                    shutil.move(str(default_onnx_path), args.output)
                    print(f"Found and moved ONNX model to: {args.output}")
                else:
                    print("Error: Could not find exported ONNX file")
                    return 1
        
        # Verify output file exists
        if not os.path.exists(args.output):
            print(f"Error: Output file was not created: {args.output}")
            return 1
        
        # Get file size
        file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        print(f"\n✓ Conversion successful!")
        print(f"  ONNX model saved to: {args.output}")
        print(f"  File size: {file_size:.2f} MB")
        
        # Additional validation
        try:
            import onnx
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print(f"  ✓ ONNX model validation passed")
            
            # Print model info
            print(f"\nModel information:")
            try:
                input_shapes = []
                for inp in onnx_model.graph.input:
                    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in inp.type.tensor_type.shape.dim]
                    input_shapes.append(shape)
                print(f"  Input shape: {input_shapes}")
                
                output_shapes = []
                for out in onnx_model.graph.output:
                    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in out.type.tensor_type.shape.dim]
                    output_shapes.append(shape)
                print(f"  Output shape: {output_shapes}")
            except Exception as e:
                print(f"  Could not extract shape info: {e}")
            
            print(f"  Opset version: {onnx_model.opset_import[0].version}")
            print(f"  IR version: {onnx_model.ir_version}")
            
        except ImportError:
            print("  Note: Install 'onnx' package to validate the exported model")
        except Exception as e:
            print(f"  Warning: Model validation failed: {e}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
