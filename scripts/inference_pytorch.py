#!/usr/bin/env python3
"""
YOLO PyTorch Inference Script

Loads a YOLO model, runs inference on an image, draws bounding boxes and labels,
and saves the annotated output image.

Usage:
    python scripts/inference_pytorch.py --image data/images/input/test.jpg
    python scripts/inference_pytorch.py --image test.jpg --model yolov8s.pt --conf 0.5
    python scripts/inference_pytorch.py --image test.jpg --output custom_output.jpg --save-txt
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO inference on an image and save annotated output'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Path to YOLO model file (.pt). Default: yolov8n.pt (downloads automatically)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output image. Default: outputs/predictions/pytorch/'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1). Default: 0.25'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (0-1). Default: 0.45'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results as .txt file with bounding box coordinates'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Include confidence scores in saved .txt file'
    )
    
    args = parser.parse_args()
    
    # Validate input image
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Set up output directory
    if args.output is None:
        output_dir = Path('outputs/predictions/pytorch')
        output_dir.mkdir(parents=True, exist_ok=True)
        input_name = Path(args.image).stem
        args.output = str(output_dir / f"{input_name}_result.jpg")
    else:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading YOLO model: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully. Model type: {model.task}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Run inference
    print(f"Running inference on: {args.image}")
    print(f"Confidence threshold: {args.conf}, IoU threshold: {args.iou}")
    
    try:
        # Run inference (don't save automatically, we'll save manually)
        results = model.predict(
            source=args.image,
            conf=args.conf,
            iou=args.iou,
            save=False,  # We'll save manually to control output path
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            project=str(output_dir),
            name='',
            exist_ok=True
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        
        # Generate annotated image with bounding boxes and labels
        annotated_img = result.plot()
        
        # Save the annotated image
        from PIL import Image
        img = Image.fromarray(annotated_img)
        img.save(args.output, quality=95)
        print(f"\n✓ Inference complete!")
        print(f"  Output saved to: {args.output}")
        
        # Print detection summary
        if result.boxes is not None and len(result.boxes) > 0:
            num_detections = len(result.boxes)
            print(f"  Detections: {num_detections}")
            
            # Count detections by class
            if result.names:
                class_counts = {}
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                print("  Class breakdown:")
                for class_name, count in sorted(class_counts.items()):
                    print(f"    - {class_name}: {count}")
        else:
            print("  No detections found")
    else:
        print("Error: No results returned from inference")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

