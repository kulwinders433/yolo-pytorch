#!/usr/bin/env python3
"""
YOLO ONNX Inference Script

Loads an ONNX model, runs inference on an image, draws bounding boxes and labels,
and saves the annotated output image.

Usage:
    python scripts/inference_onnx.py --model models/onnx/yolo11n.onnx --image data/images/input/test.jpg
    python scripts/inference_onnx.py --model model.onnx --image test.jpg --conf 0.5
"""

import argparse
import os
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import onnxruntime as ort


def preprocess_image(image_path, target_size=640):
    """Preprocess image for ONNX model input."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_height, original_width = img.shape[:2]
    
    # Resize maintaining aspect ratio
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    img_padded[top:top+new_height, left:left+new_width] = img_resized
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # Normalize and transpose for ONNX input (1, 3, H, W)
    img_array = img_rgb.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, scale, (left, top), (original_width, original_height)


def postprocess_output(output, conf_threshold=0.25, iou_threshold=0.45, scale=1.0, pad=(0, 0), orig_size=(640, 640)):
    """Post-process ONNX model output to get bounding boxes."""
    # Output shape is typically (1, num_classes + 4, num_boxes)
    # For YOLO: (1, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
    
    predictions = output[0].transpose((1, 0))  # (8400, 84)
    
    boxes = []
    scores = []
    class_ids = []
    
    # Extract boxes, scores, and class IDs
    for pred in predictions:
        # Get class scores (last 80 values)
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > conf_threshold:
            # Get box coordinates (first 4 values: x_center, y_center, width, height)
            x_center, y_center, width, height = pred[:4]
            
            # Convert from center format to corner format
            x1 = (x_center - width / 2 - pad[0]) / scale
            y1 = (y_center - height / 2 - pad[1]) / scale
            x2 = (x_center + width / 2 - pad[0]) / scale
            y2 = (y_center + height / 2 - pad[1]) / scale
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(int(class_id))
    
    # Apply NMS
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        # NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores[indices], class_ids[indices]
    
    return np.array([]), np.array([]), np.array([])


def draw_boxes(image, boxes, scores, class_ids, class_names=None):
    """Draw bounding boxes and labels on image."""
    img = image.copy()
    
    # COCO class names (default for YOLO)
    if class_names is None:
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Clip coordinates to image bounds
        x1 = max(0, min(x1, img.shape[1]))
        y1 = max(0, min(y1, img.shape[0]))
        x2 = max(0, min(x2, img.shape[1]))
        y2 = max(0, min(y2, img.shape[0]))
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name} {score:.2f}"
        
        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO ONNX inference on an image and save annotated output'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to ONNX model file (.onnx)'
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
        help='Path to save output image. Default: outputs/predictions/onnx/'
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
        '--imgsz',
        type=int,
        default=640,
        help='Input image size. Default: 640'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'tensorrt'],
        help='Inference device. Default: cpu'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Set up output directory
    if args.output is None:
        output_dir = Path('outputs/predictions/onnx')
        output_dir.mkdir(parents=True, exist_ok=True)
        input_name = Path(args.image).stem
        args.output = str(output_dir / f"{input_name}_result.jpg")
    else:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ONNX model
    print(f"Loading ONNX model: {args.model}")
    try:
        # Set up providers
        if args.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif args.device == 'tensorrt':
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(args.model, providers=providers)
        print(f"Model loaded successfully")
        print(f"  Providers: {session.get_providers()}")
        print(f"  Input shape: {session.get_inputs()[0].shape}")
        print(f"  Output shape: {session.get_outputs()[0].shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Preprocess image
    print(f"\nPreprocessing image: {args.image}")
    try:
        img_array, scale, pad, orig_size = preprocess_image(args.image, args.imgsz)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return 1
    
    # Run inference
    print(f"Running inference...")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        outputs = session.run([output_name], {input_name: img_array})
        output = outputs[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    # Post-process results
    print("Post-processing results...")
    boxes, scores, class_ids = postprocess_output(
        output, args.conf, args.iou, scale, pad, orig_size
    )
    
    # Load original image for drawing
    original_img = cv2.imread(args.image)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes
    if len(boxes) > 0:
        annotated_img = draw_boxes(original_img_rgb, boxes, scores, class_ids)
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        # Save result
        cv2.imwrite(args.output, annotated_img_bgr)
        
        print(f"\n✓ Inference complete!")
        print(f"  Output saved to: {args.output}")
        print(f"  Detections: {len(boxes)}")
        
        # Count detections by class
        if len(class_ids) > 0:
            class_counts = {}
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            for class_id in class_ids:
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("  Class breakdown:")
            for class_name, count in sorted(class_counts.items()):
                print(f"    - {class_name}: {count}")
    else:
        # Save original image if no detections
        cv2.imwrite(args.output, original_img)
        print(f"\n✓ Inference complete!")
        print(f"  Output saved to: {args.output}")
        print(f"  No detections found")
    
    return 0


if __name__ == '__main__':
    exit(main())

