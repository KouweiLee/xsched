#!/usr/bin/env python3
"""
YOLO inference program with explicit stream usage

This version attempts to use explicit CUDA streams to avoid default stream usage.
XSched GPU scheduling is transparently applied via environment variables.
"""

import time
import argparse
import os
from pathlib import Path

# Try to configure PyTorch to use explicit streams
# This must be done before importing torch
try:
    import torch
    import torch.cuda
    
    # Create a dedicated stream for inference
    if torch.cuda.is_available():
        inference_stream = torch.cuda.Stream()
        print(f"Created dedicated CUDA stream: {inference_stream}")
except ImportError:
    print("Warning: PyTorch not available, cannot create CUDA stream")

try:
    from ultralytics import YOLOE
except ImportError:
    print("Error: ultralytics is not installed. Please install it with: pip install ultralytics")
    exit(1)


def run_inference_loop(model_path, image_path, num_iterations=0, interval=0.0, device='cuda'):
    """
    Run YOLO inference in a loop with explicit stream usage
    
    Args:
        model_path: Path to YOLO model file
        image_path: Path to input image (or directory of images)
        num_iterations: Number of iterations (0 means infinite loop)
        interval: Sleep interval between iterations in seconds
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"Loading YOLO model: {model_path}")
    print("(Model will be auto-downloaded if not found locally)")
    model = YOLOE(model_path)
    print("Model loaded successfully.")
    
    # Check if image_path is a directory or a file
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not image_files:
            print(f"Error: No image files found in directory: {image_path}")
            return
        print(f"Found {len(image_files)} image(s) in directory.")
    else:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        image_files = [image_path]
    
    iteration = 0
    total_time = 0.0
    min_time = float('inf')
    max_time = 0.0
    
    print(f"\nStarting inference loop (device: {device})...")
    print(f"  - Iterations: {'infinite' if num_iterations == 0 else num_iterations}")
    print(f"  - Interval: {interval:.2f} seconds")
    print(f"  - Image(s): {len(image_files)} file(s)")
    
    # Use explicit stream if available
    use_stream = device == 'cuda' and torch.cuda.is_available()
    if use_stream:
        print(f"  - Using explicit CUDA stream for inference")
    print()
    
    try:
        while True:
            iteration += 1
            
            # Select image for this iteration (cycle through if multiple images)
            current_image = image_files[(iteration - 1) % len(image_files)]
            
            # Run inference with explicit stream if available
            start_time = time.time()
            if use_stream:
                with torch.cuda.stream(inference_stream):
                    results = model(current_image, device=device, verbose=False)
                # Synchronize the stream to ensure completion
                inference_stream.synchronize()
            else:
                results = model(current_image, device=device, verbose=False)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_time += inference_time
            min_time = min(min_time, inference_time)
            max_time = max(max_time, inference_time)
            
            # Print results
            num_detections = len(results[0].boxes) if results and len(results) > 0 else 0
            print(f"[Iteration {iteration}] Image: {os.path.basename(current_image)} | "
                  f"Time: {inference_time:.2f} ms | Detections: {num_detections}")
            
            # Print detection details for first iteration or when detections found
            if iteration == 1 or num_detections > 0:
                if results and len(results) > 0:
                    result = results[0]
                    if len(result.boxes) > 0:
                        print(f"  Detections:")
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = result.names[cls] if hasattr(result, 'names') else f"Class {cls}"
                            print(f"    - {cls_name}: {conf:.2f} confidence")
            
            # Check if we should stop
            if num_iterations > 0 and iteration >= num_iterations:
                break
            
            # Sleep between iterations
            if interval > 0:
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    # Print statistics
    if iteration > 0:
        avg_time = total_time / iteration
        print(f"\n{'='*60}")
        print(f"Inference Statistics:")
        print(f"  Total iterations: {iteration}")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Min time: {min_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Total time: {total_time/1000:.2f} s")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO inference program with explicit CUDA stream usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run infinite loop with default model (yolov8n.pt, auto-downloaded)
  python yolo_xsched_fixed.py image.jpg
  
  # Run with custom model (will auto-download if not found)
  python yolo_xsched_fixed.py yolov8s.pt image.jpg
  
  # Run 100 iterations
  python yolo_xsched_fixed.py image.jpg --iterations 100
  
  # Run with 0.5 second interval between iterations
  python yolo_xsched_fixed.py image.jpg --interval 0.5
  
  # Process all images in a directory
  python yolo_xsched_fixed.py /path/to/images/ --iterations 50
  
  # Use CPU instead of GPU
  python yolo_xsched_fixed.py image.jpg --device cpu
        """
    )
    
    parser.add_argument('model', type=str, nargs='?', default='yoloe-11l-seg-pf.pt',
                       help='Path to YOLO model file (.pt) or model name (default: yoloe-11l-seg-pf.pt, will auto-download if not found)')
    parser.add_argument('image', type=str, help='Path to input image or directory containing images')
    parser.add_argument('--iterations', type=int, default=0,
                       help='Number of iterations (0 = infinite loop, default: 0)')
    parser.add_argument('--interval', type=float, default=0.0,
                       help='Sleep interval between iterations in seconds (default: 0.0)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Run inference loop
    try:
        run_inference_loop(
            args.model,
            args.image,
            num_iterations=args.iterations,
            interval=args.interval,
            device=args.device
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

