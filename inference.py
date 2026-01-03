
"""YOLO inference module"""

import os
import sys
import argparse
import torch
from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "runs/detect/train/weights/best.pt"
TEST_IMAGES_DIR = "data/test/images"


def check_cuda_available():
    """
    Check if CUDA is available and can be used.
    
    Returns:
        tuple: (is_available: bool, device_name: str)
    """
    if not torch.cuda.is_available():
        return False, "CPU"
    
    # Check if CUDA devices are accessible
    try:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA ({device_name})"
        else:
            return False, "CPU"
    except Exception:
        return False, "CPU"


def main():
    """YOLO inference main function"""
    parser = argparse.ArgumentParser(description="Run YOLO inference on a test image")
    parser.add_argument("filename", type=str, help="Name of the image file to test (e.g., f103.jpg)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true", help="Save the inference results")
    parser.add_argument("--show", action="store_true", help="Display the inference results")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    cuda_available, device_name = check_cuda_available()
    print(f"\nRunning inference using {device_name}")
    
    # Determine device to use
    if cuda_available:
        device = 0  # Use first GPU
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU for inference")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Load the model
    print(f"\nLoading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Construct full path to image
    image_path = os.path.join(TEST_IMAGES_DIR, args.filename)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print(f"Available images in {TEST_IMAGES_DIR}:")
        if os.path.exists(TEST_IMAGES_DIR):
            images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images[:10]:  # Show first 10
                print(f"  - {img}")
            if len(images) > 10:
                print(f"  ... and {len(images) - 10} more")
        sys.exit(1)
    
    print(f"\nRunning inference on: {image_path}")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=args.conf,
        device=device,
        save=args.save,
        show=args.show,
    )
    
    # Print results
    print("\n" + "="*50)
    print("Inference Results:")
    print("="*50)
    
    for result in results:
        print(f"\nImage: {result.path}")
        print(f"Detections: {len(result.boxes)}")
        
        if len(result.boxes) > 0:
            print("\nDetected objects:")
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  {i+1}. {class_name}: {conf:.2%} confidence")
        else:
            print("No objects detected")
    
    print("\n" + "="*50)
    
    if args.save:
        print(f"\nResults saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()