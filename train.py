"""YOLO training module"""

import os
import torch
from ultralytics import YOLO

DATA = "data/train_data.yml"
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2
PLOTS = True
RESUME = False


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
    """YOLO training main function"""
    # Check CUDA availability
    cuda_available, device_name = check_cuda_available()
    
    print(f"\nTraining using {device_name}")
    
    # Determine device to use
    if cuda_available:
        device = 0  # Use first GPU
        print(f"\nUsing CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\nUsing CPU for training")
    
    print(f"PyTorch version: {torch.__version__}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")

    # Path to the last trained model (update this if needed)
    last_model_path = "runs/detect/train15/weights/last.pt"

    if os.path.exists(last_model_path):
        print(f"Continuing training from {last_model_path}")
        model = YOLO(last_model_path)
    else:
        print("No previous model found, starting from base yolo11s.pt")
        model = YOLO("yolo11s.pt")

    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,  # Use detected device (CUDA or CPU)
        workers=WORKERS,
        plots=PLOTS,
        resume=RESUME,
    )

    metrics = model.val()


if __name__ == "__main__":
    main()
