"""YOLO training module"""

import os
import torch
from ultralytics import YOLO

DATA = "data\\train_data.yml"
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2
PLOTS = True
RESUME = False


def main():
    """YOLO training main function"""
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)

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
        device=None,  # auto-select GPU if available
        workers=WORKERS,
        plots=PLOTS,
        resume=RESUME,
    )

    metrics = model.val()


if __name__ == "__main__":
    main()
