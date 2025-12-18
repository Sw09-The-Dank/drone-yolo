"""YOLO training module"""

import os
import torch
from ultralytics import YOLO


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

    results = model.train(  # pylint: disable=unused-variable
        data=r"sw9_drone\datasets\drones\data.yaml",
        epochs=200,  # additional epochs
        imgsz=640,
        batch=16,
        device=None,  # auto-select GPU if available
        workers=2,
        plots=True,
        # resume=True if os.path.exists(last_model_path) else False,
    )

    metrics = model.val()  # pylint: disable=unused-variable


if __name__ == "__main__":
    main()
