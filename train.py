"""YOLO training module"""

import subprocess
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

DATA = "data/train_data_external.yml"
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2
PLOTS = True
RESUME = False
REPO_ROOT = Path(__file__).resolve().parent


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


def convert_coco_annotations(dataset_root: Path) -> None:
    """Convert COCO JSON annotations to YOLO format if not already done."""
    train_labels = dataset_root / "train" / "labels"
    val_labels = dataset_root / "val" / "labels"

    # Skip if labels already exist
    if train_labels.exists() and val_labels.exists():
        print("YOLO labels already exist, skipping conversion")
        return

    print("Converting COCO annotations to YOLO format...")
    converter_script = REPO_ROOT / "convert_coco_to_yolo.py"

    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")

    # Run converter
    result = subprocess.run(
        [sys.executable, str(converter_script), str(dataset_root)],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("Conversion warnings/errors:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed with code {result.returncode}")


def main():
    """YOLO training main function"""
    data_path = REPO_ROOT / DATA
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    print(f"Using data config: {data_path}")

    # Resolve dataset root (Docker container path takes precedence)
    docker_dataset = Path("/app/data/swapaug7")
    if docker_dataset.exists():
        dataset_root = docker_dataset
        print(f"Running in Docker container, using: {dataset_root}")
    else:
        # Fallback for local development
        dataset_root = Path("/Documents/drone-mask-dino/dataset/swapaug7")
        if not dataset_root.exists():
            dataset_root = Path.home() / "Documents" / "drone-mask-dino" / "dataset" / "swapaug7"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")

    print(f"Dataset root: {dataset_root}")

    # Convert COCO annotations to YOLO format
    convert_coco_annotations(dataset_root)

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
    last_model_path = REPO_ROOT / "runs" / "detect" / "train15" / "weights" / "last.pt"

    if last_model_path.exists():
        print(f"Continuing training from {last_model_path}")
        model = YOLO(str(last_model_path))
    else:
        print("No previous model found, starting from base yolo11s.pt")
        model = YOLO("yolo11s.pt")

    results = model.train(
        data=str(data_path),
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