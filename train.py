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
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


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


def has_image_files(directory: Path) -> bool:
    """Return True if the directory contains at least one supported image file."""
    if not directory.exists():
        return False

    for pattern in IMAGE_EXTENSIONS:
        if any(directory.glob(pattern)):
            return True
    return False


def resolve_split_images_dir(dataset_root: Path, split: str) -> Path:
    """Resolve split image directory supporting both split/images and split layouts."""
    split_dir = dataset_root / split
    nested_images_dir = split_dir / "images"

    if has_image_files(nested_images_dir):
        return nested_images_dir
    if has_image_files(split_dir):
        return split_dir

    raise FileNotFoundError(
        f"Could not find images for split '{split}'. Checked:\n"
        f"- {nested_images_dir}\n"
        f"- {split_dir}"
    )


def build_runtime_data_config(dataset_root: Path) -> Path:
    """Build a YOLO data config using detected train/val image directory layout."""
    train_images_dir = resolve_split_images_dir(dataset_root, "train")
    val_images_dir = resolve_split_images_dir(dataset_root, "val")

    train_rel = train_images_dir.relative_to(dataset_root).as_posix()
    val_rel = val_images_dir.relative_to(dataset_root).as_posix()

    runtime_data_path = REPO_ROOT / "data" / "train_data_external_resolved.yml"
    runtime_yaml = [
        f"path: {dataset_root.as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        "",
        "nc: 1",
        "names:",
        "  0: drone",
    ]
    runtime_data_path.write_text("\n".join(runtime_yaml) + "\n", encoding="utf-8")
    return runtime_data_path


def convert_coco_annotations(dataset_root: Path) -> None:
    """Convert COCO JSON annotations to YOLO format if not already done."""
    train_labels = dataset_root / "train" / "labels"
    val_labels = dataset_root / "val" / "labels"
    resolve_split_images_dir(dataset_root, "train")
    resolve_split_images_dir(dataset_root, "val")

    train_has_labels = train_labels.exists() and any(train_labels.glob("*.txt"))
    val_has_labels = val_labels.exists() and any(val_labels.glob("*.txt"))

    # Skip if labels already exist
    if train_has_labels and val_has_labels:
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

    runtime_data_path = build_runtime_data_config(dataset_root)
    print(f"Resolved data config: {runtime_data_path}")

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
        data=str(runtime_data_path),
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