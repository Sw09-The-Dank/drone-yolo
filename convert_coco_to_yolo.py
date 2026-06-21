"""Convert COCO JSON annotations to YOLO format."""

import json
import os
from pathlib import Path
from typing import Dict, List


def convert_coco_to_yolo(
    dataset_root: Path,
    annotations_file: str,
    output_labels_dir: str,
    image_dir: str = "images",
) -> None:
    """
    Convert COCO JSON annotations to YOLO .txt format.
    Only includes annotations for drone model classes.

    Args:
        dataset_root: Root directory of the dataset split (train/ or val/)
        annotations_file: Name of the JSON annotation file (e.g., 'train.json')
        output_labels_dir: Output directory for label files (e.g., 'labels')
        image_dir: Directory containing images (default: 'images')
    """
    json_path = dataset_root / annotations_file
    labels_dir = dataset_root / output_labels_dir
    images_dir = dataset_root / image_dir

    if not json_path.exists():
        print(f"Annotation file not found: {json_path}")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Drone model class IDs from COCO (map to class 0 "drone")
    DRONE_MODELS = {5, 12, 13, 15, 16, 17, 18, 19}  # DNDN-concept, Fixed-wing-concept, Shahed, DJI-Mavic-Pro, DJI-Matrice-600-Pro, DJI-S900, DJI-Spark, U842-Sport-Racing

    # Build mappings
    images_by_id: Dict[int, dict] = {img["id"]: img for img in coco_data["images"]}

    # Verify we have images
    if not images_by_id:
        print(f"No images found in {json_path}")
        return

    # Group annotations by image (only drone models)
    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in DRONE_MODELS:
            continue  # Skip non-drone classes

        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Convert each image's annotations
    converted = 0
    for img_id, img_info in images_by_id.items():
        img_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Generate label file name
        label_name = Path(img_name).stem + ".txt"
        label_path = labels_dir / label_name

        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])

        # Write YOLO format: class_id x_center y_center width height (normalized 0-1)
        # Only write file if there are drone annotations
        if anns:
            with open(label_path, "w", encoding="utf-8") as f:
                for ann in anns:
                    # Get bounding box (COCO: [x, y, width, height])
                    bbox = ann["bbox"]
                    x, y, w, h = bbox

                    # Normalize to 0-1 range and convert to center coords
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height

                    # Clamp to valid range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_width = max(0, min(1, norm_width))
                    norm_height = max(0, min(1, norm_height))

                    # All drone models map to class 0
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

            converted += 1

    print(f"Converted {converted} images with drone annotations to YOLO format in {labels_dir}")


def main():
    """Convert both train and val datasets."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_coco_to_yolo.py <dataset_root>")
        print("Example: python convert_coco_to_yolo.py /path/to/swapaug7")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])

    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        sys.exit(1)

    # Convert train set
    train_dir = dataset_root / "train"
    if train_dir.exists():
        print(f"\nConverting train set from {train_dir}...")
        convert_coco_to_yolo(dataset_root, "train.json", "labels")
    else:
        print(f"Train directory not found: {train_dir}")

    # Convert val set
    val_dir = dataset_root / "val"
    if val_dir.exists():
        print(f"\nConverting val set from {val_dir}...")
        convert_coco_to_yolo(dataset_root, "val.json", "labels")
    else:
        print(f"Val directory not found: {val_dir}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
