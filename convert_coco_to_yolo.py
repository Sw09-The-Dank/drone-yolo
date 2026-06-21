"""Convert COCO JSON annotations to YOLO format."""

import json
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def has_image_files(directory: Path) -> bool:
    """Return True if directory contains at least one supported image file."""
    if not directory.exists():
        return False
    for pattern in IMAGE_EXTENSIONS:
        if any(directory.glob(pattern)):
            return True
    return False


def resolve_images_dir(dataset_root: Path, split: str) -> Path:
    """Resolve images directory for a split supporting split/images and split layouts."""
    split_dir = dataset_root / split
    nested_images = split_dir / "images"
    if has_image_files(nested_images):
        return nested_images
    if has_image_files(split_dir):
        return split_dir
    raise FileNotFoundError(
        f"Could not find images for split '{split}'. Checked:\n"
        f"- {nested_images}\n"
        f"- {split_dir}"
    )


def resolve_labels_dir(images_dir: Path) -> Path:
    """Resolve where labels must be written for a given images dir.

    Ultralytics maps `/.../images/...jpg` -> `/.../labels/...txt`.
    If images are in a flat split dir (no `images` segment), labels must be
    written next to images as `split/*.txt`.
    """
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return images_dir


def convert_coco_to_yolo(
    annotations_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> None:
    """
    Convert COCO JSON annotations to YOLO .txt format.
    Only includes annotations for drone model classes.

    Args:
        annotations_path: Path to COCO JSON file (e.g., train.json)
        images_dir: Directory containing split images (e.g., train/images)
        labels_dir: Directory where YOLO labels are written (e.g., train/labels)
    """
    json_path = annotations_path

    if not json_path.exists():
        print(f"Annotation file not found: {json_path}")
        return

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Drone model class IDs from COCO (map to class 0 "drone")
    DRONE_MODELS = {5, 12, 13, 15, 16, 17, 18, 19}  # DNDN-concept, Fixed-wing-concept, Shahed, DJI-Mavic-Pro, DJI-Matrice-600-Pro, DJI-S900, DJI-Spark, U842-Sport-Racing
    category_name_by_id = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

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
    multi_label_images = 0
    total_labels_written = 0
    multi_label_samples = []
    for img_id, img_info in images_by_id.items():
        img_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Generate label file name
        label_name = Path(img_name).stem + ".txt"
        label_path = labels_dir / label_name

        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])
        if len(anns) > 1:
            multi_label_images += 1
            if len(multi_label_samples) < 10:
                class_names = [
                    category_name_by_id.get(ann["category_id"], str(ann["category_id"]))
                    for ann in anns
                ]
                multi_label_samples.append((img_name, class_names))

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
                    total_labels_written += 1

            converted += 1

    print(f"Converted {converted} images with drone annotations to YOLO format in {labels_dir}")
    print(f"Wrote {total_labels_written} labels total")
    if multi_label_images:
        print(f"Images with multiple drone-model annotations in JSON: {multi_label_images}")
        print("Sample images with multiple selected classes:")
        for file_name, class_names in multi_label_samples:
            print(f"  - {file_name}: {', '.join(class_names)}")


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
        train_images_dir = resolve_images_dir(dataset_root, "train")
        train_labels_dir = resolve_labels_dir(train_images_dir)
        print(f"\nConverting train set from {train_dir}...")
        convert_coco_to_yolo(
            annotations_path=dataset_root / "train.json",
            images_dir=train_images_dir,
            labels_dir=train_labels_dir,
        )
    else:
        print(f"Train directory not found: {train_dir}")

    # Convert val set
    val_dir = dataset_root / "val"
    if val_dir.exists():
        val_images_dir = resolve_images_dir(dataset_root, "val")
        val_labels_dir = resolve_labels_dir(val_images_dir)
        print(f"\nConverting val set from {val_dir}...")
        convert_coco_to_yolo(
            annotations_path=dataset_root / "val.json",
            images_dir=val_images_dir,
            labels_dir=val_labels_dir,
        )
    else:
        print(f"Val directory not found: {val_dir}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
