"""YOLO data preparation module - converts COCO JSON to YOLO format"""

import json
import shutil
import random
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom
from clear_data import clear_data_folders

# Configuration variables
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42  # For deterministic splitting

# Paths
INPUT_IMAGES_DIR = Path("input/images")
INPUT_LABELS_DIR = Path("input/labels")
OUTPUT_BASE_DIR = Path("data")
TRAIN_DIR = OUTPUT_BASE_DIR / "train"
VAL_DIR = OUTPUT_BASE_DIR / "val"
TEST_DIR = OUTPUT_BASE_DIR / "test"


def validate_splits():
    """Validate that splits sum to approximately 1.0"""
    total = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Splits must sum to 1.0, but got {total} "
            f"(TRAIN={TRAIN_SPLIT}, VAL={VAL_SPLIT}, TEST={TEST_SPLIT})"
        )


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format.
    
    Args:
        bbox: [x, y, width, height] in absolute pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [center_x, center_y, width, height] normalized 0-1
    """
    x, y, width, height = bbox
    
    # Calculate center coordinates
    center_x = (x + width / 2.0) / img_width
    center_y = (y + height / 2.0) / img_height
    
    # Normalize width and height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return [center_x, center_y, norm_width, norm_height]


def coco_bbox_to_voc(bbox):
    """
    Convert COCO bbox format [x, y, width, height] to VOC format [xmin, ymin, xmax, ymax].
    
    Args:
        bbox: [x, y, width, height] in absolute pixels
    
    Returns:
        [xmin, ymin, xmax, ymax] in absolute pixels
    """
    x, y, width, height = bbox
    return [x, y, x + width, y + height]


def create_coco_xml(json_data):
    """
    Create COCO XML structure from COCO JSON data.
    
    Args:
        json_data: Dictionary containing COCO JSON structure
    
    Returns:
        ElementTree Element representing the XML structure
    """
    root = ET.Element("annotations")
    
    # Add images section
    images_elem = ET.SubElement(root, "images")
    for img in json_data.get("images", []):
        img_elem = ET.SubElement(images_elem, "image")
        ET.SubElement(img_elem, "id").text = str(img["id"])
        ET.SubElement(img_elem, "file_name").text = img["file_name"]
        ET.SubElement(img_elem, "width").text = str(img["width"])
        ET.SubElement(img_elem, "height").text = str(img["height"])
    
    # Add annotations section
    annotations_elem = ET.SubElement(root, "annotations")
    for ann in json_data.get("annotations", []):
        ann_elem = ET.SubElement(annotations_elem, "annotation")
        ET.SubElement(ann_elem, "id").text = str(ann["id"])
        ET.SubElement(ann_elem, "image_id").text = str(ann["image_id"])
        ET.SubElement(ann_elem, "category_id").text = str(ann["category_id"])
        
        # Bbox as space-separated values
        bbox_str = " ".join(str(x) for x in ann["bbox"])
        ET.SubElement(ann_elem, "bbox").text = bbox_str
        ET.SubElement(ann_elem, "area").text = str(ann["area"])
        ET.SubElement(ann_elem, "iscrowd").text = str(ann["iscrowd"])
    
    # Add categories section
    categories_elem = ET.SubElement(root, "categories")
    for cat in json_data.get("categories", []):
        cat_elem = ET.SubElement(categories_elem, "category")
        ET.SubElement(cat_elem, "id").text = str(cat["id"])
        ET.SubElement(cat_elem, "name").text = cat["name"]
    
    return root


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def process_json_file(json_path, output_dir):
    """
    Process a single COCO JSON file and create YOLO TXT and COCO XML files.
    
    Args:
        json_path: Path to the input JSON file
        output_dir: Base output directory (train/val/test)
    """
    # Read JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Validate JSON structure
    if "images" not in json_data or not json_data["images"]:
        print(f"Warning: No images found in {json_path}")
        return None
    
    image_info = json_data["images"][0]
    image_filename = image_info["file_name"]
    img_width = image_info["width"]
    img_height = image_info["height"]
    
    # Get base filename without extension
    base_name = Path(image_filename).stem
    
    # Copy image to appropriate directory
    input_image_path = INPUT_IMAGES_DIR / image_filename
    output_image_dir = output_dir / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    if input_image_path.exists():
        shutil.copy2(input_image_path, output_image_dir / image_filename)
    else:
        print(f"Warning: Image file not found: {input_image_path}")
        return None
    
    # Create YOLO TXT label file
    output_labels_dir = output_dir / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_labels_dir / f"{base_name}.txt"
    
    annotations = json_data.get("annotations", [])
    with open(txt_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            # Convert category_id from COCO (1-indexed) to YOLO (0-indexed)
            class_id = ann["category_id"] - 1
            
            # Convert bbox to YOLO format
            yolo_bbox = coco_bbox_to_yolo(ann["bbox"], img_width, img_height)
            
            # Write to file: class_id center_x center_y width height
            f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                   f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    # Create COCO XML file
    output_xml_dir = output_dir / "xml"
    output_xml_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_xml_dir / f"{base_name}.xml"
    
    xml_root = create_coco_xml(json_data)
    xml_string = prettify_xml(xml_root)
    
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)
    
    return base_name


def split_data(json_files):
    """
    Split JSON files into train/val/test sets based on configured ratios.
    
    Args:
        json_files: List of JSON file paths
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    # Set random seed for deterministic splitting
    random.seed(RANDOM_SEED)
    
    # Shuffle files deterministically
    shuffled = json_files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_count = int(total * TRAIN_SPLIT)
    val_count = int(total * VAL_SPLIT)
    # test_count is the remainder
    
    train_files = shuffled[:train_count]
    val_files = shuffled[train_count:train_count + val_count]
    test_files = shuffled[train_count + val_count:]
    
    return train_files, val_files, test_files


def is_directory_empty_except_gitkeep(dir_path):
    """
    Check if a directory is empty except for .gitkeep files.
    
    Args:
        dir_path: Path to the directory to check
    
    Returns:
        True if directory is empty (or only contains .gitkeep), False otherwise
    """
    if not dir_path.exists():
        return True
    
    for item in dir_path.iterdir():
        if item.name != ".gitkeep":
            return False
    
    return True


def check_test_data_populated():
    """
    Check if test data directories (labels, xml, images) are populated.
    
    Returns:
        True if any test directory contains files (other than .gitkeep), False otherwise
    """
    test_labels = TEST_DIR / "labels"
    test_xml = TEST_DIR / "xml"
    test_images = TEST_DIR / "images"
    
    return (not is_directory_empty_except_gitkeep(test_labels) or
            not is_directory_empty_except_gitkeep(test_xml) or
            not is_directory_empty_except_gitkeep(test_images))


def main():
    """Main function to process all input files and create training data."""
    print("Starting YOLO data preparation...")
    
    if check_test_data_populated():
        print("\nData folder already populated, clearing existing data...")
        return
    
    # Validate splits
    try:
        validate_splits()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Ensure input directories exist
    if not INPUT_IMAGES_DIR.exists():
        print(f"Error: Input images directory not found: {INPUT_IMAGES_DIR}")
        return
    
    if not INPUT_LABELS_DIR.exists():
        print(f"Error: Input labels directory not found: {INPUT_LABELS_DIR}")
        return
    
    # Get all JSON files
    json_files = sorted(list(INPUT_LABELS_DIR.glob("*.json")))
    
    if not json_files:
        print(f"Error: No JSON files found in {INPUT_LABELS_DIR}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Split data
    train_files, val_files, test_files = split_data(json_files)
    
    print("\nSplit configuration:")
    print(f"  Train: {len(train_files)} files ({TRAIN_SPLIT*100:.1f}%)")
    print(f"  Val: {len(val_files)} files ({VAL_SPLIT*100:.1f}%)")
    print(f"  Test: {len(test_files)} files ({TEST_SPLIT*100:.1f}%)")
    
    # Process train files
    print("\nProcessing train files...")
    train_processed = 0
    for json_path in train_files:
        result = process_json_file(json_path, TRAIN_DIR)
        if result:
            train_processed += 1
    
    # Process val files
    print("Processing val files...")
    val_processed = 0
    for json_path in val_files:
        result = process_json_file(json_path, VAL_DIR)
        if result:
            val_processed += 1
    
    # Process test files
    print("Processing test files...")
    test_processed = 0
    for json_path in test_files:
        result = process_json_file(json_path, TEST_DIR)
        if result:
            test_processed += 1
    
    print("\nProcessing complete!")
    print(f"  Train: {train_processed} files processed")
    print(f"  Val: {val_processed} files processed")
    print(f"  Test: {test_processed} files processed")


if __name__ == "__main__":
    main()
