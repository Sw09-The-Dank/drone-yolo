"""Utility to clear data folders while preserving .gitkeep files"""

import shutil
from pathlib import Path

# Paths to data directories
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


def clear_subdirectory(subdir_path, preserve_gitkeep=True):
    """
    Clear all contents of a subdirectory (images, labels, xml) while preserving .gitkeep files.
    
    Args:
        subdir_path: Path to the subdirectory to clear
        preserve_gitkeep: If True, preserve .gitkeep files (default: True)
    """
    if not subdir_path.exists():
        return
    
    # Check if .gitkeep exists
    gitkeep_path = subdir_path / ".gitkeep"
    has_gitkeep = gitkeep_path.exists()
    
    # Remove all contents
    for item in subdir_path.iterdir():
        if item.is_file():
            # Skip .gitkeep files
            if preserve_gitkeep and item.name == ".gitkeep":
                continue
            item.unlink()
        elif item.is_dir():
            # Remove subdirectories
            shutil.rmtree(item)
    
    # Restore .gitkeep if it existed
    if preserve_gitkeep and has_gitkeep:
        gitkeep_path.touch()


def clear_data_folders():
    """Clear images, labels, and xml subdirectories while preserving .gitkeep files."""
    print("Clearing data folders...")
    
    directories = [
        ("train", TRAIN_DIR),
        ("val", VAL_DIR),
        ("test", TEST_DIR),
    ]
    
    subdirs = ["images", "labels", "xml"]
    
    for name, directory in directories:
        if not directory.exists():
            print(f"  ⚠ {name} folder does not exist: {directory}")
            continue
        
        print(f"  Clearing {name} folder...")
        for subdir_name in subdirs:
            subdir_path = directory / subdir_name
            if subdir_path.exists():
                clear_subdirectory(subdir_path, preserve_gitkeep=True)
                print(f"    ✓ {subdir_name}/ cleared")
            else:
                print(f"    ⚠ {subdir_name}/ does not exist")
        
        print(f"  ✓ {name} folder cleared")
    
    print("\nData folders cleared successfully!")


def main():
    """Main function"""
    clear_data_folders()


if __name__ == "__main__":
    main()

