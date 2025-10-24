"""
Clean up the Models/dimavit folder by removing unnecessary files

Keeps only essential model files:
- config.json (model configuration)
- preprocessor_config.json (image preprocessing config)
- model.safetensors or pytorch_model.bin (model weights)

Removes:
- Checkpoint folders (training artifacts)
- Training arguments
- README files
- Cache folders
- .gitattributes
"""

import shutil
from pathlib import Path

def cleanup_dimavit_folder():
    """Remove unnecessary files from Models/dimavit"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "Models" / "dimavit"
    
    if not model_dir.exists():
        print(f"Error: {model_dir} does not exist")
        return False
    
    print("="*60)
    print("Cleaning up Models/dimavit folder")
    print("="*60)
    
    # Files to keep
    keep_files = {
        'config.json',
        'preprocessor_config.json',
        'model.safetensors',  # Preferred format
        'pytorch_model.bin',  # Alternative format
    }
    
    # Items to remove
    remove_items = []
    
    # Scan directory
    for item in model_dir.iterdir():
        if item.name not in keep_files:
            remove_items.append(item)
    
    if not remove_items:
        print("\n✓ Folder is already clean!")
        return True
    
    print(f"\nFound {len(remove_items)} items to remove:\n")
    
    total_size_before = 0
    for item in remove_items:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_size_before += size_mb
            print(f"  - {item.name} ({size_mb:.2f} MB)")
        elif item.is_dir():
            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_mb = dir_size / (1024 * 1024)
            total_size_before += size_mb
            print(f"  - {item.name}/ ({size_mb:.2f} MB)")
    
    print(f"\nTotal size to be removed: {total_size_before:.2f} MB")
    
    # Remove items
    print("\nRemoving items...")
    for item in remove_items:
        try:
            if item.is_file():
                item.unlink()
                print(f"  ✓ Removed: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed: {item.name}/")
        except Exception as e:
            print(f"  ✗ Error removing {item.name}: {e}")
    
    # Show what's left
    print("\n" + "="*60)
    print("Remaining files in Models/dimavit:")
    print("="*60)
    
    total_size_after = 0
    for item in sorted(model_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_size_after += size_mb
            print(f"  ✓ {item.name} ({size_mb:.2f} MB)")
    
    print(f"\nTotal size after cleanup: {total_size_after:.2f} MB")
    print(f"Space saved: {total_size_before:.2f} MB")
    
    print("\n" + "="*60)
    print("✓ Cleanup completed successfully!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    cleanup_dimavit_folder()
