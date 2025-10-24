"""
Additional cleanup - Remove pytorch_model.bin if model.safetensors exists

Safetensors is the preferred format and is more secure and faster to load.
If both exist, we only need to keep model.safetensors.
"""

from pathlib import Path

def remove_pytorch_model_bin():
    """Remove pytorch_model.bin if model.safetensors exists"""
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "Models" / "dimavit"
    
    safetensors_file = model_dir / "model.safetensors"
    pytorch_bin_file = model_dir / "pytorch_model.bin"
    
    print("="*60)
    print("Optimizing model storage")
    print("="*60)
    
    if not safetensors_file.exists():
        print("\n⚠ model.safetensors not found. Keeping pytorch_model.bin")
        return False
    
    if not pytorch_bin_file.exists():
        print("\n✓ pytorch_model.bin already removed!")
        return True
    
    # Both files exist - remove pytorch_model.bin
    safetensors_size = safetensors_file.stat().st_size / (1024 * 1024)
    pytorch_size = pytorch_bin_file.stat().st_size / (1024 * 1024)
    
    print(f"\nFound both model formats:")
    print(f"  - model.safetensors: {safetensors_size:.2f} MB (preferred)")
    print(f"  - pytorch_model.bin: {pytorch_size:.2f} MB (redundant)")
    
    print(f"\nRemoving pytorch_model.bin to save {pytorch_size:.2f} MB...")
    
    try:
        pytorch_bin_file.unlink()
        print(f"✓ Removed pytorch_model.bin")
        
        print("\n" + "="*60)
        print("Final model files:")
        print("="*60)
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  ✓ {item.name} ({size_mb:.2f} MB)")
        
        print("\n" + "="*60)
        print(f"✓ Optimization complete! Saved {pytorch_size:.2f} MB")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    remove_pytorch_model_bin()
