"""
Script to clean corrupt SEResNet50 model cache and trigger re-download
"""
import tempfile
import hashlib
import os
from pathlib import Path

def clean_seresnet50_cache():
    """Remove corrupt SEResNet50 cache file"""
    
    cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
    
    # SEResNet50 specific
    blob_path = "models/seresnet50/seresnet50_complete_model.pth"
    path_hash = hashlib.md5(blob_path.encode()).hexdigest()
    cache_path = cache_dir / f"{path_hash}_seresnet50_complete_model.pth"
    
    print(f"\n=== Cleaning SEResNet50 Cache ===")
    print(f"Cache path: {cache_path}")
    
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"Current size: {size_mb:.2f} MB")
        
        if size_mb < 1.0:
            print(f"⚠️  File appears corrupt (too small)")
            try:
                os.remove(cache_path)
                print(f"✅ Removed corrupt cache file")
                print(f"\n💡 Next step: Run warmup endpoint to re-download the model")
            except Exception as e:
                print(f"❌ Failed to remove file: {e}")
        else:
            print(f"✅ File appears valid")
    else:
        print(f"ℹ️  Cache file does not exist")

if __name__ == "__main__":
    clean_seresnet50_cache()
