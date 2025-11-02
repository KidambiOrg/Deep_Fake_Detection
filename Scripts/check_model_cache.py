"""
Quick diagnostic script to check model cache status
"""
import tempfile
import hashlib
from pathlib import Path

def check_cache_status():
    """Check if models are cached and where"""
    
    cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
    
    print(f"\n=== Model Cache Diagnostic ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")
    
    if not cache_dir.exists():
        print("\n❌ Cache directory does not exist!")
        print("Run warmup endpoint to download models.")
        return
    
    print(f"\n📁 Cache directory contents:")
    all_files = list(cache_dir.rglob("*"))
    
    if not all_files:
        print("   (empty)")
        return
    
    # Show all files
    for file_path in sorted(all_files):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            relative = file_path.relative_to(cache_dir)
            print(f"   {relative} ({size_mb:.2f} MB)")
    
    # Check specific models
    print(f"\n🔍 Model-specific checks:")
    
    models_to_check = [
        {
            'name': 'NoEscape Full',
            'container': 'models',
            'blob': 'nonescape/nonescape-v0.safetensors'
        },
        {
            'name': 'SEResNet50',
            'container': 'models',
            'blob': 'seresnet50/seresnet50_complete_model.pth'
        }
    ]
    
    for model_info in models_to_check:
        blob_path = f"{model_info['container']}/{model_info['blob']}"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        filename = Path(model_info['blob']).name
        cache_path = cache_dir / f"{path_hash}_{filename}"
        
        print(f"\n   {model_info['name']}:")
        print(f"      Blob path: {blob_path}")
        print(f"      Hash: {path_hash}")
        print(f"      Expected cache path: {cache_path}")
        print(f"      Exists: {'✅' if cache_path.exists() else '❌'}")
        
        if cache_path.exists():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"      Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    check_cache_status()
