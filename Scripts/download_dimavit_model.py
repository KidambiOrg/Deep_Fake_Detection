"""
Download dima806/deepfake_vs_real_image_detection model weights locally

This script downloads the pre-trained model weights from HuggingFace
and saves them to the local Models/dimavit directory.
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path

def download_dimavit_model():
    """Download dima806/deepfake_vs_real_image_detection model to local directory"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "Models" / "dimavit"
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dima806/deepfake_vs_real_image_detection model...")
    print(f"Target directory: {model_dir}")
    
    try:
        # Download the model using snapshot_download
        snapshot_download(
            repo_id="dima806/deepfake_vs_real_image_detection",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        print("\n✓ Model downloaded successfully!")
        print(f"✓ Location: {model_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(model_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    success = download_dimavit_model()
    if success:
        print("\n" + "="*60)
        print("Model is ready to use!")
        print("The model_registry.py will be updated to use local weights.")
        print("="*60)
    else:
        print("\nDownload failed. Please check your internet connection and try again.")
