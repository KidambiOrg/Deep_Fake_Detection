#!/usr/bin/env python3
"""
Download script for DistilDIRE pretrained models

This script downloads the pretrained DistilDIRE checkpoints from Hugging Face.
"""

import os
import requests
import sys
from pathlib import Path
from typing import Dict, Optional

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to output_path with progress indication
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        chunk_size: Size of chunks to download at a time
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        print(f"Saving to: {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n[SUCCESS] Successfully downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error downloading {url}: {e}")
        return False

def main():
    """Main function to download DistilDIRE models"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "Models" / "distil_dire"
    
    # DistilDIRE model URLs and filenames
    models = {
        "celebahq-distil-dire-34e.pth": "https://huggingface.co/yevvonlim/distildire/resolve/main/celebahq-distil-dire-34e.pth",
        "imagenet-distil-dire-11e.pth": "https://huggingface.co/yevvonlim/distildire/resolve/main/imagenet-distil-dire-11e.pth"
    }
    
    # ADM diffusion model (required for DistilDIRE)
    adm_models = {
        "256x256_diffusion_uncond.pt": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    }
    
    print("[START] Starting DistilDIRE model downloads...")
    print(f"[FOLDER] Models will be saved to: {models_dir}")
    
    success_count = 0
    total_count = len(models) + len(adm_models)
    
    # Download DistilDIRE models
    for filename, url in models.items():
        output_path = models_dir / filename
        if output_path.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            success_count += 1
        else:
            if download_file(url, str(output_path)):
                success_count += 1
            print()  # New line for cleaner output
    
    # Download ADM diffusion model
    for filename, url in adm_models.items():
        output_path = models_dir / filename
        if output_path.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            success_count += 1
        else:
            if download_file(url, str(output_path)):
                success_count += 1
            print()  # New line for cleaner output
    
    # Summary
    print("=" * 60)
    print(f"[STATS] Download Summary: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("\n📋 Available DistilDIRE models:")
        for filename in models.keys():
            model_path = models_dir / filename
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  • {filename} ({size_mb:.1f} MB)")
        
        adm_path = models_dir / "256x256_diffusion_uncond.pt"
        if adm_path.exists():
            size_mb = adm_path.stat().st_size / (1024 * 1024)
            print(f"  • 256x256_diffusion_uncond.pt ({size_mb:.1f} MB) [ADM Diffusion Model]")
    else:
        print("[WARNING]  Some downloads failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())