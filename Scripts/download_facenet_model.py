"""
Download FaceNet model for DeepFace and store in custom location.
"""

import os
from pathlib import Path
import shutil

# Set custom DeepFace home directory
SCRIPT_DIR = Path(__file__).parent.parent
CUSTOM_DEEPFACE_HOME = SCRIPT_DIR / "Video_Models" / "RetinaFace"
WEIGHTS_DIR = CUSTOM_DEEPFACE_HOME / ".deepface" / "weights"

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Set environment variable BEFORE importing DeepFace
os.environ['DEEPFACE_HOME'] = str(CUSTOM_DEEPFACE_HOME)

print(f"Custom DeepFace home: {CUSTOM_DEEPFACE_HOME}")
print(f"Weights directory: {WEIGHTS_DIR}")
print("-" * 50)

# Now import DeepFace (it will use the custom path)
from deepface import DeepFace

print("\nDownloading FaceNet model...")
print("This may take a few minutes (~92 MB)...")

try:
    # Build the FaceNet model (triggers download)
    model = DeepFace.build_model('Facenet')
    
    print("\n✓ FaceNet model downloaded successfully!")
    print(f"  Location: {WEIGHTS_DIR}")
    
    # List downloaded files
    if WEIGHTS_DIR.exists():
        files = list(WEIGHTS_DIR.glob('*'))
        if files:
            print(f"\nDownloaded files:")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
    
except Exception as e:
    print(f"\n✗ Error downloading model: {e}")
    raise

print("\n" + "=" * 50)
print("Model download complete!")
print("=" * 50)
