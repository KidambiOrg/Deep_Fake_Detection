"""
Download RetinaFace and VGG-Face models for video face extraction.
"""

import os
from deepface import DeepFace
from pathlib import Path

def download_models():
    """
    Download RetinaFace detector and VGG-Face recognition models.
    """
    # Set the models directories
    retinaface_models_dir = Path(__file__).parent.parent / "Video_Models" / "RetinaFace"
    vgg_models_dir = Path(__file__).parent.parent / "Video_Models" / "VGG"
    retinaface_models_dir.mkdir(parents=True, exist_ok=True)
    vgg_models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading models for video face extraction...")
    print(f"RetinaFace target directory: {retinaface_models_dir}")
    print(f"VGG-Face target directory: {vgg_models_dir}")
    print("=" * 60)
    
    # Set environment variable to use custom models directory for RetinaFace
    os.environ['DEEPFACE_HOME'] = str(retinaface_models_dir)
    
    print("\n1. Downloading RetinaFace detector model...")
    try:
        # Trigger RetinaFace download by using it
        DeepFace.extract_faces(
            img_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
            detector_backend="retinaface",
            enforce_detection=False
        )
        print("✓ RetinaFace model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading RetinaFace: {str(e)}")
    
    print("\n2. Downloading OpenCV detector (for VGG) and creating cache...")
    try:
        # Use opencv detector to create the .pkl cache file in RetinaFace folder
        os.environ['DEEPFACE_HOME'] = str(retinaface_models_dir)
        DeepFace.extract_faces(
            img_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
            detector_backend="opencv",
            enforce_detection=False
        )
        print("✓ OpenCV detector cache created successfully")
    except Exception as e:
        print(f"✗ Error creating OpenCV cache: {str(e)}")
    
    print("\n3. Downloading VGG-Face recognition model...")
    try:
        # Switch to VGG models directory
        os.environ['DEEPFACE_HOME'] = str(vgg_models_dir)
        
        # Trigger VGG-Face download by using represent (embedding extraction)
        DeepFace.represent(
            img_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
            model_name="VGG-Face",
            enforce_detection=False
        )
        print("✓ VGG-Face model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading VGG-Face: {str(e)}")
    
    print("\n4. Creating OpenCV detector cache in VGG folder...")
    try:
        # Create .pkl cache file in VGG folder too
        DeepFace.extract_faces(
            img_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
            detector_backend="opencv",
            enforce_detection=False
        )
        print("✓ OpenCV detector cache created in VGG folder")
    except Exception as e:
        print(f"✗ Error creating OpenCV cache in VGG: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Model download completed!")
    print("=" * 60)
    
    # List downloaded files from both directories
    print("\nRetinaFace files:")
    if retinaface_models_dir.exists():
        for item in retinaface_models_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(retinaface_models_dir)} ({size_mb:.2f} MB)")
    
    print("\nVGG-Face files:")
    if vgg_models_dir.exists():
        for item in vgg_models_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(vgg_models_dir)} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    download_models()
