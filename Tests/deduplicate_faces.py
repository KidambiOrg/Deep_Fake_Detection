"""
Face deduplication script using DeepFace VGG-Face model.
Compares all face images in a folder and deletes duplicates (same person).
"""

import os
from pathlib import Path
import sys
import time

# Set custom DeepFace home directory BEFORE importing DeepFace
SCRIPT_DIR = Path(__file__).parent.parent
CUSTOM_DEEPFACE_HOME = SCRIPT_DIR / "Video_Models" / "RetinaFace"
os.environ['DEEPFACE_HOME'] = str(CUSTOM_DEEPFACE_HOME)

from deepface import DeepFace


def deduplicate_faces(faces_folder: str, model_name: str = "Facenet", 
                     distance_metric: str = "cosine", threshold: float = None):
    """
    Remove duplicate faces from a folder using DeepFace verification.
    
    Logic:
    - Compare each face with all previously kept faces
    - If face matches any kept face (same person), delete it
    - If face is unique, keep it for future comparisons
    
    Args:
        faces_folder: Folder containing extracted face images
        model_name: DeepFace model to use (default: Facenet - fast & accurate)
        distance_metric: Distance metric (cosine, euclidean, euclidean_l2)
        threshold: Custom threshold for verification (None = use model default)
    
    Returns:
        Dictionary with deduplication statistics
    """
    print(f"\n{'='*50}")
    print("FACE DEDUPLICATION")
    print(f"{'='*50}")
    print(f"Folder: {faces_folder}")
    print(f"Model: {model_name}")
    print(f"Distance Metric: {distance_metric}")
    print(f"Threshold: {threshold if threshold else 'Auto (model default)'}")
    print("-" * 50)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_faces = [
        os.path.join(faces_folder, f) 
        for f in os.listdir(faces_folder) 
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not all_faces:
        print("No face images found in folder")
        return {
            'total_faces': 0,
            'unique_faces': 0,
            'duplicates_removed': 0,
            'errors': 0
        }
    
    all_faces.sort()  # Sort for consistent processing order
    total_faces = len(all_faces)
    print(f"\nTotal faces to process: {total_faces}")
    
    start_time = time.time()
    
    # Track unique faces and statistics
    unique_faces = []
    duplicates_removed = 0
    errors = 0
    
    print(f"\nProcessing faces...")
    
    for idx, current_face in enumerate(all_faces, 1):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total_faces} faces processed...")
        
        try:
            # Check if current face is a duplicate of any unique face
            is_duplicate = False
            
            for unique_face in unique_faces:
                try:
                    # Verify if current face matches this unique face
                    result = DeepFace.verify(
                        img1_path=current_face,
                        img2_path=unique_face,
                        model_name=model_name,
                        distance_metric=distance_metric,
                        enforce_detection=False  # Skip if face not detected
                    )
                    
                    # Check if they're the same person
                    if result["verified"]:
                        is_duplicate = True
                        duplicates_removed += 1
                        
                        # Delete duplicate image
                        os.remove(current_face)
                        print(f"  ✗ Removed duplicate: {os.path.basename(current_face)} "
                              f"(matches {os.path.basename(unique_face)}, "
                              f"distance: {result['distance']:.4f})")
                        break
                
                except Exception as e:
                    # If verification fails, continue checking other faces
                    continue
            
            # If not a duplicate, add to unique faces list
            if not is_duplicate:
                unique_faces.append(current_face)
                if idx % 10 == 0 or idx == 1:
                    print(f"  ✓ Kept unique face: {os.path.basename(current_face)}")
        
        except Exception as e:
            errors += 1
            print(f"  ⚠ Error processing {os.path.basename(current_face)}: {e}")
            # Keep the face if there's an error
            unique_faces.append(current_face)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*50}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total faces processed: {total_faces}")
    print(f"Unique faces kept: {len(unique_faces)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Errors encountered: {errors}")
    print(f"\nTime elapsed: {elapsed_time:.2f}s")
    print(f"Avg time per face: {elapsed_time/total_faces:.3f}s")
    print(f"{'='*50}")
    
    return {
        'total_faces': total_faces,
        'unique_faces': len(unique_faces),
        'duplicates_removed': duplicates_removed,
        'errors': errors,
        'time_elapsed': elapsed_time
    }


def main():
    """
    Main function to run face deduplication.
    
    Usage:
        py deduplicate_faces.py <faces_folder> [model_name] [distance_metric] [threshold]
    
    Examples:
        py deduplicate_faces.py C:\path\to\faces
        py deduplicate_faces.py C:\path\to\faces VGG-Face
        py deduplicate_faces.py C:\path\to\faces VGG-Face cosine
        py deduplicate_faces.py C:\path\to\faces VGG-Face cosine 0.4
    """
    if len(sys.argv) < 2:
        print("Error: Please provide faces folder path")
        print("Usage: py deduplicate_faces.py <faces_folder> [model_name] [distance_metric] [threshold]")
        sys.exit(1)
    
    faces_folder = sys.argv[1]
    
    if not os.path.exists(faces_folder):
        print(f"Error: Folder not found: {faces_folder}")
        sys.exit(1)
    
    # Optional parameters
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Facenet"
    distance_metric = sys.argv[3] if len(sys.argv) > 3 else "cosine"
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None
    
    try:
        results = deduplicate_faces(
            faces_folder=faces_folder,
            model_name=model_name,
            distance_metric=distance_metric,
            threshold=threshold
        )
        print("\nFace deduplication completed successfully!")
        
    except Exception as e:
        print(f"\nError during face deduplication: {str(e)}")
        raise


if __name__ == "__main__":
    main()
