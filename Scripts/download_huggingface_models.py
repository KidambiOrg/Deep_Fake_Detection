# """
# Download HuggingFace models to local Models folder

# This script downloads Prithiv and Organika models from HuggingFace
# and saves them to the Models folder for blob storage upload.
# """

# import os
# import sys
# from pathlib import Path

# # Add parent directory to path
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipImageProcessor, SiglipForImageClassification


# def download_model(model_name, output_dir, use_siglip=False):
#     """
#     Download a HuggingFace model and save to local directory
    
#     Args:
#         model_name: HuggingFace model identifier
#         output_dir: Local directory to save the model
#         use_siglip: If True, use SigLIP-specific classes instead of Auto classes
#     """
#     print(f"\n{'='*80}")
#     print(f"Downloading: {model_name}")
#     print(f"Output: {output_dir}")
#     if use_siglip:
#         print("Model Type: SigLIP")
#     print(f"{'='*80}\n")
    
#     try:
#         # Create output directory
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         # Download processor
#         print("Downloading processor...")
#         if use_siglip:
#             try:
#                 processor = SiglipImageProcessor.from_pretrained(model_name)
#                 processor.save_pretrained(output_dir)
#                 print("✓ SigLIP Processor saved")
#             except Exception as e:
#                 print(f"Failed with SigLIP processor, trying Auto: {e}")
#                 processor = AutoImageProcessor.from_pretrained(model_name)
#                 processor.save_pretrained(output_dir)
#                 print("✓ Auto Processor saved")
#         else:
#             processor = AutoImageProcessor.from_pretrained(model_name)
#             processor.save_pretrained(output_dir)
#             print("✓ Processor saved")
        
#         # Download model
#         print("Downloading model...")
#         if use_siglip:
#             try:
#                 model = SiglipForImageClassification.from_pretrained(model_name)
#                 model.save_pretrained(output_dir)
#                 print("✓ SigLIP Model saved")
#             except Exception as e:
#                 print(f"Failed with SigLIP model, trying Auto: {e}")
#                 model = AutoModelForImageClassification.from_pretrained(model_name)
#                 model.save_pretrained(output_dir)
#                 print("✓ Auto Model saved")
#         else:
#             model = AutoModelForImageClassification.from_pretrained(model_name)
#             model.save_pretrained(output_dir)
#             print("✓ Model saved")
        
#         print(f"\n✅ Successfully downloaded {model_name}")
#         print(f"   Saved to: {output_dir}\n")
#         return True
        
#     except Exception as e:
#         print(f"\n❌ Failed to download {model_name}: {e}\n")
#         return False


# def main():
#     """Download all required HuggingFace models"""
    
#     # Get project root
#     script_dir = Path(__file__).parent
#     project_root = script_dir.parent
#     models_dir = project_root / "Models"
    
#     # Models to download
#     models = [
#         {
#             'name': 'prithivMLMods/deepfake-detector-model-v1',
#             'output': models_dir / 'huggingface' / 'prithivMLMods' / 'deepfake-detector-model-v1',
#             'use_siglip': False
#         },
#         {
#             'name': 'Organika/sdxl-detector',
#             'output': models_dir / 'huggingface' / 'Organika' / 'sdxl-detector',
#             'use_siglip': False
#         },
#         {
#             'name': 'prithivMLmods/open-deepfake-detection',
#             'output': models_dir / 'huggingface' / 'prithiv_open_deep_fake',
#             'use_siglip': True  # This model uses SigLIP architecture
#         }
#     ]
    
#     print("\n" + "="*80)
#     print("HuggingFace Model Downloader")
#     print("="*80)
#     print(f"\nModels directory: {models_dir}")
#     print(f"Total models to download: {len(models)}\n")
    
#     successful = 0
#     failed = 0
    
#     for model_info in models:
#         use_siglip = model_info.get('use_siglip', False)
#         if download_model(model_info['name'], str(model_info['output']), use_siglip=use_siglip):
#             successful += 1
#         else:
#             failed += 1
    
#     # Summary
#     print("\n" + "="*80)
#     print("Download Summary")
#     print("="*80)
#     print(f"✓ Successful: {successful}")
#     print(f"✗ Failed: {failed}")
#     print(f"📦 Total: {successful + failed}")
#     print("="*80 + "\n")
    
#     if failed > 0:
#         sys.exit(1)
#     else:
#         print("✅ All models downloaded successfully!")
#         print("\nNext steps:")
#         print("1. Run upload script to upload models to blob storage")
#         print("2. Update model registry with blob storage paths")
#         sys.exit(0)


# if __name__ == "__main__":
#     main()
