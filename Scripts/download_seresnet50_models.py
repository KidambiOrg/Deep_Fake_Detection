# #!/usr/bin/env python3
# """
# Download SEResNet-50 models for deepfake detection

# This script downloads the pretrained SEResNet-50 model using torch.hub
# and saves it to the Models/SERESNet_50 directory.
# """

# import os
# import sys
# import torch
# import torch.hub
# from pathlib import Path
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def download_seresnet50_model(model_dir: str = None):
#     """
#     Download SEResNet-50 model using torch.hub
    
#     Args:
#         model_dir (str): Directory to save the model. If None, uses Models/SERESNet_50
#     """
#     if model_dir is None:
#         # Get the script directory and go up to the project root
#         script_dir = Path(__file__).parent
#         project_root = script_dir.parent
#         model_dir_path = project_root / "Models" / "SERESNet_50"
#     else:
#         model_dir_path = Path(model_dir)
    
#     # Create directory if it doesn't exist
#     model_dir_path.mkdir(parents=True, exist_ok=True)
    
#     logger.info(f"Downloading SEResNet-50 model to: {model_dir_path}")
    
#     try:
#         # Check if we already have the weights file
#         weights_file = model_dir_path / "seresnet50-60a8950a85b2b.pkl"
#         if weights_file.exists():
#             logger.info(f"Using existing weights file: {weights_file}")
#         else:
#             logger.info("Downloading SEResNet-50 weights from GitHub releases...")
#             import urllib.request
#             urllib.request.urlretrieve(
#                 "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl",
#                 weights_file
#             )
#             logger.info(f"Downloaded weights to: {weights_file}")
        
#         # Load the model using the weights
#         logger.info("Loading SEResNet-50 model with pretrained weights...")
        
#         # Import the se_resnet50 function from our module
#         import sys
#         sys.path.append(str(model_dir_path.parent.parent / "detectionLib"))
#         from detect_deepfake_seresnet50 import se_resnet50
        
#         model = se_resnet50(num_classes=1000, pretrained=True, weights_path=str(weights_file))
        
#         # Save the model state dict
#         model_path = model_dir_path / "seresnet50_pretrained.pth"
#         logger.info(f"Saving model state dict to: {model_path}")
#         torch.save(model.state_dict(), model_path)
        
#         # Save the complete model for easier loading
#         complete_model_path = model_dir_path / "seresnet50_complete_model.pth"
#         logger.info(f"Saving complete model to: {complete_model_path}")
#         torch.save(model, complete_model_path)
        
#         # Create a model info file
#         info_path = model_dir_path / "model_info.txt"
#         with open(info_path, 'w') as f:
#             f.write("SEResNet-50 Model Information\n")
#             f.write("============================\n\n")
#             f.write("Source: moskomule/senet.pytorch\n")
#             f.write("Model: se_resnet50\n")
#             f.write("Pretrained: True (ImageNet)\n")
#             f.write("Architecture: SEResNet-50 with Squeeze-and-Excitation blocks\n")
#             f.write("Input size: 224x224x3\n")
#             f.write("Number of classes: 1000 (ImageNet)\n")
#             f.write("Paper: Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)\n")
#             f.write("Weights URL: https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl\n\n")
#             f.write("Files:\n")
#             f.write("- seresnet50-60a8950a85b2b.pkl: Official pretrained weights from GitHub\n")
#             f.write("- seresnet50_pretrained.pth: Model state dictionary\n")
#             f.write("- seresnet50_complete_model.pth: Complete model (easier to load)\n")
#             f.write("- model_info.txt: This information file\n")
        
#         logger.info("[SUCCESS] SEResNet-50 model downloaded successfully!")
#         logger.info(f"Model files saved in: {model_dir_path}")
        
#         # Display model information
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
#         logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
#         return model_path, complete_model_path
        
#     except Exception as e:
#         logger.error(f"[ERROR] Error downloading SEResNet-50 model: {str(e)}")
#         logger.error("Make sure you have internet connection and the required dependencies.")
#         sys.exit(1)

# def main():
#     """Main function to download SEResNet-50 models"""
#     logger.info("Starting SEResNet-50 model download...")
    
#     try:
#         # Download the model
#         model_path, complete_model_path = download_seresnet50_model()
        
#         logger.info("\n" + "="*50)
#         logger.info("SEResNet-50 Model Download Complete!")
#         logger.info("="*50)
#         logger.info(f"State dict saved to: {model_path}")
#         logger.info(f"Complete model saved to: {complete_model_path}")
#         logger.info("\nYou can now use SEResNet-50 for deepfake detection!")
        
#     except KeyboardInterrupt:
#         logger.info("\n[ERROR] Download cancelled by user")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"[ERROR] Unexpected error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()