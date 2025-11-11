"""
Script to upload deepfake detection models to Azure Blob Storage

This script uploads all models to Azure Blob Storage following the structure
defined in the model registry. It supports both Managed Identity and 
connection string authentication.

Usage:
    # Using Managed Identity (recommended)
    set STORAGE_ACCOUNT_NAME=deepfakemodels
    python Scripts/upload_models_to_blob.py

    # Using Connection String
    set MODEL_BLOB_STORAGE=DefaultEndpointsProtocol=https;AccountName=...
    python Scripts/upload_models_to_blob.py

    # Dry run (show what would be uploaded without uploading)
    python Scripts/upload_models_to_blob.py --dry-run

    # Force re-upload even if file exists
    python Scripts/upload_models_to_blob.py --force
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from imagedetectionlib.model_registry import MODEL_REGISTRY


class ModelUploader:
    """Upload models to Azure Blob Storage"""
    
    def __init__(self, storage_account_name=None, connection_string=None):
        """
        Initialize blob storage client
        
        Args:
            storage_account_name: Storage account name (for Managed Identity)
            connection_string: Connection string (fallback)
        """
        self.storage_account_name = storage_account_name or os.environ.get("STORAGE_ACCOUNT_NAME")
        
        # Try Managed Identity first
        if self.storage_account_name:
            try:
                print(f"Connecting to storage account: {self.storage_account_name}")
                credential = DefaultAzureCredential()
                account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=credential
                )
                print("✓ Connected using Managed Identity (DefaultAzureCredential)")
            except Exception as e:
                print(f"Failed to connect with Managed Identity: {e}")
                self.blob_service_client = None
        else:
            self.blob_service_client = None
        
        # Fallback to connection string
        if not self.blob_service_client:
            self.connection_string = connection_string or os.environ.get("MODEL_BLOB_STORAGE")
            
            if not self.connection_string:
                raise ValueError(
                    "Neither STORAGE_ACCOUNT_NAME nor MODEL_BLOB_STORAGE configured.\n"
                    "Please set one of these environment variables."
                )
            
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                print("✓ Connected using connection string")
            except Exception as e:
                raise ValueError(f"Failed to initialize blob storage client: {e}")
    
    def create_container_if_not_exists(self, container_name):
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container()
                print(f"✓ Created container: {container_name}")
            else:
                print(f"✓ Container exists: {container_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to create/check container {container_name}: {e}")
            return False
    
    def upload_file(self, local_path, container_name, blob_name, force=False, dry_run=False):
        """
        Upload a single file to blob storage
        
        Args:
            local_path: Local file path
            container_name: Blob container name
            blob_name: Blob name (path in container)
            force: Force upload even if blob exists
            dry_run: Don't actually upload, just show what would happen
            
        Returns:
            True if uploaded/exists, False if failed
        """
        if not os.path.exists(local_path):
            print(f"✗ Local file not found: {local_path}")
            return False
        
        file_size = os.path.getsize(local_path)
        size_mb = file_size / (1024 * 1024)
        
        try:
            # Ensure container exists first
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                if not dry_run:
                    container_client.create_container()
                    print(f"   ✓ Created container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Check if blob already exists
            exists = blob_client.exists()
            
            if exists and not force:
                print(f"   ⚠️  Skipped (already exists): {blob_name} ({size_mb:.2f} MB)")
                return True
            
            if dry_run:
                action = "Would upload" if not exists or force else "Would skip"
                print(f"   [DRY RUN] {action}: {local_path} → {container_name}/{blob_name} ({size_mb:.2f} MB)")
                return True
            
            # Upload the file
            print(f"   ⬆️  Uploading: {blob_name} ({size_mb:.2f} MB)...", end=" ")
            
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=force)
            
            print("✓ Done")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed to upload {blob_name}: {e}")
            return False
    
    def upload_directory(self, local_dir, container_name, blob_prefix, force=False, dry_run=False):
        """
        Upload all files in a directory to blob storage
        
        Args:
            local_dir: Local directory path
            container_name: Blob container name
            blob_prefix: Prefix for blobs (directory path in container)
            force: Force upload even if blobs exist
            dry_run: Don't actually upload
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not os.path.exists(local_dir):
            print(f"✗ Local directory not found: {local_dir}")
            return (0, 1)
        
        successful = 0
        failed = 0
        
        # Walk through directory
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Calculate relative path from local_dir
                relative_path = os.path.relpath(local_path, local_dir)
                
                # Construct blob name (use forward slashes)
                blob_name = os.path.join(blob_prefix, relative_path).replace('\\', '/')
                
                # Upload file
                if self.upload_file(local_path, container_name, blob_name, force, dry_run):
                    successful += 1
                else:
                    failed += 1
        
        return (successful, failed)
    
    def upload_models_from_registry(self, force=False, dry_run=False):
        """
        Upload all models defined in MODEL_REGISTRY
        
        Args:
            force: Force upload even if blobs exist
            dry_run: Don't actually upload
        """
        print("\n" + "="*80)
        print("Uploading Models from Registry")
        print("="*80 + "\n")
        
        total_successful = 0
        total_failed = 0
        
        for model_key, model_info in MODEL_REGISTRY.items():
            print(f"\n📦 Processing: {model_key}")
            print(f"   {model_info.get('description', 'No description')}")
            
            # Skip if no blob_storage configuration
            if 'blob_storage' not in model_info:
                print("   ⚠️  No blob_storage configuration, skipping")
                continue
            
            blob_config = model_info['blob_storage']
            container_name = blob_config.get('container')
            
            if not container_name:
                print("   ✗ No container specified in blob_storage config")
                total_failed += 1
                continue
            
            # Handle single file upload
            if 'blob' in blob_config:
                blob_name = blob_config['blob']
                local_path = model_info.get('model_path')
                
                if not local_path:
                    print(f"   ✗ No model_path specified for {model_key}")
                    total_failed += 1
                    continue
                
                if self.upload_file(local_path, container_name, blob_name, force, dry_run):
                    total_successful += 1
                else:
                    total_failed += 1
            
            # Handle directory upload
            elif 'path' in blob_config:
                blob_prefix = blob_config['path']
                local_dir = model_info.get('local_path')
                
                if not local_dir:
                    print(f"   ✗ No local_path specified for {model_key}")
                    total_failed += 1
                    continue
                
                successful, failed = self.upload_directory(
                    local_dir, container_name, blob_prefix, force, dry_run
                )
                total_successful += successful
                total_failed += failed
            
            else:
                print("   ✗ Invalid blob_storage config (need 'blob' or 'path')")
                total_failed += 1
        
        # Summary
        print("\n" + "="*80)
        print("Upload Summary")
        print("="*80)
        print(f"✓ Successful: {total_successful}")
        print(f"✗ Failed: {total_failed}")
        print(f"📦 Total: {total_successful + total_failed}")
        print("="*80 + "\n")
        
        return total_successful, total_failed
    
    def upload_lr_voting_model(self, force=False, dry_run=False):
        """
        Upload the LR meta-classifier (voting model) to blob storage.
        This is separate from detection models.
        
        Args:
            force: Force upload even if blob exists
            dry_run: Don't actually upload
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        print("\n" + "="*80)
        print("Uploading LR Meta-Classifier (Voting Model)")
        print("="*80 + "\n")
        
        container_name = "models"
        blob_name = "LrVotingModel/lr_meta_classifier.pkl"
        local_path = "Models/LrVotingModel/lr_meta_classifier.pkl"
        
        print("📦 LR Meta-Classifier (Logistic Regression Voting Model)")
        print("   Used internally by ensemble voting to combine model predictions")
        
        if self.upload_file(local_path, container_name, blob_name, force, dry_run):
            print("\n✓ LR voting model uploaded successfully")
            return (1, 0)
        else:
            print("\n✗ Failed to upload LR voting model")
            return (0, 1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Upload deepfake detection models to Azure Blob Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all models using Managed Identity
  set STORAGE_ACCOUNT_NAME=deepfakemodels
  python Scripts/upload_models_to_blob.py

  # Upload with connection string
  set MODEL_BLOB_STORAGE=DefaultEndpointsProtocol=https;...
  python Scripts/upload_models_to_blob.py

  # Dry run to see what would be uploaded
  python Scripts/upload_models_to_blob.py --dry-run

  # Force re-upload all models
  python Scripts/upload_models_to_blob.py --force
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-upload even if files already exist in blob storage'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    
    parser.add_argument(
        '--storage-account',
        type=str,
        help='Azure Storage account name (overrides STORAGE_ACCOUNT_NAME env var)'
    )
    
    parser.add_argument(
        '--connection-string',
        type=str,
        help='Azure Storage connection string (overrides MODEL_BLOB_STORAGE env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize uploader
        uploader = ModelUploader(
            storage_account_name=args.storage_account,
            connection_string=args.connection_string
        )
        
        # Upload detection models from registry
        successful, failed = uploader.upload_models_from_registry(
            force=args.force,
            dry_run=args.dry_run
        )
        
        # Upload LR voting model (not in registry, separate internal model)
        lr_successful, lr_failed = uploader.upload_lr_voting_model(
            force=args.force,
            dry_run=args.dry_run
        )
        
        # Combine results
        total_successful = successful + lr_successful
        total_failed = failed + lr_failed
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"✓ Total Successful: {total_successful}")
        print(f"✗ Total Failed: {total_failed}")
        print(f"📦 Grand Total: {total_successful + total_failed}")
        print("="*80 + "\n")
        
        # Exit with appropriate code
        if total_failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
