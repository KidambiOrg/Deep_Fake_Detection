"""
Test script to verify Azure Blob Storage authentication
Tests both Managed Identity and Connection String authentication
"""
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.blob_storage_helper import BlobStorageModelLoader


def test_managed_identity_auth():
    """Test authentication using Managed Identity (DefaultAzureCredential)"""
    print("\n" + "="*70)
    print("Testing Managed Identity Authentication")
    print("="*70)
    
    # Set storage account name (you need to update this)
    storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME", "deepfakemodels")
    
    # if storage_account_name == "deepfakemodels":
    #     print("⚠️  Please set STORAGE_ACCOUNT_NAME environment variable or update the script")
    #     return False
    
    print(f"Storage Account: {storage_account_name}")
    
    try:
        # Initialize with Managed Identity
        loader = BlobStorageModelLoader(storage_account_name=storage_account_name)
        
        if loader.blob_service_client:
            print("✓ Blob storage client initialized with Managed Identity")
            
            # Try to list containers (tests if permissions are correct)
            try:
                containers = loader.blob_service_client.list_containers(results_per_page=5)
                container_names = [c.name for c in containers]
                print(f"✓ Successfully listed containers: {container_names}")
                return True
            except Exception as e:
                print(f"✗ Failed to list containers: {e}")
                print("  Ensure the managed identity has 'Storage Blob Data Reader' role")
                return False
        else:
            print("✗ Failed to initialize blob storage client")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_connection_string_auth():
    """Test authentication using Connection String"""
    print("\n" + "="*70)
    print("Testing Connection String Authentication")
    print("="*70)
    
    connection_string = os.environ.get("MODEL_BLOB_STORAGE")
    
    if not connection_string:
        print("⚠️  MODEL_BLOB_STORAGE environment variable not set")
        print("   Skipping connection string test")
        return None
    
    print("Connection String: ***REDACTED***")
    
    try:
        # Initialize with connection string
        loader = BlobStorageModelLoader(connection_string=connection_string)
        
        if loader.blob_service_client:
            print("✓ Blob storage client initialized with connection string")
            
            # Try to list containers
            try:
                containers = loader.blob_service_client.list_containers(results_per_page=5)
                container_names = [c.name for c in containers]
                print(f"✓ Successfully listed containers: {container_names}")
                return True
            except Exception as e:
                print(f"✗ Failed to list containers: {e}")
                return False
        else:
            print("✗ Failed to initialize blob storage client")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_download():
    """Test downloading a model file"""
    print("\n" + "="*70)
    print("Testing Model Download")
    print("="*70)
    
    from utils.blob_storage_helper import get_blob_loader
    
    loader = get_blob_loader()
    
    if not loader.blob_service_client:
        print("✗ Blob storage not configured")
        return False
    
    # Test parameters (update these to match your setup)
    test_container = "models"
    test_blob = "test/sample.txt"  # Update to an actual blob in your storage
    
    print(f"Container: {test_container}")
    print(f"Blob: {test_blob}")
    
    try:
        local_path = loader.download_model(
            container_name=test_container,
            blob_name=test_blob
        )
        
        if local_path:
            print(f"✓ Model downloaded successfully to: {local_path}")
            
            # Verify file exists
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                print(f"✓ File verified: {file_size} bytes")
                return True
            else:
                print(f"✗ File not found at: {local_path}")
                return False
        else:
            print("✗ Download failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Azure Blob Storage Authentication Test")
    print("="*70)
    print("\nThis script tests both Managed Identity and Connection String authentication")
    print("for Azure Blob Storage.\n")
    
    results = {}
    
    # Test Managed Identity
    results['managed_identity'] = test_managed_identity_auth()
    
    # # Test Connection String
    # results['connection_string'] = test_connection_string_auth()
    
    # # Test Model Download (if any auth worked)
    # if results['managed_identity'] or results['connection_string']:
    #     results['download'] = test_model_download()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else ("⚠️  SKIP" if result is None else "✗ FAIL")
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print("\n" + "="*70)
    
    # Return exit code
    if any(r is True for r in results.values()):
        print("\n✓ At least one authentication method is working!")
        return 0
    else:
        print("\n✗ No authentication method is working. Please check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
