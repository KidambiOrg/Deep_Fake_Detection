"""
Azure Blob Storage helper for downloading and caching models
"""
import os
import logging
from pathlib import Path
from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import tempfile
import hashlib

logger = logging.getLogger(__name__)


def _load_environment_settings():
    """
    Load environment variables from .env file using python-dotenv.
    Works for both standalone Python scripts and Azure Functions.
    """
    try:
        from dotenv import load_dotenv
        
        # Find .env file in project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent  # Go up one level to project root
        env_path = project_root / ".env"
        
        if env_path.exists():
            # Load .env file (doesn't override existing environment variables)
            load_dotenv(dotenv_path=env_path, override=False)
            logger.info("Loaded settings from .env file")
        else:
            logger.debug(f".env file not found at {env_path}")
            
    except ImportError:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")


# Load settings when module is imported
_load_environment_settings()


class BlobStorageModelLoader:
    """Helper class to download and cache models from Azure Blob Storage"""
    
    def __init__(self, storage_account_name: Optional[str] = None, connection_string: Optional[str] = None):
        """
        Initialize blob storage client using Managed Identity (DefaultAzureCredential) or connection string.
        Prefers Managed Identity for better security.
        
        Args:
            storage_account_name: Azure Storage account name (for Managed Identity).
                                If None, reads from environment variable STORAGE_ACCOUNT_NAME
            connection_string: Azure Storage connection string (fallback option).
                             If None, reads from environment variable MODEL_BLOB_STORAGE
        """
        # Try Managed Identity first (recommended for production)
        self.storage_account_name = storage_account_name or os.environ.get("STORAGE_ACCOUNT_NAME")
        
        if self.storage_account_name:
            try:
                # Use DefaultAzureCredential with explicit logging
                logger.info(f"Attempting to connect to storage account: {self.storage_account_name}")
                
                # For local development, prioritize Azure CLI and exclude IMDS (cloud-only)
                credential = DefaultAzureCredential(
                    exclude_managed_identity_credential=True,  # Exclude IMDS for local dev
                    exclude_shared_token_cache_credential=False,
                    exclude_visual_studio_code_credential=False,
                    exclude_azure_cli_credential=False,  # Use Azure CLI (az login)
                    exclude_interactive_browser_credential=False,
                    exclude_environment_credential=False
                )
                account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=credential
                )
                
                # Test the connection by listing containers
                try:
                    # This will verify authentication works
                    _ = list(self.blob_service_client.list_containers(results_per_page=1))
                    logger.info(f"Azure Blob Storage client initialized with Managed Identity: {self.storage_account_name}")
                except Exception as test_error:
                    logger.error(f"Managed Identity authentication failed during connection test: {test_error}")
                    logger.info("This could be due to:")
                    logger.info("  1. Missing RBAC permissions (need 'Storage Blob Data Reader' or 'Storage Blob Data Contributor')")
                    logger.info("  2. Not logged in via 'az login' for local development")
                    logger.info("  3. Wrong Azure subscription selected")
                    logger.info("  4. Network/firewall restrictions on storage account")
                    logger.info("Falling back to connection string authentication...")
                    self.blob_service_client = None
                    
            except Exception as e:
                logger.warning(f"Failed to initialize with Managed Identity: {e}")
                logger.info("Falling back to connection string authentication...")
                self.blob_service_client = None
        else:
            self.blob_service_client = None
        
        # Fallback to connection string if Managed Identity not configured or failed
        if not self.blob_service_client:
            self.connection_string = connection_string or os.environ.get("MODEL_BLOB_STORAGE")
            
            if not self.connection_string:
                logger.warning("Neither STORAGE_ACCOUNT_NAME nor MODEL_BLOB_STORAGE configured. Model loading from blob storage disabled.")
                self.blob_service_client = None
            else:
                try:
                    self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                    logger.info("Azure Blob Storage client initialized with connection string")
                except Exception as e:
                    logger.error(f"Failed to initialize blob storage client: {e}")
                    self.blob_service_client = None
        
        # Set up local cache directory
        if self.blob_service_client:
            self.cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None
    
    def _get_cache_path(self, blob_path: str) -> Path:
        """Get local cache path for a blob"""
        if not self.cache_dir:
            raise RuntimeError("Cache directory not initialized")
        # Create hash of blob path to avoid path length issues
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        filename = Path(blob_path).name
        return self.cache_dir / f"{path_hash}_{filename}"
    
    def download_model(
        self, 
        container_name: str, 
        blob_name: str, 
        force_download: bool = False
    ) -> Optional[str]:
        """
        Download model from blob storage to local cache
        
        Args:
            container_name: Name of the blob container
            blob_name: Name of the blob (file path in container)
            force_download: If True, re-download even if cached
            
        Returns:
            Local file path to downloaded model, or None if download failed
        """
        if not self.blob_service_client:
            logger.warning("Blob storage client not initialized")
            return None
        
        cache_path = self._get_cache_path(f"{container_name}/{blob_name}")
        
        # Check if already cached
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached model: {cache_path}")
            return str(cache_path)
        
        try:
            logger.info(f"Downloading model from blob storage: {container_name}/{blob_name}")
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Download to cache
            with open(cache_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"Model downloaded successfully to: {cache_path}")
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Failed to download model from blob storage: {e}")
            return None
    
    def download_directory(
        self,
        container_name: str,
        blob_prefix: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download all blobs with given prefix (directory) from blob storage
        
        Args:
            container_name: Name of the blob container
            blob_prefix: Prefix/directory path in container
            force_download: If True, re-download even if cached
            
        Returns:
            Local directory path containing downloaded files, or None if download failed
        """
        if not self.blob_service_client or not self.cache_dir:
            logger.warning("Blob storage client not initialized")
            return None
        
        try:
            # Create local directory based on prefix
            local_dir = self.cache_dir / container_name / blob_prefix
            local_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading directory from blob storage: {container_name}/{blob_prefix}")
            
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # List and download all blobs with prefix
            blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
            downloaded_count = 0
            
            for blob in blob_list:
                # Get relative path within the prefix
                relative_path = blob.name[len(blob_prefix):].lstrip('/')
                local_file_path = local_dir / relative_path
                
                # Skip if already cached and not forcing download
                if local_file_path.exists() and not force_download:
                    logger.debug(f"Using cached file: {local_file_path}")
                    continue
                
                # Create subdirectories if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download blob
                blob_client = container_client.get_blob_client(blob.name)
                with open(local_file_path, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
                
                downloaded_count += 1
                logger.debug(f"Downloaded: {blob.name}")
            
            logger.info(f"Directory download complete. Downloaded {downloaded_count} files to: {local_dir}")
            return local_dir
            
        except Exception as e:
            logger.error(f"Failed to download directory from blob storage: {e}")
            return None
    
    def clear_cache(self):
        """Clear the local model cache"""
        import shutil
        try:
            if self.cache_dir and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global singleton instance
_blob_loader: Optional[BlobStorageModelLoader] = None


def get_blob_loader() -> BlobStorageModelLoader:
    """Get or create global blob storage loader instance"""
    global _blob_loader
    if _blob_loader is None:
        _blob_loader = BlobStorageModelLoader()
    return _blob_loader
