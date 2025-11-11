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
import threading

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
    
    def __init__(self, storage_account_name: Optional[str] = None):
        """
        Initialize blob storage configuration (lazy initialization - connection happens on first use).
        Uses Managed Identity (DefaultAzureCredential) for authentication.
        
        Args:
            storage_account_name: Azure Storage account name (for Managed Identity).
                                If None, reads from environment variable STORAGE_ACCOUNT_NAME
        """
        # Store configuration but don't connect yet (lazy initialization)
        self.storage_account_name = storage_account_name or os.environ.get("STORAGE_ACCOUNT_NAME")
        
        # Blob client will be initialized on first use
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._initialization_attempted = False
        self._init_lock = threading.Lock()  # Thread-safe initialization
        
        # Set up local cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    @property
    def blob_service_client(self) -> Optional[BlobServiceClient]:
        """
        Lazy initialization of blob service client (thread-safe).
        Only connects to blob storage when actually needed (after cache check fails).
        """
        # Fast path: if already initialized, return immediately (no lock needed)
        if self._initialization_attempted:
            return self._blob_service_client
        
        # Slow path: need to initialize (with lock for thread safety)
        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialization_attempted:
                return self._blob_service_client
            
            # Try Managed Identity first (recommended for production)
            if self.storage_account_name:
                try:
                    # Use DefaultAzureCredential with explicit logging
                    logger.info(f"Connecting to storage account: {self.storage_account_name}")
                    
                    # Check if running in Azure (Container Apps, App Service, etc.)
                    is_azure_environment = os.environ.get("WEBSITE_INSTANCE_ID") or os.environ.get("CONTAINER_APP_NAME")
                    
                    if is_azure_environment:
                        # Running in Azure - use Managed Identity (IMDS)
                        logger.info("Detected Azure environment - using Managed Identity")
                        credential = DefaultAzureCredential(
                            exclude_managed_identity_credential=False,  # Enable Managed Identity for Azure
                            exclude_shared_token_cache_credential=True,
                            exclude_visual_studio_code_credential=True,
                            exclude_azure_cli_credential=True,
                            exclude_interactive_browser_credential=True,
                            exclude_environment_credential=False
                        )
                    else:
                        # Local development - prioritize Azure CLI
                        logger.info("Detected local environment - using Azure CLI credential")
                        credential = DefaultAzureCredential(
                            exclude_managed_identity_credential=True,  # Exclude IMDS for local dev
                            exclude_shared_token_cache_credential=False,
                            exclude_visual_studio_code_credential=False,
                            exclude_azure_cli_credential=False,  # Use Azure CLI (az login)
                            exclude_interactive_browser_credential=False,
                            exclude_environment_credential=False
                        )
                    
                    account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                    
                    # Enable Azure SDK logging for authentication debugging
                    import sys
                    azure_logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
                    azure_logger.setLevel(logging.INFO)
                    azure_logger.addHandler(logging.StreamHandler(sys.stdout))
                    
                    self._blob_service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                        logging_enable=True
                    )
                    
                    logger.info(f"Azure Blob Storage client initialized with Managed Identity: {self.storage_account_name}")
                    logger.info(f"Storage account URL: {account_url}")
                    
                    # Mark as initialized ONLY after successful client creation
                    self._initialization_attempted = True
                    return self._blob_service_client
                        
                except Exception as e:
                    logger.error(f"Failed to initialize with Managed Identity: {e}")
                    # Don't set _initialization_attempted - allow retry
            else:
                logger.warning("STORAGE_ACCOUNT_NAME not configured. Model loading from blob storage disabled.")
                # Mark as attempted to avoid repeated warnings
                self._initialization_attempted = True
            
            return None
    
    def _get_cache_path(self, blob_path: str) -> Path:
        """Get local cache path for a blob"""
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
        
        # Check if already cached (cache hit - skip blob connection)
        if cache_path.exists() and cache_path.stat().st_size > 0 and not force_download:
            logger.info(f"Using cached model (skipping blob connection): {cache_path}")
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
            # Clean up empty/corrupt file if created
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(f"Removed corrupt cache file: {cache_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove corrupt cache file: {cleanup_error}")
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
            # BUGFIX: Convert blob prefix forward slashes to OS path separator for Windows compatibility
            blob_prefix_normalized = blob_prefix.replace('/', os.sep)
            local_dir = self.cache_dir / container_name / blob_prefix_normalized
            
            # Check if directory already exists and has files (cache hit)
            if local_dir.exists() and any(local_dir.iterdir()) and not force_download:
                logger.info(f"Using cached model directory (skipping blob connection): {local_dir}")
                return local_dir
            
            # Cache miss or force download - create directory and download from blob
            local_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading directory from blob storage: {container_name}/{blob_prefix}")
            
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # List and download all blobs with prefix
            blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
            downloaded_count = 0
            
            for blob in blob_list:
                try:
                    # Get relative path within the prefix
                    relative_path = blob.name[len(blob_prefix):].lstrip('/')
                    
                    # BUGFIX: Replace forward slashes with OS-appropriate path separator
                    # Convert blob path separators to Windows path separators
                    relative_path = relative_path.replace('/', os.sep)
                    
                    local_file_path = local_dir / relative_path
                    
                    logger.info(f"Processing blob: {blob.name}")
                    logger.info(f"Relative path: {relative_path}")
                    logger.info(f"Local file path: {local_file_path}")
                    
                    # Skip if already cached and not forcing download
                    if local_file_path.exists() and not force_download:
                        logger.debug(f"Using cached file: {local_file_path}")
                        continue
                    
                    # Create subdirectories if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download blob
                    blob_client = container_client.get_blob_client(blob.name)
                    
                    # BUGFIX: Download to temp file first, then move to avoid file locking issues
                    import tempfile
                    import shutil
                    
                    local_file_str = str(local_file_path)
                    logger.info(f"Downloading to: {local_file_str}")
                    
                    # Create temp file in same directory to ensure same filesystem
                    temp_fd, temp_path = tempfile.mkstemp(dir=str(local_file_path.parent), suffix='.tmp')
                    
                    try:
                        # Download to temp file
                        with os.fdopen(temp_fd, 'wb') as temp_file:
                            blob_data = blob_client.download_blob()
                            blob_data.readinto(temp_file)
                        
                        # Move temp file to final location (atomic operation)
                        shutil.move(temp_path, local_file_str)
                        
                        downloaded_count += 1
                        logger.info(f"Successfully downloaded: {blob.name}")
                        
                    except Exception as temp_error:
                        # Clean up temp file on error
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        raise temp_error
                    
                except Exception as blob_error:
                    logger.error(f"Error downloading blob {blob.name}: {blob_error}")
                    logger.exception(blob_error)
                    raise  # Re-raise to trigger outer exception handler
            
            logger.info(f"Directory download complete. Downloaded {downloaded_count} files to: {local_dir}")
            return local_dir
            
        except Exception as e:
            logger.error(f"Failed to download directory from blob storage: {e}")
            return None
    
    def clear_cache(self):
        """
        Clear the local model cache.
        This includes all cached models:
        - Image detection models (HuggingFace, MobileNet, etc.)
        - LR meta-classifier (voting model)
        """
        import shutil
        try:
            if self.cache_dir and self.cache_dir.exists():
                # Log what's being cleared
                cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)
                logger.info(f"Clearing model cache ({cache_size_mb:.2f} MB): {self.cache_dir}")
                
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Model cache cleared (includes all image models and LR voting model)")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global singleton instance
_blob_loader: Optional[BlobStorageModelLoader] = None
_blob_loader_lock = threading.Lock()


def get_blob_loader() -> BlobStorageModelLoader:
    """Get or create global blob storage loader instance (thread-safe)"""
    global _blob_loader
    
    # Fast path: if already initialized, return immediately
    if _blob_loader is not None:
        return _blob_loader
    
    # Slow path: need to create instance (with lock for thread safety)
    with _blob_loader_lock:
        # Double-check after acquiring lock
        if _blob_loader is None:
            _blob_loader = BlobStorageModelLoader()
        return _blob_loader
