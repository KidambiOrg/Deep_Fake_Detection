"""Tool for downloading models from Azure Blob Storage to cache."""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def download_model_from_blob(
    container_name: str,
    blob_name: Optional[str] = None,
    blob_prefix: Optional[str] = None,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Download model files from Azure Blob Storage to local cache.
    
    This is an AI agent tool function that downloads models from blob storage
    and caches them locally for faster loading. Supports both single file
    downloads and directory downloads.
    
    Args:
        container_name: Azure blob container name (e.g., 'models', 'textmodels')
        blob_name: Optional specific blob file to download (e.g., 'nonescape/nonescape-v0.safetensors')
        blob_prefix: Optional directory prefix to download all files (e.g., 'huggingface/prithivMLMods')
        force_download: If True, re-download even if cached (default: False)
        
    Returns:
        Dictionary with status and message:
        {
            "status": "success" or "error",
            "message": "Description of result",
            "blob_name": "Blob name or prefix"
        }
    """
    try:
        from utils.blob_storage_helper import get_blob_loader
        
        # Validate inputs
        if not blob_name and not blob_prefix:
            return {
                "status": "error",
                "message": "Either blob_name or blob_prefix must be provided",
                "blob_name": None
            }
        
        if blob_name and blob_prefix:
            return {
                "status": "error",
                "message": "Cannot specify both blob_name and blob_prefix - choose one",
                "blob_name": None
            }
        
        # Get blob loader instance
        blob_loader = get_blob_loader()
        
        if not blob_loader or not blob_loader.blob_service_client:
            return {
                "status": "error",
                "message": "Blob storage client not initialized - check STORAGE_ACCOUNT_NAME configuration",
                "blob_name": blob_name or blob_prefix
            }
        
        # Download single file
        if blob_name:
            logger.info(f"Downloading single file: {container_name}/{blob_name}")
            local_path = blob_loader.download_model(
                container_name=container_name,
                blob_name=blob_name,
                force_download=force_download
            )
            
            if local_path:
                return {
                    "status": "success",
                    "message": f"Successfully downloaded model file to cache",
                    "blob_name": blob_name
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to download model file from blob storage",
                    "blob_name": blob_name
                }
        
        # Download directory
        if blob_prefix:
            logger.info(f"Downloading directory: {container_name}/{blob_prefix}")
            local_path = blob_loader.download_directory(
                container_name=container_name,
                blob_prefix=blob_prefix,
                force_download=force_download
            )
            
            if local_path:
                return {
                    "status": "success",
                    "message": f"Successfully downloaded model directory to cache",
                    "blob_name": blob_prefix
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to download model directory from blob storage",
                    "blob_name": blob_prefix
                }
        
        # Fallback (should not reach here due to validation)
        return {
            "status": "error",
            "message": "Invalid configuration - no blob_name or blob_prefix specified",
            "blob_name": None
        }
        
    except Exception as e:
        logger.error(f"Error downloading from blob storage: {e}")
        return {
            "status": "error",
            "message": f"Exception during download: {str(e)}",
            "blob_name": blob_name or blob_prefix
        }


def download_multiple_models(
    model_configs: List[Dict[str, str]],
    force_download: bool = True
) -> Dict[str, Any]:
    """
    Download multiple models from Azure Blob Storage in sequence.
    Always overwrites cache by default.
    
    Args:
        model_configs: List of model configurations, each containing:
            - container_name: Container name
            - blob_name: Optional blob file name
            - blob_prefix: Optional blob directory prefix
        force_download: If True, re-download even if cached (default: True to always overwrite)
        
    Returns:
        Dictionary with overall status and per-model results:
        {
            "status": "success" or "partial" or "error",
            "total_models": int,
            "successful": int,
            "failed": int,
            "results": [list of individual download results]
        }
    """
    results = []
    successful = 0
    failed = 0
    
    for config in model_configs:
        container = config.get('container_name')
        blob_name = config.get('blob_name')
        blob_prefix = config.get('blob_prefix')
        
        # Skip if container is missing
        if not container:
            results.append({
                "status": "error",
                "message": "Missing container_name in configuration",
                "blob_name": None
            })
            failed += 1
            continue
        
        result = download_model_from_blob(
            container_name=container,
            blob_name=blob_name,
            blob_prefix=blob_prefix,
            force_download=force_download
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    # Determine overall status
    if failed == 0:
        overall_status = "success"
    elif successful == 0:
        overall_status = "error"
    else:
        overall_status = "partial"
    
    return {
        "status": overall_status,
        "total_models": len(model_configs),
        "successful": successful,
        "failed": failed,
        "results": results
    }
