"""
Utility to clear cached text detection models downloaded from Azure Blob Storage.
This removes all cached text models from the temporary directory.
"""
import sys
import os
from pathlib import Path
import logging
import shutil

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from textdetectionlib.model_registry import TEXT_MODEL_REGISTRY
from utils.blob_storage_helper import get_blob_loader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_text_model_cache(dry_run: bool = False) -> dict:
    """
    Clear cached text detection models from local cache.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
        
    Returns:
        Dictionary with statistics about cleared cache
    """
    stats = {
        'total_files_removed': 0,
        'total_size_freed_mb': 0.0,
        'models_affected': [],
        'cache_dir': None,
        'errors': []
    }
    
    try:
        # Get blob loader to access cache directory
        blob_loader = get_blob_loader()
        
        if not blob_loader or not blob_loader.cache_dir:
            logger.warning("Blob storage not configured or cache directory not available")
            return stats
        
        cache_dir = blob_loader.cache_dir
        stats['cache_dir'] = str(cache_dir)
        
        if not cache_dir.exists():
            logger.info("Cache directory does not exist - nothing to clear")
            return stats
        
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"{'DRY RUN - ' if dry_run else ''}Clearing text model cache...")
        
        # Track text model blob names from registry
        text_model_blobs = set()
        text_model_files = set()
        
        for model_key, model_info in TEXT_MODEL_REGISTRY.items():
            model_files = model_info.get('model_files', {})
            
            for file_type, file_config in model_files.items():
                blob_config = file_config.get('blob_storage')
                if blob_config:
                    container = blob_config['container']
                    blob_name = blob_config['blob']
                    blob_path = f"{container}/{blob_name}"
                    text_model_blobs.add(blob_path)
                    
                    # Extract just the filename for matching cached files
                    filename = Path(blob_name).name
                    text_model_files.add(filename)
                    
                    if model_key not in stats['models_affected']:
                        stats['models_affected'].append(model_key)
        
        logger.info(f"Found {len(stats['models_affected'])} text models in registry: {stats['models_affected']}")
        logger.info(f"Tracking {len(text_model_files)} text model files: {text_model_files}")
        
        # Scan cache directory for text model files
        files_to_remove = []
        total_size = 0
        
        for cache_file in cache_dir.rglob('*'):
            if cache_file.is_file():
                # Check if this file belongs to a text model
                # Cache files are named with hash prefix: {hash}_{filename}
                # Extract the actual filename after the hash
                
                filename = cache_file.name
                
                # Check if in textmodels container subdirectory
                relative_path = cache_file.relative_to(cache_dir)
                in_textmodels_dir = str(relative_path).startswith('textmodels/')
                
                # Check if filename matches any text model file (after hash prefix)
                matches_text_model = any(
                    text_file in filename 
                    for text_file in text_model_files
                )
                
                if in_textmodels_dir or matches_text_model:
                    # This is a text model cache file
                    file_size = cache_file.stat().st_size
                    files_to_remove.append(cache_file)
                    total_size += file_size
                    logger.debug(f"Found cached file: {cache_file.name} ({file_size / 1024 / 1024:.2f} MB)")
        
        stats['total_files_removed'] = len(files_to_remove)
        stats['total_size_freed_mb'] = total_size / 1024 / 1024
        
        logger.info(f"Found {len(files_to_remove)} cached text model files")
        logger.info(f"Total size: {stats['total_size_freed_mb']:.2f} MB")
        
        # Remove files
        if not dry_run:
            for cache_file in files_to_remove:
                try:
                    cache_file.unlink()
                    logger.debug(f"Removed: {cache_file.name}")
                except Exception as e:
                    error_msg = f"Failed to remove {cache_file.name}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            # Remove empty directories in textmodels container
            textmodels_dir = cache_dir / 'textmodels'
            if textmodels_dir.exists():
                try:
                    # Remove empty subdirectories
                    for subdir in textmodels_dir.rglob('*'):
                        if subdir.is_dir() and not any(subdir.iterdir()):
                            subdir.rmdir()
                            logger.debug(f"Removed empty directory: {subdir}")
                    
                    # Remove textmodels directory if empty
                    if textmodels_dir.exists() and not any(textmodels_dir.iterdir()):
                        textmodels_dir.rmdir()
                        logger.info("Removed empty 'textmodels' directory")
                except Exception as e:
                    logger.warning(f"Failed to clean up empty directories: {e}")
            
            logger.info(f"✓ Cleared {stats['total_files_removed']} files, freed {stats['total_size_freed_mb']:.2f} MB")
        else:
            logger.info(f"DRY RUN: Would remove {len(files_to_remove)} files ({stats['total_size_freed_mb']:.2f} MB)")
        
        return stats
        
    except Exception as e:
        error_msg = f"Error clearing text model cache: {e}"
        logger.error(error_msg)
        stats['errors'].append(error_msg)
        return stats


def clear_all_text_models():
    """Clear all cached text models (convenience function)"""
    return clear_text_model_cache(dry_run=False)


def preview_text_cache_clear():
    """Preview what would be cleared without actually deleting (dry run)"""
    return clear_text_model_cache(dry_run=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear cached text detection models")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "="*60)
    print("Text Model Cache Cleaner")
    print("="*60 + "\n")
    
    stats = clear_text_model_cache(dry_run=args.dry_run)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Models Affected: {', '.join(stats['models_affected']) if stats['models_affected'] else 'None'}")
    print(f"Files {'to be removed' if args.dry_run else 'removed'}: {stats['total_files_removed']}")
    print(f"Space {'to be freed' if args.dry_run else 'freed'}: {stats['total_size_freed_mb']:.2f} MB")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors']:
            print(f"  - {error}")
    
    print("="*60 + "\n")
    
    if args.dry_run:
        print("This was a DRY RUN. No files were actually deleted.")
        print("Run without --dry-run to actually clear the cache.\n")
