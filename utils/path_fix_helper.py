"""
Helper utility to fix duplicated file paths.

Azure Agent Framework bug: On retries, image_path can get duplicated like:
'C:\\path\\image.jpgC:\\path\\image.jpg'

This helper detects and fixes such duplications.
"""
import logging

logger = logging.getLogger(__name__)


def fix_duplicated_path(path: str) -> str:
    """
    Fix duplicated file paths that may occur during Azure Agent Framework retries.
    
    Detects patterns like: "path.jpgpath.jpg" → "path.jpg"
    
    Args:
        path: Input path (may be duplicated)
        
    Returns:
        Fixed path (de-duplicated)
    """
    # Check if path contains multiple file extensions (sign of duplication)
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    for ext in extensions:
        count = path.lower().count(ext.lower())
        if count > 1:
            logger.warning(f"Detected {count} occurrences of '{ext}' in path: {path[:100]}...")
            
            # Check if it's a simple duplication (path repeated twice)
            half_len = len(path) // 2
            first_half = path[:half_len]
            second_half = path[half_len:]
            
            if first_half == second_half:
                fixed_path = first_half
                logger.info(f"Fixed duplicated path: {fixed_path}")
                return fixed_path
            
            # If not exact duplication, try to find the first complete path
            # Look for the last occurrence of the extension
            last_ext_pos = path.lower().rfind(ext.lower())
            if last_ext_pos > 0:
                # Find the second-to-last occurrence
                second_last_pos = path.lower()[:last_ext_pos].rfind(ext.lower())
                if second_last_pos > 0:
                    # Extract path ending at first extension occurrence
                    fixed_path = path[:second_last_pos + len(ext)]
                    logger.info(f"Fixed overlapping path: {fixed_path}")
                    return fixed_path
    
    # No duplication detected, return original
    return path
