"""Helper functions for image file operations."""

import base64
import io
import logging
import tempfile
import os
import uuid
from datetime import datetime
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


def load_and_save_to_temp(
    image_path: Optional[str] = None,
    base64_image: Optional[str] = None,
    original_filename: Optional[str] = None
) -> str:
    """
    Load an image from path or base64 and save it to a temporary file.
    If original_filename is provided, generates a unique filename with datetime and GUID.
    
    Args:
        image_path: Path to image file (optional if base64_image provided)
        base64_image: Base64-encoded image (optional if image_path provided)
        original_filename: Original filename to use for generating unique name (e.g., "photo.jpg")
        
    Returns:
        Full path to the saved temporary file
        
    Raises:
        ValueError: If neither image_path nor base64_image is provided
    """
    logger.info("Loading and encoding image...")
    image = None
    
    if base64_image:
        # Decode to verify and get size
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
    elif image_path:
        # Load from file path
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        raise ValueError("Either image_path or base64_image must be provided")
    
    logger.info(f"Loaded image: {image.size}, mode: {image.mode}")
    
    # Generate unique filename if original_filename is provided
    if original_filename:
        # Create unique filename: originalname_YYYYMMDD_HHMMSS_GUID.ext (preserves original extension)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_guid = str(uuid.uuid4())
        name_without_ext, ext = os.path.splitext(original_filename)
        # Keep original extension, default to .png if none
        if not ext:
            ext = '.png'
        unique_filename = f"{name_without_ext}_{timestamp}_{file_guid}{ext}"
        
        # Save in temp directory with unique filename
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, unique_filename)
        logger.info(f"Generated unique filename: {unique_filename}")
        
        # Determine save format based on extension
        save_format = ext[1:].upper() if ext else 'PNG'
        if save_format == 'JPG':
            save_format = 'JPEG'
    else:
        # Use default temp file naming
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False)
        temp_image_path = temp_file.name
        temp_file.close()
        save_format = 'PNG'
    
    try:
        image.save(temp_image_path, format=save_format)
        logger.info(f"Saved image to temporary file: {temp_image_path}")
        return temp_image_path
    except Exception as e:
        logger.error(f"Failed to save temporary image: {e}")
        raise
