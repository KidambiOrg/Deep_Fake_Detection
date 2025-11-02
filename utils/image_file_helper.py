"""Helper functions for image file operations."""

import base64
import io
import logging
import tempfile
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


def load_and_save_to_temp(
    image_path: Optional[str] = None,
    base64_image: Optional[str] = None
) -> str:
    """
    Load an image from path or base64 and save it to a temporary file.
    
    Args:
        image_path: Path to image file (optional if base64_image provided)
        base64_image: Base64-encoded image (optional if image_path provided)
        
    Returns:
        Temporary file path
        
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
    
    # Save image to temporary file and pass reference instead of base64
    # This avoids corruption issues when passing large strings through LLM
    temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False)
    try:
        image.save(temp_file, format='PNG')
        temp_image_path = temp_file.name
        temp_file.close()
        logger.info(f"Saved image to temporary file: {temp_image_path}")
        return temp_image_path
    except Exception as e:
        logger.error(f"Failed to save temporary image: {e}")
        temp_file.close()
        raise
