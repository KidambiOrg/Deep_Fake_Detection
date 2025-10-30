"""
EXIF Data Extractor Utility
Extracts EXIF metadata from images and PNG metadata
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Any, Optional
import os
import struct


class ExifExtractor:
    """Extract EXIF metadata from images"""
    
    # Mapping of EXIF tag names to standardized field names
    TAG_MAPPING = {
        'Make': 'make',
        'Model': 'model',
        'Software': 'software',
        'DateTime': 'dateTime',
        'DateTimeOriginal': 'dateTimeOriginal',
        'DateTimeDigitized': 'dateTimeDigitized',
        'ImageWidth': 'imageWidth',
        'ImageLength': 'imageHeight',
        'Orientation': 'orientation',
        'XResolution': 'xResolution',
        'YResolution': 'yResolution',
        'ResolutionUnit': 'resolutionUnit',
        'ColorSpace': 'colorSpace',
        'ExifVersion': 'exifVersion',
        'FlashPixVersion': 'flashPixVersion',
    }
    
    # PNG color type descriptions
    PNG_COLOR_TYPES = {
        0: 'Grayscale',
        2: 'RGB',
        3: 'Palette',
        4: 'Grayscale with Alpha',
        6: 'RGBA'
    }
    
    @staticmethod
    def extract_exif(image_path: str) -> Dict[str, Any]:
        """
        Extract EXIF data from an image file, plus PNG-specific metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing EXIF data and format-specific metadata
        """
        exif_data = {
            'make': None,
            'model': None,
            'software': None,
            'dateTime': None,
            'dateTimeOriginal': None,
            'dateTimeDigitized': None,
            'imageWidth': None,
            'imageHeight': None,
            'orientation': None,
            'xResolution': None,
            'yResolution': None,
            'resolutionUnit': None,
            'colorSpace': None,
            'exifVersion': None,
            'flashPixVersion': None,
            'gpsInfo': None,
            'bitDepth': None,
            'bitsPerChannel': None,
            'channels': None,
            'colorType': None,
            'additionalData': {},
            'hasExif': False
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return exif_data
            
            # Open the image
            with Image.open(image_path) as img:
                # Get basic image info
                exif_data['imageWidth'] = img.width
                exif_data['imageHeight'] = img.height
                
                # Extract bit depth and color information
                ExifExtractor._extract_bit_depth_info(img, exif_data)
                
                # Extract format-specific metadata (JFIF for JPEG, PNG chunks for PNG)
                ExifExtractor._extract_format_metadata(img, image_path, exif_data)
                
                # Try to get EXIF data
                exif = img.getexif()
                
                if exif is not None and len(exif) > 0:
                    exif_data['hasExif'] = True
                    
                    # Process EXIF tags
                    for tag_id, value in exif.items():
                        # Get the tag name
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Handle GPS data separately
                        if tag_name == 'GPSInfo':
                            exif_data['gpsInfo'] = ExifExtractor._process_gps_info(value)
                            continue
                        
                        # Map to standardized field names
                        if tag_name in ExifExtractor.TAG_MAPPING:
                            field_name = ExifExtractor.TAG_MAPPING[tag_name]
                            exif_data[field_name] = ExifExtractor._convert_value(value)
                        else:
                            # Store in additionalData for unmapped tags
                            exif_data['additionalData'][tag_name] = ExifExtractor._convert_value(value)
                
        except Exception as e:
            # If there's an error, return the default data structure
            exif_data['additionalData']['error'] = str(e)
        
        return exif_data
    
    @staticmethod
    def _extract_bit_depth_info(img: Image.Image, exif_data: Dict[str, Any]) -> None:
        """
        Extract bit depth information from the image
        
        Args:
            img: PIL Image object
            exif_data: Dictionary to store the extracted data
        """
        # Mapping of PIL mode to bits per channel
        mode_to_bits = {
            '1': 1,      # 1-bit pixels, black and white
            'L': 8,      # 8-bit pixels, grayscale
            'P': 8,      # 8-bit pixels, mapped using a color palette
            'RGB': 8,    # 3x8-bit pixels, true color
            'RGBA': 8,   # 4x8-bit pixels, true color with transparency
            'CMYK': 8,   # 4x8-bit pixels, color separation
            'YCbCr': 8,  # 3x8-bit pixels, color video format
            'LAB': 8,    # 3x8-bit pixels, L*a*b color space
            'HSV': 8,    # 3x8-bit pixels, Hue, Saturation, Value
            'I': 32,     # 32-bit signed integer pixels
            'F': 32      # 32-bit floating point pixels
        }
        
        bits_per_channel = mode_to_bits.get(img.mode, 8)
        channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
        
        exif_data['bitsPerChannel'] = bits_per_channel
        exif_data['channels'] = channels
        exif_data['bitDepth'] = bits_per_channel * channels
        exif_data['colorType'] = img.mode
    
    @staticmethod
    def _extract_format_metadata(img: Image.Image, image_path: str, exif_data: Dict[str, Any]) -> None:
        """
        Extract format-specific metadata (JFIF for JPEG, PNG chunks for PNG)
        
        Args:
            img: PIL Image object
            image_path: Path to the image file
            exif_data: Dictionary to store the extracted data
        """
        img_format = img.format
        
        if img_format == 'JPEG':
            # Extract JFIF metadata for JPEG files
            ExifExtractor._extract_jfif_metadata(img, exif_data)
        elif img_format == 'PNG':
            # Extract PNG chunk metadata
            ExifExtractor._extract_png_metadata(image_path, exif_data)
    
    @staticmethod
    def _extract_jfif_metadata(img: Image.Image, exif_data: Dict[str, Any]) -> None:
        """
        Extract JFIF metadata from JPEG files
        
        Args:
            img: PIL Image object
            exif_data: Dictionary to store the extracted data
        """
        info = img.info
        
        # Check for JFIF density information
        if 'jfif_density' in info:
            x_density, y_density = info['jfif_density']
            unit = info.get('jfif_unit', 0)
            
            # unit: 0 = no units, 1 = dots per inch, 2 = dots per cm
            if unit == 1:  # DPI
                exif_data['xResolution'] = x_density
                exif_data['yResolution'] = y_density
                exif_data['resolutionUnit'] = 'inches'
            elif unit == 2:  # dots per cm
                # Convert to DPI (1 inch = 2.54 cm)
                exif_data['xResolution'] = x_density * 2.54
                exif_data['yResolution'] = y_density * 2.54
                exif_data['resolutionUnit'] = 'inches'
            else:
                # No unit specified, store raw values
                exif_data['additionalData']['jfif_density_x'] = x_density
                exif_data['additionalData']['jfif_density_y'] = y_density
        
        # Store JFIF version if available
        if 'jfif_version' in info:
            version = info['jfif_version']
            exif_data['additionalData']['jfif_version'] = f"{version[0]}.{version[1]}"
    
    @staticmethod
    def _extract_png_metadata(image_path: str, exif_data: Dict[str, Any]) -> None:
        """
        Extract PNG-specific metadata from PNG chunks
        
        Args:
            image_path: Path to the PNG file
            exif_data: Dictionary to store the extracted data
        """
        try:
            with open(image_path, 'rb') as f:
                # Check PNG signature
                signature = f.read(8)
                if signature != b'\x89PNG\r\n\x1a\n':
                    return
                
                # Read chunks
                while True:
                    # Read chunk length
                    length_data = f.read(4)
                    if len(length_data) < 4:
                        break
                    
                    length = struct.unpack('>I', length_data)[0]
                    
                    # Read chunk type
                    chunk_type = f.read(4)
                    if len(chunk_type) < 4:
                        break
                    
                    chunk_name = chunk_type.decode('ascii', errors='ignore')
                    
                    # Read chunk data
                    data = f.read(length)
                    
                    # Read CRC
                    crc = f.read(4)
                    
                    # Parse specific chunks
                    if chunk_name == 'IHDR' and length == 13:
                        # Image header - contains bit depth
                        width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', data)
                        exif_data['bitsPerChannel'] = bit_depth
                        
                        # Update color type description
                        color_desc = ExifExtractor.PNG_COLOR_TYPES.get(color_type, f'Unknown ({color_type})')
                        exif_data['additionalData']['png_color_type'] = color_desc
                        exif_data['additionalData']['png_compression'] = compression
                        exif_data['additionalData']['png_interlace'] = interlace
                        
                        # Recalculate total bit depth based on PNG color type
                        channels_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
                        channels = channels_map.get(color_type, exif_data['channels'])
                        exif_data['channels'] = channels
                        exif_data['bitDepth'] = bit_depth * channels
                    
                    elif chunk_name == 'pHYs' and length == 9:
                        # Physical pixel dimensions
                        pixels_per_unit_x, pixels_per_unit_y, unit = struct.unpack('>IIB', data)
                        
                        if unit == 1:  # meter
                            # Convert to DPI (1 meter = 39.3701 inches)
                            exif_data['xResolution'] = pixels_per_unit_x / 39.3701
                            exif_data['yResolution'] = pixels_per_unit_y / 39.3701
                            exif_data['resolutionUnit'] = 'inches'
                        else:
                            exif_data['additionalData']['png_pixels_per_unit_x'] = pixels_per_unit_x
                            exif_data['additionalData']['png_pixels_per_unit_y'] = pixels_per_unit_y
                    
                    elif chunk_name == 'IEND':
                        break
        except Exception as e:
            exif_data['additionalData']['png_read_error'] = str(e)
    
    @staticmethod
    def _process_gps_info(gps_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process GPS information from EXIF data
        
        Args:
            gps_data: Raw GPS data from EXIF
            
        Returns:
            Dictionary containing GPS information
        """
        if not gps_data:
            return None
        
        gps_info = {}
        
        try:
            for tag_id, value in gps_data.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                gps_info[tag_name] = ExifExtractor._convert_value(value)
        except Exception:
            return None
        
        return gps_info if gps_info else None
    
    @staticmethod
    def _convert_value(value: Any) -> Any:
        """
        Convert EXIF value to JSON-serializable format
        
        Args:
            value: Raw EXIF value
            
        Returns:
            JSON-serializable value
        """
        # Handle bytes
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        
        # Handle tuples (like resolution)
        if isinstance(value, tuple):
            if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                # Likely a rational number (numerator, denominator)
                if value[1] != 0:
                    return float(value[0]) / float(value[1])
                return value[0]
            return list(value)
        
        # Handle other types
        if isinstance(value, (str, int, float, bool)):
            return value
        
        # Default: convert to string
        return str(value)
    
    @staticmethod
    def get_exif_summary(image_path: str) -> str:
        """
        Get a human-readable summary of image metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown formatted string with metadata summary
        """
        exif = ExifExtractor.extract_exif(image_path)
        
        lines = []
        lines.append("### 📷 Image Metadata")
        lines.append("")
        
        # Image dimensions
        if exif['imageWidth'] and exif['imageHeight']:
            lines.append(f"**Dimensions:** {exif['imageWidth']} x {exif['imageHeight']} pixels")
            lines.append("")
        
        # Bit depth and color information
        if exif['bitDepth']:
            lines.append("**Color Information:**")
            lines.append(f"- Color Type: {exif['colorType']}")
            lines.append(f"- Bits per Channel: {exif['bitsPerChannel']}")
            lines.append(f"- Number of Channels: {exif['channels']}")
            lines.append(f"- Total Bit Depth: {exif['bitDepth']} bits per pixel")
            lines.append("")
        
        # Resolution
        if exif['xResolution'] and exif['yResolution']:
            lines.append("**Resolution:**")
            lines.append(f"- Horizontal: {exif['xResolution']:.2f} DPI")
            lines.append(f"- Vertical: {exif['yResolution']:.2f} DPI")
            lines.append("")
        
        # Camera info (from EXIF)
        if exif['make'] or exif['model']:
            lines.append("**Camera:**")
            if exif['make']:
                lines.append(f"- Make: {exif['make']}")
            if exif['model']:
                lines.append(f"- Model: {exif['model']}")
            lines.append("")
        
        # Date/Time
        if exif['dateTimeOriginal']:
            lines.append(f"**Date Taken:** {exif['dateTimeOriginal']}")
            lines.append("")
        elif exif['dateTime']:
            lines.append(f"**Date:** {exif['dateTime']}")
            lines.append("")
        
        # Software
        if exif['software']:
            lines.append(f"**Software:** {exif['software']}")
            lines.append("")
        
        # GPS
        if exif['gpsInfo']:
            lines.append("**GPS Data:** Present")
            lines.append("")
        
        # EXIF status
        if exif['hasExif']:
            lines.append("**EXIF Data:** Present")
        else:
            lines.append("**EXIF Data:** Not available")
        
        return "\n".join(lines)
