"""
EXIF Data Extractor Utility
Extracts EXIF metadata from images
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Any, Optional
import os


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
    
    @staticmethod
    def extract_exif(image_path: str) -> Dict[str, Any]:
        """
        Extract EXIF data from an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing EXIF data
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
                
                # Try to get EXIF data
                exif = img.getexif()
                
                if exif is None or len(exif) == 0:
                    # No EXIF data available
                    return exif_data
                
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
        Get a human-readable summary of EXIF data
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown formatted string with EXIF summary
        """
        exif = ExifExtractor.extract_exif(image_path)
        
        if not exif['hasExif']:
            return "No EXIF data available"
        
        lines = []
        lines.append("### 📷 EXIF Data")
        lines.append("")
        
        # Camera info
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
        
        # Image dimensions
        if exif['imageWidth'] and exif['imageHeight']:
            lines.append(f"**Dimensions:** {exif['imageWidth']} x {exif['imageHeight']} pixels")
            lines.append("")
        
        # Software
        if exif['software']:
            lines.append(f"**Software:** {exif['software']}")
            lines.append("")
        
        # GPS
        if exif['gpsInfo']:
            lines.append("**GPS Data:** Present")
            lines.append("")
        
        return "\n".join(lines)
