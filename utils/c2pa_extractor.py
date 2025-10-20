"""
C2PA (Content Authenticity Initiative) metadata extractor
Uses the c2pa-python package to read and validate content credentials
"""
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging

try:
    from c2pa import Reader
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False
    logging.warning("c2pa-python package not installed. C2PA functionality will be disabled.")


class C2paExtractor:
    """
    Extracts and validates C2PA (Content Credentials) metadata from media files.
    
    C2PA provides tamper-evident information about content provenance and history,
    helping to verify the authenticity and origin of digital media.
    """
    
    def __init__(self):
        """Initialize the C2PA extractor."""
        self.available = C2PA_AVAILABLE
    
    def extract_c2pa_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract C2PA metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing C2PA metadata including:
            - hasC2pa: Whether C2PA data is present
            - validationState: Overall validation status
            - activeManifest: Information about the active manifest
            - manifests: All available manifests
            - validationResults: Detailed validation results
            - error: Error message if extraction failed
        """
        result = {
            'hasC2pa': False,
            'validationState': None,
            'activeManifest': None,
            'manifests': None,
            'validationResults': None,
            'isEmbedded': None,
            'remoteUrl': None,
            'error': None
        }
        
        if not self.available:
            result['error'] = "c2pa-python package not installed"
            return result
        
        try:
            # Get MIME type from file extension
            mime_type = self._get_mime_type(image_path)
            if not mime_type:
                result['error'] = f"Unsupported file format: {Path(image_path).suffix}"
                return result
            
            # Open the file and create a Reader
            with open(image_path, 'rb') as file:
                with Reader(mime_type, file) as reader:
                    # Get the full manifest as JSON
                    manifest_json = reader.json()
                    manifest_data = json.loads(manifest_json)
                    
                    # Mark that C2PA data is present
                    result['hasC2pa'] = True
                    
                    # Get validation state
                    result['validationState'] = reader.get_validation_state()
                    
                    # Get validation results
                    result['validationResults'] = reader.get_validation_results()
                    
                    # Get active manifest
                    active_manifest_data = reader.get_active_manifest()
                    if active_manifest_data:
                        result['activeManifest'] = self._extract_manifest_summary(active_manifest_data)
                    
                    # Store all manifests (limited info to avoid large payloads)
                    if 'manifests' in manifest_data:
                        result['manifests'] = list(manifest_data['manifests'].keys())
                    
                    # Check if manifest is embedded or remote
                    result['isEmbedded'] = reader.is_embedded()
                    result['remoteUrl'] = reader.get_remote_url()
                    
        except FileNotFoundError:
            result['error'] = f"File not found: {image_path}"
        except Exception as e:
            error_msg = str(e)
            # Check if it's a "no manifest found" error (expected for images without C2PA)
            if 'ManifestNotFound' in error_msg or 'no JUMBF data found' in error_msg:
                # This is expected - image simply doesn't have C2PA credentials
                result['hasC2pa'] = False
                result['error'] = None  # Not actually an error
            else:
                # Unexpected error
                result['error'] = f"Error extracting C2PA data: {error_msg}"
                logging.error(f"C2PA extraction error for {image_path}: {error_msg}", exc_info=True)
        
        return result
    
    def _extract_manifest_summary(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from a C2PA manifest.
        
        Args:
            manifest: Full manifest dictionary
            
        Returns:
            Summary dictionary with key manifest information
        """
        summary = {
            'claimGenerator': manifest.get('claim_generator'),
            'title': manifest.get('title'),
            'format': manifest.get('format'),
            'instanceId': manifest.get('instance_id'),
            'claimGeneratorInfo': None,
            'assertions': [],
            'ingredients': [],
            'signature': None
        }
        
        # Extract claim generator info
        if 'claim_generator_info' in manifest:
            claim_gen_info = manifest['claim_generator_info'][0] if isinstance(
                manifest['claim_generator_info'], list) else manifest['claim_generator_info']
            summary['claimGeneratorInfo'] = {
                'name': claim_gen_info.get('name'),
                'version': claim_gen_info.get('version'),
                'icon': claim_gen_info.get('icon')
            }
        
        # Extract assertions summary
        if 'assertions' in manifest:
            for assertion in manifest['assertions']:
                summary['assertions'].append({
                    'label': assertion.get('label'),
                    'kind': assertion.get('kind')
                })
        
        # Extract ingredients summary
        if 'ingredients' in manifest:
            for ingredient in manifest['ingredients']:
                summary['ingredients'].append({
                    'title': ingredient.get('title'),
                    'format': ingredient.get('format'),
                    'relationship': ingredient.get('relationship')
                })
        
        # Extract signature info
        if 'signature_info' in manifest:
            sig_info = manifest['signature_info']
            summary['signature'] = {
                'issuer': sig_info.get('issuer'),
                'time': sig_info.get('time'),
                'alg': sig_info.get('alg')
            }
        
        return summary
    
    def _get_mime_type(self, file_path: str) -> Optional[str]:
        """
        Get MIME type from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string or None if unsupported
        """
        extension_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.pdf': 'application/pdf'
        }
        
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext)
    
    def validate_c2pa(self, image_path: str) -> bool:
        """
        Quick validation check for C2PA credentials.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if C2PA data is present and valid, False otherwise
        """
        if not self.available:
            return False
        
        try:
            c2pa_data = self.extract_c2pa_data(image_path)
            return (c2pa_data['hasC2pa'] and 
                   c2pa_data['validationState'] in ['Valid', 'valid'])
        except Exception:
            return False
