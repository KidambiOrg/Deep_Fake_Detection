"""
C2PA (Content Authenticity Initiative) metadata extractor
Uses the c2pa-python package to read and validate content credentials

This module extracts comprehensive C2PA manifest metadata according to the 
manifest schema defined at:
https://github.com/contentauth/c2pa-rs/blob/main/cli/schemas/manifest-definition.json

Extracted metadata includes:
- Claim fields: vendor, claim_generator, claim_version, title, format, instance_id
- Claim generator info: name, version, icon, operating_system, custom fields
- Metadata: dateTime, reference, actor, reviewRatings, custom fields
- Credentials: W3C verifiable credentials
- Assertions (including Actions for process/edit history):
  * label: assertion type (e.g., c2pa.actions, c2pa.metadata)
  * kind: data format (Cbor/Json)
  * created: whether attributed to signer
  * data: full assertion data (including Actions for process information)
    - Actions assertion contains: actions list, softwareAgents, allActionsIncluded, templates, metadata
    - Each action includes: action type, parameters, when, softwareAgent, digitalSourceType, actors, changes
- Ingredients: title, format, instanceId, documentId, relationship, c2pa_manifest,
               validation_status, thumbnail, data, description, informational_uri,
               data_types, validation_results, active_manifest, claim_signature,
               soft_bindings_matched, soft_binding_algorithms_matched, metadata
- Signature info: alg, issuer, issuer_org, common_name, time, date, serial_number, cert_chain
- Redactions: redacted assertions list
- Validation: validation_state, validation_results, is_embedded, remote_url

Note: Process/edit history is captured in the Actions assertion (c2pa.actions),
not as a separate "process" section.
"""
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging

try:
    from c2pa import Reader as C2PAReader
    C2PA_AVAILABLE = True
except ImportError as e:
    C2PAReader = None  # type: ignore
    C2PA_AVAILABLE = False
    logging.warning(f"c2pa-python package import failed: {str(e)}. C2PA functionality will be disabled.")
except Exception as e:
    C2PAReader = None  # type: ignore
    C2PA_AVAILABLE = False
    logging.warning(f"c2pa-python package failed to load: {str(e)}. C2PA functionality will be disabled.")


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
        Extract ALL C2PA metadata from image per Reader JSON reference.
        Reference: https://opensource.contentauthenticity.org/docs/manifest/json-ref/reader
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing complete C2PA Reader structure:
            - hasC2pa: Whether C2PA data is present
            - activeManifest: Label for the active (most recent) manifest (String)
            - manifests: HashMap of all Manifests (Object)
            - validationResults: ValidationResults with activeManifest and ingredientDeltas
            - validationState: ValidationState enum (Invalid/Valid/Trusted)
            - validationStatus: Array of ValidationStatus from loading
            - isEmbedded: Whether manifest is embedded or remote
            - remoteUrl: Remote URL if not embedded
            - error: Error message if extraction failed
        """
        result = {
            'hasC2pa': False,
            'activeManifest': None,  # String: label for active manifest
            'manifests': None,  # Object: HashMap of Manifests
            'validationResults': None,  # ValidationResults object
            'validationState': None,  # ValidationState enum
            'validationStatus': None,  # Array of ValidationStatus
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
                with C2PAReader(mime_type, file) as reader:  # type: ignore
                    # Get the full manifest as JSON (complete Reader structure)
                    manifest_json = reader.json()
                    manifest_data = json.loads(manifest_json)
                    
                    # Mark that C2PA data is present
                    result['hasC2pa'] = True
                    
                    # Extract Reader top-level fields per spec
                    # active_manifest: String - label for active manifest
                    result['activeManifest'] = manifest_data.get('active_manifest')
                    
                    # validation_state: ValidationState enum (Invalid/Valid/Trusted)
                    result['validationState'] = manifest_data.get('validation_state')
                    
                    # validation_status: Array of ValidationStatus
                    result['validationStatus'] = manifest_data.get('validation_status')
                    
                    # validation_results: ValidationResults object
                    # Contains activeManifest (StatusCodes) and ingredientDeltas array
                    result['validationResults'] = self._extract_validation_results(
                        manifest_data.get('validation_results')
                    )
                    
                    # manifests: Object - HashMap of all Manifests
                    if 'manifests' in manifest_data:
                        result['manifests'] = {}
                        for manifest_label, manifest_content in manifest_data['manifests'].items():
                            result['manifests'][manifest_label] = self._extract_manifest_summary(manifest_content)
                    
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
        Extract comprehensive information from a C2PA manifest according to the Reader JSON spec.
        Reference: https://opensource.contentauthenticity.org/docs/manifest/json-ref/reader
        
        Args:
            manifest: Full manifest dictionary
            
        Returns:
            Dictionary with complete manifest information per C2PA spec
        """
        summary = {
            # Basic Manifest fields
            'vendor': manifest.get('vendor'),
            'claimGenerator': manifest.get('claim_generator'),
            'claimVersion': manifest.get('claim_version'),
            'title': manifest.get('title'),
            'format': manifest.get('format'),
            'instanceId': manifest.get('instance_id'),
            'label': manifest.get('label'),
            
            # Complex nested structures
            'claimGeneratorInfo': None,
            'metadata': None,
            'credentials': None,
            'assertions': [],
            'ingredients': [],
            'signatureInfo': None,
            'redactions': None,
            'thumbnail': None,
            'resources': None,
            
            # Algorithm fields
            'alg': manifest.get('alg'),
            'algSoft': manifest.get('alg_soft'),
            'redactedAssertions': manifest.get('redacted_assertions'),
        }
        
        # Extract claim generator info with all fields
        if 'claim_generator_info' in manifest:
            claim_gen_list = manifest['claim_generator_info']
            if isinstance(claim_gen_list, list) and len(claim_gen_list) > 0:
                summary['claimGeneratorInfo'] = []
                for claim_gen_info in claim_gen_list:
                    info = {
                        'name': claim_gen_info.get('name'),
                        'version': claim_gen_info.get('version'),
                        'icon': claim_gen_info.get('icon'),
                        'operatingSystem': claim_gen_info.get('operating_system'),
                    }
                    # Include any additional custom fields
                    for key, value in claim_gen_info.items():
                        if key not in ['name', 'version', 'icon', 'operating_system']:
                            info[key] = value
                    summary['claimGeneratorInfo'].append(info)
        
        # Extract metadata with all fields
        if 'metadata' in manifest:
            metadata_list = manifest['metadata']
            if isinstance(metadata_list, list):
                summary['metadata'] = []
                for metadata_item in metadata_list:
                    meta = {
                        'dateTime': metadata_item.get('dateTime'),
                        'reference': metadata_item.get('reference'),
                        'actor': metadata_item.get('actor'),
                        'reviewRatings': metadata_item.get('reviewRatings'),
                    }
                    # Include any additional custom fields
                    for key, value in metadata_item.items():
                        if key not in ['dateTime', 'reference', 'actor', 'reviewRatings']:
                            meta[key] = value
                    summary['metadata'].append(meta)
        
        # Extract credentials (W3C verifiable credentials)
        if 'credentials' in manifest:
            summary['credentials'] = manifest['credentials']
        
        # Extract assertions with detailed data
        if 'assertions' in manifest:
            for assertion in manifest['assertions']:
                assertion_data = {
                    'label': assertion.get('label'),
                    'kind': assertion.get('kind'),
                    'created': assertion.get('created'),
                }
                
                # Extract full data for important assertions
                if 'data' in assertion:
                    data = assertion['data']
                    label = assertion.get('label', '')
                    
                    # For Actions assertion, include full process/edit information
                    if 'actions' in label.lower():
                        assertion_data['data'] = data  # Include full actions data
                    # For other assertions, include selective data
                    elif isinstance(data, dict):
                        # For dictionaries, include the structure
                        assertion_data['data'] = data
                    elif isinstance(data, list):
                        assertion_data['dataCount'] = len(data)
                        assertion_data['hasData'] = True
                    else:
                        assertion_data['hasData'] = True
                        
                summary['assertions'].append(assertion_data)
        
        # Extract ingredients with ALL fields per C2PA Reader spec
        if 'ingredients' in manifest:
            for ingredient in manifest['ingredients']:
                ingredient_data = {
                    # Basic ingredient fields
                    'title': ingredient.get('title'),
                    'format': ingredient.get('format'),
                    'instanceId': ingredient.get('instance_id'),
                    'documentId': ingredient.get('document_id'),
                    'relationship': ingredient.get('relationship', 'componentOf'),
                    'description': ingredient.get('description'),
                    'informationalURI': ingredient.get('informational_URI'),
                    'provenance': ingredient.get('provenance'),
                    'label': ingredient.get('label'),
                    'hash': ingredient.get('hash'),
                    
                    # Resource references
                    'thumbnail': ingredient.get('thumbnail'),
                    'data': ingredient.get('data'),
                    'manifestData': ingredient.get('manifest_data'),
                    
                    # Data types
                    'dataTypes': ingredient.get('data_types'),
                    
                    # Validation fields
                    'validationStatus': ingredient.get('validation_status'),
                    'validationResults': ingredient.get('validation_results'),
                    
                    # Manifest references
                    'activeManifest': ingredient.get('active_manifest'),
                    'c2paManifest': ingredient.get('c2pa_manifest'),
                    'claimSignature': ingredient.get('claim_signature'),
                    
                    # Soft binding fields
                    'softBindingsMatched': ingredient.get('soft_bindings_matched'),
                    'softBindingAlgorithmsMatched': ingredient.get('soft_binding_algorithms_matched'),
                    
                    # Metadata and resources
                    'metadata': ingredient.get('metadata'),
                    'resources': ingredient.get('resources'),
                }
                
                summary['ingredients'].append(ingredient_data)
        
        # Extract comprehensive SignatureInfo per C2PA spec
        if 'signature_info' in manifest:
            sig_info = manifest['signature_info']
            summary['signatureInfo'] = {
                'alg': sig_info.get('alg'),  # SigningAlg enum (Es256, Es384, Es512, Ps256, Ps384, Ps512, Ed25519)
                'issuer': sig_info.get('issuer'),
                'certSerialNumber': sig_info.get('cert_serial_number'),
                'revocationStatus': sig_info.get('revocation_status'),
                'time': sig_info.get('time'),
                # Legacy/additional fields
                'issuerOrg': sig_info.get('issuer_org'),
                'commonName': sig_info.get('common_name'),
                'date': sig_info.get('date'),
                'serialNumber': sig_info.get('serial_number'),
                'certChain': sig_info.get('cert_chain'),
            }
        
        # Extract thumbnail (ResourceRef)
        if 'thumbnail' in manifest:
            summary['thumbnail'] = self._extract_resource_ref(manifest['thumbnail'])
        
        # Extract resources (ResourceStore)
        if 'resources' in manifest:
            resources = manifest['resources']
            summary['resources'] = {
                'basePath': resources.get('base_path'),
                'label': resources.get('label'),
                'resources': resources.get('resources'),  # HashMap of resources
            }
        
        # Extract redactions
        if 'redactions' in manifest:
            summary['redactions'] = manifest['redactions']
        
        return summary
    
    def _extract_resource_ref(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ResourceRef structure.
        
        Args:
            resource: ResourceRef dictionary
            
        Returns:
            Extracted resource reference information
        """
        return {
            'identifier': resource.get('identifier'),
            'format': resource.get('format'),
            'hash': resource.get('hash'),
            'alg': resource.get('alg'),
            'dataTypes': resource.get('data_types'),
        }
    
    def _extract_validation_results(self, validation_results_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract ValidationResults structure per Reader spec.
        
        ValidationResults contains:
        - activeManifest: StatusCodes object with success/informational/failure arrays
        - ingredientDeltas: Array of IngredientDeltaValidationResult objects
        
        Args:
            validation_results_data: Dictionary containing ValidationResults fields
            
        Returns:
            Dictionary with ValidationResults structure or None
        """
        if not validation_results_data:
            return None
        
        result = {}
        
        # Extract activeManifest StatusCodes
        # Contains arrays: success, informational, failure
        if 'activeManifest' in validation_results_data:
            active_manifest_codes = validation_results_data['activeManifest']
            result['activeManifest'] = {
                'success': active_manifest_codes.get('success', []),
                'informational': active_manifest_codes.get('informational', []),
                'failure': active_manifest_codes.get('failure', [])
            }
        
        # Extract ingredientDeltas array
        # Each entry contains:
        # - label: String
        # - statusCodes: StatusCodes object
        if 'ingredientDeltas' in validation_results_data:
            ingredient_deltas = []
            for delta in validation_results_data['ingredientDeltas']:
                delta_info = {
                    'label': delta.get('label')
                }
                
                # Extract StatusCodes for this ingredient delta
                if 'statusCodes' in delta:
                    status_codes = delta['statusCodes']
                    delta_info['statusCodes'] = {
                        'success': status_codes.get('success', []),
                        'informational': status_codes.get('informational', []),
                        'failure': status_codes.get('failure', [])
                    }
                
                ingredient_deltas.append(delta_info)
            
            result['ingredientDeltas'] = ingredient_deltas
        
        return result
    
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
    
    def extract_actions(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract Actions (process/edit history) from C2PA manifest.
        
        Actions assertions describe what processes, edits, or modifications
        were performed on the asset.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing actions data or None if not available.
            Includes:
            - actions: List of individual actions performed
            - softwareAgents: Software that performed the actions
            - allActionsIncluded: Whether all actions are included
            - templates: Action templates if present
            - metadata: Additional action metadata
        """
        c2pa_data = self.extract_c2pa_data(image_path)
        
        if not c2pa_data.get('hasC2pa') or not c2pa_data.get('activeManifest'):
            return None
        
        # Look for Actions assertion in the active manifest
        assertions = c2pa_data['activeManifest'].get('assertions', [])
        
        for assertion in assertions:
            label = assertion.get('label', '')
            if 'actions' in label.lower() and 'data' in assertion:
                return assertion['data']
        
        return None
    
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
