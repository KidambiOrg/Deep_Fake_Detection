"""
API utility functions for calling Azure Function endpoints
"""

import streamlit as st
import requests
from PIL import Image
import io
from urllib.parse import urlparse
import os

def get_endpoints():
    """Get API endpoints based on base URL"""
    base_url = st.session_state.azure_function_url
    return {
        'warmup': f"{base_url}/warmup",
        'analyze': f"{base_url}/analyzeImage"
    }

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_image_from_url(url):
    """Load image from URL and return tuple of (image, filename)"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'image.png'
        
        return image, filename
    except Exception as e:
        st.error(f"Failed to load image from URL: {str(e)}")
        return None, None

def load_image_from_upload(uploaded_file):
    """Load image from uploaded file"""
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"Failed to load uploaded image: {str(e)}")
        return None

def image_to_bytes(image):
    """Convert PIL Image to bytes"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"Failed to convert image to bytes: {str(e)}")
        return None

def call_warmup_endpoint():
    """Call the warmup endpoint to preload models"""
    try:
        endpoints = get_endpoints()
        response = requests.get(endpoints['warmup'], timeout=300)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'message': f"Failed with status code: {response.status_code}"
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def analyze_image(image_bytes, filename=None):
    """Call the analyze endpoint with image
    
    Args:
        image_bytes: Image data as bytes
        filename: Original filename (optional, defaults to 'image.png')
    """
    try:
        endpoints = get_endpoints()
        # Use provided filename or default to 'image.png'
        file_name = filename if filename else 'image.png'
        files = {'image': (file_name, image_bytes, 'image/png')}
        
        response = requests.post(
            endpoints['analyze'],
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed with status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Analysis request failed: {str(e)}")
        return None
