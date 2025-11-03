"""
Streamlit Web Application for Deepfake Detection - Main Entrypoint
This is the main navigation file that routes to different pages.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'image_data' not in st.session_state:
    st.session_state.image_data = None

if 'image_source' not in st.session_state:
    st.session_state.image_source = None  # 'upload' or 'url'

if 'preload_status' not in st.session_state:
    st.session_state.preload_status = None

if 'azure_function_url' not in st.session_state:
    # Try to get from secrets, otherwise use default
    try:
        st.session_state.azure_function_url = st.secrets.get("AZURE_FUNCTION_URL", "http://localhost:7071/api")
    except:
        st.session_state.azure_function_url = "http://localhost:7071/api"

if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

if 'show_settings_modal' not in st.session_state:
    st.session_state.show_settings_modal = False

if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = ""

if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

if 'text_data' not in st.session_state:
    st.session_state.text_data = None

if 'text_analysis_result' not in st.session_state:
    st.session_state.text_analysis_result = None

# Define pages
analyze_image_page = st.Page(
    "pages/analyze_image.py",
    title="Analyze Image",
    icon="🖼️",
    default=True
)

analyze_text_page = st.Page(
    "pages/analyze_text.py",
    title="Analyze Text",
    icon="📝"
)

chat_with_image_page = st.Page(
    "pages/ChatWithImage.py",
    title="Chat with Image",
    icon="💬"
)

# Create navigation
pg = st.navigation([analyze_image_page, analyze_text_page, chat_with_image_page])

# Run the selected page
pg.run()
