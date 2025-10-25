"""
Streamlit Web Application for Deepfake Detection
This application allows users to upload images or provide URLs to detect deepfakes using Azure Functions.
"""

import streamlit as st
import requests
from PIL import Image
import io
import time
import plotly.graph_objects as go
from urllib.parse import urlparse

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    /* Compact button styling */
    .stButton button {
        width: auto !important;
        padding: 0.25rem 1rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Reduce margins */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* Remove extra spacing */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
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

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Analyze Image"

if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

if 'show_settings_modal' not in st.session_state:
    st.session_state.show_settings_modal = False

if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = ""

# Helper functions
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
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Failed to load image from URL: {str(e)}")
        return None

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

def analyze_image(image_bytes):
    """Call the analyze endpoint with image"""
    try:
        endpoints = get_endpoints()
        files = {'image': ('image.png', image_bytes, 'image/png')}
        
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

def create_pie_chart(model_result):
    """Create a pie chart for individual model results with enhanced tooltip"""
    model_name = model_result.get('model', 'Unknown')
    prediction = model_result.get('prediction', 'Unknown')
    confidence = model_result.get('confidence', 0)
    fake_score = model_result.get('fakeScore', 0)
    real_score = model_result.get('realScore', 0)
    
    # Create pie chart data
    labels = ['Fake', 'Real']
    values = [fake_score * 100, real_score * 100]
    colors = ['#ff6b6b', '#51cf66']
    
    # Create custom hover text (Plotly supports basic HTML)
    hover_template = (
        f"<b>{model_name}</b><br><br>"
        f"<b>%{{label}}</b><br>"
        f"Percentage: %{{percent}}<br>"
        f"<br>"
        f"<b>Overall Results:</b><br>"
        f"Prediction: <b>{prediction}</b><br>"
        f"Confidence: {confidence:.1%}<br>"
        f"Fake Score: {fake_score:.1%}<br>"
        f"Real Score: {real_score:.1%}<br>"
        "<extra></extra>"
    )
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate=hover_template,
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"{model_name}<br><sub>Prediction: {prediction} ({confidence:.1%})</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=300,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    return fig

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Analyze Image button
        if st.button("Analyze Image", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Analyze Image" else "secondary",
                    disabled=st.session_state.is_analyzing):
            st.session_state.current_page = "Analyze Image"
            st.rerun()
        
        # Analyze Text button
        if st.button("Analyze Text", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Analyze Text" else "secondary",
                    disabled=st.session_state.is_analyzing):
            st.session_state.current_page = "Analyze Text"
            st.rerun()
    
    return st.session_state.current_page

@st.dialog("Settings", width="large")
def show_settings_dialog():
    """Show settings dialog for configuring Azure Function URL"""
    st.markdown("### Azure Function Configuration")
    
    # Input for Azure Function URL
    current_url = st.session_state.azure_function_url
    new_url = st.text_input(
        "Azure Function Base URL",
        value=current_url,
        help="Enter the base URL of your Azure Function (e.g., http://localhost:7071/api or https://your-function.azurewebsites.net/api)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save", use_container_width=True, type="primary"):
            st.session_state.azure_function_url = new_url
            st.success("✅ Settings saved!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def render_header(page_title: str):
    """Render the header section with title, PreLoad button, and settings gear"""
    # Create header with horizontal layout
    with st.container():
        col_title, col_buttons = st.columns([6, 2])
        
        with col_title:
            # Page title
            st.markdown(f'<div class="main-header" style="text-align: left;">{page_title}</div>', unsafe_allow_html=True)
        
        with col_buttons:
            # Buttons in horizontal layout with compact spacing
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2, gap="small")
            
            with btn_col1:
                if st.button("PreLoad", type="secondary", 
                            key=f"preload_{page_title}", disabled=st.session_state.is_analyzing):
                    with st.spinner("Loading models... This may take 2-5 minutes for first time."):
                        result = call_warmup_endpoint()
                        st.session_state.preload_status = result
                        
                        if result.get('status') == 'ready':
                            st.success("✅ Models loaded successfully!")
                        elif result.get('status') == 'loading':
                            st.info("⏳ Models are loading...")
                        else:
                            st.error(f"❌ {result.get('message', 'Unknown error')}")
            
            with btn_col2:
                if st.button("⚙️", key=f"settings_{page_title}", help="Settings", 
                            disabled=st.session_state.is_analyzing):
                    show_settings_dialog()
    
    # Separator line with reduced margin
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

def render_analyze_image_page():
    """Render the main image analysis page"""
    # Render header
    render_header("Analyze Image")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method",
        ["Upload Image", "Image URL"],
        horizontal=True,
        label_visibility="collapsed",
        disabled=st.session_state.is_analyzing
    )
    
    # Help text below radio buttons
    if input_method == "Upload Image":
        st.caption("Supported: JPG, JPEG, PNG, WEBP • Max size: 10MB")
    else:
        st.caption("Enter a valid image URL (http:// or https://)")
    
    # File uploader or URL input
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Browse files",
            type=['jpg', 'jpeg', 'png', 'webp'],
            disabled=st.session_state.is_analyzing,
            accept_multiple_files=False,
            key="file_uploader",
            on_change=lambda: None  # Track when file is cleared
        )
        
        # Handle file removal (when user clicks X)
        if uploaded_file is None and st.session_state.uploaded_filename:
            st.session_state.image_data = None
            st.session_state.image_source = None
            st.session_state.analysis_result = None
            st.session_state.uploaded_filename = ""
        elif uploaded_file is not None:
            image = load_image_from_upload(uploaded_file)
            if image:
                st.session_state.image_data = image
                st.session_state.image_source = "upload"
                st.session_state.uploaded_filename = uploaded_file.name
    
    # Image input controls - Action buttons with compact horizontal layout
    if input_method == "Upload Image":
        # Show action buttons when file is uploaded
        if st.session_state.uploaded_filename and st.session_state.image_data is not None:
            # Use columns with minimal gap for compact button layout
            col_clear, col_analyze, col_spinner = st.columns([0.7, 1, 5], gap="small")
            
            with col_clear:
                # Clear button
                if st.button("🗑️ Clear", disabled=st.session_state.is_analyzing, key="clear_upload"):
                    st.session_state.image_data = None
                    st.session_state.image_source = None
                    st.session_state.analysis_result = None
                    st.session_state.uploaded_filename = ""
                    st.rerun()
            
            with col_analyze:
                # Analyze button
                if st.session_state.is_analyzing:
                    st.button("🔍 Analyze", type="primary", disabled=True, key="analyze_upload_disabled")
                else:
                    if st.button("🔍 Analyze", type="primary", 
                                disabled=st.session_state.is_analyzing, key="analyze_upload"):
                        # Set analyzing state to disable page
                        st.session_state.is_analyzing = True
                        st.rerun()
            
            with col_spinner:
                # Show spinner when analyzing
                if st.session_state.is_analyzing:
                    st.markdown("🔄 Analyzing...")
    
    else:  # Image URL
        # URL input
        url_input = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            label_visibility="collapsed",
            disabled=st.session_state.is_analyzing
        )
        
        # Auto-load image when URL is entered
        if url_input and is_valid_url(url_input):
            if st.session_state.image_source != "url" or st.session_state.image_data is None:
                image = load_image_from_url(url_input)
                if image:
                    st.session_state.image_data = image
                    st.session_state.image_source = "url"
        
        # Show action buttons when image is loaded
        if st.session_state.image_data is not None:
            # Use columns with minimal gap for compact button layout
            col_clear, col_analyze, col_spinner = st.columns([0.7, 1, 5], gap="small")
            
            with col_clear:
                # Clear button
                if st.button("🗑️ Clear", key="clear_url", disabled=st.session_state.is_analyzing):
                    st.session_state.image_data = None
                    st.session_state.image_source = None
                    st.session_state.analysis_result = None
                    st.rerun()
            
            with col_analyze:
                # Analyze button
                if st.session_state.is_analyzing:
                    st.button("🔍 Analyze", type="primary", disabled=True, key="analyze_url_disabled")
                else:
                    if st.button("🔍 Analyze", type="primary", 
                                disabled=st.session_state.is_analyzing, key="analyze_url"):
                        # Set analyzing state to disable page
                        st.session_state.is_analyzing = True
                        st.rerun()
            
            with col_spinner:
                # Show spinner when analyzing
                if st.session_state.is_analyzing:
                    st.markdown("🔄 Analyzing...")
    
    # Main content area - Image preview and Results
    if st.session_state.image_data is not None:
        # Use columns with small gap for tighter layout
        col_image, col_result = st.columns([1, 1], gap="small")
        
        with col_image:
            st.markdown("#### Image Preview")
            # Display image with fixed size 300x300
            st.image(st.session_state.image_data, width=300)
        
        with col_result:
            st.markdown("#### Markdown Result")
            
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                
                # Display markdown result
                markdown_content = result.get('markdown', 'No results available')
                st.markdown(markdown_content)
            else:
                st.info("Click 'Analyze' to see results here")
        
        # Show progress indicator if analyzing
        if st.session_state.is_analyzing:
            # Convert image to bytes
            image_bytes = image_to_bytes(st.session_state.image_data)
            
            # Call analysis endpoint
            result = analyze_image(image_bytes)
            
            if result:
                st.session_state.analysis_result = result
                st.success("✅ Analysis complete!")
                time.sleep(1)
            else:
                st.error("❌ Analysis failed. Please try again.")
            
            # Reset analyzing state
            st.session_state.is_analyzing = False
            st.rerun()
        
        # Individual Model Results - Display below image
        if st.session_state.analysis_result:
            st.markdown("#### Individual Model Results")
            
            result = st.session_state.analysis_result
            
            # Get individual model results
            images = result.get('images', [])
            if images:
                individual_results = images[0].get('individualModelResults', [])
                
                # Create responsive columns for pie charts
                num_models = len(individual_results)
                if num_models > 0:
                    # Responsive layout: 4 columns on desktop, auto-wrap on mobile
                    cols_per_row = min(4, num_models)
                    rows = (num_models + cols_per_row - 1) // cols_per_row
                    
                    for row in range(rows):
                        cols = st.columns(cols_per_row, gap="small")
                        for col_idx in range(cols_per_row):
                            model_idx = row * cols_per_row + col_idx
                            if model_idx < num_models:
                                with cols[col_idx]:
                                    model_result = individual_results[model_idx]
                                    fig = create_pie_chart(model_result)
                                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No individual model results available")
    else:
        # Show placeholder when no image is loaded
        col1, col2 = st.columns([1, 1], gap="small")
        
        with col1:
            st.markdown("#### Image Preview")
            st.info("👆 Please select an image to preview")
        
        with col2:
            st.markdown("#### Markdown Result")
            st.info("Results will appear here after analysis")

def render_analyze_text_page():
    """Render the text analysis page (placeholder)"""
    # Render header
    render_header("Analyze Text")
    
    st.info("🚧 Text analysis feature coming soon!")

def main():
    """Main application entry point"""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render the selected page
    if page == "Analyze Image":
        render_analyze_image_page()
    elif page == "Analyze Text":
        render_analyze_text_page()

if __name__ == "__main__":
    main()
