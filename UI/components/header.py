"""
Header component with PreLoad and Settings buttons
"""

import streamlit as st
import time
from components.api_utils import call_warmup_endpoint, call_warmup_text_endpoint

@st.dialog("Settings", width="small")
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

    # Horizontal container with right alignment and minimal gap for buttons
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel"):
            st.rerun()
        
        if st.button("Save", type="primary"):
            st.session_state.azure_function_url = new_url
            st.success("✅ Settings saved!")
            time.sleep(1)
            st.rerun()

def render_header(page_title: str):
    """Render the header section with title, PreLoad button, and settings gear"""
    # Add small space above header to prevent button trimming
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # Create header with horizontal layout
    col_title, col_buttons = st.columns([6, 2])
    
    with col_title:
        # Page title
        st.markdown(f'<div class="main-header" style="text-align: left;">{page_title}</div>', unsafe_allow_html=True)
    
    with col_buttons:
        # Horizontal container with right alignment and minimal gap
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button("PreLoad", type="secondary", 
                        key=f"preload_{page_title}", disabled=st.session_state.is_analyzing):
                with st.spinner("Loading models... This may take 2-5 minutes for first time."):
                    # Call different warmup endpoints based on the page
                    if page_title == "Analyze Text":
                        result = call_warmup_text_endpoint()
                    else:
                        result = call_warmup_endpoint()
                    
                    st.session_state.preload_status = result
                    
                    if result.get('status') == 'ready':
                        st.success("✅ Models loaded successfully!")
                    elif result.get('status') == 'loading':
                        st.info("⏳ Models are loading...")
                    else:
                        st.error(f"❌ {result.get('message', 'Unknown error')}")
            
            if st.button("⚙️", key=f"settings_{page_title}", help="Settings", 
                        disabled=st.session_state.is_analyzing):
                show_settings_dialog()
    
    # Separator line with reduced margin
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
