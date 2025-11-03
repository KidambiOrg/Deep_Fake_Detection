"""
Header component with PreLoad and Settings buttons
"""

import streamlit as st
import time
from components.api_utils import call_warmup_endpoint, call_warmup_text_endpoint

@st.dialog("Settings", width="large")
def show_settings_dialog(page_title: str = ""):
    """Show settings dialog for configuring Azure Function URL and other settings"""
    st.markdown("### ⚙️ Settings")
    
    # Check if this is the Chat with Image page
    is_chat_page = page_title == "Chat with Image"
    
    if is_chat_page:
        # Create tabs for Chat with Image page
        tab1, tab2, tab3 = st.tabs(["📡 API Configuration", "🔀 Chat Mode", "🤖 Azure OpenAI"])
        
        with tab1:
            st.markdown("#### Azure Function Base URL")
            current_url = st.session_state.azure_function_url
            new_url = st.text_input(
                "Base URL",
                value=current_url,
                help="Enter the base URL of your Azure Function (e.g., http://localhost:7071/api)"
            )
            st.session_state.azure_function_url = new_url
        
        with tab2:
            st.markdown("#### Analysis Mode")
            st.caption("Choose how images should be analyzed in chat")
            
            mode = st.radio(
                "Select Mode",
                options=["analyze", "analyze_with_detect"],
                format_func=lambda x: "🤖 Analyze (Azure OpenAI Vision)" if x == "analyze" else "🔍 Analyze with Detection",
                key="settings_chat_mode",
                index=0 if st.session_state.get('chat_mode', 'analyze') == "analyze" else 1
            )
            
            st.session_state.chat_mode = mode
            
            if mode == "analyze":
                st.info("💡 Uses Azure OpenAI to answer questions about the image")
            else:
                st.info("🔬 Analyzes image for detection and displays results")
        
        with tab3:
            st.markdown("#### Azure OpenAI Settings")
            st.caption("Required for 'Analyze' mode")
            
            st.session_state.azure_openai_endpoint = st.text_input(
                "Azure OpenAI Endpoint",
                value=st.session_state.get('azure_openai_endpoint', ''),
                placeholder="https://your-resource.openai.azure.com",
                key="settings_openai_endpoint"
            )
            
            st.session_state.azure_openai_deployment = st.text_input(
                "Deployment Name",
                value=st.session_state.get('azure_openai_deployment', ''),
                placeholder="gpt-4o",
                key="settings_openai_deployment"
            )
            
            st.session_state.azure_openai_api_key = st.text_input(
                "API Key",
                value=st.session_state.get('azure_openai_api_key', ''),
                type="password",
                key="settings_openai_key"
            )
    else:
        # Original settings for other pages
        st.markdown("#### Azure Function Configuration")
        
        # Input for Azure Function URL
        current_url = st.session_state.azure_function_url
        new_url = st.text_input(
            "Azure Function Base URL",
            value=current_url,
            help="Enter the base URL of your Azure Function (e.g., http://localhost:7071/api or https://your-function.azurewebsites.net/api)"
        )
        st.session_state.azure_function_url = new_url

    # Horizontal container with right alignment and minimal gap for buttons
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel"):
            st.rerun()
        
        if st.button("Save", type="primary"):
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
                show_settings_dialog(page_title)
    
    # Separator line with reduced margin
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
