"""
Analyze Image Page
This page allows users to upload images or provide URLs to detect deepfakes.
"""

import streamlit as st
import time
from components.styles import apply_custom_css
from components.header import render_header
from components.api_utils import (
    is_valid_url,
    load_image_from_url,
    load_image_from_upload,
    image_to_bytes,
    analyze_image
)
from components.charts import create_pie_chart

# Apply custom styling
apply_custom_css()

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
    # Always use columns layout with buttons area
    col_uploader, col_buttons = st.columns([7, 1.5], gap="medium")
    
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Browse files",
            type=['jpg', 'jpeg', 'png', 'webp'],
            disabled=st.session_state.is_analyzing,
            accept_multiple_files=False,
            key=f"file_uploader_{st.session_state.file_uploader_key}",
            on_change=lambda: None
        )
    
    # Handle file state changes - only update if file is newly uploaded
    # Don't clear when uploaded_file is None (could be from page navigation)
    if uploaded_file is not None:
        # New file uploaded
        image = load_image_from_upload(uploaded_file)
        if image:
            st.session_state.image_data = image
            st.session_state.image_source = "upload"
            st.session_state.uploaded_filename = uploaded_file.name
    
    # Always show buttons
    with col_buttons:
        # Empty space to align with file uploader
        st.write("")
        st.write("")
        
        # Place buttons in sub-columns
        col_clear, col_analyze = st.columns(2, gap="small")
        
        # Check if file is uploaded
        has_file = uploaded_file is not None or (st.session_state.uploaded_filename and st.session_state.image_data is not None)
        
        with col_clear:
            # Clear button - always enabled
            if st.button("🗑️ Clear", key="clear_upload"):
                # Only clear if there's a file
                if has_file:
                    st.session_state.image_data = None
                    st.session_state.image_source = None
                    st.session_state.analysis_result = None
                    st.session_state.uploaded_filename = ""
                    st.session_state.file_uploader_key += 1  # Change key to reset file uploader
                    st.rerun()
        
        with col_analyze:
            # Analyze button - disabled when no file or when analyzing
            if st.session_state.is_analyzing:
                st.button("🔍 Analyze", type="primary", disabled=True, key="analyze_upload_disabled")
            else:
                if st.button("🔍 Analyze", type="primary", 
                            disabled=not has_file or st.session_state.is_analyzing, 
                            key="analyze_upload"):
                    st.session_state.is_analyzing = True
                    st.rerun()

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
            image, filename = load_image_from_url(url_input)
            if image:
                st.session_state.image_data = image
                st.session_state.image_source = "url"
                st.session_state.uploaded_filename = filename  # Store filename from URL
    
    # Show action buttons when image is loaded
    if st.session_state.image_data is not None:
        # Horizontal container with right alignment and minimal gap
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            # Clear button
            if st.button("🗑️ Clear", key="clear_url", disabled=st.session_state.is_analyzing):
                st.session_state.image_data = None
                st.session_state.image_source = None
                st.session_state.analysis_result = None
                st.session_state.uploaded_filename = ""  # Clear filename too
                st.rerun()
            
            # Analyze button
            if st.session_state.is_analyzing:
                st.button("🔍 Analyze", type="primary", disabled=True, key="analyze_url_disabled")
            else:
                if st.button("🔍 Analyze", type="primary", 
                            disabled=st.session_state.is_analyzing, key="analyze_url"):
                    # Set analyzing state to disable page
                    st.session_state.is_analyzing = True
                    st.rerun()

# Main content area - Image preview and Results
if st.session_state.image_data is not None:
    # Use columns with narrow image preview and wide markdown result (Example 2 pattern)
    col_image, col_result = st.columns([1, 3], gap="small")
    
    col_image.markdown("#### Image Preview")
    # Display image with fixed size 300x300
    col_image.image(st.session_state.image_data, width=300)
    
    col_result.markdown("#### Markdown Result")
    
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Display markdown result
        markdown_content = result.get('markdown', 'No results available')
        col_result.markdown(markdown_content)
    else:
        col_result.info("Click 'Analyze' to see results here")
    
    # Show progress indicator if analyzing
    if st.session_state.is_analyzing:
        with st.spinner("🔄 Analyzing image... This may take a few moments."):
            # Convert image to bytes
            image_bytes = image_to_bytes(st.session_state.image_data)
            
            # Get filename from session state if available (for uploads)
            filename = st.session_state.get('uploaded_filename', None)
            
            # Call analysis endpoint with filename
            result = analyze_image(image_bytes, filename=filename)
            
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
        
        # Get individual model results directly from top level (ImageDetectionResult structure)
        individual_results = result.get('individualModelResults', [])
        
        if individual_results and len(individual_results) > 0:
            # Create responsive columns for pie charts
            num_models = len(individual_results)
            
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
    col_image, col_result = st.columns([1, 3], gap="small")
    
    col_image.markdown("#### Image Preview")
    col_image.info("👆 Please select an image to preview")
    
    col_result.markdown("#### Markdown Result")
    col_result.info("Results will appear here after analysis")
