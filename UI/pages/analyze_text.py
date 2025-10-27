"""
Analyze Text Page
This page allows users to analyze text for AI-generated content detection.
"""

import streamlit as st
import time
from components.styles import apply_custom_css
from components.header import render_header
from components.api_utils import analyze_text
from components.charts import create_text_pie_chart

# Apply custom styling
apply_custom_css()

# Render header
render_header("Analyze Text")

# Main layout: Text input on left, results on right
col_input, col_result = st.columns([1, 2], gap="medium")

with col_input:
    st.markdown("#### Enter text below to analyze")
    
    # Text area for input
    text_input = st.text_area(
        "Text Input",
        placeholder="Copy and paste text here.",
        height=400,
        label_visibility="collapsed",
        disabled=st.session_state.is_analyzing,
        key="text_input_area"
    )
    
    # Buttons row
    col_clear, col_analyze = st.columns(2, gap="small")
    
    with col_clear:
        if st.button("clear", key="clear_text", use_container_width=True):
            st.session_state.text_data = None
            st.session_state.text_analysis_result = None
            st.rerun()
    
    with col_analyze:
        has_text = text_input and text_input.strip()
        
        if st.session_state.is_analyzing:
            st.button("Analyze", type="primary", disabled=True, key="analyze_text_disabled", use_container_width=True)
        else:
            if st.button("Analyze", type="primary", 
                        disabled=not has_text or st.session_state.is_analyzing, 
                        key="analyze_text",
                        use_container_width=True):
                st.session_state.text_data = text_input
                st.session_state.is_analyzing = True
                st.rerun()

with col_result:
    st.markdown("#### Markdown result")
    
    if st.session_state.text_analysis_result:
        result = st.session_state.text_analysis_result
        
        # Display markdown result in a scrollable container
        markdown_content = result.get('markdown', 'No results available')
        
        # Create a container with fixed height for scrolling
        with st.container(height=400):
            st.markdown(markdown_content)
    else:
        st.info("Click 'Analyze' to see results here")

# Show progress indicator if analyzing
if st.session_state.is_analyzing and st.session_state.text_data:
    with st.spinner("🔄 Analyzing text... This may take a few moments."):
        # Call analysis endpoint
        result = analyze_text(st.session_state.text_data)
        
        if result:
            st.session_state.text_analysis_result = result
            st.success("✅ Analysis complete!")
            time.sleep(1)
        else:
            st.error("❌ Analysis failed. Please try again.")
    
    # Reset analyzing state
    st.session_state.is_analyzing = False
    st.rerun()

# Individual Model Results - Display below in pie charts
if st.session_state.text_analysis_result:
    st.markdown("---")
    
    result = st.session_state.text_analysis_result
    
    # Get individual model results
    individual_scores = result.get('individual_scores', [])
    
    if individual_scores:
        # Create responsive columns for pie charts
        num_models = len(individual_scores)
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
                            model_result = individual_scores[model_idx]
                            fig = create_text_pie_chart(model_result)
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No individual model results available")

