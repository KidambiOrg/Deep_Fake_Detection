"""
Shared CSS styles for the Streamlit application
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the application"""
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
