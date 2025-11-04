"""
Chat With Image Page
This page allows users to upload images and ask questions about them using Azure OpenAI chat.
Supports two modes: Analyze (Azure OpenAI vision) and Analyze with Detection (deepfake detection)
"""

import streamlit as st
import base64
import io
from PIL import Image
from components.styles import apply_custom_css
from components.header import render_header
from components.api_utils import get_endpoints, image_to_bytes
from components.charts import create_pie_chart
import requests
import os

# Apply custom styling
apply_custom_css()

# Render header
render_header("Chat with Image")

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "analyze"  # "analyze" or "analyze_with_detect"

if 'azure_openai_endpoint' not in st.session_state:
    st.session_state.azure_openai_endpoint = ""

if 'azure_openai_deployment' not in st.session_state:
    st.session_state.azure_openai_deployment = ""

if 'azure_openai_api_key' not in st.session_state:
    st.session_state.azure_openai_api_key = ""

def clear_chat():
    """Clear all chat messages and session state"""
    st.session_state.chat_messages = []
    st.session_state.chat_mode = "analyze"
    st.session_state.show_chat_config = False


def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_azure_openai_vision(image_base64, question):
    """Call Azure OpenAI vision API with image and question"""
    try:
        # Check if Azure OpenAI credentials are configured
        if not st.session_state.azure_openai_endpoint or not st.session_state.azure_openai_api_key:
            return {
                "error": "Azure OpenAI not configured",
                "message": "Please configure Azure OpenAI endpoint and API key in settings"
            }
        
        # Prepare the request
        endpoint = st.session_state.azure_openai_endpoint.rstrip('/')
        deployment = st.session_state.azure_openai_deployment or "gpt-4o"
        api_version = "2024-02-15-preview"
        
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": st.session_state.azure_openai_api_key
        }
        
        # Prepare the message with image
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_completion_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return {
                    "success": True,
                    "answer": result['choices'][0]['message']['content']
                }
            else:
                return {
                    "error": "No response",
                    "message": "Azure OpenAI returned no content"
                }
        else:
            return {
                "error": f"API Error {response.status_code}",
                "message": response.text
            }
            
    except Exception as e:
        return {
            "error": "Request failed",
            "message": str(e)
        }


def call_deepfake_detection(image_bytes, filename):
    """Call the Azure Function deepfake detection endpoint"""
    try:
        endpoints = get_endpoints()
        files = {'image': (filename, image_bytes, 'image/png')}
        
        response = requests.post(
            endpoints['analyze'],
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Analysis failed with status {response.status_code}",
                "message": response.text
            }
    except Exception as e:
        return {
            "error": "Analysis request failed",
            "message": str(e)
        }


def display_detection_result(result):
    """Display deepfake detection result in the chat with markdown and charts"""
    if 'error' in result:
        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
        return
    
    # Display markdown result
    st.markdown("#### 📊 Analysis Results")
    
    # Check if prediction is FAKE and add warning
    prediction = result.get('prediction', '')
    confidence = result.get('confidence', 0)
    
    if prediction.upper() == "FAKE":
        confidence_percent = int(confidence * 100)
        
        # Determine warning level and styling based on confidence
        if confidence >= 0.90:
            # High confidence - Red warning
            icon = "🛡️"
            warning_color = "#dc3545"  # Red
            border_color = "#dc3545"
        else:
            # Moderate confidence - Amber/Orange warning
            icon = "⚠️"
            warning_color = "#fd7e14"  # Orange
            border_color = "#fd7e14"
        
        # Display warning banner
        st.markdown(f"""
        <div style="
            padding: 15px;
            border-left: 5px solid {border_color};
            background-color: rgba(253, 126, 20, 0.1);
            border-radius: 5px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; color: {warning_color}; font-weight: bold; font-size: 16px;">
                {icon} Warning: This image may have been altered.
            </p>
            <p style="margin: 5px 0 0 0; color: #333; font-size: 14px;">
                Our analysis shows a {confidence_percent}% likelihood of manipulation. Please verify the source before using it.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    markdown_content = result.get('markdown', 'No results available')
    st.markdown(markdown_content)
    
    # Display individual model results with pie charts
    st.markdown("#### 🎯 Individual Model Results")
    
    # Get individual model results directly from top level (ImageDetectionResult structure)
    individual_results = result.get('individualModelResults', [])
    
    if individual_results and len(individual_results) > 0:
        # Create responsive columns for pie charts (same as AnalyzeImage page)
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


# Sidebar with Clear Chat button
with st.sidebar:
    st.markdown("### 💬 Chat Actions")
    
    if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
        clear_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.markdown("""
    **Configuration:**
    - Click the ⚙️ Settings button in the header
    - Configure API URL, Chat Mode, and Azure OpenAI credentials
    
    **Chat Modes:**
    - 🤖 **Analyze**: Ask questions about images using Azure OpenAI Vision
    - 🔍 **Analyze with Detection**: Run detection on images
    
    **Usage:**
    - Upload an image using the 📎 icon in the chat input
    - Type your question and press Enter
    """)

# Display current mode
mode_emoji = "🤖" if st.session_state.chat_mode == "analyze" else "🔍"
mode_text = "Azure OpenAI Vision" if st.session_state.chat_mode == "analyze" else "Analyze with Detection"
st.info(f"{mode_emoji} **Current Mode:** {mode_text}")

# Chat display area
st.markdown("### 💬 Chat")

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display image if present (reconstruct from base64)
        if "image_base64" in message and message["image_base64"] is not None:
            try:
                image_data = base64.b64decode(message["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                st.image(image, width=300, caption="Uploaded Image")
            except Exception as e:
                st.caption(f"[Image display error: {str(e)}]")

# Chat input with file upload
prompt = st.chat_input(
    "Ask a question and/or attach an image",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
    key="chat_input"
)

# Process user input
if prompt:
    user_message = ""
    uploaded_image = None
    image_filename = "image.png"
    
    # Check if prompt has text and/or files
    if prompt.text:
        user_message = prompt.text
    
    if prompt.files:
        # Get the first uploaded file
        uploaded_file = prompt.files[0]
        image_filename = uploaded_file.name if uploaded_file.name else "image.png"
        
        try:
            uploaded_image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load image: {str(e)}")
            uploaded_image = None
    
    # Add user message to chat
    if user_message or uploaded_image:
        # Convert image to base64 for storage (to avoid JSON serialization issues)
        image_base64_for_storage = None
        if uploaded_image:
            image_base64_for_storage = encode_image_to_base64(uploaded_image)
        
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message if user_message else "[Image uploaded]",
            "image_base64": image_base64_for_storage
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message if user_message else "[Image uploaded]")
            if uploaded_image:
                st.image(uploaded_image, width=300, caption="Uploaded Image")
        
        # Process based on mode
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                if st.session_state.chat_mode == "analyze":
                    # Use Azure OpenAI for vision analysis
                    if uploaded_image and user_message:
                        # Convert image to base64
                        image_base64 = encode_image_to_base64(uploaded_image)
                        
                        # Call Azure OpenAI
                        result = call_azure_openai_vision(image_base64, user_message)
                        
                        if "success" in result and result["success"]:
                            response_text = result["answer"]
                            st.markdown(response_text)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                        else:
                            response_text = f"❌ Error: {result.get('message', 'Unknown error')}"
                            st.error(response_text)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                    elif uploaded_image and not user_message:
                        response_text = "Please provide a question about the image."
                        st.info(response_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    elif not uploaded_image and user_message:
                        response_text = "Please upload an image to analyze."
                        st.info(response_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    else:
                        response_text = "Please provide both an image and a question."
                        st.info(response_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                
                elif st.session_state.chat_mode == "analyze_with_detect":
                    # Use deepfake detection
                    if uploaded_image:
                        # Convert image to bytes
                        image_bytes = image_to_bytes(uploaded_image)
                        
                        if image_bytes:
                            # Call deepfake detection
                            result = call_deepfake_detection(image_bytes, image_filename)
                            
                            # Display results with markdown and charts
                            display_detection_result(result)
                            
                            # Store simplified message for chat history
                            if 'error' not in result:
                                markdown_content = result.get('markdown', 'Analysis complete')
                                response_text = f"✅ Deepfake detection analysis complete.\n\n{markdown_content}"
                                
                                # Add assistant response to chat history (without storing complex result object)
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": response_text
                                })
                            else:
                                response_text = f"❌ Error: {result.get('message', 'Unknown error')}"
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": response_text
                                })
                        else:
                            response_text = "❌ Failed to process image"
                            st.error(response_text)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                    else:
                        response_text = "Please upload an image for deepfake detection."
                        st.info(response_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                else:
                    response_text = "Invalid mode selected."
                    st.error(response_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
