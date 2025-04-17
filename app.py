import streamlit as st
import os
from PIL import Image
from model_connector import predict_skin_disease

# Configure page settings
st.set_page_config(
    page_title="Skin Diseases Classifier",
    page_icon="🔬",
    layout="centered"
)

# Add custom CSS for Italiana font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Italiana&display=swap');
    .italiana-font {
        font-family: 'Italiana', serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


def set_page(page_name):
    st.session_state.page = page_name


def restart_app():
    # Reset all session state variables
    st.session_state.page = 'welcome'
    st.session_state.uploaded_image = None


# Welcome page
def welcome_page():
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>Skin Disease Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='italiana-font' style='text-align: center; font-size: 60px;'>WELCOME</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start", key="start_btn", use_container_width=True, on_click=lambda: set_page('upload')):
            pass  # Action is handled by on_click

    st.markdown("<div style='position: fixed; bottom: 20px; width: 100%; text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 15px;'>AI 50.021</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Upload page
def upload_page():
    st.title("Image Preview")
    
    # Create a placeholder for the image preview
    preview_container = st.empty()
    preview_container.markdown(
        """
        <div style='background-color: #757061; height: 400px; display: flex; 
        justify-content: center; align-items: center; text-align: center;'>
        Please select an image to upload
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Upload button
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")
    
    # If user uploads a file
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            preview_container.image(image, use_container_width=True)
            
            # Show upload button with on_click handler
            if st.button("Upload Image", use_container_width=True, on_click=lambda: set_page('results')):
                pass  # Action is handled by on_click
        except Exception as e:
            st.error(f"Error opening image: {e}")


# Results page
def results_page():
    st.title("Results")
    
    # Only proceed if an image is available
    if st.session_state.uploaded_image:
        # 1. Display the uploaded image with placeholder for GradCAM overlay
        st.subheader("Analysis Visualization")
        
        # For now, just display the original image
        # In the future, this would be replaced with GradCAM visualization
        st.image(st.session_state.uploaded_image, use_container_width=True)
        
        st.info("Note: The above is a GradCAM feature to highlight the areas of interest for the model's analysis.")
        
        # 2. Display the model results in a grey box below the image
        st.subheader("Model Output")
        st.markdown(
            """
            <div style='background-color: #757061; padding: 20px; border-radius: 5px;'>
            <ol>
                <li>Skin disease classification</li>
                <li>Confidence Score</li>
            </ol>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Here we would call the model for prediction
        # But since the model is not ready, we'll show placeholder values
        classification, confidence = predict_skin_disease(st.session_state.uploaded_image)
        
        # Display the classification and confidence score
        st.markdown(f"**Classification:** {classification}")
        st.markdown(f"**Confidence Score:** {confidence:.2%}" if isinstance(confidence, float) else f"**Confidence Score:** {confidence}")
        
        # Add restart button
        st.button("Start Over", on_click=restart_app, use_container_width=True)
    else:
        st.error("No image available. Please go back and upload an image.")
        # Add restart button even if there's an error
        st.button("Return to Home", on_click=restart_app, use_container_width=True)


# Main app logic - determine which page to show
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'upload':
    upload_page()
elif st.session_state.page == 'results':
    results_page()