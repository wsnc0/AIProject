# Key change: Modified the results page to show the original uploaded image
# alongside the GradCAM heatmap and overlay images

import streamlit as st
import os
import base64
import io
from PIL import Image

# Import the model connector for classification and GradCAM
from model_connector import predict_skin_disease

# Import only the description function from Gemini integration
from gemini_integration import get_gemini_description

# Configure page settings
st.set_page_config(
    page_title="Skin Diseases Classifier",
    page_icon="🔬",
    layout="centered"
)

# Disable the default Streamlit top bar margin to make buttons look better
st.markdown("""
    <style>
        div.stButton > button {
            margin: 0 auto;
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)

# Add custom CSS for Italiana font and button styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Italiana&display=swap');
    .italiana-font {
        font-family: 'Italiana', serif;
    }
    .stButton button {
        background-color: #333;
        color: white;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: normal;
    }
    .grad-cam-container img {
        max-height: 400px;
        width: auto;
    }
    div[data-testid="stImage"] img {
        display: block;
        margin: 0 auto;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
    }
    .disclaimer {
        margin-top: 20px;
        margin-bottom: 30px;  /* Added margin to create space before button */
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #f39c12;
    }
    .disclaimer p {
        margin: 0;
        color: #555;
    }
    .image-caption {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .welcome-note {
        text-align: center;
        font-style: italic;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 10px;
        padding: 10px;
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 5px;
        display: inline-block;
    }
    .main-title {
        font-size: 100px !important;  /* Increased from 50px to 100px */
        text-align: center;
        margin-top: 40px;
        margin-bottom: 60px;
        font-family: 'Italiana', serif;
        color: #fff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .description-box {
        background-color: rgba(30, 30, 30, 0.7);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    /* Add spacing for button */
    .button-spacing {
        margin-top: 30px;
    }
    .description-panel {
        background-color: rgba(45, 45, 45, 0.7);
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #3498db;
    }
    .description-panel p {
        color: #eee;
        font-size: 1rem;
        line-height: 1.5;
    }
    .error-panel {
        background-color: rgba(200, 30, 30, 0.3);
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #e74c3c;
    }
    .error-panel p {
        color: #eee;
        font-size: 1rem;
        line-height: 1.5;
    }
    .info-panel {
        background-color: rgba(52, 152, 219, 0.3);
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #3498db;
    }
    .info-panel p {
        color: #eee;
        font-size: 1rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'gradcam_overlay' not in st.session_state:
    st.session_state.gradcam_overlay = None
if 'gradcam_heatmap' not in st.session_state:
    st.session_state.gradcam_heatmap = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'heatmap_description' not in st.session_state:
    st.session_state.heatmap_description = ""
if 'disease_description' not in st.session_state:
    st.session_state.disease_description = ""
if 'use_gemini' not in st.session_state:
    st.session_state.use_gemini = True  # Default to using Gemini
if 'gemini_error' not in st.session_state:
    st.session_state.gemini_error = None
if 'model_download_error' not in st.session_state:
    st.session_state.model_download_error = None


# Callback functions to handle button clicks
def set_page_callback(page_name):
    # Set the page immediately and rerun
    st.session_state.page = page_name
    st.session_state.button_clicked = True
    st.rerun()  # Use st.rerun() for newer versions, or st.experimental_rerun() for older versions


def restart_app_callback():
    # Reset all session state variables
    st.session_state.page = 'welcome'
    st.session_state.uploaded_image = None
    st.session_state.gradcam_overlay = None
    st.session_state.gradcam_heatmap = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.heatmap_description = ""
    st.session_state.disease_description = ""
    st.session_state.analysis_complete = False
    st.session_state.button_clicked = True
    st.session_state.gemini_error = None
    st.session_state.model_download_error = None
    st.rerun()  # Use st.rerun() for newer versions


def process_image_callback():
    if st.session_state.uploaded_image:
        # Navigate to results page immediately
        st.session_state.page = 'results'
        st.session_state.analysis_complete = False
        st.session_state.button_clicked = True
        # Reset any previous errors
        st.session_state.model_download_error = None
        # Force rerun to immediately update the page
        st.rerun()
    else:
        st.error("No image available for analysis. Please upload an image first.")


def toggle_gemini_callback():
    st.session_state.use_gemini = not st.session_state.use_gemini


# Welcome page
def welcome_page():
    # Main Title - Much larger and removed the WELCOME text
    st.markdown("<h1 class='main-title'>Skin Disease Classifier</h1>", unsafe_allow_html=True)
    
    # Description box
    st.markdown("""
    <div class="description-box">
        <p style="color: white; font-size: 18px; text-align: center;">Developed by Celest, Ivan, Joel and Sofeanna.</p>
        <p style="color: white; font-size: 18px; text-align: center;">This application uses AI to analyze images and identify potential skin conditions.</p>
        <p style="color: white; font-size: 18px; text-align: center;">Simply upload a clear image of the skin condition, and let our AI provide an assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add info about remote model
    st.markdown("""
    <div class="info-panel">
        <p><strong>Note:</strong> The AI model will be downloaded on first use. This may take a moment depending on your internet connection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gemini AI toggle
    st.checkbox("Use Gemini AI for enhanced descriptions", value=st.session_state.use_gemini, 
                on_change=toggle_gemini_callback)
    
    if st.session_state.use_gemini:
        st.markdown("""
        <div style="background-color: rgba(50, 50, 50, 0.7); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <p style="color: white; font-size: 14px;">Enhanced descriptions will provide more detailed information about the detected skin condition using Google's Gemini AI model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Start button - properly centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start", key="start_btn", use_container_width=True):
            set_page_callback('upload')

    # Footer
    st.markdown("<div style='position: fixed; bottom: 40px; width: 100%; text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 15px;'>AI 50.021</h3>", unsafe_allow_html=True)
    
    # Note below the footer
    st.markdown("<div style='position: fixed; bottom: 10px; width: 100%; text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<p class='welcome-note'>Note: This tool is for educational purposes only and should not replace professional medical advice.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Upload page
def upload_page():
    st.title("Image Preview")
    
    # Create a placeholder for the image preview
    preview_container = st.empty()
    preview_container.markdown(
        """
        <div style='background-color: #616161; height: 400px; display: flex; 
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
            
            # Modified this line to handle older versions of Streamlit
            # that don't support use_container_width
            try:
                preview_container.image(image, use_container_width=True)
            except TypeError:
                # Fallback for older streamlit versions
                preview_container.image(image)
            
            # Show image details
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Image size: {image.size}")
            with col2:
                st.write(f"Image format: {image.format or 'JPEG'}")
            
            # Process image with the model when user clicks "Analyze Image"
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Analyze Image", use_container_width=True):
                    process_image_callback()
    
        except Exception as e:
            st.error(f"Error opening image: {e}")
    
    # Add a back button - centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Back to Home", key="back_home_btn", use_container_width=True):
            restart_app_callback()


# Results page
def results_page():
    st.title("Results")
    
    # If analysis hasn't been done yet but we have an image, run the analysis
    if not st.session_state.analysis_complete and st.session_state.uploaded_image:
        with st.spinner('Analyzing image...'):
            try:
                # Create a placeholder for the progress
                progress_placeholder = st.empty()
                progress_placeholder.write("Loading model (this may take a moment the first time)...")
                
                # Process the image and get results using your model_connector
                prediction, confidence, overlay_image, heatmap_image = predict_skin_disease(st.session_state.uploaded_image)
                
                # Reset any previous errors
                st.session_state.gemini_error = None
                st.session_state.model_download_error = None
                
                # Get descriptions based on user preference
                with st.spinner('Generating descriptions...'):
                    # Try to use Gemini if enabled
                    if st.session_state.use_gemini:
                        try:
                            # Call Gemini for enhanced descriptions
                            disease_description, heatmap_description = get_gemini_description(prediction, confidence)
                            progress_placeholder.write("Generating enhanced descriptions with Gemini AI...")
                        except Exception as e:
                            # Store the error message
                            st.session_state.gemini_error = f"Error getting description from Gemini AI: {str(e)}"
                            # Don't provide fallback descriptions - just use empty strings
                            disease_description = ""
                            heatmap_description = ""
                    else:
                        # If Gemini is disabled for descriptions, show an error
                        st.session_state.gemini_error = "Gemini AI descriptions are disabled. Please enable them in the settings."
                        disease_description = ""
                        heatmap_description = ""
                                
                # Store results in session state
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.gradcam_overlay = overlay_image
                st.session_state.gradcam_heatmap = heatmap_image
                st.session_state.heatmap_description = heatmap_description
                st.session_state.disease_description = disease_description
                st.session_state.analysis_complete = True
                
                # Clear the progress placeholder
                progress_placeholder.empty()
                
            except FileNotFoundError as e:
                # Specific error for model file not found
                error_msg = str(e)
                st.session_state.model_download_error = "The model file could not be downloaded or found. Please check your internet connection and try again."
                st.error(st.session_state.model_download_error)
                st.error(f"Error details: {error_msg}")
                
                # Add a button to return to the upload page - centered
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("Try Again", key="error_back_btn", use_container_width=True):
                        set_page_callback('upload')
                
            except Exception as e:
                st.error(f"Error during image analysis: {e}")
                import traceback
                st.code(traceback.format_exc())
                
                # Add a button to return to the upload page - centered
                col1, col2, col3 = st.columns([1, 1, 1]) 
                with col2:
                    if st.button("Go Back", key="error_back_btn2", use_container_width=True):
                        set_page_callback('upload')
    
    # Show specific error message for model download issues
    if st.session_state.model_download_error:
        st.markdown(f"""
        <div class="error-panel">
            <p><strong>Model Download Error:</strong> {st.session_state.model_download_error}</p>
            <p>Possible solutions:</p>
            <ul>
                <li>Check your internet connection</li>
                <li>Ensure the Google Drive link is accessible</li>
                <li>Try running the application again</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a button to retry - centered
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Try Again", key="retry_btn", use_container_width=True):
                process_image_callback()
        
    # Once analysis is complete, display results
    elif st.session_state.analysis_complete and st.session_state.prediction and st.session_state.gradcam_overlay and st.session_state.gradcam_heatmap:
        # Display the classification and confidence score
        st.markdown(f"**Classification:** {st.session_state.prediction}")
        
        # Format confidence as percentage if it's a float
        if isinstance(st.session_state.confidence, float):
            st.markdown(f"**Confidence Score:** {st.session_state.confidence:.2%}")
        else:
            st.markdown(f"**Confidence Score:** {st.session_state.confidence}")
            
        # Display Gemini usage indicator
        if st.session_state.use_gemini:
            st.markdown("<p style='color: #3498db; font-size: 0.8rem;'>✓ Enhanced descriptions with Gemini AI enabled</p>", unsafe_allow_html=True)
        
        # Display error message if there was an error with Gemini
        if st.session_state.gemini_error:
            st.markdown(f"""
            <div class="error-panel">
                <p><strong>Gemini AI Error:</strong> {st.session_state.gemini_error}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display disease description only if we have one
        if st.session_state.disease_description:
            st.markdown(f"""
            <div class="description-panel">
                <p><strong>About this condition:</strong> {st.session_state.disease_description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create three columns for the images (original, heatmap, overlay)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            # Display the original uploaded image
            try:
                st.image(st.session_state.uploaded_image, use_container_width=True)
            except TypeError:
                st.image(st.session_state.uploaded_image)
            st.markdown("<p class='image-caption'>Original uploaded image.</p>", unsafe_allow_html=True)
            
        with col2:
            st.subheader("Heatmap")
            # Display the heatmap
            try:
                st.image(st.session_state.gradcam_heatmap, use_container_width=True)
            except TypeError:
                st.image(st.session_state.gradcam_heatmap)
            st.markdown("<p class='image-caption'>Areas the AI focused on for prediction. Red indicates regions of high importance.</p>", unsafe_allow_html=True)
            
        with col3:
            st.subheader("Overlay")
            # Display the overlay
            try:
                st.image(st.session_state.gradcam_overlay, use_container_width=True)
            except TypeError:
                st.image(st.session_state.gradcam_overlay)
            st.markdown("<p class='image-caption'>Relevant areas highlighted on the original image to show what influenced the diagnosis.</p>", unsafe_allow_html=True)
        
        # Display heatmap analysis only if we have one
        if st.session_state.heatmap_description:
            st.markdown(f"""
            <div class="description-panel">
                <p><strong>Analysis of highlighted areas:</strong> {st.session_state.heatmap_description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add disclaimer
        st.markdown("""
        <div class="disclaimer">
            <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult with a dermatologist for proper diagnosis and treatment of skin conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add empty space
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Add restart button (with added spacing) - centered
        st.markdown("<div class='button-spacing'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Start Over", key="restart_btn", use_container_width=True):
                restart_app_callback()
    else:
        # Show error message if no prediction or image data is available
        if not st.session_state.uploaded_image and not st.session_state.model_download_error:
            st.error("No analysis results available. Please go back and upload an image.")
            
            # Add a button to return to the upload page - centered
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Upload Another Image", key="back_to_upload", use_container_width=True):
                    set_page_callback('upload')


# Main app logic - determine which page to show
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'upload':
    upload_page()
elif st.session_state.page == 'results':
    results_page()