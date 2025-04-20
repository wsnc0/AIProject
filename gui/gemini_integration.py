import google.generativeai as genai
import os

# Initialize the Gemini client with your API key
genai.configure(api_key="AIzaSyDCDw2YmiMKIIPB2VNsrvdXPIdqFUxqk4I")

# Print available models at startup to help troubleshoot
try:
    print("Listing available Gemini models:")
    available_models = genai.list_models()
    for model in available_models:
        print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

def get_gemini_description(disease_name, confidence_score):
    """
    Get a detailed description of the skin disease using Gemini AI API
    
    Args:
        disease_name: The name of the diagnosed skin disease
        confidence_score: The confidence score of the diagnosis (0-1)
        
    Returns:
        A tuple containing (disease_description, heatmap_analysis)
    """
    # Print debugging info
    print(f"Requesting Gemini description for: {disease_name}")
    
    try:
        # Use the newer gemini-1.5-flash model for text generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # System instructions as context (combined into the prompt for Gemini)
        system_instructions = """You are a dermatology assistant providing accurate medical information 
        about skin conditions. Keep descriptions concise but informative, focusing on key symptoms, 
        appearance, potential causes, and general treatment approaches. Avoid medical jargon unless 
        necessary and never provide definitive diagnosis or specific treatment recommendations."""
        
        # Create the prompt for the disease description
        disease_prompt = f"""{system_instructions}
        
        Provide a detailed description (5-6 sentences) of the skin condition '{disease_name}'. 
        Include:
        1. What it is (including if it's infectious, cancerous, etc.)
        2. Its typical appearance and how it presents on the skin
        3. Common symptoms or sensations (pain, itching, etc.)
        4. General treatment approaches
        5. When someone should urgently see a doctor
        Be specific and informative. The AI model detected this with {confidence_score:.2%} confidence.
        """
        
        # Create the prompt for the GradCAM analysis
        gradcam_prompt = f"""{system_instructions}
        
        Explain in 3-4 detailed sentences what visual features a medical AI model would focus on when 
        identifying '{disease_name}' in a skin image. What specific visual characteristics 
        (color, texture, border patterns, shapes, etc.) would the AI look for in the GradCAM 
        heatmap to make this particular diagnosis? Be specific to this condition rather than generic.
        """
        
        # Make the first API call for disease description
        print("Sending disease description request to Gemini...")
        disease_response = model.generate_content(disease_prompt)
        
        # Make the second API call for GradCAM analysis
        print("Sending GradCAM analysis request to Gemini...")
        gradcam_response = model.generate_content(gradcam_prompt)
        
        # Extract the responses
        disease_description = disease_response.text.strip()
        heatmap_analysis = gradcam_response.text.strip()
        
        # Print response for debugging
        print(f"Gemini response received. Description length: {len(disease_description)} characters")
        
        # Verify we got actual content, not just generic text
        if len(disease_description) < 50 or "please consult a dermatologist" in disease_description.lower():
            print("Warning: Response seems too short or generic")
            raise ValueError("Gemini returned a generic or too short response")
            
        return disease_description, heatmap_analysis
    
    except Exception as e:
        print(f"Error getting Gemini description: {e}")
        import traceback
        traceback.print_exc()
        
        # Instead of fallback, raise an error to the calling code
        raise Exception(f"Unable to get description from Gemini: {str(e)}")