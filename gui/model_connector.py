import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import os
import sys
import datetime
import requests
import gdown
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt

# Debug flag - set to False to disable image saving
SAVE_DEBUG_IMAGES = False

# Add debug prints to help troubleshooting
print("Loading model_connector.py")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Google Drive model file ID - from your shared link
MODEL_FILE_ID = "1qPEB2qYfkWYbASdFAPi-dafjM2HEd5lY"
MODEL_FILENAME = "best_model.pth"

# Define classes for skin diseases
DISEASE_CLASSES = [
    'Eczema',
    'Warts Molluscum',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma',
    'Melanocytic Nevi',
    'Benign Keratosis',
    'Psoriasis',
    'Seborrheic Keratoses',
    'Tinea Ringworm'
]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure debug directories exist
def ensure_debug_dirs():
    if not SAVE_DEBUG_IMAGES:
        return None
        
    # Create a unique timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base debug directory
    debug_dir = os.path.join("debug_images")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Create session directory with timestamp
    session_dir = os.path.join(debug_dir, f"session_{timestamp}")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        
    # Create subdirectories
    orig_dir = os.path.join(session_dir, "original")
    result_dir = os.path.join(session_dir, "results")
    
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    return {
        "base": session_dir,
        "original": orig_dir,
        "results": result_dir
    }

# Function to download model from Google Drive
def download_model_from_gdrive():
    """Download the model file from Google Drive if not already present"""
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model file already exists at {model_path}")
        return model_path
    
    # Also check in models directory
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(models_dir):
        model_in_dir = os.path.join(models_dir, MODEL_FILENAME)
        if os.path.exists(model_in_dir):
            print(f"Model file already exists at {model_in_dir}")
            return model_in_dir
    
    # Create a models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    download_path = os.path.join(models_dir, MODEL_FILENAME)
    
    # Download using gdown (handles Google Drive links)
    try:
        print(f"Downloading model from Google Drive (ID: {MODEL_FILE_ID})...")
        # Direct download URL format for Google Drive
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        
        # First attempt: use gdown
        success = gdown.download(url, download_path, quiet=False)
        
        if not success or not os.path.exists(download_path) or os.path.getsize(download_path) < 1000:  # Check if file is too small (likely an error)
            print("First download attempt failed, trying alternative method...")
            
            # Alternative method: using requests with cookies
            try:
                import requests
                print("Trying alternative download method with requests...")
                
                # Try direct download with confirm=t parameter
                confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={MODEL_FILE_ID}"
                response = requests.get(confirm_url, stream=True)
                
                # Check if we got a valid response
                if response.status_code == 200 and int(response.headers.get('content-length', 0)) > 1000000:  # At least 1MB
                    with open(download_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    print(f"Downloaded model using requests: {os.path.getsize(download_path)} bytes")
                else:
                    print(f"Failed to download with requests. Status code: {response.status_code}")
                    print(f"Content length: {response.headers.get('content-length', 'unknown')}")
                    raise Exception("Download failed with both methods")
            except Exception as req_err:
                print(f"Alternative download method failed: {req_err}")
                raise Exception(f"Failed to download model with multiple methods: {req_err}")
        
        # Verify the downloaded file
        if os.path.exists(download_path):
            file_size = os.path.getsize(download_path)
            print(f"Model downloaded successfully to {download_path} (Size: {file_size} bytes)")
            
            # Check if the file size is reasonable (at least 10MB for a deep learning model)
            if file_size < 10 * 1024 * 1024:  # 10MB
                print(f"Warning: Downloaded file seems too small ({file_size} bytes). Might not be a valid model file.")
                
            return download_path
        else:
            print("Model download failed - file does not exist")
            raise FileNotFoundError(f"Could not download model to {download_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        raise FileNotFoundError(f"Failed to download model from Google Drive: {str(e)}")

# Define PadToSquare transform
class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        return transforms.functional.pad(
            image,
            (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h),
            fill=0
        )

# Define normalization stats (same as training)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Define preprocessing transform that matches val_test_transform
preprocess = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Define DenseNet model architecture - now simplified to ensure correct loading
class DenseNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Load base DenseNet model without pretrained weights first
        self.densenet = models.densenet121()
        
        # Get feature dimension
        num_ftrs = self.densenet.classifier.in_features  # 1024 for DenseNet121
        
        # Create custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),     # Hidden layer 1
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),          # Hidden layer 2
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),          # Hidden layer 3
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)   # Final output layer
        )
        
        # Replace the classifier in densenet
        self.densenet.classifier = self.classifier
    
    def forward(self, x):
        # Use the built-in forward method
        return self.densenet(x)

# Initialize model to None (it will be loaded when needed)
model = None

def find_model_file():
    """Find the model file in current directory or models directory"""
    # Check the main directory
    model_files = [f for f in os.listdir() if f.endswith('.pth')]
    
    # Check models directory if it exists
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(models_dir):
        model_files.extend([os.path.join("models", f) for f in os.listdir(models_dir) if f.endswith('.pth')])
    
    print(f"Found model files: {model_files}")
    return model_files[0] if model_files else None

def inspect_model_structure(model_path):
    """Inspect the structure of the saved model to help with debugging"""
    try:
        state_dict = torch.load(model_path, map_location=device)
        print(f"Model file contains {len(state_dict)} parameters")
        
        # Print some key information
        keys = list(state_dict.keys())
        print(f"First 10 keys: {keys[:10]}")
        
        # Check for common prefixes
        prefixes = set()
        for key in keys:
            parts = key.split('.')
            if len(parts) > 1:
                prefixes.add(parts[0])
        
        print(f"Key prefixes in state dict: {prefixes}")
        
        # Count key types
        feature_keys = sum(1 for k in keys if 'features' in k)
        classifier_keys = sum(1 for k in keys if 'classifier' in k)
        
        print(f"Feature keys: {feature_keys}")
        print(f"Classifier keys: {classifier_keys}")
        
        # Return useful information
        return {
            'total_params': len(state_dict),
            'prefixes': prefixes,
            'feature_keys': feature_keys,
            'classifier_keys': classifier_keys,
            'first_keys': keys[:10]
        }
    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model():
    """Load the model if it's not already loaded with improved error handling"""
    global model
    
    if model is None:
        try:
            print("Creating DenseNetModel architecture...")
            
            # Try to find a local model file first
            model_path = None
            local_model = find_model_file()
            
            if local_model and os.path.exists(local_model):
                print(f"Found existing model file: {local_model}")
                model_path = local_model
                
                # Inspect the model structure to help with debugging
                print("Inspecting model structure...")
                model_info = inspect_model_structure(model_path)
                
                if model_info and 'prefixes' in model_info:
                    print(f"Model has prefixes: {model_info['prefixes']}")
            else:
                # Try to download model if not found locally
                print("No local model found. Attempting to download...")
                try:
                    model_path = download_model_from_gdrive()
                    
                    # Inspect downloaded model
                    print("Inspecting downloaded model structure...")
                    model_info = inspect_model_structure(model_path)
                except Exception as download_err:
                    print(f"Failed to download model: {download_err}")
                    raise FileNotFoundError(f"Could not download or find model: {str(download_err)}")
            
            # Create model instance
            model = DenseNetModel(num_classes=len(DISEASE_CLASSES)).to(device)
            print(f"DenseNetModel created with {len(DISEASE_CLASSES)} classes")
            
            if model_path and os.path.exists(model_path):
                try:
                    # Load the model weights
                    print(f"Loading model weights from {model_path}...")
                    state_dict = torch.load(model_path, map_location=device)
                    
                    # Print model keys and state dict keys for comparison
                    model_keys = set(model.state_dict().keys())
                    state_dict_keys = set(state_dict.keys())
                    
                    # Check for key mismatches
                    missing_keys = model_keys - state_dict_keys
                    extra_keys = state_dict_keys - model_keys
                    
                    print(f"Model has {len(model_keys)} parameters")
                    print(f"State dict has {len(state_dict_keys)} parameters")
                    
                    if missing_keys:
                        print(f"Missing {len(missing_keys)} keys in state dict")
                        print(f"First few missing keys: {list(missing_keys)[:5]}")
                    
                    if extra_keys:
                        print(f"Extra {len(extra_keys)} keys in state dict not in model")
                        print(f"First few extra keys: {list(extra_keys)[:5]}")
                    
                    # Check if keys need remapping (common with DataParallel)
                    if len(extra_keys) > 0 and all('module.' in k for k in extra_keys):
                        print("State dict appears to be from DataParallel. Removing 'module.' prefix...")
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
                    # Special handling for different key formats
                    if 'densenet.' in list(model_keys)[0] and 'densenet.' not in list(state_dict_keys)[0]:
                        print("Adding 'densenet.' prefix to state dict keys...")
                        state_dict = {'densenet.' + k: v for k, v in state_dict.items()}
                    
                    # Load state dict with strict=False to allow partial loading
                    model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded model weights with strict=False")
                    
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    import traceback
                    traceback.print_exc()
                    raise FileNotFoundError(f"Could not load model weights: {str(e)}")
            else:
                print(f"Model file not found and could not be downloaded.")
                model = None
                raise FileNotFoundError("Model file not found and could not be downloaded.")
                
            # Set model to evaluation mode
            model.eval()
            print("Model loaded successfully and set to evaluation mode")
        except Exception as e:
            print(f"Critical error creating model: {e}")
            model = None
            import traceback
            traceback.print_exc()
            raise e
        
        if model_path:
            print(f"Model successfully loaded from: {os.path.abspath(model_path)}")
    
    return model

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # Use the last convolutional layer of DenseNet's last dense block
        self.target_layer = target_layer or model.densenet.features.denseblock4.denselayer16.conv2

        self.activations = None
        self.gradients = None

        self.handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient)
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def release(self):
        for h in self.handles:
            h.remove()

    def __call__(self, x, class_idx=None):
        orig_mode = self.model.training

        # Disable all in-place ReLUs
        import torch.nn as nn
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        # Patch functional relu to avoid inplace
        import torch.nn.functional as F
        orig_frelu = F.relu
        F.relu = lambda inp, inplace=True: orig_frelu(inp, inplace=False)

        try:
            def _bn_eval(m):
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
            self.model.apply(_bn_eval)

            x = x.clone().requires_grad_(True)
            out = self.model(x)

            if class_idx is None:
                class_idx = torch.argmax(out, dim=1).item()

            one_hot = torch.zeros_like(out)
            one_hot[0, class_idx] = 1
            self.model.zero_grad()
            out.backward(gradient=one_hot, retain_graph=True)

            weights = torch.mean(self.gradients, dim=(2, 3))  # shape: [1, C]

            cam = torch.zeros_like(self.activations[0, 0])
            for i, w in enumerate(weights[0]):
                cam += w * self.activations[0, i]

            cam = F.relu(cam)
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
            return cam.squeeze().cpu().numpy()

        finally:
            F.relu = orig_frelu
            self.model.train(orig_mode)

def generate_gradcam_images(image_tensor, predicted_class=None):
    """Generate GradCAM visualization images - both heatmap and overlay"""
    try:
        # Create debug directories if needed
        debug_dirs = ensure_debug_dirs() if SAVE_DEBUG_IMAGES else None
        
        model = load_model()
        
        # Target the right layer based on your model architecture
        target_layer = model.densenet.features.denseblock4.denselayer16.conv2
        print(f"GradCAM targeting layer: {target_layer}")
        
        # Create GradCAM instance
        grad_cam = GradCAM(model, target_layer=target_layer)
        
        # Get original image for display (before normalization)
        orig_img = image_tensor.squeeze().cpu().clone().detach()
        
        # Get prediction if not provided
        if predicted_class is None:
            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class].item()
                print(f"Predicted class: {DISEASE_CLASSES[predicted_class]} with {confidence:.2%} confidence")
        
        # Generate CAM for predicted class
        print(f"Generating GradCAM for class {predicted_class}")
        cam = grad_cam(image_tensor, predicted_class)
        print(f"GradCAM generated with shape: {cam.shape}, min: {cam.min()}, max: {cam.max()}")
        
        # Release hooks
        grad_cam.release()
        
        # Convert image tensor to numpy for visualization
        img_np = orig_img.permute(1, 2, 0).numpy()
        
        # Unnormalize the image
        img_np = img_np * np.array(imagenet_std) + np.array(imagenet_mean)
        img_np = np.clip(img_np, 0, 1)
        
        # Make heatmap & overlay (using matplotlib colormap)
        print("Creating heatmap from matplotlib's jet colormap...")
        heatmap = plt.get_cmap('jet')(cam)[:, :, :3]  # H×W×3
        
        # Create overlay
        print("Creating overlay...")
        overlay = 0.6 * img_np + 0.4 * heatmap
        overlay = overlay / np.maximum(overlay.max(), 1e-8)  # Avoid division by zero
        
        # Convert result to PIL Image (overlay)
        result_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        # Convert heatmap to PIL Image (standalone heatmap)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        
        # Save intermediate images for debugging (only if enabled)
        if SAVE_DEBUG_IMAGES and debug_dirs:
            try:
                # Save original
                orig_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                orig_pil.save(os.path.join(debug_dirs["original"], 'debug_original.png'))
                
                # Save heatmap and result to results dir
                heatmap_img.save(os.path.join(debug_dirs["results"], 'debug_heatmap.png'))
                result_img.save(os.path.join(debug_dirs["results"], 'debug_result.png'))
                
                print(f"Debug images saved to {debug_dirs['base']}")
            except Exception as e:
                print(f"Could not save debug images: {e}")
        
        return result_img, heatmap_img
    except Exception as e:
        print(f"Error generating GradCAM: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_image(image):
    """Preprocess the image for model input"""
    try:
        # Convert to RGB if not already (important for proper processing)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Apply preprocessing pipeline
        img_tensor = preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
        print(f"Image preprocessed successfully with shape: {img_tensor.shape}")
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        raise e

def predict_skin_disease(image):
    """
    Predict skin disease from an image and return classification, confidence, and GradCAM visualizations.
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (disease_classification, confidence_score, overlay_image, heatmap_image)
    """
    try:
        # Create debug directories if needed
        debug_dirs = ensure_debug_dirs() if SAVE_DEBUG_IMAGES else None
        
        print(f"Starting prediction for image size: {image.size}")
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            print(f"Converted image to RGB mode")
            
        # Load model
        model = load_model()
        if model is None:
            print("Model loading failed")
            return "Model loading failed", 0.0, None, None
        
        # Preprocess the image
        img_tensor = preprocess_image(image)
        print(f"Image preprocessed to tensor shape: {img_tensor.shape}")
        
        # Get prediction
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
            
        print(f"Prediction: class {pred_idx} ({DISEASE_CLASSES[pred_idx]}) with confidence {confidence:.2f}")
        
        # Get class name
        disease_class = DISEASE_CLASSES[pred_idx]
        
        # Generate GradCAM visualizations
        print("Generating GradCAM...")
        overlay_img, heatmap_img = generate_gradcam_images(img_tensor, pred_idx)
        
        if overlay_img is None or heatmap_img is None:
            print("GradCAM generation failed, using original image")
            copy_img = image.copy()  # Create a copy to avoid modifying the original
            return disease_class, confidence, copy_img, copy_img
        else:
            print("GradCAM generation successful!")
            
            # Save the final GradCAM images for debugging (only if enabled)
            if SAVE_DEBUG_IMAGES and debug_dirs:
                try:
                    overlay_img.save(os.path.join(debug_dirs["results"], 'gradcam_overlay.png'))
                    heatmap_img.save(os.path.join(debug_dirs["results"], 'gradcam_heatmap.png'))
                    
                    # Save a text file with prediction details
                    with open(os.path.join(debug_dirs["base"], 'prediction_results.txt'), 'w') as f:
                        f.write(f"Prediction: {disease_class}\n")
                        f.write(f"Confidence: {confidence:.4f}\n")
                        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"Saved final results to {debug_dirs['results']}")
                except Exception as e:
                    print(f"Could not save final results: {e}")
        
        print("Prediction completed successfully")
        return disease_class, confidence, overlay_img, heatmap_img
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Error in classification", 0.0, None, None

# Add a simple test function to check model loading and prediction
def test_model_prediction(test_image_path=None):
    """Test model loading and prediction with a sample image"""
    try:
        # Load the model
        model = load_model()
        if model is None:
            print("Model loading failed")
            return False
        
        # Use a test image, if provided
        if test_image_path and os.path.exists(test_image_path):
            print(f"Testing with image: {test_image_path}")
            image = Image.open(test_image_path)
        else:
            # Create a simple test image if none provided
            print("Creating test image...")
            image = Image.new('RGB', (300, 300), color=(100, 150, 200))
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess the image
        img_tensor = preprocess_image(image)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        print(f"Test prediction: {DISEASE_CLASSES[pred_idx]} with {confidence:.2%} confidence")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# If this file is run directly, test the model
if __name__ == "__main__":
    print("Running model test...")
    test_model_prediction()