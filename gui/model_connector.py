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
import gdown  # Add this new import for Google Drive downloads

# Debug flag - set to False to disable image saving
SAVE_DEBUG_IMAGES = False

# Add debug prints to help troubleshooting
print("Loading model_connector.py")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Google Drive model file ID - extracted from your shared link
MODEL_FILE_ID = "1rY6XJTDIvn90ZheaQ1N7Hk4B8LqVkNDs"
MODEL_FILENAME = "efficientnet_phase4_epoch6-3.pth"

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
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,  # left
            (max_side - h) // 2,  # top
            (max_side - w + 1) // 2,  # right
            (max_side - h + 1) // 2,  # bottom
        )
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

# Define image preprocessing transform
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# This matches your val_test_transform from the notebook
preprocess = transforms.Compose([
    PadToSquare(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Define model architecture
class EfficientNetWithLSTM(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=10):
        super().__init__()
        
        # Important: use weights parameter for newer PyTorch versions
        try:
            self.backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")
        except TypeError:
            # Fallback for older PyTorch versions
            self.backbone = models.efficientnet_b4(pretrained=True)
            
        self.backbone.classifier = nn.Identity()
        
        self.lstm = nn.LSTM(
            input_size=1792,  
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: [B, 3, 380, 380]
        feats = self.backbone.features(x)  # [B, 1792, H, W]
        B, C, H, W = feats.size()
        feats = feats.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, 1792]
        
        lstm_out, _ = self.lstm(feats)  # [B, H*W, 2*hidden_dim]
        pooled = lstm_out.mean(dim=1)  # [B, 2*hidden_dim]
        
        return self.classifier(pooled)

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

def load_model():
    """Load the model if it's not already loaded"""
    global model
    
    if model is None:
        try:
            print("Creating model architecture...")
            model = EfficientNetWithLSTM(num_classes=len(DISEASE_CLASSES)).to(device)
            print(f"Model architecture created with {len(DISEASE_CLASSES)} classes")
            
            model_path = None
            
            # First, try to find model file locally
            local_model = find_model_file()
            if local_model and os.path.exists(local_model):
                print(f"Found existing model file: {local_model}")
                model_path = local_model
            else:
                # Try to download model from Google Drive if not available locally
                print("No local model found, attempting to download from Google Drive...")
                try:
                    model_path = download_model_from_gdrive()
                except Exception as download_err:
                    print(f"Failed to download model: {download_err}")
                    raise FileNotFoundError(f"Could not download model: {str(download_err)}")
            
            if model_path and os.path.exists(model_path):
                try:
                    # Try to load the model weights
                    print(f"Loading model weights from {model_path}...")
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    print(f"Successfully loaded model from {model_path}")
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    print("Will try with strict=False...")
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded with strict=False")
                    except Exception as e2:
                        print(f"Still failed with strict=False: {e2}")
                        model = None
                        raise FileNotFoundError(f"Could not load model weights: {str(e2)}")
            else:
                print(f"Model file not found and could not be downloaded.")
                model = None
                raise FileNotFoundError("Model file not found and could not be downloaded.")
                
            model.eval()
            print("Model loaded successfully and set to evaluation mode")
        except Exception as e:
            print(f"Critical error creating model: {e}")
            model = None
            raise e
        print(f"Model successfully loaded from: {os.path.abspath(model_path)}")
    
    return model

# GradCAM implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.target_layer = model.backbone.features[-1]
        self.gradients = None
        self.activations = None
        
        self.handles = []
        self.handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def release(self):
        # Remove hooks when done
        for handle in self.handles:
            handle.remove()
    
    def __call__(self, x, class_idx=None):
        # Store original model mode
        original_mode = self.model.training
        
        try:
            # Switch to train mode (required for RNN backward pass)
            self.model.train()
            
            # Disable BatchNorm tracking (since using batch_size=1)
            def set_bn_eval(module):
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            self.model.apply(set_bn_eval)
            
            # Enable gradients
            with torch.enable_grad():
                x.requires_grad_()  # Enable gradients for input
                
                # Forward pass through the whole model to get predictions
                model_output = self.model(x)
                
                if class_idx is None:
                    # Use the predicted class if none is provided
                    class_idx = torch.argmax(model_output, dim=1).item()
                
                # Create one-hot encoding for the target class
                one_hot = torch.zeros_like(model_output)
                one_hot[0, class_idx] = 1
                
                # Zero gradients
                self.model.zero_grad()
                model_output.backward(gradient=one_hot, retain_graph=True)
                
                # Get weights
                # Global average pooling of gradients
                weights = torch.mean(self.gradients, dim=(2, 3))
                
                # Create class activation map
                batch_size, channels, height, width = self.activations.size()
                cam = torch.zeros(height, width, dtype=torch.float32, device=x.device)
                
                # Weight the channels by corresponding gradients
                for i, w in enumerate(weights[0]):
                    cam += w * self.activations[0, i, :, :]
                
                # Apply ReLU to focus on features that have a positive influence
                cam = F.relu(cam)
                # Normalize
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                # Resize to input image size
                cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
                cam = cam.squeeze()
            return cam.cpu().detach().numpy()
        
        finally:
            # Restore original model mode
            self.model.train(original_mode)

def generate_gradcam_images(image_tensor, predicted_class=None):
    """Generate GradCAM visualization images - both heatmap and overlay"""
    try:
        # Create debug directories if needed
        debug_dirs = ensure_debug_dirs() if SAVE_DEBUG_IMAGES else None
        
        model = load_model()
        
        # Create GradCAM instance
        grad_cam = GradCAM(model)
        
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
        cam = grad_cam(image_tensor, predicted_class)
        
        # Release hooks
        grad_cam.release()
        
        # Convert image tensor to numpy for visualization
        img_np = orig_img.permute(1, 2, 0).numpy()
        
        # Unnormalize the image
        img_np = img_np * np.array(imagenet_std) + np.array(imagenet_mean)
        img_np = np.clip(img_np, 0, 1)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Resize heatmap to match image dimensions if needed
        if heatmap.shape[:2] != img_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Overlay heatmap on original image
        result = 0.6 * img_np + 0.4 * heatmap
        result = result / result.max()
        
        # Convert result to PIL Image (overlay)
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        
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
            
        # Apply preprocessing pipeline matching val_test_transform
        img_tensor = preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, 380, 380]
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