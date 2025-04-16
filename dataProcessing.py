# import os
# from sklearn.model_selection import train_test_split
# from PIL import Image
# # Function to create the necessary directories
# def create_dirs(base_dir, classes):
#     os.makedirs(base_dir, exist_ok=True)
#     for subset in ['train', 'valid', 'test']:
#         subset_path = os.path.join(base_dir, subset)
#         os.makedirs(subset_path, exist_ok=True)
#         for class_name in classes:
#             os.makedirs(os.path.join(subset_path, class_name), exist_ok=True)

# # Function to split and save images into train and test directories
# def split_and_process_images(raw_data_dir:str = 'data', processed_data_dir:str = 'processedData', randomState=42) -> None:
#     """
#     * Processes Image Datasets that are already split into class folders.
#     * Does a Training / Validation / Teseting split on the dataset with a 8:1:1 ratio

#     Params:
#     * raw_data_dir: path to the raw dataset
#     * processed_data_dir: path to where the split dataset is to be saved
#     * randomState: for reproducibility
#     """
#     # Get all class names (subfolder names)
#     classes = os.listdir(raw_data_dir)
#     print("Subfolders:",classes)
#     create_dirs(processed_data_dir, classes)
    
#     # Iterate through each class folder
#     for class_name in classes:
#         class_folder = os.path.join(raw_data_dir, class_name)
#         if os.path.isdir(class_folder):
#             # Get all image filenames
#             image_filenames = os.listdir(class_folder)
#             print(f"{class_name} : {len(image_filenames)} files.")
#             # Split into train and test sets
#             train_files, testval_files = train_test_split(image_filenames, test_size=0.2, random_state=randomState)
#             print(f" Train: {len(train_files)}, Test&Val: {len(testval_files)}")
#             # Process and save training images
#             for filename in train_files:
#                 image_path = os.path.join(class_folder, filename)
#                 img = Image.open(image_path)
#                 # Save the training image
#                 train_save_path = os.path.join(processed_data_dir, 'train', class_name, filename)
#                 img.save(train_save_path)

#             # Split into validation and test sets
#             val_files, test_files = train_test_split(testval_files, test_size=0.5, random_state=randomState)

#             # Process and save validation images
#             for filename in val_files:
#                 image_path = os.path.join(class_folder, filename)
#                 img = Image.open(image_path)
#                 # Save the validation image
#                 val_save_path = os.path.join(processed_data_dir, 'valid', class_name, filename)
#                 img.save(val_save_path)

#             # Process and save test images
#             for filename in test_files:
#                 image_path = os.path.join(class_folder, filename)
#                 img = Image.open(image_path)
#                 # Save the testing image
#                 test_save_path = os.path.join(processed_data_dir, 'test', class_name, filename)
#                 img.save(test_save_path)

#         else:
#             print(f"Error on {class_name}")


# if __name__ == "__main__":
#     split_and_process_images()

"""
Corrected preprocessing script for skin disease dataset.
This script handles the correct directory structure shown in the image.
"""

import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define paths
data_dir = './data/IMG_CLASSES'  # This is the path to the directory containing all class folders
processed_data_dir = './processedData'
target_size = (224, 224)

# Set random seed
np.random.seed(42)

# Create directories
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(os.path.join(processed_data_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(processed_data_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(processed_data_dir, 'test'), exist_ok=True)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

print("Starting preprocessing...")

# List all class directories
class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print(f"Found {len(class_dirs)} class directories: {class_dirs}")

# Process each class
for class_dir in class_dirs:
    class_path = os.path.join(data_dir, class_dir)
    print(f"\nProcessing class: {class_dir}")
    
    # Create class directories in train, val, test folders
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_data_dir, split, class_dir), exist_ok=True)
    
    # Get all image files in the class directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(class_path) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images for {class_dir}")
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {class_path}")
        continue
    
    # Shuffle and split the data
    np.random.shuffle(image_files)
    
    train_end = int(len(image_files) * train_ratio)
    val_end = train_end + int(len(image_files) * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Split for {class_dir}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Process and copy images to respective directories
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for img_name in tqdm(files, desc=f"Processing {split} - {class_dir}", leave=False):
            src_path = os.path.join(class_path, img_name)
            dest_path = os.path.join(processed_data_dir, split, class_dir, img_name)
            
            try:
                with Image.open(src_path) as img:
                    img = img.convert('RGB')  # Convert to RGB to ensure 3 channels
                    img = img.resize(target_size)
                    img.save(dest_path)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

print("\nPreprocessing complete!")

# Verify data was processed correctly
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(processed_data_dir, split)
    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    print(f"\n{split.capitalize()} set:")
    for class_dir in sorted(class_dirs):
        class_path = os.path.join(split_dir, class_dir)
        num_images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
        print(f"  - {class_dir}: {num_images} images")