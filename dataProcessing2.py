import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps  # Or use OpenCV depending on your preferred library
import numpy as np


# def pad_to_square(img):
#     """
#     Pads the image to a square size, keeping the center intact.
#     Args:
#     - img: The input PIL Image.
#     Returns:
#     - Padded image (PIL Image).
#     """
#     width, height = img.size

#     if width == height:
#         return img
#     else:
#         size = max(width,height)

#         l_pad = (size-width)//2
#         r_pad = size - width - l_pad
#         t_pad = (size-height)//2
#         b_pad = size - height - t_pad

#         padding = (l_pad,t_pad,r_pad,b_pad)
#         return ImageOps.expand(img,padding)
    
# def resize_img(img,size:int =(224,224)):
#     """
#     Resizes the image to the specified size.
#     Params:
#     - img: The input PIL Image.
#     - size: The target size to resize the image to. Default is 224x224.

#     Returns:
#     - Resized image (PIL Image).
#     """
#     return img.resize(size, Image.ANTIALIAS)


    


# Function to create the necessary directories
def create_dirs(base_dir, classes):
    os.makedirs(base_dir, exist_ok=True)
    for subset in ['train', 'test']:
        subset_path = os.path.join(base_dir, subset)
        os.makedirs(subset_path, exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(subset_path, class_name), exist_ok=True)

# Function to split and save images into train and test directories
def split_and_process_images(raw_data_dir, processed_data_dir, test_size=0.2):
    # Get all class names (subfolder names)
    classes = os.listdir(raw_data_dir)
    print("Subfolders:",classes)
    create_dirs(processed_data_dir, classes)
    
    # Iterate through each class folder
    for class_name in classes:
        class_folder = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_folder):
            # Get all image filenames
            image_filenames = os.listdir(class_folder)
            print(f"{class_name} : {len(image_filenames)} files.")
            # Split into train and test sets
            train_files, test_files = train_test_split(image_filenames, test_size=test_size, random_state=42)
            
            # Process and save training images
            for filename in train_files:
                image_path = os.path.join(class_folder, filename)
                img = Image.open(image_path)

                # Save the training image
                train_save_path = os.path.join(processed_data_dir, 'train', class_name, filename)
                img.save(train_save_path)

            # Process and save testing images
            for filename in test_files:
                image_path = os.path.join(class_folder, filename)
                img = Image.open(image_path)

                # Save the testing image
                test_save_path = os.path.join(processed_data_dir, 'test', class_name, filename)
                img.save(test_save_path)

        else:
            print(f"Error on {class_name}")

# Define directories
raw_data_dir = 'data'
processed_data_dir = 'processedData'

# Split and process the images
split_and_process_images(raw_data_dir, processed_data_dir)
