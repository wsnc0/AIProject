import os
from sklearn.model_selection import train_test_split
from PIL import Image
# Function to create the necessary directories
def create_dirs(base_dir, classes):
    os.makedirs(base_dir, exist_ok=True)
    for subset in ['train', 'valid', 'test']:
        subset_path = os.path.join(base_dir, subset)
        os.makedirs(subset_path, exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(subset_path, class_name), exist_ok=True)

# Function to split and save images into train and test directories
def split_and_process_images(raw_data_dir:str = 'data', processed_data_dir:str = 'processedData', randomState=42) -> None:
    """
    * Processes Image Datasets that are already split into class folders.
    * Does a Training / Validation / Teseting split on the dataset with a 8:1:1 ratio

    Params:
    * raw_data_dir: path to the raw dataset
    * processed_data_dir: path to where the split dataset is to be saved
    * randomState: for reproducibility
    """
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
            train_files, testval_files = train_test_split(image_filenames, test_size=0.2, random_state=randomState)
            print(f" Train: {len(train_files)}, Test&Val: {len(testval_files)}")
            # Process and save training images
            for filename in train_files:
                image_path = os.path.join(class_folder, filename)
                img = Image.open(image_path)
                # Save the training image
                train_save_path = os.path.join(processed_data_dir, 'train', class_name, filename)
                img.save(train_save_path)

            # Split into validation and test sets
            val_files, test_files = train_test_split(testval_files, test_size=0.5, random_state=randomState)

            # Process and save validation images
            for filename in val_files:
                image_path = os.path.join(class_folder, filename)
                img = Image.open(image_path)
                # Save the validation image
                val_save_path = os.path.join(processed_data_dir, 'valid', class_name, filename)
                img.save(val_save_path)

            # Process and save test images
            for filename in test_files:
                image_path = os.path.join(class_folder, filename)
                img = Image.open(image_path)
                # Save the testing image
                test_save_path = os.path.join(processed_data_dir, 'test', class_name, filename)
                img.save(test_save_path)

        else:
            print(f"Error on {class_name}")


if __name__ == "__main__":
    split_and_process_images()

