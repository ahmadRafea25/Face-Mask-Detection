import os
from PIL import Image
import numpy as np

# Define the root directory for the dummy dataset
DATASET_ROOT = "dummy_dataset"

# Define the required directory structure
structure = {
    "incorrect_mask": ["mc", "mmc"],
    "with_mask": ["simple", "complex"],
    "without_mask": ["simple", "complex"],
}

def create_dummy_image(path, size=(10, 10)):
    """Creates a simple black image and saves it to the given path."""
    try:
        # Create a black image (10x10 pixels)
        img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(path)
        print(f"Created dummy image at: {path}")
    except Exception as e:
        print(f"Error creating image at {path}: {e}")

def setup_dummy_dataset():
    """Creates the directory structure and populates it with dummy images."""
    print(f"Setting up dummy dataset in: {DATASET_ROOT}")
    
    # Create the root directory
    os.makedirs(DATASET_ROOT, exist_ok=True)

    for class_name, subfolders in structure.items():
        for subfolder in subfolders:
            # Construct the full path for the deepest subdirectory
            dir_path = os.path.join(DATASET_ROOT, class_name, subfolder)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create a few dummy images in each deepest folder
            for i in range(3):
                image_path = os.path.join(dir_path, f"dummy_img_{i}.jpg")
                create_dummy_image(image_path)

    print("Dummy dataset setup complete.")

if __name__ == "__main__":
    setup_dummy_dataset()
