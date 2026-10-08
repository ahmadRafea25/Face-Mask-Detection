import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from data_preprocessing import IMAGE_SIZE, DATASET_PATH, load_data_from_directory, SEED

# --- Configuration ---
NUM_CLASSES = 3
MODEL_INPUT_SHAPE = IMAGE_SIZE + (3,)

# --- Model Definition (Phase 2 Requirement) ---

def create_custom_cnn(input_shape, num_classes):
    """
    Creates a custom, simple CNN model for educational clarity.
    Architecture: Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Output
    """
    input_tensor = Input(shape=input_shape, name="input_layer")
    
    # Block 1: Feature Learning (Edges, Textures)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input_tensor)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Dropout(0.25, name='dropout1')(x)
    
    # Block 2: Feature Learning (Shapes, Patterns)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = Dropout(0.25, name='dropout2')(x)
    
    # Block 3: Feature Learning (Facial Regions, High-level features)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    x = Dropout(0.25, name='dropout3')(x)
    
    # Classification Head (Phase 3 preparation)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout4')(x)
    output_tensor = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor, name="Custom_FaceMask_CNN")
    
    print(model.summary())
    return model

# --- Visualization (Phase 2 Requirement) ---

def visualize_feature_maps(model, image_path, layer_names, save_dir="feature_maps"):
    """
    Visualizes the feature maps output by specified layers of the CNN model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load and preprocess the image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) # Add batch dimension
    img_tensor /= 255.0 # Normalize as done in data_preprocessing

    # 2. Create a model that outputs the feature maps of the selected layers
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_map_model = Model(inputs=model.inputs, outputs=outputs)
    
    # 3. Get the feature maps
    feature_maps = feature_map_model.predict(img_tensor)
    
    # 4. Plot the feature maps
    for layer_name, fmap in zip(layer_names, feature_maps):
        # fmap shape is (1, H, W, D) -> (H, W, D)
        fmap = fmap[0] 
        n_features = fmap.shape[-1]
        
        # We will only visualize a subset of the features (e.g., first 8)
        n_cols = 4
        n_rows = int(np.ceil(min(n_features, 8) / n_cols))
        
        plt.figure(figsize=(n_cols * 2, n_rows * 2))
        plt.suptitle(f"Feature Maps from Layer: {layer_name}", fontsize=12)
        
        for i in range(min(n_features, 8)):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            # Display the i-th channel
            plt.imshow(fmap[:, :, i], cmap='viridis')
            plt.title(f"Channel {i}", fontsize=8)
            ax.axis('off')
            
        save_path = os.path.join(save_dir, f"{layer_name}_feature_maps.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.savefig(save_path)
        plt.close()
        print(f"Saved feature maps for {layer_name} to {save_path}")

def main():
    # 1. Create the model
    model = create_custom_cnn(MODEL_INPUT_SHAPE, NUM_CLASSES)
    
    # 2. Get a sample image path for visualization
    image_paths, _, _ = load_data_from_directory(DATASET_PATH)
    if len(image_paths) == 0:
        print("Error: No images found for feature map visualization.")
        return
        
    # Use a random image from the dataset
    np.random.seed(SEED)
    sample_image_path = np.random.choice(image_paths)
    print(f"\nVisualizing feature maps for sample image: {sample_image_path}")

    # 3. Define layers for visualization
    # Early layer (conv1) for low-level features (edges, textures)
    # Deeper layer (conv3) for high-level features (facial regions)
    layers_to_visualize = ['conv1', 'conv3']
    
    # 4. Visualize feature maps
    visualize_feature_maps(model, sample_image_path, layers_to_visualize)
    
    print("\nPhase 2: CNN Feature Learning complete.")

if __name__ == "__main__":
    # Change working directory to project root for consistent pathing
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
