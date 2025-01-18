import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.decomposition import PCA
from tqdm import tqdm

# Step 1: Load Images and Extract Features
def load_images(folder_path):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            images.append(preprocess_input(img_array))
    return np.array(images), image_paths

def extract_features(images):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model.predict(images)

# Step 2: Reduce Dimensionality
def reduce_dimensionality(features, n_components=None):
    # Determine the maximum possible n_components
    max_components = min(features.shape[0], features.shape[1])
    if n_components is None or n_components > max_components:
        n_components = max_components - 1  # Leave some margin
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


# Step 3: Cluster Features
def cluster_features(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Step 4: Save Grouped Images
def save_grouped_images(image_paths, labels, output_folder):
    for i, label in enumerate(labels):
        group_folder = os.path.join(output_folder, f'group_{label}')
        os.makedirs(group_folder, exist_ok=True)
        output_path = os.path.join(group_folder, os.path.basename(image_paths[i]))
        os.rename(image_paths[i], output_path)

# Main Program
if __name__ == "__main__":
    input_folder = "D:\projects\Grouping Images\Dataset"
    output_folder = "D:\projects\Grouping Images\groups"
    n_clusters = 5  # Set the number of groups/clusters
    
    # Load images
    print("Loading images...")
    images, image_paths = load_images(input_folder)
    
    # Extract features
    print("Extracting features...")
    features = extract_features(images)
    
    # Dimensionality reduction
    print("Reducing dimensionality...")
    reduced_features = reduce_dimensionality(features)
    
    # Clustering
    print("Clustering images...")
    labels = cluster_features(reduced_features, n_clusters=n_clusters)
    
    # Save grouped images
    print("Saving grouped images...")
    save_grouped_images(image_paths, labels, output_folder)
    
    print("Image grouping completed!")
