import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
from shutil import copy2

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser(description='Cluster similar images using unsupervised learning.')
parser.add_argument('image_folder', help='Path to the folder containing images')
parser.add_argument('--output_folder', default='clustered_images', help='Output folder to save clusters')
parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')
args = parser.parse_args()

IMAGE_FOLDER = args.image_folder
OUTPUT_FOLDER = args.output_folder
NUM_CLUSTERS = args.clusters
IMAGE_SIZE = (224, 224)

# === LOAD PRETRAINED MODEL ===
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# === LOAD & EMBED IMAGES ===
features = []
image_paths = []

for filename in os.listdir(IMAGE_FOLDER):
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.isfile(filepath):
        try:
            img = cv2.imread(filepath)
            img = cv2.resize(img, IMAGE_SIZE)
            img = preprocess_input(img.astype(np.float32))
            img = np.expand_dims(img, axis=0)
            feature = base_model.predict(img, verbose=0)
            features.append(feature[0])
            image_paths.append(filepath)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

features = np.array(features)

# === CLUSTER IMAGES ===
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features)

# === SAVE TO CLUSTER FOLDERS ===
for cluster_id in range(NUM_CLUSTERS):
    cluster_path = os.path.join(OUTPUT_FOLDER, f'cluster_{cluster_id}')
    os.makedirs(cluster_path, exist_ok=True)

for img_path, label in zip(image_paths, labels):
    output_path = os.path.join(OUTPUT_FOLDER, f'cluster_{label}')
    copy2(img_path, output_path)

print(f"✅ Images clustered into {NUM_CLUSTERS} folders under '{OUTPUT_FOLDER}'")
