from deepface import DeepFace
import numpy as np
import os
from collections import defaultdict
from sklearn.cluster import DBSCAN
import tensorflow as tf

# Silence TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Force GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU enabled")
else:
    print("CPU fallback")

def cluster_faces(folder_path, eps=0.6, min_samples=2):
    embeddings = []
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            try:
                embedding = DeepFace.represent(
                    img_path,
                    model_name='ArcFace'
                )[0]['embedding']
                embeddings.append(embedding)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipped {filename}: {str(e)}")
    
    if not embeddings:
        print("No valid faces found in folder")
        return 0, {}
    
    embeddings = np.array(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_
    
    unique_labels = set(labels) - {-1}
    num_people = len(unique_labels)
    
    people_images = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            people_images[label].append(image_paths[idx])
    
    return num_people, people_images

# Usage
folder = 'high_confidence'
num_people, people_images = cluster_faces(folder)
print(f"Number of unique people: {num_people}")
for label, images in people_images.items():
    print(f"Person {label}: {len(images)} images")
    for img_path in images:
        print(f"  - {img_path}")