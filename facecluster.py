from insightface.app import FaceAnalysis
import os
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
import cv2

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def cluster_faces(folder_path, eps=0.55, min_samples=1):
    embeddings = []
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                embedding = faces[0].normed_embedding
                embeddings.append(embedding)
                image_paths.append(img_path)
    
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