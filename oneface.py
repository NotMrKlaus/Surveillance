from insightface.app import FaceAnalysis
import cv2
import numpy as np

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def check_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return False, 0.0
    faces = app.get(img)
    if faces:
        confidence = faces[0].det_score
        print(f"Face detected in {image_path} with confidence: {confidence:.2f}")
        return True, confidence
    print(f"No face detected in {image_path}")
    return False, 0.0

# Usage
image_path = 'high_confidence/face_1761074501.jpg'
detected, confidence = check_face(image_path)

