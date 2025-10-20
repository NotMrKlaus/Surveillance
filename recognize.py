from deepface import DeepFace
import numpy as np

print("lol")
emb = DeepFace.represent("received_face1.jpg", model_name="Facenet", enforce_detection=False)
print(emb)

'''
emb1 = DeepFace.represent("received_face1.jpg", model_name="Facenet")['embedding']
emb2 = DeepFace.represent("received_face2.jpg", model_name="Facenet")['embedding']

# Compare (cosine similarity > 0.4 = same person)
result = DeepFace.verify("received_face.jpg", "received_face1.jpg", model_name="Facenet", enforce_detection=False)
print(result['verified'])  # True/False
print(result['distance'])  # Lower = more similar
'''