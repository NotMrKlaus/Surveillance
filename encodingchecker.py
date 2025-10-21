from deepface import DeepFace
import os
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

def check_face_encoding(image_path):
    try:
        embedding = DeepFace.represent(
            image_path,
            model_name='ArcFace'
        )[0]['embedding']
        return True
    except Exception as e:
        return False

# Usage
folder = 'high_confidence'
results = []
for filename in os.listdir(folder):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(folder, filename)
        detected = check_face_encoding(img_path)
        results.append((filename, detected))

# Pretty print
print("\n=== Face Detection Results ===")
print(f"Total images: {len(results)}")
print("\nDetected Faces:")
for filename, detected in results:
    if detected:
        print(f"  ✓ {filename}")
print("\nNo Faces Detected:")
for filename, detected in results:
    if not detected:
        print(f"  ✗ {filename}")
print("==============================")