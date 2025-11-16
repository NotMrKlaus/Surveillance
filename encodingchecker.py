from insightface.app import FaceAnalysis
import os
import shutil
import cv2

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def debug_filter_face_images(folder_path):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            print(f"\n--- {filename} ---")
            print(f"Image loaded: {img is not None}")
            
            if img is None:
                print("❌ Image failed to load")
                results.append((filename, False))
                continue
                
            faces = app.get(img)
            print(f"Faces found: {len(faces)}")
            
            if faces:
                for i, face in enumerate(faces):
                    print(f"  Face {i}: det_score={face.det_score:.3f}")
                results.append((filename, True))
            else:
                print("❌ No faces detected")
                results.append((filename, False))
    
    print(f"\n=== SUMMARY ===")
    print(f"Success: {sum(1 for _, r in results if r)}")
    print(f"Failed: {sum(1 for _, r in results if not r)}")

folder = 'clemovitch_high_confidence'
debug_filter_face_images(folder)