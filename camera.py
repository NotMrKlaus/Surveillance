import cv2
import requests
from streamlink import Streamlink
from ultralytics import YOLO
import time
import os

def get_stream_url(channel):
    session = Streamlink()
    streams = session.streams(f'https://www.twitch.tv/{channel}')
    return streams['best'].url if streams else None

channel = 'jasontheween'
server_url = 'http://127.0.0.2:8000/upload'
model = YOLO('yolov8n-face.pt')
SAVE_LOCAL_TEST = True

if SAVE_LOCAL_TEST:
    os.makedirs('low_confidence', exist_ok=True)
    os.makedirs('high_confidence', exist_ok=True)

while True:
    url = get_stream_url(channel)
    if url:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            results = model(frame, conf=0.5)
            faces = results[0].boxes
            if faces:
                for i, face in enumerate(faces):
                    conf = float(face.conf)
                    x1, y1, x2, y2 = map(int, face.xyxy[0])
                    
                    # Expand 20%
                    w, h = x2-x1, y2-y1
                    x1 = max(0, int(x1 - w*0.2))
                    y1 = max(0, int(y1 - h*0.2))
                    x2 = min(frame.shape[1], int(x2 + w*0.2))
                    y2 = min(frame.shape[0], int(y2 + h*0.2))
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Send to server
                    files = {'image': (f'face_{i}.jpg', cv2.imencode('.jpg', face_crop)[1].tobytes(), 'image/jpeg')}
                    requests.post(server_url, files=files)
                    
                    # Local save
                    if SAVE_LOCAL_TEST:
                        if 0.5 <= conf < 0.7:
                            folder = 'low_confidence'
                        elif 0.7 <= conf <= 1.0:
                            folder = 'high_confidence'
                        else:
                            folder = None
                        
                        if folder:
                            filename = f"{folder}/face_{int(time.time())}_{i}.jpg"
                            cv2.imwrite(filename, face_crop)
                            print(f"Saved to {folder}: {conf:.2f}")
                    
                    print(f"Sent face {i} to server: {conf:.2f}")
                
                time.sleep(5)
            else:
                print("No faces found")
        else:
            print("Failed to capture frame")
    
    time.sleep(1)