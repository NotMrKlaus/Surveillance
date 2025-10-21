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

channel = 'ronnyberger'
server_url = 'http://127.0.0.2:8000/upload'  # Replace with server URL
model = YOLO('yolov8n-face.pt')

os.makedirs('low_confidence', exist_ok=True)
os.makedirs('high_confidence', exist_ok=True)


SAVE_LOCAL_TEST = True  # Set False to disable local saving

if SAVE_LOCAL_TEST:
    os.makedirs('low_confidence', exist_ok=True)
    os.makedirs('high_confidence', exist_ok=True)



while True:
    url = get_stream_url(channel)
    if url:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            results = model(frame, conf=0.5)
            if results[0].boxes:
                best_face = max(results[0].boxes, key=lambda x: x.conf)
                conf = float(best_face.conf)
                x1, y1, x2, y2 = map(int, best_face.xyxy[0])
                
                w, h = x2-x1, y2-y1
                x1 = max(0, int(x1 - w*0.2))
                y1 = max(0, int(y1 - h*0.2))
                x2 = min(frame.shape[1], int(x2 + w*0.2))
                y2 = min(frame.shape[0], int(y2 + h*0.2))
                
                face_crop = frame[y1:y2, x1:x2]
                
                # ALWAYS send to server
                files = {'image': ('best_face.jpg', cv2.imencode('.jpg', face_crop)[1].tobytes(), 'image/jpeg')}
                requests.post(server_url, files=files)
                
                # OPTIONAL local save for testing
                if SAVE_LOCAL_TEST:
                    if 0.5 <= conf < 0.7:
                        folder = 'low_confidence'
                    elif 0.7 <= conf <= 1.0:
                        folder = 'high_confidence'
                    else:
                        folder = None
                    
                    if folder:
                        filename = f"{folder}/face_{int(time.time())}.jpg"
                        cv2.imwrite(filename, face_crop)
                        print(f"Saved to {folder}: {conf:.2f}")
                
                print(f"Sent to server: {conf:.2f}")
                time.sleep(5)
            else:
                print("No good face found")
    
    time.sleep(1)