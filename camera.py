import cv2
import requests
from streamlink import Streamlink
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import psycopg2
import numpy as np
import time
import os
import threading
from sklearn.metrics.pairwise import cosine_similarity
import json

# Postgres connection
conn = psycopg2.connect("dbname=face_db user=guri password=1004")
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def get_stream_url(channel):
    session = Streamlink()
    streams = session.streams(f'https://www.twitch.tv/{channel}')
    return streams['best'].url if streams else None



def save_embedding(embedding, conf, channel, filename):
    cur = conn.cursor()
    emb_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
    cur.execute(
        "INSERT INTO faces (channel, embedding, confidence, image_path) VALUES (%s, %s::vector, %s, %s) RETURNING id",
        (channel, emb_str, conf, filename)
    )
    id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return id

def camera_thread(channel, server_url):
    model = YOLO('yolov8n-face.pt')
    
    if SAVE_LOCAL_TEST:
        os.makedirs(f'{channel}_low_confidence', exist_ok=True)
        os.makedirs(f'{channel}_high_confidence', exist_ok=True)
    
    while True:
        url = get_stream_url(channel)
        cap = cv2.VideoCapture(url)
        if url:
            
            #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            
            
            if ret:
                results = model(frame, conf=0.5)
                faces = results[0].boxes
                if faces:
                    for i, face in enumerate(faces):
                        conf = float(face.conf)
                        box = tuple(map(int, face.xyxy[0]))
                        x1, y1, x2, y2 = box
                        w, h = (int (2*(x2-x1))), (int (2*(y2-y1)))
                        x1 = max(0, int(x1 - w*0.5))
                        y1 = max(0, int(y1 - h*0.5))
                        x2 = min(frame.shape[1], int(x2 + w*0.5))
                        y2 = min(frame.shape[0], int(y2 + h*0.5))
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Only process high-quality faces (conf >= 0.7) for DB
                        if conf >= 0.7:
                            faces_insight = app.get(face_crop)
                            if not faces_insight:
                                print(f"{channel} Face {i}: No embedding")
                                continue
                            embedding = faces_insight[0].normed_embedding
                            
                            save_embedding(embedding, conf, channel, f"{channel}_face_{int(time.time())}_{i}.jpg")
                            print(f"{channel} New person {person_id} saved")
                        else:
                            person_id = None  # Low-quality face, no DB entry
                        
                        # Send to server (all faces)
                        #face_crop = frame[box[1]:box[3], box[0]:box[2]]
                        face_crop = frame[y1:y2, x1:x2]
                        files = {'image': (f'{channel}_face_{i}.jpg', cv2.imencode('.jpg', face_crop)[1].tobytes(), 'image/jpeg')}
                        try:
                            requests.post(server_url, files=files, timeout=5)
                            print(f"{channel} Sent face {i} (Person {person_id if person_id else 'None'}): {conf:.2f}")
                        except requests.exceptions.ConnectionError:
                            print(f"{channel} Failed to send face {i}: Server not running")
                        
                        # Local save (all faces)
                        if SAVE_LOCAL_TEST:
                            if 0.5 <= conf < 0.7:
                                folder = f'{channel}_low_confidence'
                            elif 0.7 <= conf <= 1.0:
                                folder = f'{channel}_high_confidence'
                            else:
                                folder = None
                            if folder:
                                save_path = f"{folder}/{channel}_face_{int(time.time())}_{i}.jpg"
                                cv2.imwrite(save_path, face_crop)
                                print(f"{channel} Saved to {folder}: {conf:.2f}")
                    
                    time.sleep(5)
                else:
                    print(f"{channel}: No faces found")
            else:
                print(f"{channel}: Failed to capture frame")
        
        time.sleep(1)
    cap.release()

server_url = 'http://127.0.0.1:8000/upload'
SAVE_LOCAL_TEST = True
channels = ['cinna']

threads = []
for channel in channels:
    t = threading.Thread(target=camera_thread, args=(channel, server_url))
    t.start()
    threads.append(t)

for t in threads:
    t.join()