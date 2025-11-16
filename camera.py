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
import json



app = FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(320, 320))  # or (320, 320)



class FaceRecognitionCamera:
    def __init__(self, channel, server_url, db_config, save_local=False):
        self.channel = channel
        self.server_url = server_url
        self.save_local = save_local
        self.conn = psycopg2.connect(**db_config)
        #self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        #self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.app = app
        self.model = YOLO('yolov8n-face.pt')

        if self.save_local:
            os.makedirs(f'{channel}_low_confidence', exist_ok=True)
            os.makedirs(f'{channel}_high_confidence', exist_ok=True)

    def get_stream_url(self):
        session = Streamlink()
        streams = session.streams(f'https://www.twitch.tv/{self.channel}')
        return streams['best'].url if streams else None
    
    
    def save_embedding(self, embedding, confidence, filename, threshold=0.7):
        cur = self.conn.cursor()
        emb_str = '[' + ','.join(map(str, embedding.tolist())) + ']'

        # Search for similar person (using canonical embeddings only)
        cur.execute("""
            SET ivfflat.probes = 10;
            SELECT person_id, embedding <-> %s::vector AS dist
            FROM people
            WHERE person_id IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT 1
        """, (emb_str, emb_str))
        row = cur.fetchone()
        if row and row[1] < threshold:
            person_id = row[0] # Match found
            cur.execute("""
            UPDATE people 
            SET count = count + 1 
            WHERE person_id = %s
            RETURNING confidence
        """, (person_id,))

        else:
            # Get next person_id
            person_id = None
            cur.execute("""
                INSERT INTO people (confidence, embedding, image_path, count)
                VALUES(%s, %s::vector, %s, 1)
                RETURNING person_id

            """, (confidence, emb_str, filename))
            person_id = cur.fetchone()[0]
        # Insert new face (non-canonical)
        cur.execute("""
            INSERT INTO faces (channel, embedding, confidence, image_path, person_id)
            VALUES (%s, %s::vector, %s, %s, %s)
            RETURNING id
        """, (self.channel, emb_str, confidence, filename, person_id))
        new_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return new_id

    
    """def save_embedding(self, embedding, conf, filename):
        cur = self.conn.cursor()
        emb_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        cur.execute(
            "INSERT INTO faces (channel, embedding, confidence, image_path) VALUES (%s, %s::vector, %s, %s) RETURNING id",
            (self.channel, emb_str, conf, filename)
        )
        face_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return face_id
"""
    def process_face(self, face, frame, framenumber):
        conf = float(face.conf)
        box = tuple(map(int, face.xyxy[0]))
        x1, y1, x2, y2 = box
        w, h = (x2 - x1) * 2, (y2 - y1) * 2
        x1 = max(0, int(x1 - w * 0.5))
        y1 = max(0, int(y1 - h * 0.5))
        x2 = min(frame.shape[1], int(x2 + w * 0.5))
        y2 = min(frame.shape[0], int(y2 + h * 0.5))
        face_crop = frame[y1:y2, x1:x2]

        person_id = None
        if conf >= 0.7:
            faces_insight = self.app.get(face_crop)
            if faces_insight:
                embedding = faces_insight[0].normed_embedding
                filename = f"{self.channel}_face_({framenumber}){int(time.time())}.jpg"
                new_capture_id = self.save_embedding(embedding, conf, filename)
                print(f"{self.channel} New person capture {new_capture_id} saved")

        # Send to server
        _, img_encoded = cv2.imencode('.jpg', face_crop)
        files = {'image': (f"{self.channel}_face_({framenumber}){int(time.time())}.jpg", img_encoded.tobytes(), 'image/jpeg')}
        try:
            requests.post(self.server_url, files=files, timeout=5)
            print(f"{self.channel} Sent face: {conf:.2f}")
        except requests.exceptions.ConnectionError:
            print(f"{self.channel} Failed to send face")

        # Save locally
        if self.save_local:
            folder = f'{self.channel}_low_confidence' if 0.5 <= conf < 0.7 else f'{self.channel}_high_confidence' if conf >= 0.7 else None
            if folder:
                save_path = f"{folder}/{self.channel}_face_({framenumber}){int(time.time())}.jpg"
                cv2.imwrite(save_path, face_crop)
                print(f"{self.channel} Saved to {folder}: {conf:.2f}")

    def run(self):
        url = self.get_stream_url()
        cap = cv2.VideoCapture(url)
        while True:
            if not url:
                print(f"{self.channel}: Stream offline. Retrying in 15 minutes...")
                cap = cv2.VideoCapture(url)
                time.sleep(900)  # 15 minutes = 900 seconds
                continue
            cap = cv2.VideoCapture(url)

            ret, frame = cap.read()
            if not ret:
                print(f"{self.channel}: Failed to capture frame")
                time.sleep(1)
                continue

            results = self.model(frame, conf=0.5)
            faces = results[0].boxes if results[0].boxes is not None else []

            if faces:
                for i, face in enumerate(faces):
                    self.process_face(face, frame, i)
            else:
                print(f"{self.channel}: No faces found")

            time.sleep(20)
        cap.release()


# === USAGE ===
if __name__ == "__main__":
    """db_config = {
        "dbname": "face_db",
        "user": "guri",
        "password": "1004"
    }"""

    db_config = {
    "host": "db.aulbtbeaabfwlnwycvsz.supabase.co",
    "port": 5432,        # ← Standard PostgreSQL port
    "dbname": "postgres",
    "user": "postgres",
    "password": "everythingislowercasee"
    }


    

    server_url = 'http://127.0.0.1:8000/upload'
    channels = ['karii', 'ninadaddyisblack', 'nahyunworld', 'jinnytty', 'hello_kiko', 'fanfan', 'joeykaotyk', 'maimaittv', 'sunnys', 'michaaam']
    cameras = [FaceRecognitionCamera(channel, server_url, db_config, save_local=True) for channel in channels]
    threads = [threading.Thread(target=cam.run) for cam in cameras]

    for t in threads:
        t.start()
    for t in threads:
        t.join()