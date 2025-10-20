import cv2
from ultralytics import YOLO
from streamlink import streams
import time
import requests
import io
import subprocess
import numpy as np

model = YOLO('yolov8n-face.pt')
twitch_url = 'https://www.twitch.tv/gogoborgor'
available = streams(twitch_url)
stream_url = list(available.values())[0].url

# Simple subprocess pipe
process = subprocess.Popen([
    'ffmpeg', '-i', stream_url, 
    '-map', '0:v:0',           # Video only
    '-f', 'rawvideo', 
    '-pix_fmt', 'bgr24', 
    '-r', '30',                # 30 FPS
    '-'
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

server_url = 'http://127.0.0.1:8000/upload'  # Replace with server URL

while True:
    in_bytes = process.stdout.read(1920*1080*3)
    if len(in_bytes) == 1920*1080*3:
        frame = np.frombuffer(in_bytes, np.uint8).reshape(1080, 1920, 3)
        
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                
                _, buffer = cv2.imencode('.jpg', face)
                io_buf = io.BytesIO(buffer)
                requests.post(server_url, files={'file': ('face.jpg', io_buf, 'image/jpeg')})
    
    time.sleep(3)

process.terminate()