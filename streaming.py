import cv2
import requests
from streamlink import Streamlink
import time

server_url = 'http://127.0.0.1:8000/upload'  # Replace with server URL

def get_stream_url(channel):
    session = Streamlink()
    streams = session.streams(f'https://www.twitch.tv/{channel}')
    return streams['best'].url if streams else None

channel = 'n3on'

while True:
    url = get_stream_url(channel)
    if url:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            files = {'image': ('frame.jpg', cv2.imencode('.jpg', frame)[1].tobytes(), 'image/jpeg')}
            requests.post(server_url, files=files)
    
    time.sleep(1)