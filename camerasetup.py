
import sys
import os
from dotenv import load_dotenv
from camera import FaceRecognitionCamera
from insightface.app import FaceAnalysis
import threading

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python camerasetup.py channel1 channel2 channel3 ...")
    sys.exit(1)

channels = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]

if not channels:
    print("No channels provided.")
    sys.exit(1)

print(f"Starting face recognition for: {', '.join(channels)}")

db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Shared InsightFace app
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Create and start cameras
cameras = [FaceRecognitionCamera(ch, db_config, app, save_local=True) for ch in channels]
threads = [threading.Thread(target=cam.run, daemon=True) for cam in cameras]

for t in threads:
    t.start()

print(f"Monitoring {len(channels)} channels. Press Ctrl+C to stop.")

try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("\nShutting down...")
    sys.exit(0)