from flask import Flask, request, send_from_directory
import cv2
import numpy as np

app = Flask(__name__)
counter = 1

@app.route('/')
def home():
    return 'Face Detection Server Running'

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        file.save('latest_frame.jpg')
        print("Saved: latest_frame.jpg")
        return 'OK', 200
    print("No image in files")
    return 'No image', 400

@app.route('/files')
def files():
    return send_from_directory('.', 'received_face.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)