from flask import Flask, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite('latest_frame.jpg', img)
        print("Saved: latest_frame.jpg")
        return 'OK', 200
    return 'No image', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)