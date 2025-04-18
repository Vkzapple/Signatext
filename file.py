import os
import torch
import cv2
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.common import DetectMultiBackend
import psycopg2
from io import BytesIO
from PIL import Image
import numpy as np
import argparse
import json
import requests

# Inisialisasi Flask app\app = Flask(__name__)
CORS(app)

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dljflfis5"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "688273164556944"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "qiak8-mntievgXgTH6XUPg5b0S0")
)

# Konfigurasi database PostgreSQL
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "signatext")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")

conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cursor = conn.cursor()

# Path ke model YOLOv5
model = DetectMultiBackend("bisindo_best.pt", device='cpu')
names = model.names

def detect_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    results = model(tensor)[0]
    output = []
    for detection in results:
        *xyxy, conf, cls = detection
        x1, y1, x2, y2 = map(int, map(torch.Tensor.item, xyxy))
        cls = int(cls.item())
        label = names[cls]
        output.append({
            'label': label,
            'confidence': float(conf),
            'bbox': [x1, y1, x2, y2]
        })
    return output

def predict_image(path):
    frame = cv2.imread(path)
    detections = detect_frame(frame)
    return {"file": path, "detections": detections}

def predict_video(path, frame_skip=10):
    cap = cv2.VideoCapture(path)
    results = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip == 0:
            detections = detect_frame(frame)
            results.append({ 'frame': frame_id, 'detections': detections })
    cap.release()
    return {"file": path, "video_detections": results}

def download_from_url(url):
    response = requests.get(url)
    content_type = response.headers.get('Content-Type', '')
    if 'image' in content_type:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, frame_bgr)
        return predict_image(temp_path)
    elif 'video' in content_type:
        temp_path = "temp_video.mp4"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return predict_video(temp_path)
    else:
        return {"error": "Unsupported file type from URL."}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    # Simpan bytes untuk re-upload ke Cloudinary
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ambil user context\    user_id = request.form.get('user_id', type=int)
    session_id = request.form.get('session_id')

    # Deteksi
    detections = detect_frame(frame)

    # Simpan hasil deteksi ke PostgreSQL
    saved_ids = []
    if user_id:
        for det in detections:
            letter = det['label']
            accuracy = det['confidence'] * 100
            cursor.execute(
                """
                INSERT INTO letters (user_id, session_id, letter, accuracy)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (user_id, session_id, letter, accuracy)
            )
            det_id = cursor.fetchone()[0]
            saved_ids.append(det_id)
        conn.commit()

    # Upload file ke Cloudinary
    upload_stream = BytesIO(file_bytes)
    upload_result = cloudinary.uploader.upload(upload_stream)
    image_url = upload_result.get('secure_url', '')

    return jsonify({
        'detections': detections,
        'saved_ids': saved_ids,
        'image_url': image_url
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 BISINDO Detection Service")
    parser.add_argument('--image', type=str, help='Path to local image file')
    parser.add_argument('--video', type=str, help='Path to local video file')
    parser.add_argument('--url', type=str, help='URL of image or video file')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    args = parser.parse_args()

    if args.cli:
        if args.image:
            print(json.dumps(predict_image(args.image), indent=2))
        elif args.video:
            print(json.dumps(predict_video(args.video), indent=2))
        elif args.url:
            print(json.dumps(download_from_url(args.url), indent=2))
        else:
            print(json.dumps({"error": "Provide --image, --video, or --url"}, indent=2))
    else:
        app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
