import torch
import cv2
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from models.common import DetectMultiBackend
import os
from io import BytesIO
from PIL import Image
import numpy as np
import argparse
import json
import requests

##############################################
# KONFIGURASI CLOUDINARY & MODEL YOLOv5
##############################################

# Inisialisasi Flask app
app = Flask(__name__)

# Setup Cloudinary: ganti 'YOUR_CLOUD_NAME', 'YOUR_API_KEY', 'YOUR_API_SECRET' 
cloudinary.config(
    cloud_name='dljflfis5', 
    api_key='688273164556944',
    api_secret='qiak8-mntievgXgTH6XUPg5b0S0'
)

# Path Model
model = DetectMultiBackend("bisindo_best.pt", device='cpu')
names = model.names

##############################################
# FUNGSI UTAMA UNTUK DETEKSI
##############################################

def detect_frame(frame):
    """
    Fungsi untuk mendeteksi objek pada satu frame menggunakan model YOLOv5.
    Input: frame (array NumPy)
    Output: list deteksi berupa dictionary {label, confidence, bbox}
    """
    # Konversi BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Buat tensor untuk model
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    results = model(tensor)[0]
    output = []
    for detection in results:
        # Ambil koordinat bounding box, confidence, dan kelas
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

##############################################
# MODE PREDIKSI CLI (Command Line Interface)
##############################################

def predict_image(path):
    """
    Prediksi dari file gambar lokal.
    """
    frame = cv2.imread(path)
    detections = detect_frame(frame)
    return {"file": path, "detections": detections}

def predict_video(path, frame_skip=10):
    """
    Prediksi dari file video lokal.
    Mengambil 1 frame setiap 'frame_skip' frame.
    """
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
    """
    Prediksi untuk file yang diambil dari URL. 
    Secara otomatis mendeteksi apakah file berupa image atau video.
    """
    response = requests.get(url)
    content_type = response.headers.get('Content-Type', '')
    
    # Jika URL mengarah ke file gambar
    if 'image' in content_type:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Simpan ke file sementara
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, frame_bgr)
        return predict_image(temp_path)
    
    # Jika URL mengarah ke file video
    elif 'video' in content_type:
        temp_path = "temp_video.mp4"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return predict_video(temp_path)
    else:
        return {"error": "Unsupported file type from URL."}

##############################################
# ENDPOINT API FLASK
##############################################

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint untuk menerima file gambar/video melalui request POST,
    melakukan deteksi menggunakan model YOLOv5, meng-upload file ke Cloudinary,
    dan mengembalikan hasil prediksi serta URL file di Cloudinary.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file:
        # Membaca file (dalam bytes) dan mengonversi menjadi image
        img = file.read()
        nparr = np.frombuffer(img, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Lakukan deteksi
        detections = detect_frame(frame)
        
        # Upload file ke Cloudinary
        upload_result = cloudinary.uploader.upload(file)
        image_url = upload_result.get('secure_url', '')
        
        return jsonify({
            'detections': detections,
            'image_url': image_url
        })
    else:
        return jsonify({'error': 'File not received'}), 400

##############################################
# MAIN: PILIH MODE API / CLI
##############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 BISINDO Detection")
    parser.add_argument('--image', type=str, help='Path to local image file')
    parser.add_argument('--video', type=str, help='Path to local video file')
    parser.add_argument('--url', type=str, help='URL of image or video file')
    parser.add_argument('--cli', action='store_true', help='Run in command line mode (non-API)')
    args = parser.parse_args()

    # Jika dijalankan sebagai CLI (misal untuk testing manual)
    if args.cli:
        if args.image:
            result = predict_image(args.image)
        elif args.video:
            result = predict_video(args.video)
        elif args.url:
            result = download_from_url(args.url)
        else:
            result = {"error": "Please provide --image or --video or --url"}
        print(json.dumps(result, indent=2))
    else:
        # Jalankan API Flask di port 5000
        app.run(host='0.0.0.0', port=5000)
