import pathlib
import sys
import argparse
import cv2
import torch
import numpy as np
import json
import requests
from io import BytesIO
from PIL import Image
import os

# Kompatibilitas path model
pathlib.PosixPath = pathlib.WindowsPath
sys.path.append(r'https://github.com/Vkzapple/Signatext/blob/2fbd820f9168492061ed51c2945de4cb7748babc/bisindo_best.pt')  # Ganti path ke repo YOLO kamu

from models.common import DetectMultiBackend

# Load model
model = DetectMultiBackend('bisindo_best.pt', device='cpu')
names = model.names

# Deteksi frame menjadi label
def detect_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    results = model(tensor)[0]
    output = []

    for detection in results:
        *xyxy, conf, cls = detection
        cls = int(cls.item())
        label = names[cls]
        output.append(label)
    return output

# Prediksi gambar
def predict_image(path):
    frame = cv2.imread(path)
    detections = detect_frame(frame)
    return detections

# Prediksi video
def predict_video(path, frame_skip=10):
    cap = cv2.VideoCapture(path)
    detected_labels = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip == 0:
            detections = detect_frame(frame)
            detected_labels.extend(detections)
    cap.release()
    return detected_labels

# Prediksi dari URL
def download_from_url(url):
    response = requests.get(url)
    content_type = response.headers['Content-Type']
    
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
        return ["Unsupported file type from URL."]

# Konversi label ke teks
def labels_to_text(labels):
    text = ''
    for label in labels:
        if label.upper() == "SPACE":
            text += ' '
        else:
            text += label.upper()
    return text

# Simpan hasil ke JSON
def save_to_json(labels, filename="hasil_prediksi.json"):
    data = {
        "labels": labels,
        "text": labels_to_text(labels)
    }
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Hasil berhasil disimpan di: {filename}")

# Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--url', type=str, help='URL to image or video file')
    args = parser.parse_args()

    if args.image:
        result = predict_image(args.image)
    elif args.video:
        result = predict_video(args.video)
    elif args.url:
        result = download_from_url(args.url)
    else:
        result = ["Please provide --image or --video or --url"]

    save_to_json(result)
