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
    return [{"frame": 1, "detections": detect_frame(frame)}]

def predict_video(path, frame_skip=10):
    cap = cv2.VideoCapture(path)
    results = []
    frame_id = 0
    out_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip == 0:
            detections = detect_frame(frame)
            results.append({
                'frame': frame_id,
                'detections': detections
            })
            out_id += 1

    cap.release()
    return results

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
        return [{"error": "Unsupported file type from URL."}]

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
        result = [{"error": "Please provide --image or --video or --url"}]

    print(json.dumps(result, indent=2))
