import pathlib
import sys
import cv2
import torch

# Patch biar PosixPath dari model Colab bisa kebaca di Windows
pathlib.PosixPath = pathlib.WindowsPath

sys.path.append(r'D:\Noding\aiojwaioje\yolov5')
from models.common import DetectMultiBackend

# Load model hasil training
model = DetectMultiBackend(r'D:\Noding\aiojwaioje\bisindo_best.pt', device='cpu')

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Inference
    results = model(tensor)
    
    # Ambil nama kelas
    names = model.names

    # Proses hasil deteksi
    detections = results[0]  # Format output: [x1, y1, x2, y2, confidence, class]
    
    if detections.shape[0] > 0:
        for detection in detections:
            *xyxy, conf, cls = detection
            
            # Konversi tensor ke tipe data Python
            cls = int(cls.item())  
            label = names[cls]
            
            # Konversi koordinat
            x1, y1, x2, y2 = map(int, map(torch.Tensor.item, xyxy))
            
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tampilkan label
            cv2.putText(
                frame,
                f'{label} {conf.item():.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow('Deteksi Bisindo YOLOv5', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
