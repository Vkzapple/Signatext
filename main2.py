import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import mediapipe as mp

# Inisialisasi kamera dan model
cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=2)  # Deteksi dua tangan
classifier = Classifier("D:/Noding/HandDetectionn/keras_model.h5", "D:/Noding/HandDetectionn/labels.txt")

# Parameter gambar
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Deteksi tangan

    if len(hands) == 2:  # Jika dua tangan terdeteksi
        # Ambil koordinat bounding box kedua tangan
        x1, y1, w1, h1 = hands[0]['bbox']
        x2, y2, w2, h2 = hands[1]['bbox']

        # Buat bounding box besar yang mencakup kedua tangan
        x_min = max(0, min(x1, x2) - offset)
        y_min = max(0, min(y1, y2) - offset)
        x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + offset)
        y_max = min(img.shape[0], max(y1 + h1, y2 + h2) + offset)

        # Pastikan koordinat valid sebelum cropping
        if x_max > x_min and y_max > y_min:
            imgCrop = img[y_min:y_max, x_min:x_max]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            h, w, _ = imgCrop.shape
            if w > 0 and h > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                # Pre-processing untuk meningkatkan akurasi
                imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
                _, imgThresh = cv2.threshold(imgBlur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                imgWhite = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)

                # Augmentasi untuk meningkatkan variasi data
                flip = np.random.choice([None, 1, 0])
                if flip is not None:
                    imgWhite = cv2.flip(imgWhite, flip)

                rows, cols, _ = imgWhite.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.uniform(-10, 10), 1)
                imgWhite = cv2.warpAffine(imgWhite, M, (cols, rows))

                # Prediksi gerakan tangan
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(f"Prediction: {prediction}, Index: {index}, Label: {labels[index]}")

                # Gambar bounding box besar
                cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)
                cv2.putText(imgOutput, labels[index], (x_min, y_min - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                # Tampilkan gambar crop dan hasil klasifikasi
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
