# SignaText

SignaText adalah web app yang menerjemahkan bahasa isyarat ke teks secara real-time menggunakan Model AI. Proyek ini menggabungkan Machine Learning, Backend, dan Frontend dalam satu ekosistem yang terintegrasi.

## 🛠 Tech Stack

- **Machine Learning**: Python, TensorFlow
- **Database**: SQL
- **Backend**: Hapi.js, Flask (untuk model AI)
- **Frontend**: HTML, CSS, JavaScript

## 🎯 Flow Integrasi

1. **User membuka web app** → Frontend menampilkan UI
2. **User melakukan isyarat di depan kamera** → Frontend menangkap video frame
3. **Frame dikirim ke backend** → Flask memproses dengan model AI
4. **Hasil dikembalikan ke frontend** → Ditampilkan sebagai teks

### **Branch dalam Repository**

1. **Main Branch (`main`)**
   → **Hanya berisi kode yang sudah stabil dan teruji.**
2. **Develop Branch (`develop`)**
   → **Tempat semua fitur dari backend, frontend, dan ML diuji sebelum ke main.**
3. **Branch Backend (`backend`)**
   → **Branch khusus tim backend untuk pengembangan API.**
4. **Branch Frontend (`frontend`)**
   → **Branch khusus tim frontend untuk UI/UX.**
5. **Branch Machine Learning (`ml-model`)**
   → **Branch khusus model AI untuk bahasa isyarat.**

### **Workflow Git untuk Tim**

1. **Clone repo hanya untuk branch masing-masing**
2. **Kerjakan kode di branch sendiri**
   ```bash
   git add .
   git commit -m "Menambahkan fitur X"
   git push origin backend  # atau frontend/ml-model sesuai tim
   ```
3. **Merge ke `develop` Setelah untuk testing Model + backend + frontend**
   ```bash
   git checkout develop
   git merge backend
   git merge frontend
   git merge ml-model
   git push origin develop
   ```
4. **Merge `develop` ke `main` setelah fitur stabil**
   ```bash
   git checkout main
   git merge develop
   git push origin main
   ```

## 📌 Catatan Tim

- **Setiap tim hanya bekerja di branch masing-masing** agar tidak tabrakan.
- **Gunakan `develop` hanya untuk integrasi fitur** setelah diuji.
- # **Jangan langsung push ke `main`** kecuali sudah disetujui tim.

1. **User membuka web app** → Frontend menampilkan UI
2. **User melakukan isyarat di depan kamera** → Frontend menangkap video frame
3. **Frame dikirim ke backend** → Flask memproses dengan model AI
4. **Hasil dikembalikan ke frontend** → Ditampilkan sebagai teks

## tim

main
