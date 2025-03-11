# SignaText

SignaText adalah web app yang menerjemahkan bahasa isyarat ke teks secara real-time menggunakan Model AI. Proyek ini menggabungkan Machine Learning, Backend, dan Frontend dalam satu ekosistem yang terintegrasi.

## 🛠 Tech Stack

- **Machine Learning**: Python, TensorFlow
- **Database**: SQL
- **Backend**: Hapi.js, Flask (untuk model AI)
- **Frontend**: HTML, CSS, JavaScript

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

## 🎯 Flow Integrasi

1. **User membuka web app** → Frontend menampilkan UI
2. **User melakukan isyarat di depan kamera** → Frontend menangkap video frame
3. **Frame dikirim ke backend** → Flask memproses dengan model AI
4. **Hasil dikembalikan ke frontend** → Ditampilkan sebagai teks

## tim
