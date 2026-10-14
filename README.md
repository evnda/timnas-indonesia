# ⚽ Proyek Analisis Data X (Twitter) - Timnas Indonesia

Aplikasi Dashboard Interaktif berbasis **Streamlit** untuk menganalisis data perbincangan publik mengenai **Timnas Indonesia** di platform X. Proyek ini mengintegrasikan pemrosesan teks, analisis sentimen, pengelompokan topik (K-Means), dan estimasi engagement (Regresi Linear).

---

## 🎯 Pertanyaan Analisis Proyek
Proyek ini dibangun untuk menjawab 4 pertanyaan utama sesuai petunjuk praktikum:
1. Bagaimana distribusi sentimen pada postingan yang dikumpulkan?
2. Kelompok pembahasan (cluster) apa saja yang muncul?
3. Apa ciri utama dari setiap cluster?
4. Seberapa baik jumlah kata (`word_count`) dan hashtag (`hashtag_count`) dapat digunakan untuk memperkirakan jumlah likes (`like_count`)?

---

## 👥 Pembagian Tugas Kelompok
* **Anggun:** Pengambilan Data Crawling X & Cleaning (`dataset_timnas.csv`)
* **Fian:** Analisis Sentimen & Evaluasi Label Manual (`sentimen.py`, `evaluasi_sentimen.csv`)
* **Kartika:** Pengelompokan Pembahasan dengan K-Means (`clustering.py`, `hasil_cluster.csv`)
* **Ladraz:** Pemodelan Regresi Linear & Pembangunan Dashboard Streamlit Utama (`regresi.py`, `regresi.ipynb`, `app.py`)

---

## 🛠️ Cara Jalankan Aplikasi Secara Lokal

### 1. Install Dependensi
Buka Terminal di VS Code dan jalankan:
```bash
pip install -r requirements.txt

Buka Terminal di VS Code, lalu jalankan aplikasi web:
```bash
streamlit run app.py

Ketika sudah berjalan, bisa mencoba menambahkan atau melakukan dataset lain dengan menambahkan file dataset ke fitur menu yang tersedia