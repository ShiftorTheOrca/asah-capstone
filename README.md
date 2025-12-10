## Segmentasi Pelanggan Berdasarkan Perilaku Pembelian menggunakan Metode RFM dan Clustering

---

## Deskripsi Singkat Proyek

Proyek ini adalah hasil pengerjaan capstone untuk melakukan segmentasi pelanggan guna mendapatkan strategi marketing terbaik untuk setiap segmen. Segmentasi pelanggan dilakukan menggunakan metode **RFM (Recency, Frequency, Monetary)** dengan algoritma clustering **HDBSCAN**.

Dataset yang digunakan berasal dari [Online Retail Dataset milik UCI](https://archive.ics.uci.edu/dataset/352/online+retail), yang berisi data transaksi pelanggan online retail. Melalui analisis clustering, kami dapat mengidentifikasi pola perilaku pembelian pelanggan dan membagi mereka ke dalam beberapa segmen yang berbeda untuk strategi pemasaran yang lebih efektif.

---

## Petunjuk Setup Environment

### Prasyarat

- Python 3.8 atau lebih baru
- pip (Python package manager)
- Git

### Langkah-Langkah Setup

1. **Clone repository ke lokal Anda:**

   ```bash
   git clone https://github.com/ShiftorTheOrca/asah-capstone.git
   cd asah-capstone
   ```

2. **Buat virtual environment (opsional tetapi disarankan):**

   **Untuk Windows:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **Untuk macOS/Linux:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   Navigasi ke folder `src` dan install semua library yang diperlukan:

   ```bash
   pip install -r requirements.txt
   ```

   Library utama yang diinstall meliputi:

   - `streamlit`: Framework untuk membuat web aplikasi
   - `pandas`: Manipulasi dan analisis data
   - `scikit-learn`: Machine learning dan preprocessing
   - `matplotlib`, `seaborn`, `plotly`: Visualisasi data
   - `numpy`: Komputasi numerik
   - Dan library pendukung lainnya

4. **Download dataset (opsional):**
   Download dataset menggunakan Kaggle API, pastikan file `kaggle.json` sudah dikonfigurasi dengan credentials Kaggle Anda dan diletakkan di root project. Dataset untuk streamlit app sudah tersedia pada folder `assets/`.

---

## Model Machine Learning

Proyek ini menggunakan algoritma **HDBSCAN** untuk clustering yang lebih robust dibanding K-Means dalam menangani data dengan density yang bervariasi. Model clustering HDBSCAN yang digunakan telah dilatih pada dataset Online Retail dan hasilnya disimpan dalam file CSV processed:

- **Dataset yang sudah diproses:** `assets/online_retail_uci_clustering.csv` - Berisi data dengan hasil clustering dan nilai RFM scores

Model dapat dilatih ulang menggunakan script yang tersedia di dalam notebook:

- `customer_segmentation_capstone.ipynb` - Notebook lengkap untuk training dan evaluation model

---

## Cara Menjalankan Aplikasi

### Metode 1: Menjalankan Streamlit Web App

1. **Pastikan sudah berada di dalam virtual environment (jika menggunakan):**

   ```bash
   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

2. **Navigasi ke folder project root:**

   ```bash
   cd c:\Project\asah-capstone
   ```

3. **Jalankan aplikasi Streamlit:**

   ```bash
   streamlit run src/app.py
   ```

   Anda akan melihat output di terminal yang menunjukkan aplikasi telah dimulai:

   ```
   Local URL: http://localhost:8501
   ```

4. **Akses aplikasi di browser:**

   - Buka browser web Anda
   - Masukkan URL: `http://localhost:8501` atau `localhost:8501`
   - Aplikasi web akan menampilkan dashboard dengan visualisasi hasil clustering dan analisis segmentasi pelanggan

5. **Menghentikan aplikasi:**
   - Tekan `Ctrl + C` pada terminal untuk menghentikan server

### Metode 2: Menjalankan Notebook Jupyter

Untuk melakukan training ulang atau exploratory data analysis, Anda dapat menjalankan notebook.

**Install Jupyter terlebih dahulu (jika belum):**

```bash
pip install notebook
```

Kemudian jalankan notebook menggunakan:

```bash
jupyter notebook src/customer_segmentation_capstone.ipynb
```

Jika ingin melihat notebook sistem rekomendasi:

```bash
jupyter notebook src/Sistem_Rekomendasi.ipynb
```

---

## Fitur Aplikasi

Aplikasi Streamlit menyediakan:

- **Dashboard Interaktif**: Visualisasi data dan hasil clustering
- **Analisis RFM**: Menampilkan nilai Recency, Frequency, dan Monetary untuk setiap customer
- **Segmentasi Pelanggan**: Visualisasi segmen yang dihasilkan dari algoritma HDBSCAN
- **Statistik Deskriptif**: Ringkasan statistik data pelanggan
- **Word Cloud**: Visualisasi kata kunci dari data

#### Sistem Rekomendasi Produk

Aplikasi juga dilengkapi dengan **Sistem Rekomendasi Produk** yang menggunakan metode **Content-Based Filtering** untuk memberikan rekomendasi produk kepada pelanggan berdasarkan preferensi pembelian mereka.

**Fitur Rekomendasi:**

- Rekomendasi berdasarkan deskripsi produk
- Rekomendasi untuk pelanggan individual berdasarkan riwayat pembelian
- Scoring similarity untuk setiap rekomendasi

---

## Struktur Folder

```
asah-capstone/
├── assets/
│   ├── logo.png
│   ├── Online Retail.csv (not included)
│   └── online_retail_uci_clustering.csv
├── src/
│   ├── app.py (Aplikasi Streamlit)
│   ├── customer_segmentation_capstone.ipynb
│   ├── Sistem_Rekomendasi.ipynb
│   └── online-retail-uci-dataset.zip (not included)
├── requirements.txt
├── .gitignore
├── README.md
├── kaggle.json (Credentials Kaggle, not included)
└── venv/ (Virtual environment, not included)
```

---

## Troubleshooting

- **Port 8501 sudah digunakan**: Gunakan `streamlit run src/app.py --server.port=8502` untuk menggunakan port berbeda
- **Module tidak ditemukan**: Pastikan virtual environment sudah diaktifkan dan dependencies sudah diinstall dengan benar
- **kaggle.json tidak ditemukan**: Pastikan mengkonfigurasi file kaggle.json pada root project

---
