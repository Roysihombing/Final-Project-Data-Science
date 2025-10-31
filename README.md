# 🧩 PRODUCT SEGMENTATION ANALYSIS USING CLUSTERING

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## 🧠 Deskripsi Proyek

Proyek ini merupakan **Final Project Data Science** yang berfokus pada **segmentasi produk menggunakan metode unsupervised learning (clustering)**.  
Tujuannya adalah mengelompokkan produk berdasarkan **pola penjualan, profitabilitas, dan rating pelanggan** untuk membantu pengambilan keputusan strategis dalam **manajemen stok dan pemasaran**.

Metode clustering yang digunakan:
- **K-Means Clustering**
- **Agglomerative (Hierarchical) Clustering**
- **DBSCAN**

Aplikasi interaktif dibangun menggunakan **Streamlit**, sementara proses analisis dilakukan dengan **Python (Jupyter Notebook)**.

---

## 🚀 Akses Aplikasi Streamlit

Klik logo di bawah ini untuk mencoba dashboard interaktif:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## ⚙️ Fitur Utama

- 📈 Segmentasi produk otomatis berdasarkan data penjualan dan performa  
- 🔍 Perbandingan tiga algoritma clustering: *K-Means*, *Agglomerative*, dan *DBSCAN*  
- 📊 Visualisasi PCA untuk melihat distribusi tiap cluster  
- 💬 Interpretasi karakteristik tiap segmen produk  
- 🧮 Dashboard analitik interaktif berbasis **Streamlit**  

---

## 🧩 Perbandingan Model Clustering

| Model | Silhouette Score | Jumlah Cluster | Catatan |
|:------|:----------------:|:---------------:|:--------|
| **K-Means** | 0.600 | 5 | Cukup baik, namun beberapa produk tumpang tindih antar cluster |
| **Agglomerative** | **0.658** | **9** | Hasil paling stabil dan interpretatif |
| **DBSCAN** | 0.620 | 9 | Konsisten dengan Agglomerative dan mampu deteksi outlier |

> Berdasarkan evaluasi **Silhouette Score**, model **Agglomerative Clustering** memberikan hasil paling representatif dan stabil dibandingkan model lainnya.

---

## 🔍 Visualisasi PCA – Agglomerative Clustering

<p align="center">
  <img src="https://github.com/Roysihombing/Final-Project-Data-Science/blob/main/images/Visual-Agglo.png" alt="Visualisasi PCA Agglomerative" width="700px"/>
</p>

Visualisasi menunjukkan hasil reduksi dimensi PCA (2 komponen utama)  
di mana setiap warna merepresentasikan kelompok produk hasil **Agglomerative Clustering (9 cluster)**.

---

## 📋 Ringkasan Tiap Cluster – Agglomerative Clustering (Product Level)

| Cluster | Dominant_Category | Dominant_Product | Avg_Total_Purchases | Avg_Ratings | Avg_Total_Amount | Avg_Profitability_Index |
|:--------|:------------------|:-----------------|--------------------:|-------------:|-----------------:|------------------------:|
| 0 | Electronics | Action | 17.22 | 3.17 | 4,317.78 | 763.88 |
| 1 | Electronics | Acer Swift | 3,475.08 | 3.24 | 886,534.17 | 164,853.40 |
| 2 | Home Decor | Bathtub | 3,243.54 | 3.11 | 829,082.86 | 153,858.61 |
| 3 | Clothing | A-line dress | 3,263.34 | 3.10 | 830,536.94 | 154,102.32 |
| 4 | Books | Biography | 9,779.74 | 3.12 | 2,505,839.60 | 461,560.87 |
| 5 | Electronics | 4K TV | 6,489.45 | 3.12 | 1,652,005.42 | 306,216.28 |
| 6 | Clothing | Boots | 6,537.85 | 3.11 | 1,668,121.83 | 307,273.68 |
| 7 | Home Decor | Bed | 6,460.85 | 3.11 | 1,644,557.10 | 304,625.83 |
| 8 | Grocery | Adventure | 13,081.15 | 3.25 | 3,334,116.59 | 612,850.09 |

---

## 💬 Interpretasi Segmentasi Produk

| Cluster | Kategori & Produk Dominan | Karakteristik Produk | Implikasi Inventory | Nama Segmen |
|:--------|:--------------------------|:--------------------|:--------------------|:-------------|
| **0** | Electronics – Action | Penjualan & profit rendah, pelanggan sedikit, rating standar. | Butuh promosi aktif & reposisi produk. | 🟠 **Low-Performing Electronics** |
| **1** | Electronics – Acer Swift | Penjualan & profit tinggi, rating stabil, loyalitas pelanggan kuat. | Pertahankan stok dan promosi nilai tambah. | 🔵 **Mid-Tier Tech Segment** |
| **2** | Home Decor – Bathtub | Penjualan stabil, profit cukup tinggi, pelanggan loyal. | Fokus pada stok unggulan & variasi desain. | 🟢 **Stable Home Decor** |
| **3** | Clothing – A-line Dress | Penjualan stabil & profit menengah. Cocok untuk fashion kasual. | Pertahankan desain populer & promosi musiman. | 🔵 **Consistent Fashion Segment** |
| **4** | Books – Biography | Penjualan & profit sangat tinggi, permintaan luas & stabil. | Pertahankan stok populer & perluas distribusi. | 🟢 **High-Value Literature** |
| **5** | Electronics – 4K TV | Produk premium dengan profit besar & rating stabil. | Fokus pada inovasi & kampanye fitur teknologi. | 🟣 **Premium Visual Tech** |
| **6** | Clothing – Boots | Profit tinggi, rating stabil, cocok untuk tren musiman. | Terapkan limited edition & stok adaptif. | 🔵 **Trend-Driven Apparel** |
| **7** | Home Decor – Bed | Produk furnitur bernilai tinggi, pelanggan luas & loyal. | Jaga kualitas bahan & waktu pengiriman. | 🟢 **Premium Furniture Segment** |
| **8** | Grocery – Adventure | Penjualan, profit, & pelanggan tertinggi. | Pastikan pasokan & distribusi efisien. | 🟢 **Core Grocery Segment** |

---

## 💼 Rekomendasi Bisnis

- **Perbaiki kinerja segmen lemah** (*Low-Performing Electronics*) dengan promosi aktif dan peningkatan kualitas produk.  
- **Pertahankan segmen unggulan** (*Core Grocery*, *High-Value Literature*) melalui stok terjaga dan distribusi efisien.  
- **Lakukan promosi fleksibel** pada segmen menengah (*Mid-Tier Tech*, *Trend-Driven Apparel*) dengan inovasi dan kampanye mengikuti tren.  
- **Tingkatkan citra segmen premium** (*Premium Visual Tech*, *Premium Furniture*) dengan kualitas unggul dan pelayanan pelanggan terbaik.  

---

## 🧩 Kesimpulan

- Model terbaik: **Agglomerative Clustering (9 cluster, Silhouette Score: 0.658)**  
- Segmentasi produk berhasil mengungkap perbedaan performa antar kategori.  
- Produk dengan performa tinggi dan stabil (Books & Grocery) layak diprioritaskan.  
- Hasil analisis membantu perusahaan dalam menentukan **strategi stok, pricing, dan promosi** berbasis data.

---

## 📓 Notebook Analisis

📘 [`Final_Project_DS_Roy.ipynb`](./Final_Project_DS_Roy.ipynb)

---

## 🧮 Tools & Library yang Digunakan

- **Python 3.10+**
- **Streamlit**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Plotly Express**
- **Matplotlib**, **Seaborn**
- **Joblib**, **Requests**

---

## 📈 Tahapan Analisis Data

1. **Data Preparation** – Pembersihan dan standarisasi dataset  
2. **Feature Engineering** – Pembuatan metrik performa penjualan dan profit  
3. **PCA** – Reduksi dimensi data  
4. **Modeling** – K-Means, Agglomerative, dan DBSCAN  
5. **Evaluasi Model** – Berdasarkan *Silhouette Score*  
6. **Interpretasi & Insight Bisnis** – Analisis karakter tiap cluster  
7. **Deployment** – Dashboard interaktif menggunakan Streamlit  

---

## 🧰 Cara Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/Roysihombing/Final-Project-Data-Science.git
cd Final-Project-Data-Science

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
