# 📊 Product Segmentation & Sales Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## 🧠 Deskripsi Proyek

Proyek ini merupakan **Final Project Data Science** yang bertujuan melakukan **segmentasi produk dan analisis penjualan** menggunakan beberapa metode *unsupervised learning* (clustering):

- **K-Means**
- **Agglomerative (Hierarchical) Clustering**
- **DBSCAN**

Analisis ini membantu memahami pola performa produk, preferensi pelanggan, dan profitabilitas untuk mendukung keputusan bisnis berbasis data.  
Aplikasi interaktif dibangun menggunakan **Streamlit**.

---

## 🚀 Akses Aplikasi Streamlit

Klik logo di bawah untuk membuka versi interaktif:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## ⚙️ Fitur Utama

- 📈 **Segmentasi Produk Otomatis** berdasarkan data pembelian dan performa
- 🔍 **Perbandingan 3 Algoritma Clustering** (K-Means, Agglomerative, DBSCAN)
- 📊 **Visualisasi PCA** untuk melihat distribusi tiap cluster
- 💬 **Interpretasi dan Insight Bisnis** untuk tiap segmen
- 🧮 **Dashboard Interaktif** yang menampilkan tren penjualan, rating, dan profitabilitas

---

## 🧩 Perbandingan Model Clustering

![Perbandingan Model](https://github.com/Roysihombing/Final-Project-Data-Science/blob/main/images/Perbandingan-Segment.png)

> Berdasarkan evaluasi **Silhouette Score**, model **Agglomerative Clustering** memberikan hasil paling stabil dan interpretatif dibandingkan K-Means dan DBSCAN.

---

## 🔍 Visualisasi PCA – Agglomerative Clustering

![Visualisasi PCA Agglomerative](https://github.com/Roysihombing/Final-Project-Data-Science/blob/main/images/Visual-Agglo.png)

Visualisasi PCA menampilkan distribusi tiap produk berdasarkan dua komponen utama hasil reduksi dimensi.  
Setiap warna menunjukkan kelompok hasil **Hierarchical Agglomerative Clustering**.

---

## 📋 Ringkasan Tiap Cluster – Agglomerative Clustering

| Cluster | Dominant_Category | Dominant_Brand | Avg_Total_Purchases | Avg_Total_Amount | Avg_Ratings | Avg_Unique_Customers | Avg_Profitability_Index |
|----------|------------------|----------------|----------------------|------------------|--------------|----------------------|--------------------------|
| 0 | Books | Adidas | 101,845 | 25,995,338 | 3.12 | 16,969 | 4,331,222 |
| 1 | Electronics | BlueStar | 12,679 | 3,210,780 | 3.64 | 2,273 | 575,483 |
| 2 | Electronics | Pepsi | 191 | 47,035 | 3.01 | 35 | 8,487 |

---

## 💬 Interpretasi Segmentasi Produk

| Cluster | Kategori & Brand Dominan | Karakteristik Produk | Implikasi Bisnis | Nama Segmen |
|:--------|:-------------------------|:--------------------|:-----------------|:-------------|
| **Cluster 0** | Books – Adidas | Volume penjualan dan profit tertinggi. Rating stabil menunjukkan popularitas tinggi. | Pertahankan stok dan promosi agar stabilitas tetap terjaga. | 🔵 **High-Performer Segment** |
| **Cluster 1** | Electronics – BlueStar | Rating tinggi dengan volume menengah. Menunjukkan loyalitas pelanggan. | Fokus pada kualitas dan ulasan positif. | 🟣 **Trusted Mid-Tier Electronics** |
| **Cluster 2** | Electronics – Pepsi | Volume dan profit rendah, pelanggan terbatas. | Perlu promosi atau inovasi harga untuk meningkatkan daya saing. | 🟠 **Low-Tier Electronics** |

---

## 📓 Notebook Analisis

Notebook lengkap proses eksplorasi data, pemodelan, evaluasi, dan interpretasi tersedia di:  
📘 [`Fiks_Final_Project_DS.ipynb`](./Final_Project_DS-Roy.ipynb)

---

## 🧮 Tools & Library yang Digunakan

- **Python 3.10+**
- **Streamlit**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Plotly Express**
- **Matplotlib / Seaborn**
- **Joblib**, **Requests**

---

## 📈 Tahapan Analisis Data

1. **Persiapan Dataset** – Import, pembersihan, dan penggabungan data produk & penjualan  
2. **Feature Engineering** – Membentuk metrik seperti total pembelian, profit, rating rata-rata  
3. **Standarisasi & PCA** – Reduksi dimensi agar clustering lebih optimal  
4. **Clustering Modeling** – K-Means, Agglomerative, dan DBSCAN  
5. **Evaluasi & Visualisasi** – Menggunakan *Silhouette Score* dan PCA plot  
6. **Interpretasi Bisnis** – Menentukan insight dan strategi per cluster  
7. **Deployment Streamlit** – Dashboard analitik interaktif  

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
