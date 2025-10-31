# 🧩 PRODUCT SEGMENTATION ANALYSIS USING CLUSTERING

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## 🧠 Deskripsi Proyek

Proyek ini merupakan **Final Project Data Science** yang berfokus pada **segmentasi produk menggunakan metode unsupervised learning (clustering)**.  
Tujuan utama analisis ini adalah mengelompokkan produk berdasarkan pola penjualan, profitabilitas, dan rating pelanggan untuk membantu pengambilan keputusan strategis dalam manajemen stok dan pemasaran.

Metode clustering yang digunakan:
- **K-Means Clustering**
- **Agglomerative (Hierarchical) Clustering**
- **DBSCAN**

Aplikasi interaktif dibangun menggunakan **Streamlit**, sedangkan proses analisis dilakukan dengan **Python (Jupyter Notebook)**.

---

## 🚀 Akses Aplikasi Streamlit

Klik logo di bawah untuk mencoba dashboard interaktif:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-roy.streamlit.app/)

---

## ⚙️ Fitur Utama

- 📈 Segmentasi produk otomatis berdasarkan data penjualan dan performa  
- 🔍 Perbandingan 3 algoritma clustering: *K-Means*, *Agglomerative*, dan *DBSCAN*  
- 📊 Visualisasi PCA untuk melihat distribusi tiap cluster  
- 💬 Interpretasi karakteristik tiap segmen produk  
- 🧮 Dashboard analitik interaktif dengan visual dinamis  

---

## 🧩 Perbandingan Model Clustering

![Perbandingan Model](https://github.com/Roysihombing/Final-Project-Data-Science/blob/main/images/Perbandingan-Segment.png)

> Berdasarkan evaluasi **Silhouette Score**, model **Agglomerative Clustering** memberikan hasil paling stabil dan interpretatif dibandingkan K-Means dan DBSCAN.

---

## 🔍 Visualisasi PCA – Agglomerative Clustering

![Visualisasi PCA Agglomerative](https://github.com/Roysihombing/Final-Project-Data-Science/blob/main/images/Visual-Agglo.png)

Visualisasi di atas menunjukkan distribusi produk hasil reduksi dimensi menggunakan PCA (2 komponen utama).  
Setiap warna merepresentasikan kelompok produk hasil **Hierarchical Agglomerative Clustering**.

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
| **Cluster 0** | Books – Adidas | Volume penjualan dan profit tertinggi. Rating stabil menunjukkan popularitas tinggi. | Pertahankan stok dan promosi untuk menjaga stabilitas penjualan. | 🔵 **High-Performer Segment** |
| **Cluster 1** | Electronics – BlueStar | Rating tertinggi dengan volume menengah. Pelanggan loyal dengan kepuasan tinggi. | Jaga kualitas dan gunakan ulasan positif untuk memperkuat branding. | 🟣 **Trusted Mid-Tier Electronics** |
| **Cluster 2** | Electronics – Pepsi | Penjualan dan profit rendah, pelanggan terbatas. | Perlu strategi promosi atau evaluasi harga agar lebih kompetitif. | 🟠 **Low-Tier Electronics** |

---

## 🧩 Kesimpulan

Hasil analisis menunjukkan bahwa:
- **Agglomerative Clustering** menghasilkan segmentasi paling representatif dengan 3 kelompok utama.
- Produk dengan performa tinggi (Cluster 0) layak diprioritaskan untuk distribusi dan promosi berkelanjutan.
- Produk dengan loyalitas pelanggan tinggi (Cluster 1) dapat dijadikan fokus branding jangka panjang.
- Produk berperforma rendah (Cluster 2) perlu strategi diskon, promo, atau perbaikan kualitas.

Insight ini membantu perusahaan menentukan **strategi stok, pricing, dan marketing** secara lebih efektif berdasarkan data objektif.

---

## 📓 Notebook Analisis

Notebook lengkap mencakup proses eksplorasi data, pemodelan, evaluasi, dan interpretasi hasil:  
📘 [`Fiks_Final_Project_DS.ipynb`](./Fiks_Final_Project_DS.ipynb)

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

1. **Persiapan Dataset** – Import dan pembersihan data  
2. **Feature Engineering** – Pembentukan metrik performa (penjualan, profit, rating)  
3. **Standarisasi & PCA** – Reduksi dimensi untuk meningkatkan hasil clustering  
4. **Clustering Modeling** – Menggunakan K-Means, Agglomerative, dan DBSCAN  
5. **Evaluasi Model** – Menggunakan *Silhouette Score*  
6. **Interpretasi & Visualisasi** – Analisis cluster dan insight bisnis  
7. **Deployment** – Dashboard analitik di Streamlit  

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
