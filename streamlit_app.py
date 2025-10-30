# ==========================================================
# 📊 STREAMLIT DASHBOARD – PRODUCT SEGMENTATION & SALES ANALYSIS
# by Roy Sihombing | Final Project – Data Science
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from io import BytesIO
import requests
from kneed import KneeLocator
import random

# -------------------------------------------------
# ⚙️ PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Product Segmentation Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Product Segmentation & Sales Analytics Dashboard")
st.caption("Analisis segmentasi produk dan pergerakan penjualan berdasarkan berbagai algoritma clustering.")
st.divider()

# -------------------------------------------------
# 📂 SIDEBAR MENU
# -------------------------------------------------
st.sidebar.header("⚙️ Pilihan Analisis")
mode = st.sidebar.radio(
    "Pilih Mode Analisis:",
    ("📊 Segmentasi Produk", "📈 Pergerakan Penjualan")
)

# ==========================================================
# 📊 SEGMENTASI PRODUK
# ==========================================================
if mode == "📊 Segmentasi Produk":

    # -------------------------------------------------
    # 🔄 LOAD DATASET PRODUK
    # -------------------------------------------------
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/dataset/product_df.xlsx"
        df = pd.read_excel(url, engine="openpyxl")
        return df

    df = load_data()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    st.subheader("📦 Dataset Produk")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Jumlah data: **{df.shape[0]} baris**, **{df.shape[1]} kolom**")

    numeric_features = [
        'Total_Purchases', 'Total_Amount', 'Ratings',
        'Avg_Amount_per_Purchase', 'Profitability_Index'
    ]

    # -------------------------------------------------
    # 🔧 DATA PREPARATION (dikunci supaya hasil identik)
    # -------------------------------------------------
    np.random.seed(42)
    random.seed(42)

    # Pastikan float64
    scaled_data = np.array(df[numeric_features], dtype=np.float64)
    scaler = StandardScaler(copy=True)
    scaled_data = scaler.fit_transform(scaled_data)

    # Tahap 1: PCA n_components=0.95
    pca1 = PCA(n_components=0.95, random_state=42, copy=True)
    stage1 = pca1.fit_transform(scaled_data)

    # Tahap 2: PCA n_components=3
    pca2 = PCA(n_components=3, random_state=42, copy=True)
    scaled_pca = pca2.fit_transform(stage1)

    # -------------------------------------------------
    # 🔍 PILIH ALGORITMA
    # -------------------------------------------------
    algo_choice = st.sidebar.multiselect(
        "Pilih Algoritma Clustering:",
        ["K-Means", "Agglomerative", "DBSCAN"],
        default=[]
    )

    if not algo_choice:
        algo_choice = ["K-Means", "Agglomerative", "DBSCAN"]

    results = []
    summaries = {}

    # -------------------------------------------------
    # 🧩 K-MEANS
    # -------------------------------------------------
    if "K-Means" in algo_choice:
        sse = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300, tol=1e-4)
            km.fit(scaled_pca)
            sse.append(km.inertia_)

        kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
        optimal_k = 4  # hasil tetap dari notebook kamu

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300, tol=1e-4)
        labels = kmeans.fit_predict(scaled_pca)
        df['KMeans_Cluster'] = labels
        sil = silhouette_score(scaled_pca, labels)

        results.append({"Model": "K-Means", "Silhouette Score": round(sil, 3), "Jumlah Cluster": optimal_k})

        kmeans_summary = (
            df.groupby('KMeans_Cluster')
            .agg({
                'Product_Category': lambda x: x.mode()[0],
                'Product_Brand': lambda x: x.mode()[0],
                'Total_Purchases': 'mean',
                'Ratings': 'mean',
                'Total_Amount': 'mean',
                'Profitability_Index': 'mean'
            })
            .reset_index()
            .rename(columns={
                'KMeans_Cluster': 'Cluster',
                'Product_Category': 'Dominant_Category',
                'Product_Brand': 'Dominant_Brand'
            })
        )
        summaries['K-Means'] = kmeans_summary

    # -------------------------------------------------
    # 🧬 AGGLOMERATIVE
    # -------------------------------------------------
    if "Agglomerative" in algo_choice:
        best_n_agg = 3  # hasil dari Python
        agg_model = AgglomerativeClustering(n_clusters=best_n_agg)
        labels = agg_model.fit_predict(scaled_pca)
        df['Agg_Cluster'] = labels
        sil = silhouette_score(scaled_pca, labels)

        results.append({"Model": "Agglomerative", "Silhouette Score": round(sil, 3), "Jumlah Cluster": best_n_agg})

        agg_summary = (
            df.groupby('Agg_Cluster')
            .agg({
                'Product_Category': lambda x: x.mode()[0],
                'Product_Brand': lambda x: x.mode()[0],
                'Total_Purchases': 'mean',
                'Ratings': 'mean',
                'Total_Amount': 'mean',
                'Profitability_Index': 'mean'
            })
            .reset_index()
            .rename(columns={
                'Agg_Cluster': 'Cluster',
                'Product_Category': 'Dominant_Category',
                'Product_Brand': 'Dominant_Brand'
            })
        )
        summaries['Agglomerative'] = agg_summary

    # -------------------------------------------------
    # 🌐 DBSCAN
    # -------------------------------------------------
    if "DBSCAN" in algo_choice:
        db = DBSCAN(eps=1.2, min_samples=5)
        labels = db.fit_predict(scaled_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(scaled_pca, labels) if n_clusters > 1 else 0
        df['DBSCAN_Cluster'] = labels

        results.append({"Model": "DBSCAN", "Silhouette Score": round(sil, 3), "Jumlah Cluster": n_clusters})

        db_summary = (
            df[df['DBSCAN_Cluster'] != -1]
            .groupby('DBSCAN_Cluster')
            .agg({
                'Product_Category': lambda x: x.mode()[0],
                'Product_Brand': lambda x: x.mode()[0],
                'Total_Purchases': 'mean',
                'Ratings': 'mean',
                'Total_Amount': 'mean',
                'Profitability_Index': 'mean'
            })
            .reset_index()
            .rename(columns={
                'DBSCAN_Cluster': 'Cluster',
                'Product_Category': 'Dominant_Category',
                'Product_Brand': 'Dominant_Brand'
            })
        )
        summaries['DBSCAN'] = db_summary

    # -------------------------------------------------
    # 📊 PERBANDINGAN MODEL
    # -------------------------------------------------
    st.markdown("### 📈 Perbandingan Model Clustering – Product Segmentation")
    compare_df = pd.DataFrame(results)
    st.dataframe(compare_df.style.background_gradient(cmap='YlGnBu', subset=['Silhouette Score']))

    # -------------------------------------------------
    # 🧭 CENTROID DAN INTERPRETASI
    # -------------------------------------------------
    for model, summary in summaries.items():
        st.markdown(f"### 🧩 Ringkasan Tiap Cluster – {model}")
        st.dataframe(
            summary.style.background_gradient(
                cmap='YlGnBu',
                subset=['Total_Purchases', 'Total_Amount', 'Profitability_Index']
            )
        )

        st.markdown(f"### 🧭 Interpretasi Segmentasi ({model})")
        interpret_html = f"""
        <table style="width:100%; border-collapse:collapse; color:#f5f5f5; font-family:Arial;">
          <thead style="background-color:#1E293B;">
            <tr>
              <th style="padding:10px;">Cluster</th>
              <th style="padding:10px;">Kategori & Brand Dominan</th>
              <th style="padding:10px;">Karakteristik Produk</th>
              <th style="padding:10px;">Strategi & Rekomendasi</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background-color:#2D3748;">
              <td>0</td><td>Books – Adidas</td>
              <td>Produk dengan penjualan tinggi, profit besar, pelanggan loyal.</td>
              <td>Pertahankan stok & program loyalitas pelanggan.</td>
            </tr>
            <tr style="background-color:#374151;">
              <td>1</td><td>Electronics – BlueStar</td>
              <td>Produk dengan rating tinggi, profit sedang, potensial untuk promosi.</td>
              <td>Optimalkan digital marketing & kolaborasi marketplace.</td>
            </tr>
            <tr style="background-color:#2D3748;">
              <td>2</td><td>Electronics – Pepsi</td>
              <td>Produk dengan profit rendah & permintaan kecil.</td>
              <td>Evaluasi harga, kurangi stok, atau reposisi produk.</td>
            </tr>
          </tbody>
        </table>
        """
        st.markdown(interpret_html, unsafe_allow_html=True)

# ==========================================================
# 📈 PERGERAKAN PENJUALAN PRODUK
# ==========================================================
else:
    st.subheader("📈 Analisis Pergerakan Penjualan Produk")

    @st.cache_data
    def load_sales_data():
        file_id = "1hFrTkoZfwcH8ltdK254ZGdyx_FbIHneZ"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = requests.get(url)
        if resp.status_code != 200 or "text/html" in resp.headers.get("Content-Type", ""):
            st.error("⚠️ Gagal memuat data penjualan. Pastikan file Drive publik & valid (.xlsx)")
            return None
        return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

    sales_df = load_sales_data()
    if sales_df is None:
        st.stop()

    st.dataframe(sales_df.head(10), use_container_width=True)
    product_list = sales_df['Product_Name'].unique()
    selected_product = st.selectbox("Pilih Produk:", product_list)

    product_data = sales_df[sales_df['Product_Name'] == selected_product]
    fig = px.line(product_data, x='Date', y='Sales', title=f"📈 Tren Penjualan – {selected_product}",
                  markers=True, line_shape='spline')
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 🧾 FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 Roy Sihombing | Final Project – Data Science</p>",
    unsafe_allow_html=True
)
