import streamlit as st
import pandas as pd
import plotly.express as px

# Judul aplikasi
st.set_page_config(page_title="Product Segmentation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Product Segmentation Dashboard")
st.caption("Analisis produk berdasarkan hasil Agglomerative Clustering")

# --- Contoh data (bisa diganti dengan hasil cluster kamu) ---
data = {
    'Cluster': [0, 1, 2],
    'Dominant_Category': ['Books', 'Electronics', 'Electronics'],
    'Dominant_Brand': ['Adidas', 'BlueStar', 'Pepsi'],
    'Avg_Total_Purchases': [101845, 12679, 191],
    'Avg_Total_Amount': [25995338, 3210780, 47035],
    'Avg_Ratings': [3.12, 3.64, 3.01],
    'Avg_Unique_Customers': [16969, 2273, 35],
    'Avg_Profitability_Index': [4331222, 575483, 8487]
}

df = pd.DataFrame(data)

# --- Tampilkan tabel hasil cluster ---
st.subheader("📋 Ringkasan Tiap Cluster")
st.dataframe(df.style.background_gradient(cmap="Blues"))

# --- Visualisasi bar chart ---
st.subheader("📈 Perbandingan Rata-Rata Fitur per Cluster")
fig = px.bar(
    df.melt(id_vars='Cluster', value_vars=['Avg_Total_Purchases', 'Avg_Total_Amount', 'Avg_Profitability_Index'],
            var_name='Fitur', value_name='Nilai'),
    x='Cluster', y='Nilai', color='Fitur', barmode='group',
    title="Perbandingan Rata-Rata Fitur per Cluster"
)
st.plotly_chart(fig, use_container_width=True)

# --- Interpretasi Segmen ---
st.subheader("🧩 Interpretasi Segmen Produk")
st.markdown("""
| Cluster | Segmen | Karakteristik | Strategi |
|----------|---------|----------------|-----------|
| 0 | 🟢 Premium & Bestseller | Produk dengan volume & profit tinggi | Pertahankan stok & loyalitas |
| 1 | 🔵 Mid-Tier Electronics | Produk menengah dengan rating tinggi | Fokus promosi & ekspansi |
| 2 | 🔴 Low-Performance | Produk kurang laku & profit kecil | Evaluasi produk atau reposisi harga |
""")
