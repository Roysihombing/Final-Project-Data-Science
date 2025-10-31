import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import requests
import tempfile
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Product Segmentation Dashboard", page_icon="📈", layout="wide")
st.title("📊 Product Segmentation & Sales Analytics Dashboard")
st.caption("Segmentasi produk (KMeans / Agglomerative / DBSCAN) + analisa pergerakan penjualan")
st.divider()

PRODUCT_URL = "https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/dataset/product_clustered.csv"
SALES_URL = "https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/dataset/sales_sampled.csv"

MODEL_RAW = {
    "scaler":  "https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/scaler.pkl",
    "pca_stage1":"https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/pca_stage1.pkl",
    "pca_stage2":"https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/pca_stage2.pkl",
    "kmeans":  "https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/kmeans.pkl",
    "agg_best":"https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/agg_best.pkl",
    "dbscan_best":"https://raw.githubusercontent.com/Roysihombing/Final-Project-Data-Science/main/models/dbscan_best.pkl"
}

# Load CSVs from GitHub raw
@st.cache_data(show_spinner=False)
def load_csvs():
    df = pd.read_csv(PRODUCT_URL)
    sales_df = pd.read_csv(SALES_URL)
    return df, sales_df

@st.cache_resource(show_spinner=False)
def load_models_from_github(model_raw_map):
    models = {}
    for key, url in model_raw_map.items():
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.content:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(r.content)
                    tmp_path = tmp.name
                models[key] = joblib.load(tmp_path)
            else:
                models[key] = None
        except Exception:
            models[key] = None
    return models

try:
    df, sales_df = load_csvs()
except Exception as e:
    st.error(f"⚠️ Gagal memuat dataset dari GitHub: {e}")
    st.stop()

models = load_models_from_github(MODEL_RAW)
models_available = any(v is not None for v in models.values())

st.sidebar.header("⚙️ Pilihan Analisis")
mode = st.sidebar.radio("Pilih Mode Analisis:", ("📊 Segmentasi Produk", "📈 Pergerakan Penjualan"))

if mode == "📊 Segmentasi Produk":
    algo_choice = st.sidebar.multiselect(
        "Pilih Algoritma Clustering:",
        ["K-Means", "Agglomerative", "DBSCAN"],
        default=[]
    )
    if not algo_choice:
        algo_choice = ["K-Means", "Agglomerative", "DBSCAN"]
else:
    algo_choice = []

def detect_cluster_col(df, algo_name):
    patterns = {
        "K-Means": ["K-Means_Cluster", "KMeans_Cluster", "KMeans", "K-Means"],
        "Agglomerative": ["Agglomerative_Cluster", "Agg_Cluster", "AggCluster", "Agglomerative"],
        "DBSCAN": ["DBSCAN_Cluster", "DBSCANCluster", "DBSCAN"]
    }
    for p in patterns.get(algo_name, []):
        if p in df.columns:
            return p
    for col in df.columns:
        if algo_name.replace("-", "").lower() in col.replace("_", "").lower():
            return col
    return None

# Updated numeric features to match notebook (7 features)
numeric_features = [
    'Total_Purchases', 'Total_Amount', 'Ratings',
    'Unique_Customers', 'Total_Transactions',
    'Avg_Amount_per_Purchase', 'Profitability_Index'
]

for c in numeric_features:
    if c not in df.columns:
        st.error(f"Kolom numerik `{c}` tidak ditemukan di product_clustered.csv — periksa dataset.")
        st.stop()

def compute_scaled_pca_safe(df, numeric_features, models):
    X_num_df = df.copy()
    X_num = df[numeric_features]  # DataFrame of numeric base features

    scaler = models.get('scaler', None)
    pca1 = models.get('pca_stage1', None)
    pca2 = models.get('pca_stage2', None)

    if scaler is not None:
        feature_names = getattr(scaler, "feature_names_in_", None)
        n_features_in = getattr(scaler, "n_features_in_", None)

        if feature_names is not None:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                for m in missing:
                    X_num_df[m] = 0.0
            X_for_scaler = X_num_df[feature_names].astype(float).values
            try:
                Xs = scaler.transform(X_for_scaler)
            except Exception as e:
                st.warning(f"⚠️ Gagal transform dengan scaler (feature_names_in_ path): {e}. Akan fallback ke fit scaler lokal.")
                scaler = None
        elif n_features_in is not None:
            if n_features_in == X_num.shape[1]:
                try:
                    Xs = scaler.transform(X_num.values.astype(float))
                except Exception as e:
                    st.warning(f"⚠️ Gagal transform dengan scaler (n_features match): {e}. Akan fallback ke fit scaler lokal.")
                    scaler = None
            else:
                st.warning("⚠️ Scaler expects different number of features than `numeric_features`. "
                           "Attempting to create placeholder columns to match scaler's expected input.")
                try:
                    needed = int(n_features_in)
                    cols = list(X_num.columns)
                    idx = 0
                    while len(cols) < needed:
                        cand = f"__placeholder_{idx}"
                        if cand not in X_num_df.columns:
                            X_num_df[cand] = 0.0
                            cols.append(cand)
                        idx += 1
                    X_for_scaler = X_num_df[cols[:needed]].astype(float).values
                    Xs = scaler.transform(X_for_scaler)
                except Exception as e:
                    st.warning(f"⚠️ Gagal menyesuaikan input untuk scaler: {e}. Akan fallback ke fit scaler lokal.")
                    scaler = None
        else:
            scaler = None

    if scaler is None or pca1 is None or pca2 is None:
        st.warning("⚠️ Menggunakan fallback: scaler/PCA lokal dibuat ulang. Hasil clustering bisa berbeda dari Colab jika model asli tidak tersedia.")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        sc_local = StandardScaler()
        Xs = sc_local.fit_transform(X_num.values.astype(float))
        p1_local = PCA(n_components=0.95, random_state=42)
        s1 = p1_local.fit_transform(Xs)
        p2_local = PCA(n_components=3, random_state=42)
        s2 = p2_local.fit_transform(s1)
        return s2
    try:
        s1 = pca1.transform(Xs)
        s2 = pca2.transform(s1)
        return s2
    except Exception as e:
        st.warning(f"⚠️ Gagal transform dengan pca_stage1/2 dari model: {e}. Akan fallback ke PCA lokal.")
        from sklearn.decomposition import PCA
        p1_local = PCA(n_components=0.95, random_state=42)
        s1 = p1_local.fit_transform(Xs)
        p2_local = PCA(n_components=3, random_state=42)
        s2 = p2_local.fit_transform(s1)
        return s2

scaled_pca = compute_scaled_pca_safe(df, numeric_features, models)

if mode == "📊 Segmentasi Produk":
    st.subheader("📦 Dataset Produk (sample)")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Jumlah data: **{df.shape[0]} baris**, **{df.shape[1]} kolom**")
    st.divider()

    results = []    
    summaries = {} 
    visuals = {}    

    for algo in algo_choice:
        col = detect_cluster_col(df, algo)
        labels = None
        if col is not None and col in df.columns:
            labels = df[col].values
        else:
            if algo == "K-Means" and models.get('kmeans') is not None:
                try:
                    labels = models['kmeans'].predict(scaled_pca)
                    col = "K-Means_Cluster"
                    df[col] = labels
                except Exception:
                    labels = None
            elif algo == "Agglomerative" and models.get('agg_best') is not None:
                m = models['agg_best']
                try:
                    if hasattr(m, "labels_") and len(getattr(m, "labels_", [])) == len(df):
                        labels = m.labels_
                        col = "Agglomerative_Cluster"
                        df[col] = labels
                    else:
                        labels = m.fit_predict(scaled_pca)
                        col = "Agglomerative_Cluster"
                        df[col] = labels
                except Exception:
                    labels = None
            elif algo == "DBSCAN" and models.get('dbscan_best') is not None:
                m = models['dbscan_best']
                try:
                    if hasattr(m, "labels_") and len(getattr(m, "labels_", [])) == len(df):
                        labels = m.labels_
                        col = "DBSCAN_Cluster"
                        df[col] = labels
                    else:
                        labels = m.fit_predict(scaled_pca)
                        col = "DBSCAN_Cluster"
                        df[col] = labels
                except Exception:
                    labels = None

        if labels is None:
            st.warning(f"⚠️ Tidak menemukan label cluster untuk `{algo}` (csv tidak punya kolom & model tidak tersedia). Lewati {algo}.")
            continue

        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1]) if algo == "DBSCAN" else len(unique_labels)
        sil = np.nan
        try:
            if algo == "DBSCAN":
                if n_clusters > 1:
                    sil = silhouette_score(scaled_pca, labels)
            else:
                if len(unique_labels) > 1:
                    sil = silhouette_score(scaled_pca, labels)
        except Exception:
            sil = np.nan

        results.append({"Model": algo, "Silhouette Score": round(float(sil) if not np.isnan(sil) else np.nan, 3), "Jumlah Cluster": int(n_clusters)})

        if algo == "DBSCAN":
            df_valid = df[df[col] != -1].copy()
            if df_valid.empty:
                summary = pd.DataFrame(columns=["Cluster", "Dominant_Category", "Dominant_Brand", "Total_Purchases", "Ratings", "Total_Amount", "Profitability_Index"])
            else:
                summary = (
                    df_valid.groupby(col)
                    .agg({
                        'Product_Category': lambda x: x.mode().iloc[0] if not x.mode().empty else "",
                        'products': lambda x: x.mode().iloc[0] if not x.mode().empty else "",
                        'Total_Purchases': 'mean',
                        'Ratings': 'mean',
                        'Total_Amount': 'mean',
                        'Profitability_Index': 'mean'
                    })
                    .reset_index()
                    .rename(columns={col: "Cluster", "Product_Category": "Dominant_Category", "products": "Dominant_Brand"})
                )
        else:
            summary = (
                df.groupby(col)
                .agg({
                    'Product_Category': lambda x: x.mode().iloc[0] if not x.mode().empty else "",
                    'products': lambda x: x.mode().iloc[0] if not x.mode().empty else "",
                    'Total_Purchases': 'mean',
                    'Ratings': 'mean',
                    'Total_Amount': 'mean',
                    'Profitability_Index': 'mean'
                })
                .reset_index()
                .rename(columns={col: "Cluster", "Product_Category": "Dominant_Category", "products": "Dominant_Brand"})
            )

        summaries[algo] = summary

        vis = pd.DataFrame(scaled_pca[:, :2], columns=["PCA1", "PCA2"])
        vis["Cluster"] = labels.astype(str)
        visuals[algo] = px.scatter(vis, x="PCA1", y="PCA2", color="Cluster", title=f"PCA Visualization — {algo}", hover_data=[df.get('products')])

    if results:
        compare_df = pd.DataFrame(results)
        st.markdown("### 📈 Perbandingan Model Clustering")
        st.dataframe(compare_df.style.background_gradient(cmap='YlGnBu', subset=["Silhouette Score"]) if "Silhouette Score" in compare_df.columns else compare_df, use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada algoritma yang valid untuk ditampilkan.")

    st.divider()

    for algo in algo_choice:
        if algo in visuals:
            st.markdown(f"## 🔍 {algo}")
            st.plotly_chart(visuals[algo], use_container_width=True)
            st.markdown(f"### Ringkasan Cluster — {algo}")
            st.dataframe(summaries[algo].style.background_gradient(cmap="YlGnBu"), use_container_width=True)

            st.markdown("#### Interpretasi & Rekomendasi")
            summary = summaries[algo]
            if summary.empty:
                st.write("Tidak ada cluster (atau semuanya noise).")
            else:
                overall = {
                    "Total_Purchases": df["Total_Purchases"].mean(),
                    "Profitability_Index": df["Profitability_Index"].mean(),
                    "Ratings": df["Ratings"].mean(),
                    "Total_Amount": df["Total_Amount"].mean()
                }
                for _, row in summary.iterrows():
                    cl = row["Cluster"]
                    tp = row["Total_Purchases"]
                    pi = row["Profitability_Index"]
                    rt = row["Ratings"]

                    traits = []
                    recs = []
                    if tp >= overall["Total_Purchases"]:
                        traits.append("penjualan tinggi")
                        recs.append("pertahankan stok & program loyalitas")
                    else:
                        traits.append("penjualan rendah")
                        recs.append("promosi & bundling")

                    if pi >= overall["Profitability_Index"]:
                        traits.append("profitabilitas tinggi")
                        recs.append("pertahankan margin")
                    else:
                        traits.append("profitabilitas rendah")
                        recs.append("evaluasi harga / biaya")

                    if rt >= overall["Ratings"]:
                        traits.append("rating pelanggan tinggi")
                        recs.append("tingkatkan eksposur produk")
                    else:
                        traits.append("rating rendah")
                        recs.append("perbaiki kualitas / deskripsi produk")

                    st.markdown(f"- **Cluster {cl}** — karakter: {', '.join(traits)}. Rekomendasi: {', '.join(dict.fromkeys(recs))}.")

    st.divider()

    st.markdown("## 🔎 Cek Produk & Rekomendasi")
    if 'products' in df.columns:
        product_list = df['products'].dropna().unique().tolist()
    else:
        product_list = (df.get('products', pd.Series(["unknown"])) + " – " + df.get('Product_Category', pd.Series(["unknown"]))).unique().tolist()

    selected_product = st.selectbox("Pilih Produk:", product_list)

    if selected_product:
        available_results = [r for r in results if r["Model"] in algo_choice]
        if not available_results:
            st.warning("Tidak ada model valid untuk rekomendasi.")
        else:
            if len(algo_choice) == 1:
                pick_algo = algo_choice[0]
            else:
                best = max(available_results, key=lambda x: (np.nan if np.isnan(x["Silhouette Score"]) else x["Silhouette Score"]))
                pick_algo = best["Model"]

            pick_col = detect_cluster_col(df, pick_algo)
            if pick_col is None:
                st.warning(f"Tidak menemukan kolom label untuk model {pick_algo}.")
            else:
                if 'products' in df.columns:
                    sel_row = df[df['products'] == selected_product]
                else:
                    sel_row = df[(df.get('products', "") + " – " + df.get('Product_Category', "") ) == selected_product]

                if sel_row.empty:
                    st.warning("Produk tidak ditemukan di dataset clustering.")
                else:
                    cluster_label = sel_row[pick_col].iloc[0]
                    st.success(f"Produk **{selected_product}** ada di **Cluster {cluster_label}** menurut model **{pick_algo}**.")

                    if pick_algo in summaries and not summaries[pick_algo].empty:
                        ssum = summaries[pick_algo]
                        matched = ssum[ssum['Cluster'].astype(str) == str(cluster_label)]
                        if not matched.empty:
                            row = matched.iloc[0]
                            rec_list = []
                            if row['Total_Purchases'] >= df['Total_Purchases'].mean():
                                rec_list.append("Pertahankan stok & program loyalitas")
                            else:
                                rec_list.append("Tingkatkan promosi & bundling")
                            if row['Profitability_Index'] >= df['Profitability_Index'].mean():
                                rec_list.append("Pertahankan margin & fokus cross-sell")
                            else:
                                rec_list.append("Evaluasi harga atau sourcing untuk tingkatkan margin")
                            if row['Ratings'] >= df['Ratings'].mean():
                                rec_list.append("Naikkan eksposur di channel berbayar & marketplace")
                            else:
                                rec_list.append("Perbaiki kualitas produk & deskripsi, kumpulkan ulasan")
                            st.markdown("**Rekomendasi Bisnis:**")
                            for r in rec_list:
                                st.write(f"- {r}")
                        else:
                            st.info("Tidak ada ringkasan cluster untuk rekomendasi detil.")
                    else:
                        st.info("Ringkasan cluster tidak tersedia untuk model ini.")

else:
    st.subheader("📈 Analisis Pergerakan Penjualan Produk (sample 20%)")
    st.dataframe(sales_df.head(10), use_container_width=True)

    product_col = None
    for cand in ['Product_Name', 'products', 'Product_Type', 'products']:
        if cand in sales_df.columns:
            product_col = cand
            break

    if product_col is None:
        st.error("Tidak ditemukan kolom produk yang sesuai (misalnya 'Product_Name' atau 'products').")
    else:
        product_list = sales_df[product_col].dropna().unique().tolist()
        sel = st.selectbox("Pilih Produk:", product_list)

        if sel:
            prod_df = sales_df[sales_df[product_col] == sel].copy()

            if 'Date' in prod_df.columns:
                prod_df['Date'] = pd.to_datetime(prod_df['Date'])

                y_col = None
                for cand in ['Total_Amount', 'Amount', 'Total_Purchases']:
                    if cand in prod_df.columns:
                        y_col = cand
                        break

                if y_col is None:
                    st.error("Tidak ditemukan kolom metrik penjualan seperti 'Total_Amount', 'Amount', atau 'Total_Purchases'.")
                else:
                    prod_df['Month'] = prod_df['Date'].dt.to_period('M')
                    monthly = prod_df.groupby('Month')[y_col].sum().reset_index()
                    monthly['Month'] = monthly['Month'].astype(str)

                    fig_line = px.line(
                        monthly,
                        x='Month', y=y_col,
                        title=f"📈 Tren Penjualan Bulanan – {sel}",
                        markers=True,
                        line_shape='spline'
                    )
                    fig_line.update_layout(xaxis_title="Bulan", yaxis_title=y_col, template="plotly_white")
                    st.plotly_chart(fig_line, use_container_width=True)

                    if len(monthly) >= 2:
                        diff = monthly[y_col].iloc[-1] - monthly[y_col].iloc[-2]
                        if diff > 0:
                            st.success(f"📊 Penjualan {sel} meningkat {diff:,.0f} dibanding bulan sebelumnya.")
                        elif diff < 0:
                            st.warning(f"📉 Penjualan {sel} menurun {abs(diff):,.0f} dibanding bulan sebelumnya.")
                        else:
                            st.info(f"➡️ Penjualan {sel} stabil dibanding bulan sebelumnya.")

                    st.divider()
                    if 'Product_Category' in sales_df.columns:
                        cat_df = sales_df.groupby('Product_Category')[y_col].sum().reset_index().sort_values(y_col, ascending=False)
                        fig_bar = px.bar(
                            cat_df, x='Product_Category', y=y_col,
                            title="🏷️ Total Penjualan per Kategori Produk",
                            color=y_col, color_continuous_scale='Viridis'
                        )
                        fig_bar.update_layout(xaxis_title="Kategori", yaxis_title=y_col, template="plotly_white")
                        st.plotly_chart(fig_bar, use_container_width=True)

                    if 'products' in sales_df.columns:
                        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
                        sales_df['Month'] = sales_df['Date'].dt.strftime('%b-%Y')
                        heat = sales_df.groupby(['Month', 'products'])[y_col].sum().reset_index()
                        pivot_heat = heat.pivot(index='products', columns='Month', values=y_col).fillna(0)
                        fig_heat = px.imshow(
                            pivot_heat,
                            labels=dict(x="Bulan", y="Brand", color=y_col),
                            title="🔥 Heatmap Penjualan per Brand & Bulan",
                            color_continuous_scale="YlGnBu"
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

                    if 'Ratings' in sales_df.columns:
                        fig_hist = px.histogram(
                            sales_df, x='Ratings', nbins=20,
                            title="⭐ Distribusi Rating Produk",
                            color='Product_Category' if 'Product_Category' in sales_df.columns else None,
                            marginal="box"
                        )
                        fig_hist.update_layout(template="plotly_white")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    top_cat = None
                    if 'Product_Category' in sales_df.columns:
                        top_cat = sales_df.groupby('Product_Category')[y_col].sum().idxmax()
                    st.markdown("### 💡 Insight Otomatis:")
                    st.write("- Produk dengan penjualan tertinggi secara keseluruhan:",
                             f"**{top_cat}**" if top_cat else "Data tidak lengkap.")
                    if 'Ratings' in sales_df.columns:
                        avg_rating = sales_df['Ratings'].mean()
                        st.write(f"- Rata-rata rating semua produk: **{avg_rating:.2f} / 5**")
                    if 'City' in sales_df.columns:
                        top_city = sales_df['City'].value_counts().idxmax()
                        st.write(f"- Kota dengan transaksi terbanyak: **{top_city}**")
            else:
                st.error("Kolom 'Date' tidak ditemukan di sales_sampled.csv")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 Roy Sihombing | Final Project – Data Science</p>", unsafe_allow_html=True)
