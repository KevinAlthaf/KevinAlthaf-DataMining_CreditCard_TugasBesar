
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CreditCard.AI – Kelompok 6",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.15) !important;
        border-color: rgba(255,255,255,0.25) !important;
    }
    /* Section header */
    .section-header {
        background: linear-gradient(135deg, #1a237e, #1565c0);
        color: white !important;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #1565c0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    /* Cluster card */
    .cluster-card-0 {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #2e7d32;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .cluster-card-1 {
        background: linear-gradient(135deg, #fce4ec, #f8bbd0);
        border-left: 5px solid #c62828;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    /* Tujuan analisis columns separator */
    .tujuan-col {
        border-right: 1px solid #e0e0e0;
        padding-right: 16px;
    }
    hr { margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CACHED DATA & MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('CC GENERAL.csv')
    return df


@st.cache_data
def prepare_data(_df):
    df_clean = _df.drop(columns=['CUST_ID']).copy()
    df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(
        df_clean['MINIMUM_PAYMENTS'].median())
    df_clean['CREDIT_LIMIT'] = df_clean['CREDIT_LIMIT'].fillna(
        df_clean['CREDIT_LIMIT'].median())
    # IQR Capping
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(
            lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df_clean


@st.cache_data
def run_clustering(_df_clean):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(
        _df_clean), columns=_df_clean.columns)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    # 5-cluster (eksplorasi awal)
    km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_5 = km5.fit_predict(X_pca)

    # Elbow + Silhouette
    inertias, sil_scores, k_range = [], [], range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        from sklearn.metrics import silhouette_score
        sil_scores.append(silhouette_score(X_pca, lbl))

    # Final 2-cluster
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_2 = km2.fit_predict(X_pca)

    return X_pca, labels_5, labels_2, list(k_range), inertias, sil_scores


@st.cache_data
def run_models(_df_clean):
    scaler2 = joblib.load('scaler.pkl')
    df_scaled = pd.DataFrame(scaler2.transform(
        _df_clean), columns=_df_clean.columns)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_2 = km2.fit_predict(X_pca)

    df_model = _df_clean.copy()
    df_model['Cluster'] = labels_2

    X = df_model.drop(columns='Cluster')
    y = df_model['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train_sc = scaler2.transform(X_train)
    X_test_sc = scaler2.transform(X_test)

    # GridSearchCV Logistic Regression
    param_grid_lr = {
        'C':       [0.01, 0.1, 1, 10],
        'solver':  ['lbfgs', 'liblinear'],
        'max_iter': [200, 500]
    }
    gs_lr = GridSearchCV(LogisticRegression(), param_grid_lr,
                         cv=5, scoring='accuracy', n_jobs=-1)
    gs_lr.fit(X_train_sc, y_train)

    # GridSearchCV Naive Bayes
    param_grid_gnb = {'var_smoothing': np.logspace(-12, -1, 20)}
    gs_gnb = GridSearchCV(GaussianNB(), param_grid_gnb,
                          cv=5, scoring='accuracy', n_jobs=-1)
    gs_gnb.fit(X_train_sc, y_train)

    y_pred_lr = gs_lr.predict(X_test_sc)
    y_pred_gnb = gs_gnb.predict(X_test_sc)

    y_proba_lr = gs_lr.predict_proba(X_test_sc)[:, 1]
    y_proba_gnb = gs_gnb.predict_proba(X_test_sc)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_gnb = accuracy_score(y_test, y_pred_gnb)

    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_proba_lr)
    fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_proba_gnb)
    auc_lr = auc(fpr_lr,  tpr_lr)
    auc_gnb = auc(fpr_gnb, tpr_gnb)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_gnb = confusion_matrix(y_test, y_pred_gnb)

    cr_lr = classification_report(y_test, y_pred_lr,  target_names=[
                                  'Cluster 0', 'Cluster 1'], output_dict=True)
    cr_gnb = classification_report(y_test, y_pred_gnb, target_names=[
                                   'Cluster 0', 'Cluster 1'], output_dict=True)

    return {
        'best_params_lr':  gs_lr.best_params_,
        'best_params_gnb': gs_gnb.best_params_,
        'acc_lr':  acc_lr,  'acc_gnb':  acc_gnb,
        'fpr_lr':  fpr_lr,  'tpr_lr':   tpr_lr,  'auc_lr':  auc_lr,
        'fpr_gnb': fpr_gnb, 'tpr_gnb':  tpr_gnb, 'auc_gnb': auc_gnb,
        'cm_lr':   cm_lr,   'cm_gnb':   cm_gnb,
        'cr_lr':   cr_lr,   'cr_gnb':   cr_gnb,
    }


# ─────────────────────────────────────────
# SESSION STATE – active menu
# ─────────────────────────────────────────
if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Homepage"

MENUS = [
    ("Homepage",           "🏠", "Homepage"),
    ("EDA & Preparation",  "📊", "EDA & Preparation"),
    ("Clustering",         "🔵", "Clustering"),
    ("Perbandingan Model", "📈", "Perbandingan Model"),
    ("Simulasi Prediksi",  "🤖", "Simulasi Prediksi"),
]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 8px 12px 8px;text-align:center;">
        <div style="font-size:2.4rem;">💳</div>
        <div style="font-size:1.2rem;font-weight:800;letter-spacing:1px;color:white;">CreditCard.AI</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.6);margin-top:2px;">Kelompok 6 · SI4802</div>
    </div>
    <div style="height:1px;background:rgba(255,255,255,0.2);margin:0 0 14px 0;"></div>
    <div style="font-size:0.68rem;font-weight:700;letter-spacing:2.5px;color:rgba(255,255,255,0.45);
                padding:0 6px;margin-bottom:10px;">NAVIGASI</div>
    """, unsafe_allow_html=True)

    for key, icon, label in MENUS:
        is_active = st.session_state.active_menu == key
        btn_style = (
            "background:rgba(255,255,255,0.92);color:#1a237e !important;"
            "font-weight:700;border:2px solid rgba(255,255,255,0.95);"
            "transform:scale(1.02);box-shadow:0 4px 14px rgba(0,0,0,0.25);"
        ) if is_active else (
            "background:rgba(255,255,255,0.07);color:rgba(255,255,255,0.9) !important;"
            "font-weight:500;border:1px solid rgba(255,255,255,0.15);"
        )
        st.markdown(f"""
        <div style="margin-bottom:6px;">
            <form action="" method="get">
                <button name="nav" value="{key}" type="submit" style="
                    width:100%;text-align:left;padding:11px 14px;
                    border-radius:10px;cursor:pointer;
                    font-size:0.9rem;transition:all 0.18s ease;
                    {btn_style}">
                    {icon}&nbsp;&nbsp;{label}
                </button>
            </form>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_menu = key
            st.rerun()

    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,0.2);margin:14px 0 10px 0;"></div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.4);text-align:center;">
        Data Mining · SI4802<br>Kelompok 6 · 2026
    </div>
    """, unsafe_allow_html=True)

# ── hide default streamlit button styles, keep only our sidebar buttons custom ──
st.markdown("""
<style>
[data-testid="stSidebar"] .stButton button {
    width: 100% !important;
    text-align: left !important;
    padding: 11px 16px !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: all 0.18s ease !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.9) !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.16) !important;
    border-color: rgba(255,255,255,0.3) !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(255,255,255,0.92) !important;
    color: #1a237e !important;
    font-weight: 700 !important;
    border: 2px solid white !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2) !important;
    transform: scale(1.02) !important;
}
/* hide the HTML form buttons — only real st.button used */
[data-testid="stSidebar"] form { display: none !important; }
</style>
""", unsafe_allow_html=True)

menu = st.session_state.active_menu


# ═══════════════════════════════════════════
# MENU 1 – HOMEPAGE
# ═══════════════════════════════════════════
if menu == "Homepage":
    df = load_data()
    st.image("Header_Web.webp", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Identitas Kelompok ──
    st.markdown('<div class="section-header"><h3 style="margin:0;color:white">👥 Identitas Kelompok</h3></div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        **Mata Kuliah:** Data Mining  
        **Kelas:** SI4802  
        **Kelompok:** 6
        """)
    with col2:
        st.markdown("""
        Muhammad Daffa Izzati &nbsp;·&nbsp; 102022430036  
        Muhammad Daffa Reifansya &nbsp;·&nbsp; 102022400329  
        Kevin Muhammad Althaf &nbsp;·&nbsp; 102022400325  
        Sekar Erliana Putri &nbsp;·&nbsp; 102022430069
        """)

    st.divider()

    # ── Alasan Pemilihan Dataset ──
    st.markdown('<div class="section-header"><h3 style="margin:0;color:white">💡 Mengapa Dataset Ini?</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    Dataset **CC GENERAL** berisi data perilaku transaksi **8.950 nasabah** kartu kredit selama 6 bulan terakhir.
    Kami memilih dataset ini karena:
    - Relevan dengan industri keuangan perbankan yang terus berkembang.
    - Memiliki variasi fitur yang kaya sehingga cocok untuk eksplorasi clustering dan klasifikasi.
    - Dapat digunakan untuk membantu bank memahami segmentasi nasabah secara otomatis dan data-driven.
    - Ukuran dataset yang cukup besar (~9K baris, 18 kolom) memberikan hasil analisis yang lebih robust.
    """)

    st.divider()

    # ── Tujuan Analisis ──
    st.markdown('<div class="section-header"><h3 style="margin:0;color:white">🎯 Tujuan Analisis</h3></div>',
                unsafe_allow_html=True)
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        with st.container(border=True):
            st.markdown("**1. Segmentasi Nasabah**")
            st.markdown(
                "Mengelompokkan nasabah berdasarkan pola transaksi menggunakan algoritma K-Means Clustering.")
    with row1_c2:
        with st.container(border=True):
            st.markdown("**2. Pemahaman Perilaku**")
            st.markdown(
                "Memahami karakteristik tiap segmen nasabah — apakah cenderung berbelanja atau menarik uang tunai.")
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        with st.container(border=True):
            st.markdown("**3. Perbandingan Model**")
            st.markdown(
                "Membandingkan performa Logistic Regression dan Gaussian Naive Bayes untuk klasifikasi segmen nasabah.")
    with row2_c2:
        with st.container(border=True):
            st.markdown("**4. Simulasi Prediksi**")
            st.markdown(
                "Membangun sistem simulasi interaktif yang dapat memprediksi segmen nasabah baru berdasarkan data inputan.")

    st.divider()

    # ── Penjelasan Variabel ──
    st.markdown('<div class="section-header"><h3 style="margin:0;color:white">📌 Penjelasan Variabel Dataset</h3></div>', unsafe_allow_html=True)
    st.markdown(
        f"Dataset memiliki **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**, dikelompokkan menjadi 4 kategori variabel berikut.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris 1: Saldo & Pembelian (full width) ──
    with st.container(border=True):
        st.markdown("##### 💰 Saldo & Pembelian")
        st.markdown("---")
        var_list_1 = [
            ("BALANCE",
             "Sisa saldo kartu kredit nasabah pada periode observasi."),
            ("BALANCE_FREQUENCY",
             "Seberapa sering saldo diperbarui (skala 0–1; semakin mendekati 1 = semakin sering)."),
            ("PURCHASES",
             "Total nilai pembelian yang dilakukan nasabah selama periode berjalan."),
            ("ONEOFF_PURCHASES",
             "Nilai pembelian yang dibayar sekali lunas, bukan cicilan."),
            ("INSTALLMENTS_PURCHASES",
             "Nilai pembelian yang dibayar secara cicilan/angsuran."),
            ("PURCHASES_FREQUENCY",
             "Frekuensi nasabah melakukan pembelian (skala 0–1)."),
            ("ONEOFF_PURCHASES_FREQUENCY",
             "Frekuensi pembelian sekali bayar (skala 0–1)."),
            ("PURCHASES_INSTALLMENTS_FREQUENCY",
             "Frekuensi pembelian cicilan (skala 0–1)."),
            ("PURCHASES_TRX",
             "Jumlah total transaksi pembelian yang tercatat."),
        ]
        # Tampilkan dalam 3 kolom agar tidak terlalu panjang ke bawah
        col_v1, col_v2, col_v3 = st.columns(3)
        for i, (name, desc) in enumerate(var_list_1):
            target = [col_v1, col_v2, col_v3][i % 3]
            with target:
                st.markdown(f"**`{name}`**  \n{desc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris 2: Penarikan Tunai | Identitas ──
    vg2, vg4 = st.columns(2)
    with vg2:
        with st.container(border=True):
            st.markdown("##### 🏧 Penarikan Tunai")
            st.markdown("---")
            var_list_2 = [
                ("CASH_ADVANCE",
                 "Total uang tunai yang ditarik nasabah menggunakan kartu kredit."),
                ("CASH_ADVANCE_FREQUENCY",
                 "Seberapa sering nasabah melakukan penarikan tunai (skala 0–1)."),
                ("CASH_ADVANCE_TRX",
                 "Jumlah transaksi penarikan uang tunai yang dilakukan."),
            ]
            for name, desc in var_list_2:
                st.markdown(f"**`{name}`**  \n{desc}")

    with vg4:
        with st.container(border=True):
            st.markdown("##### 🪪 Identitas")
            st.markdown("---")
            var_list_4 = [
                ("CUST_ID", "ID unik setiap nasabah. Bersifat kategorik dan tidak digunakan dalam proses pemodelan."),
                ("TENURE",  "Lama nasabah menggunakan layanan kartu kredit, dalam satuan bulan."),
            ]
            for name, desc in var_list_4:
                st.markdown(f"**`{name}`**  \n{desc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris 3: Kredit & Pembayaran (full width) ──
    with st.container(border=True):
        st.markdown("##### 💳 Kredit & Pembayaran")
        st.markdown("---")
        var_list_3 = [
            ("CREDIT_LIMIT",     "Batas maksimum kredit yang diberikan bank kepada nasabah."),
            ("PAYMENTS",         "Total pembayaran tagihan yang dilakukan nasabah."),
            ("MINIMUM_PAYMENTS",
             "Jumlah pembayaran minimum yang dilakukan (jika tidak melunasi penuh)."),
            ("PRC_FULL_PAYMENT",
             "Proporsi bulan di mana nasabah membayar tagihan secara penuh (skala 0–1)."),
        ]
        col_w1, col_w2 = st.columns(2)
        for i, (name, desc) in enumerate(var_list_3):
            target = col_w1 if i % 2 == 0 else col_w2
            with target:
                st.markdown(f"**`{name}`**  \n{desc}")


# ═══════════════════════════════════════════
# MENU 2 – EDA & PREPARATION
# ═══════════════════════════════════════════
elif menu == "EDA & Preparation":
    df = load_data()
    df_clean = prepare_data(df)
    st.markdown('<div class="section-header"><h2 style="margin:0;color:white">📊 Exploratory Data Analysis & Data Preparation</h2></div>', unsafe_allow_html=True)

    tab_raw, tab_stats, tab_missing, tab_outlier, tab_hist, tab_corr = st.tabs([
        "📁 Raw Data", "📐 Statistik Deskriptif", "🔍 Missing Values",
        "📦 Outlier Handling", "📊 Distribusi Fitur", "🌡️ Korelasi"
    ])

    # ── Raw Data ──
    with tab_raw:
        st.subheader("Preview Dataset Asli")
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
        c2.metric("Jumlah Kolom", f"{df.shape[1]}")
        c3.metric("Duplikat", f"{df.duplicated().sum()}")

    # ── Statistik Deskriptif ──
    with tab_stats:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().T.style.format(
            "{:.3f}"), use_container_width=True)

    # ── Missing Values ──
    with tab_missing:
        st.subheader("Pengecekan Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        df_missing = pd.DataFrame({
            'Jumlah Missing': missing,
            'Persentase (%)': missing_pct
        })
        df_missing = df_missing[df_missing['Jumlah Missing'] > 0]
        if df_missing.empty:
            st.success("Tidak ada missing value pada dataset!")
        else:
            st.dataframe(df_missing, use_container_width=True)
            st.markdown("""
            **Penanganan Missing Values:**
            - `MINIMUM_PAYMENTS` → diisi dengan **median** (robust terhadap outlier)
            - `CREDIT_LIMIT` → diisi dengan **median** (robust terhadap outlier)
            """)
            fig_mv, ax_mv = plt.subplots(figsize=(6, 3))
            ax_mv.barh(df_missing.index,
                       df_missing['Jumlah Missing'], color='#1565c0')
            ax_mv.set_xlabel('Jumlah Missing')
            ax_mv.set_title('Missing Values per Kolom')
            plt.tight_layout()
            st.pyplot(fig_mv)

    # ── Outlier Handling ──
    with tab_outlier:
        st.subheader("Outlier Handling dengan IQR Capping")
        st.markdown("""
        Outlier ditangani menggunakan metode **IQR Capping** — nilai di luar batas `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
        di-clip ke batas tersebut tanpa menghapus data. Berikut perbandingan **sebelum** dan **sesudah** capping untuk semua fitur:
        """)

        cols_num = df_clean.columns.tolist()
        n_cols = 3
        n_rows = (len(cols_num) + n_cols - 1) // n_cols

        fig_bp, axes_bp = plt.subplots(
            n_rows, n_cols * 2, figsize=(22, n_rows * 3.2))
        axes_bp = axes_bp.reshape(n_rows, n_cols * 2)

        for idx, col in enumerate(cols_num):
            row = idx // n_cols
            col_pair_start = (idx % n_cols) * 2

            ax_before = axes_bp[row, col_pair_start]
            ax_after = axes_bp[row, col_pair_start + 1]

            raw_col = df[col].dropna()
            Q1 = raw_col.quantile(0.25)
            Q3 = raw_col.quantile(0.75)
            IQR = Q3 - Q1
            n_out = ((raw_col < Q1 - 1.5*IQR) | (raw_col > Q3 + 1.5*IQR)).sum()

            ax_before.boxplot(raw_col, patch_artist=True,
                              boxprops=dict(facecolor='#ffcdd2'),
                              medianprops=dict(color='#c62828', linewidth=2),
                              whiskerprops=dict(color='#c62828'),
                              capprops=dict(color='#c62828'),
                              flierprops=dict(marker='o', markerfacecolor='#ef9a9a', markersize=2))
            ax_before.set_title(f'{col}\nSebelum ({n_out} outlier)',
                                fontsize=7.5, fontweight='bold', color='#c62828')
            ax_before.tick_params(labelsize=6)

            ax_after.boxplot(df_clean[col], patch_artist=True,
                             boxprops=dict(facecolor='#c8e6c9'),
                             medianprops=dict(color='#2e7d32', linewidth=2),
                             whiskerprops=dict(color='#2e7d32'),
                             capprops=dict(color='#2e7d32'))
            ax_after.set_title(f'{col}\nSetelah (0 outlier)',
                               fontsize=7.5, fontweight='bold', color='#2e7d32')
            ax_after.tick_params(labelsize=6)

        # Sembunyikan subplot kosong
        for idx in range(len(cols_num), n_rows * n_cols):
            row = idx // n_cols
            col_pair_start = (idx % n_cols) * 2
            fig_bp.delaxes(axes_bp[row, col_pair_start])
            fig_bp.delaxes(axes_bp[row, col_pair_start + 1])

        plt.suptitle('Perbandingan Boxplot Sebelum & Sesudah IQR Capping',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig_bp)

    # ── Distribusi Histogram ──
    with tab_hist:
        st.subheader("Distribusi Setiap Fitur (Histogram)")
        fig_hist, axes_hist = plt.subplots(4, 5, figsize=(20, 14))
        axes_hist = axes_hist.flatten()
        cols = df_clean.columns.tolist()
        for i, col in enumerate(cols):
            axes_hist[i].hist(df_clean[col], bins=30,
                              color='#1565c0', edgecolor='white', alpha=0.85)
            axes_hist[i].set_title(col, fontsize=9, fontweight='bold')
            axes_hist[i].set_xlabel('')
            axes_hist[i].tick_params(labelsize=7)
        # Sembunyikan subplot kosong
        for j in range(len(cols), len(axes_hist)):
            fig_hist.delaxes(axes_hist[j])
        plt.suptitle('Distribusi Semua Fitur (Setelah Preprocessing)',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig_hist)

    # ── Heatmap Korelasi ──
    with tab_corr:
        st.subheader("Heatmap Korelasi Antar Fitur")
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        corr_matrix = df_clean.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='Blues', ax=ax_corr, linewidths=0.5,
            annot_kws={'size': 7}
        )
        ax_corr.set_title('Correlation Heatmap (Lower Triangle)',
                          fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_corr)


# ═══════════════════════════════════════════
# MENU 3 – CLUSTERING
# ═══════════════════════════════════════════
elif menu == "Clustering":
    df = load_data()
    df_clean = prepare_data(df)
    st.markdown('<div class="section-header"><h2 style="margin:0;color:white">🔵 K-Means Clustering – Segmentasi Nasabah</h2></div>', unsafe_allow_html=True)

    with st.spinner("Menjalankan proses clustering... ⏳"):
        X_pca, labels_5, labels_2, k_range, inertias, sil_scores = run_clustering(
            df_clean)

    tab_5, tab_eval, tab_final = st.tabs([
        "🔬 Eksplorasi Awal (5 Cluster)",
        "📉 Evaluasi Jumlah Cluster",
        "✅ Hasil Final (2 Cluster)"
    ])

    # ── Eksplorasi 5 Cluster ──
    with tab_5:
        st.subheader("Percobaan Awal: 5 Cluster")
        st.markdown("""
        Pada tahap awal, kami mencoba membagi data menjadi **5 cluster** untuk melihat sebaran awal nasabah
        dalam ruang fitur yang telah direduksi dengan PCA menjadi 2 dimensi.
        """)
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_5,
                              cmap='tab10', s=15, alpha=0.6)
        plt.colorbar(scatter, ax=ax5, label='Cluster')
        ax5.set_title('K-Means Clustering dengan 5 Cluster (PCA 2D)',
                      fontsize=13, fontweight='bold')
        ax5.set_xlabel('Principal Component 1')
        ax5.set_ylabel('Principal Component 2')
        plt.tight_layout()
        st.pyplot(fig5)

        st.markdown("""
        <p style="font-size:1.05rem; line-height:1.8; margin-top:16px;">
        Dengan 5 cluster, beberapa kelompok tampak <strong>berdekatan dan saling tumpang tindih</strong> satu sama lain.
        Hal ini menunjukkan bahwa pembagian menjadi 5 kelompok terlalu granular dan kurang bermakna secara bisnis.
        Oleh karena itu, perlu dilakukan evaluasi lebih lanjut menggunakan <strong>Elbow Method</strong> dan
        <strong>Silhouette Score</strong> untuk menentukan jumlah cluster yang benar-benar optimal.
        </p>
        """, unsafe_allow_html=True)

    # ── Evaluasi ──
    with tab_eval:
        st.subheader("Mencari Jumlah Cluster Optimal")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
            ax_elbow.plot(k_range, inertias, 'bo-', linewidth=2,
                          markersize=7, color='#1565c0')
            ax_elbow.axvline(x=2, color='red', linestyle='--',
                             linewidth=1.5, label='k=2 (dipilih)')
            ax_elbow.set_xlabel('Jumlah Cluster (k)')
            ax_elbow.set_ylabel('Inertia (Within-cluster SSE)')
            ax_elbow.set_title('Elbow Method', fontsize=12, fontweight='bold')
            ax_elbow.legend()
            ax_elbow.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_elbow)

        with col_e2:
            fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
            ax_sil.plot(k_range, sil_scores, 's-', linewidth=2,
                        markersize=7, color='#2e7d32')
            ax_sil.axvline(x=2, color='red', linestyle='--',
                           linewidth=1.5, label='k=2 (dipilih)')
            ax_sil.set_xlabel('Jumlah Cluster (k)')
            ax_sil.set_ylabel('Silhouette Score')
            ax_sil.set_title('Silhouette Score',
                             fontsize=12, fontweight='bold')
            ax_sil.legend()
            ax_sil.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sil)

        # Tabel Skor — highlight hanya baris k=2
        df_eval = pd.DataFrame(
            {'k': k_range, 'Inertia': inertias, 'Silhouette Score': sil_scores})
        df_eval = df_eval.set_index('k')

        def highlight_k2(row):
            return ['background-color: #fff9c4; font-weight: bold'] * len(row) if row.name == 2 else [''] * len(row)

        st.dataframe(
            df_eval.style.apply(highlight_k2, axis=1).format('{:.4f}'),
            use_container_width=True
        )
        st.caption(
            "🟡 Baris yang di-highlight menunjukkan k=2 yang dipilih sebagai jumlah cluster optimal.")

        st.success("**Kesimpulan:** Berdasarkan Elbow Method dan Silhouette Score, nilai **k=2** dipilih sebagai jumlah cluster optimal karena memberikan pemisahan antar kelompok yang paling jelas dan bermakna secara bisnis.")

    # ── Hasil Final 2 Cluster ──
    with tab_final:
        st.subheader("Hasil Akhir: 2 Cluster Nasabah")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        colors_2 = ['#1565c0', '#c62828']
        for cl, color in zip([0, 1], colors_2):
            mask = labels_2 == cl
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, s=15, alpha=0.6, label=f'Cluster {cl}')
        ax2.set_title('K-Means Clustering – 2 Cluster Final (PCA 2D)',
                      fontsize=13, fontweight='bold')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)

        # Distribusi cluster
        unique, counts = np.unique(labels_2, return_counts=True)
        st.markdown("**Distribusi Nasabah per Cluster:**")
        cc1, cc2 = st.columns(2)
        cc1.metric("Cluster 0 – Active Shoppers 🛒",
                   f"{counts[0]:,} nasabah", f"{counts[0]/len(labels_2)*100:.1f}%")
        cc2.metric("Cluster 1 – Cash Advance Users 🏧",
                   f"{counts[1]:,} nasabah", f"{counts[1]/len(labels_2)*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_cl1, col_cl2 = st.columns(2)

        with col_cl1:
            st.markdown("""
            <div style="background:#dbeafe; border-left:6px solid #1565c0; border-radius:10px; padding:18px 22px; margin-bottom:12px;">
                <h4 style="color:#000000; margin-top:0;">🛒 Cluster 0 – Active Shoppers</h4>
                <b style="color:#000000;">Karakteristik Utama:</b>
                <ul style="color:#000000;">
                    <li>Sering melakukan transaksi pembelian langsung dengan kartu kredit</li>
                    <li>Frekuensi pembelian (one-off maupun cicilan) yang tinggi</li>
                    <li>Sangat jarang melakukan penarikan uang tunai (cash advance)</li>
                    <li>Cenderung memiliki jumlah transaksi pembelian yang banyak</li>
                    <li>Memanfaatkan kartu kredit sebagai alat belanja utama</li>
                </ul>
                <b style="color:#000000;">Rekomendasi Bank:</b> <span style="color:#000000;">Tawarkan program cashback, reward poin, dan promosi merchant.</span>
            </div>
            """, unsafe_allow_html=True)

        with col_cl2:
            st.markdown("""
            <div style="background:#fee2e2; border-left:6px solid #c62828; border-radius:10px; padding:18px 22px; margin-bottom:12px;">
                <h4 style="color:#000000; margin-top:0;">🏧 Cluster 1 – Cash Advance Users</h4>
                <b style="color:#000000;">Karakteristik Utama:</b>
                <ul style="color:#000000;">
                    <li>Jarang melakukan pembelian menggunakan kartu kredit</li>
                    <li>Sering menggunakan kartu kredit untuk menarik uang tunai</li>
                    <li>Frekuensi cash advance yang tinggi dengan jumlah transaksi besar</li>
                    <li>Cenderung memiliki sisa saldo tagihan yang lebih tinggi</li>
                    <li>Lebih bergantung pada fungsi pinjaman tunai kartu kredit</li>
                </ul>
                <b style="color:#000000;">Rekomendasi Bank:</b> <span style="color:#000000;">Tawarkan program cicilan ringan, edukasi keuangan, dan konversi ke KTA.</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# MENU 4 – PERBANDINGAN MODEL
# ═══════════════════════════════════════════
elif menu == "Perbandingan Model":
    df = load_data()
    df_clean = prepare_data(df)
    st.markdown('<div class="section-header"><h2 style="margin:0;color:white">📈 Perbandingan Model: Logistic Regression vs Naive Bayes</h2></div>', unsafe_allow_html=True)

    with st.spinner("Menjalankan GridSearchCV dan evaluasi model... ⏳ (mungkin 30–60 detik pertama kali)"):
        results = run_models(df_clean)

    # ── Best Params ──
    st.subheader("🔧 Best Parameters (GridSearchCV)")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Logistic Regression**")
        params_lr = results['best_params_lr']
        for k, v in params_lr.items():
            st.markdown(f"- `{k}` = **{v}**")
    with col_p2:
        st.markdown("**Gaussian Naive Bayes**")
        params_gnb = results['best_params_gnb']
        for k, v in params_gnb.items():
            st.markdown(f"- `{k}` = **{v:.2e}**")

    st.divider()

    # ── Accuracy ──
    st.subheader("🎯 Perbandingan Akurasi")
    col_a1, col_a2 = st.columns(2)
    delta_acc = results['acc_lr'] - results['acc_gnb']
    col_a1.metric("Logistic Regression", f"{results['acc_lr']:.4f}",
                  delta=f"+{delta_acc:.4f} vs GNB" if delta_acc > 0 else f"{delta_acc:.4f} vs GNB")
    col_a2.metric("Gaussian Naive Bayes", f"{results['acc_gnb']:.4f}")

    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    models = ['Logistic Regression', 'Gaussian Naive Bayes']
    accs = [results['acc_lr'], results['acc_gnb']]
    bar_colors = ['#1565c0', '#ff7f0e']
    bars = ax_acc.bar(models, accs, color=bar_colors,
                      width=0.45, edgecolor='white', linewidth=1.2)
    ax_acc.set_ylim(0, 1.12)
    ax_acc.set_ylabel('Accuracy Score')
    ax_acc.set_title('Perbandingan Akurasi Model', fontweight='bold')
    for bar, val in zip(bars, accs):
        ax_acc.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', fontweight='bold', fontsize=12)
    ax_acc.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_acc)

    st.divider()

    # ── ROC Curve ──
    st.subheader("📉 ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    ax_roc.plot(results['fpr_lr'],  results['tpr_lr'],  color='#1565c0', lw=2.5,
                label=f"Logistic Regression (AUC = {results['auc_lr']:.4f})")
    ax_roc.plot(results['fpr_gnb'], results['tpr_gnb'], color='#ff7f0e', lw=2.5,
                label=f"Gaussian Naive Bayes  (AUC = {results['auc_gnb']:.4f})")
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.5,
                linestyle='--', label='Random Classifier')
    ax_roc.fill_between(results['fpr_lr'],
                        results['tpr_lr'], alpha=0.08, color='#1565c0')
    ax_roc.fill_between(
        results['fpr_gnb'], results['tpr_gnb'], alpha=0.08, color='#ff7f0e')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(
        'Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_roc)

    st.divider()

    # ── Confusion Matrix ──
    st.subheader("🧮 Confusion Matrix")
    col_cm1, col_cm2 = st.columns(2)
    for col_cm, cm, title in [
        (col_cm1, results['cm_lr'],  "Logistic Regression"),
        (col_cm2, results['cm_gnb'], "Gaussian Naive Bayes")
    ]:
        with col_cm:
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Cluster 0', 'Cluster 1'],
                        yticklabels=['Cluster 0', 'Cluster 1'],
                        linewidths=0.5)
            ax_cm.set_title(title, fontweight='bold')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            plt.tight_layout()
            st.pyplot(fig_cm)

    st.divider()

    # ── Classification Report ──
    st.subheader("📋 Classification Report")
    col_cr1, col_cr2 = st.columns(2)
    for col_cr, cr, title in [
        (col_cr1, results['cr_lr'],  "Logistic Regression"),
        (col_cr2, results['cr_gnb'], "Gaussian Naive Bayes")
    ]:
        with col_cr:
            st.markdown(f"**{title}**")
            df_cr = pd.DataFrame(cr).T
            st.dataframe(df_cr.style.format("{:.4f}").background_gradient(
                subset=['precision', 'recall', 'f1-score'], cmap='Blues'),
                use_container_width=True)

    # ── Kesimpulan ──
    st.divider()
    st.subheader("💡 Kesimpulan Perbandingan")
    winner = "Logistic Regression" if results['acc_lr'] >= results['acc_gnb'] else "Gaussian Naive Bayes"
    st.success(f"""
    **Model terbaik: {winner}**  
    - Logistic Regression Accuracy: **{results['acc_lr']:.4f}** | AUC: **{results['auc_lr']:.4f}**  
    - Gaussian Naive Bayes Accuracy: **{results['acc_gnb']:.4f}** | AUC: **{results['auc_gnb']:.4f}**  

    Model disimpan menggunakan `logreg_model.pkl` dan `gnb_model.pkl` untuk keperluan deployment dan simulasi prediksi.
    """)


# ═══════════════════════════════════════════
# MENU 5 – SIMULASI PREDIKSI
# ═══════════════════════════════════════════
elif menu == "Simulasi Prediksi":
    st.markdown('<div class="section-header"><h2 style="margin:0;color:white">🤖 Simulasi Prediksi Segmen Nasabah</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "Masukkan data nasabah baru di bawah ini, lalu klik **Prediksi** untuk mengetahui segmen nasabah tersebut.")

    try:
        scaler_sim = joblib.load('scaler.pkl')
        logreg_sim = joblib.load('logreg_model.pkl')
        gnb_sim = joblib.load('gnb_model.pkl')
    except FileNotFoundError as e:
        st.error(
            f"❌ File model tidak ditemukan: {e}. Pastikan `scaler.pkl`, `logreg_model.pkl`, dan `gnb_model.pkl` ada di folder yang sama.")
        st.stop()

    # ── Input Form ──
    with st.container():
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("**💰 Informasi Saldo & Pembelian**")
            with st.container(border=True):
                balance = st.number_input(
                    "BALANCE – Sisa Saldo (Rp)", min_value=0.0, value=1000.0, step=100.0)
                balance_freq = st.slider("BALANCE_FREQUENCY", 0.0, 1.0, 1.0, 0.05,
                                         help="Seberapa sering saldo diperbarui (0–1)")
                purchases = st.number_input(
                    "PURCHASES – Total Pembelian (Rp)", min_value=0.0, value=500.0, step=100.0)
                oneoff_purch = st.number_input(
                    "ONEOFF_PURCHASES – Pembelian Sekali Bayar (Rp)", min_value=0.0, value=200.0, step=50.0)
                inst_purch = st.number_input(
                    "INSTALLMENTS_PURCHASES – Pembelian Cicilan (Rp)", min_value=0.0, value=300.0, step=50.0)
                purch_freq = st.slider(
                    "PURCHASES_FREQUENCY", 0.0, 1.0, 0.5, 0.05)
                oneoff_freq = st.slider(
                    "ONEOFF_PURCHASES_FREQUENCY", 0.0, 1.0, 0.2, 0.05)
                inst_freq = st.slider(
                    "PURCHASES_INSTALLMENTS_FREQUENCY", 0.0, 1.0, 0.4, 0.05)
                purch_trx = st.number_input(
                    "PURCHASES_TRX – Jumlah Transaksi Beli", min_value=0, value=5, step=1)

        with col_f2:
            st.markdown("**🏧 Informasi Tunai & Pembayaran**")
            with st.container(border=True):
                cash_adv = st.number_input(
                    "CASH_ADVANCE – Penarikan Tunai (Rp)", min_value=0.0, value=0.0, step=100.0)
                cash_adv_freq = st.slider(
                    "CASH_ADVANCE_FREQUENCY", 0.0, 1.0, 0.0, 0.05)
                cash_adv_trx = st.number_input(
                    "CASH_ADVANCE_TRX – Jumlah Transaksi Tarik Tunai", min_value=0, value=0, step=1)

                st.markdown("**💳 Informasi Kredit & Tagihan**")
                credit_limit = st.number_input(
                    "CREDIT_LIMIT – Batas Kredit (Rp)", min_value=0.0, value=3000.0, step=500.0)
                payments = st.number_input(
                    "PAYMENTS – Total Pembayaran (Rp)", min_value=0.0, value=800.0, step=100.0)
                min_payments = st.number_input(
                    "MINIMUM_PAYMENTS – Pembayaran Minimum (Rp)", min_value=0.0, value=200.0, step=50.0)
                prc_full = st.slider(
                    "PRC_FULL_PAYMENT – % Bayar Penuh", 0.0, 1.0, 0.1, 0.05)
                tenure = st.selectbox("TENURE – Lama Menjadi Nasabah (bulan)", [
                                      6, 7, 8, 9, 10, 11, 12])

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button(
            "🔍 Prediksi Segmen Nasabah", use_container_width=True, type="primary")

    if predict_btn:
        user_input = pd.DataFrame([{
            'BALANCE':                         balance,
            'BALANCE_FREQUENCY':               balance_freq,
            'PURCHASES':                       purchases,
            'ONEOFF_PURCHASES':                oneoff_purch,
            'INSTALLMENTS_PURCHASES':          inst_purch,
            'CASH_ADVANCE':                    cash_adv,
            'PURCHASES_FREQUENCY':             purch_freq,
            'ONEOFF_PURCHASES_FREQUENCY':      oneoff_freq,
            'PURCHASES_INSTALLMENTS_FREQUENCY': inst_freq,
            'CASH_ADVANCE_FREQUENCY':          cash_adv_freq,
            'CASH_ADVANCE_TRX':                cash_adv_trx,
            'PURCHASES_TRX':                   purch_trx,
            'CREDIT_LIMIT':                    credit_limit,
            'PAYMENTS':                        payments,
            'MINIMUM_PAYMENTS':                min_payments,
            'PRC_FULL_PAYMENT':                prc_full,
            'TENURE':                          tenure,
        }])

        user_scaled = scaler_sim.transform(user_input)
        pred_lr = logreg_sim.predict(user_scaled)[0]
        pred_gnb = gnb_sim.predict(user_scaled)[0]
        proba_lr = logreg_sim.predict_proba(user_scaled)[0]
        proba_gnb = gnb_sim.predict_proba(user_scaled)[0]

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        res_col1, res_col2 = st.columns(2)

        for res_col, pred, proba, model_name in [
            (res_col1, pred_lr,  proba_lr,  "Logistic Regression"),
            (res_col2, pred_gnb, proba_gnb, "Gaussian Naive Bayes"),
        ]:
            with res_col:
                st.markdown(f"**Model: {model_name}**")
                if pred == 0:
                    card_class = "cluster-card-0"
                    label = "🛒 Cluster 0 – Active Shoppers"
                    deskripsi = "Nasabah ini cenderung sering berbelanja menggunakan kartu kredit secara langsung dan sangat jarang melakukan penarikan uang tunai."
                else:
                    card_class = "cluster-card-1"
                    label = "🏧 Cluster 1 – Cash Advance Users"
                    deskripsi = "Nasabah ini jarang berbelanja namun sering menarik uang tunai menggunakan kartu kredit. Cenderung menyisakan saldo tagihan yang tinggi."

                st.markdown(f"""
                <div class="{card_class}">
                    <h4 style="margin-top:0">{label}</h4>
                    <p>{deskripsi}</p>
                    <hr>
                    <b>Probabilitas:</b><br>
                    &nbsp;&nbsp;• Cluster 0: <b>{proba[0]:.2%}</b><br>
                    &nbsp;&nbsp;• Cluster 1: <b>{proba[1]:.2%}</b>
                </div>
                """, unsafe_allow_html=True)

        # Gauge chart probabilitas
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "**Visualisasi Probabilitas Prediksi (Logistic Regression)**")
        fig_prob, ax_prob = plt.subplots(figsize=(6, 2.5))
        categories = ['Cluster 0\n(Active Shoppers)',
                      'Cluster 1\n(Cash Advance Users)']
        bar_c = ['#1565c0' if i == pred_lr else '#b0bec5' for i in range(2)]
        bars_prob = ax_prob.barh(categories, proba_lr,
                                 color=bar_c, height=0.4, edgecolor='white')
        for bar, val in zip(bars_prob, proba_lr):
            ax_prob.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                         f'{val:.2%}', va='center', fontweight='bold', fontsize=12)
        ax_prob.set_xlim(0, 1.1)
        ax_prob.set_xlabel('Probabilitas')
        ax_prob.set_title('Confidence Score Prediksi', fontweight='bold')
        ax_prob.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_prob)

        if pred_lr == 0:
            st.balloons()
        else:
            st.snow()
