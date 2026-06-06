
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Credit Card Clustering", page_icon="💳", layout="wide")

st.title("Dashboard Prediksi Tipe Pengguna Kartu Kredit")
st.markdown("Aplikasi ini menggunakan model **K-Means Clustering** dan **Logistic Regression** untuk memprediksi kebiasaan transaksi nasabah berdasarkan data historis kartu kredit.")
# st.markdown("Tugas Besar Data Mining Kelompok 6")
# st.markdown("Muhammad Daffa Izzati    - 102022430036")
# st.markdown("Kevin Muhammad Althaf    - 102022400325")
# st.markdown("Muhammad Daffa Reifansya - 102022400329")
# st.markdown("Sekar Erliana Putri      - 102022430069")
st.divider()

tab1, tab2 = st.tabs(["Form Prediksi", "Visualisasi Kinerja Model"])

with tab1:
    st.header("Masukkan Data Nasabah Baru")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informasi Saldo & Pembelian")
        with st.container(border=True):
            balance = st.number_input("Jumlah Sisa Saldo (BALANCE)", min_value=0.0, value=1000.0, step=100.0)
            balance_frequency = st.slider("Frekuensi Saldo (BALANCE_FREQUENCY)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
            purchases = st.number_input("Total Pembelian (PURCHASES)", min_value=0.0, value=500.0, step=100.0)
            oneoff_purchases = st.number_input("Pembelian Sekali Bayar (ONEOFF)", min_value=0.0, value=200.0, step=50.0)
            installments_purchases = st.number_input("Pembelian Cicilan (INSTALLMENTS)", min_value=0.0, value=300.0, step=50.0)

            st.markdown("**Intensitas Pembelian**")
            purchases_frequency = st.slider("Frekuensi Pembelian", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            oneoff_purchases_frequency = st.slider("Frekuensi Pembelian Sekali Bayar", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            purchases_installments_frequency = st.slider("Frekuensi Pembelian Cicilan", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            purchases_trx = st.number_input("Jumlah Transaksi Pembelian (TRX)", min_value=0, value=5, step=1)

    with col2:
        st.subheader("Informasi Tunai & Pembayaran")
        with st.container(border=True):
            cash_advance = st.number_input("Penarikan Tunai (CASH_ADVANCE)", min_value=0.0, value=0.0, step=100.0)
            cash_advance_frequency = st.slider("Frekuensi Penarikan Tunai", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
            cash_advance_trx = st.number_input("Jumlah Transaksi Penarikan (TRX)", min_value=0, value=0, step=1)

            st.markdown("**Informasi Kredit & Tagihan**")
            credit_limit = st.number_input("Batas Kredit (CREDIT_LIMIT)", min_value=0.0, value=3000.0, step=500.0)
            payments = st.number_input("Jumlah Pembayaran (PAYMENTS)", min_value=0.0, value=800.0, step=100.0)
            minimum_payments = st.number_input("Pembayaran Minimum (MIN_PAYMENTS)", min_value=0.0, value=200.0, step=50.0)
            prc_full_payment = st.slider("Persentase Pembayaran Penuh", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            tenure = st.selectbox("Pilih Tenure:", [6, 7, 8, 9, 10, 11, 12])

    st.write("")
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        submit_button = st.button("Prediksi Cluster Nasabah", use_container_width=True)

    if submit_button:
        user_input = pd.DataFrame({
            'BALANCE': [balance],
            'BALANCE_FREQUENCY': [balance_frequency],
            'PURCHASES': [purchases],
            'ONEOFF_PURCHASES': [oneoff_purchases],
            'INSTALLMENTS_PURCHASES': [installments_purchases],
            'CASH_ADVANCE': [cash_advance],
            'PURCHASES_FREQUENCY': [purchases_frequency],
            'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
            'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
            'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
            'CASH_ADVANCE_TRX': [cash_advance_trx],
            'PURCHASES_TRX': [purchases_trx],
            'CREDIT_LIMIT': [credit_limit],
            'PAYMENTS': [payments],
            'MINIMUM_PAYMENTS': [minimum_payments],
            'PRC_FULL_PAYMENT': [prc_full_payment],
            'TENURE': [tenure]
        })

        try:
            scaler = joblib.load('scaler.pkl')
            logreg_model = joblib.load('logreg_model.pkl')

            user_input_scaled = scaler.transform(user_input)
            prediction = logreg_model.predict(user_input_scaled)
            hasil_prediksi = prediction[0]

            if hasil_prediksi == 0:
                nama_cluster = "Cluster 0: Pengguna Aktif Pembelian (Active Shoppers) 🛒"
                deskripsi = "Pengguna ini memiliki kecenderungan untuk sering menggunakan kartu kredit mereka secara langsung untuk berbelanja (transaksi tinggi) dan sangat jarang melakukan penarikan uang tunai."
            elif hasil_prediksi == 1:
                nama_cluster = "Cluster 1: Pengguna Tarik Tunai (Cash Advance Users) 🏧"
                deskripsi = "Pengguna ini jarang berbelanja menggunakan kartu kredit, namun sering menggunakannya untuk menarik uang tunai. Mereka cenderung menyisakan saldo tagihan yang tinggi."
            else:
                nama_cluster = f"Cluster {hasil_prediksi}"
                deskripsi = ""

            st.write("---")
            st.subheader("📈 Hasil Analisis")
            st.success("**Nasabah ini masuk ke dalam kategori:**")
            st.metric(label="Prediksi Logistic Regression", value=nama_cluster)
            st.info(f"**Karakteristik:** {deskripsi}")
            st.balloons()

        except FileNotFoundError:
            st.error("⚠️ File `scaler.pkl` atau `logreg_model.pkl` tidak ditemukan.")

with tab2:
    st.header("Visualisasi Kinerja Model")

    model = joblib.load('logreg_model.pkl')
    scaler_viz = joblib.load('scaler.pkl')

    df = pd.read_csv('CC GENERAL.csv')

    # Data Preparation
    df_viz = df.drop(columns=['CUST_ID'])
    df_viz['MINIMUM_PAYMENTS'] = df_viz['MINIMUM_PAYMENTS'].fillna(df_viz['MINIMUM_PAYMENTS'].median())
    df_viz['CREDIT_LIMIT'] = df_viz['CREDIT_LIMIT'].fillna(df_viz['CREDIT_LIMIT'].median())

    # Outlier handling (IQR capping)
    for col in df_viz.columns:
        Q1 = df_viz[col].quantile(0.25)
        Q3 = df_viz[col].quantile(0.75)
        IQR = Q3 - Q1
        df_viz[col] = df_viz[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    df_scaled_viz = pd.DataFrame(scaler_viz.transform(df_viz), columns=df_viz.columns)

    # KMeans
    st.subheader("1. Visualisasi Cluster (Live Plot)")
    pca = PCA(n_components=2)
    X_pca_viz = pca.fit_transform(df_scaled_viz)
    k_means_viz = KMeans(n_clusters=2, random_state=42)
    clusters_viz = k_means_viz.fit_predict(X_pca_viz)

    fig, ax = plt.subplots(figsize=(7, 4))  
    ax.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=clusters_viz, cmap='plasma', s=50, alpha=0.7)
    centroids = k_means_viz.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    ax.set_title('Hasil Segmentasi Nasabah Menggunakan K-Means', fontsize=14)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False) 

    st.divider()

    # Logistic Regression Evaluation
    st.subheader("2. Evaluasi Model & ROC Curve")

    df_viz['Kategori_Cluster'] = clusters_viz
    X_eval = df_viz.drop(columns='Kategori_Cluster')
    y_eval = df_viz['Kategori_Cluster']

    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
    X_train_e = scaler_viz.transform(X_train_e)
    X_test_e = scaler_viz.transform(X_test_e)

    model.fit(X_train_e, y_train_e)
    y_pred_e = model.predict(X_test_e)
    y_proba_e = model.predict_proba(X_test_e)[:, 1]

    acc_score = accuracy_score(y_test_e, y_pred_e)
    class_report = classification_report(y_test_e, y_pred_e, output_dict=True)
    conf_matrix = confusion_matrix(y_test_e, y_pred_e)
    fpr, tpr, _ = roc_curve(y_test_e, y_proba_e)
    roc_auc = auc(fpr, tpr)

    st.markdown("**A. Skor Performa Model**")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label="Accuracy Score", value=f"{acc_score:.4f}")
    col_m2.metric(label="ROC AUC Score", value=f"{roc_auc:.4f}")

    st.markdown("**B. Classification Report**")
    df_report = pd.DataFrame(class_report).transpose()
    st.dataframe(df_report.style.format("{:.4f}"), use_container_width=True)

    st.markdown("**C. Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))  
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    st.pyplot(fig_cm, use_container_width=False)  

    st.markdown("**D. ROC Curve**")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))  
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc, use_container_width=False)  
