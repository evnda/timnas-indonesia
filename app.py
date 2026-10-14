import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul buatan anggota kelompok
from regresi import train_and_evaluate, plot_actual_vs_pred, predict_likes

st.set_page_config(
    page_title="Dashboard Analisis Data X: Timnas Indonesia", 
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ Dashboard Analisis Data X: Timnas Indonesia")

# Sidebar - Navigasi Menu & File Uploader
st.sidebar.header("📌 Navigasi Menu")

menu_options = ["Home", "Data", "Sentimen", "Clustering", "Regresi", "Temuan & Keterbatasan"]
choice = st.sidebar.selectbox("Pilih Halaman:", menu_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Unggah Dataset (Opsional)")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV baru (misal: dataset_timnas.csv)", type=["csv"])

# Fungsi pencarian dataset otomatis
@st.cache_data
def load_dataset(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    
    # Mencari lokasi file bawaan
    possible_paths = [
        'dataset_timnas.csv', 
        'Timnas_Indonesia/dataset_timnas.csv',
        'data mentah crawling/dataset_timnas.csv'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df = load_dataset(uploaded_file)

# 1. HALAMAN HOME

if choice == "Home":
    st.header("Beranda & Informasi Proyek")
    st.markdown("""
    Selamat datang di **Dashboard Analisis Data X (Twitter) mengenai Timnas Indonesia**.
    
    Aplikasi ini mengintegrasikan seluruh tahapan pemrosesan data kelompok:
    1. **Data:** Pratinjau dataset hasil crawling, pembersihan data, dan fitur unggah CSV secara langsung.
    2. **Sentimen:** Visualisasi distribusi sentimen (Positif, Negatif, Netral) dan tabel evaluasi manual.
    3. **Clustering:** Pengelompokan topik teks perbincangan publik menggunakan algoritma K-Means.
    4. **Regresi:** Pemodelan estimasi *likes* berdasarkan jumlah kata dan hashtag.
    
    ---
    *Gunakan menu dropdown di sidebar sebelah kiri untuk berpindah halaman atau mengunggah dataset alternatif.*
    """)

# 2. HALAMAN DATA

elif choice == "Data":
    st.header("Pratinjau & Manajemen Dataset Utama")
    
    if df is not None:
        st.success("✅ Dataset berhasil dimuat ke dalam aplikasi.")
        st.write(f"**Total Postingan (Baris Data):** {len(df)} data")
        st.dataframe(df, use_container_width=True)
        
        # Check kolom wajib sesuai rubrik
        required_cols = ['text', 'created_at', 'like_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            st.success("✔ Seluruh kolom wajib ('text', 'created_at', 'like_count') tersedia.")
        else:
            st.warning(f"⚠️ Kolom wajib belum lengkap. Kolom hilang: {missing_cols}")
    else:
        st.error("❌ Dataset 'dataset_timnas.csv' belum ditemukan. Silakan unggah file CSV melalui menu di sidebar sebelah kiri.")

# 3. HALAMAN SENTIMEN (Lengkap dengan Word Cloud)
elif choice == "Sentimen":
    st.header("Analisis Sentimen & Distribusi Teks")
    
    # Mencari lokasi file evaluasi sentimen buatan Anggota 2
    sentiment_path = 'evaluasi_sentimen.csv'
    if not os.path.exists(sentiment_path) and os.path.exists('Timnas_Indonesia/analisis-sentiment/output/evaluasi_sentimen.csv'):
        sentiment_path = 'Timnas_Indonesia/analisis-sentiment/output/evaluasi_sentimen.csv'
        
    if os.path.exists(sentiment_path):
        df_sentimen = pd.read_csv(sentiment_path)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Grafik Distribusi Sentimen")
            
            # Cek nama kolom label sentimen (sentiment atau label_manual)
            if 'sentiment' in df_sentimen.columns:
                counts = df_sentimen['sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
                ax.set_ylabel("Jumlah Tweet")
                st.pyplot(fig)
            elif 'label_manual' in df_sentimen.columns:
                counts = df_sentimen['label_manual'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
                ax.set_ylabel("Jumlah Tweet")
                st.pyplot(fig)
            else:
                st.info("Kolom 'sentiment' atau 'label_manual' tidak ditemukan pada CSV.")
                
        with col2:
            st.subheader("☁️ Word Cloud (Awan Kata)")
            try:
                from wordcloud import WordCloud
                
                # Mendeteksi nama kolom teks (teks, text, atau full_text)
                text_col = None
                for col in ['teks', 'text', 'full_text']:
                    if col in df_sentimen.columns:
                        text_col = col
                        break
                
                if text_col:
                    text_combined = " ".join(df_sentimen[text_col].dropna().astype(str))
                    wc = WordCloud(width=600, height=400, background_color='black').generate(text_combined)
                    fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.info("Kolom teks tidak ditemukan untuk Word Cloud.")
            except Exception as e:
                st.info("Visualisasi Word Cloud siap ditampilkan (jalankan 'pip install wordcloud' di terminal).")
                
        st.markdown("---")
        st.subheader("📋 Tabel Evaluasi Label Manual (20 Sampel Data)")
        st.dataframe(df_sentimen, use_container_width=True)
    else:
        st.info("File 'evaluasi_sentimen.csv' dari Anggota 2 belum diunggah.")

# 4. HALAMAN CLUSTERING (K-Means)

elif choice == "Clustering":
    st.header("Pengelompokan Pembahasan Teks (K-Means Clustering)")
    
    cluster_path = 'hasil_cluster.csv'
    if not os.path.exists(cluster_path) and os.path.exists('Timnas_Indonesia/hasil_cluster.csv'):
        cluster_path = 'Timnas_Indonesia/hasil_cluster.csv'
        
    if os.path.exists(cluster_path):
        df_cluster = pd.read_csv(cluster_path)
        
        if 'cluster' in df_cluster.columns:
            st.subheader("📊 Distribusi Jumlah Data per Cluster Topik")
            cluster_counts = df_cluster['cluster'].value_counts()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='magma', ax=ax)
            ax.set_xlabel("ID Cluster Topik")
            ax.set_ylabel("Jumlah Postingan")
            st.pyplot(fig)
            
        st.subheader("📑 Tabel Hasil Clustering")
        st.dataframe(df_cluster, use_container_width=True)
    else:
        st.info("File 'hasil_cluster.csv' dari Anggota 3 belum diunggah.")

# 5. HALAMAN REGRESI (Bagian Anggota 4)

elif choice == "Regresi":
    st.header("Regresi Linear: Estimasi Jumlah Likes")
    
    if df is not None and 'like_count' in df.columns and 'text' in df.columns:
        model, mae, r2, y_test, y_pred = train_and_evaluate(df)
        
        col1, col2 = st.columns(2)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        st.subheader("📈 Grafik Perbandingan Likes Aktual vs Prediksi")
        st.pyplot(plot_actual_vs_pred(y_test, y_pred))
        
        st.markdown("---")
        st.subheader("🧮 Formulir Interaktif Estimasi Likes")
        with st.form("prediction_form"):
            user_words = st.number_input("Input Jumlah Kata (Word Count):", min_value=1, value=12)
            user_hashtags = st.number_input("Input Jumlah Hashtag:", min_value=0, value=2)
            submit_button = st.form_submit_button("Hitung Estimasi Likes")
            
            if submit_button:
                pred = predict_likes(model, user_words, user_hashtags)
                st.success(f"Estimasi Jumlah Likes: **{max(0, pred):.1f}**")
                if pred < 0:
                    st.warning("Catatan: Hasil estimasi mentah bernilai negatif karena batas linieritas model.")

        with st.expander("📌 Penjelasan Evaluasi & Keterbatasan Model Regresi"):
            st.write("""
            - **MAE:** Mengukur rata-rata selisih mutlak antara jumlah likes asli dengan hasil estimasi.
            - **R² Score:** Menunjukkan seberapa besar variansi likes yang dapat dijelaskan oleh jumlah kata dan hashtag.
            - **Keterbatasan:** Jumlah likes di platform sosial media lebih dominan dipengaruhi oleh faktor visual, algoritma jangkauan, dan popularitas akun.
            """)
    else:
        st.error("Dataset 'dataset_timnas.csv' atau kolom 'like_count' belum siap untuk regresi.")

# 6. HALAMAN TEMUAN & KETERBATASAN

elif choice == "Temuan & Keterbatasan":
    st.header("Temuan Utama Analisis & Keterbatasan Proyek")
    
    st.subheader("💡 3 Temuan Utama Analisis Data:")
    st.markdown("""
    1. **Dominasi Sentimen:** Perbincangan masyarakat di X mengenai Timnas Indonesia didominasi oleh respons positif dan antusiasme dukungan.
    2. **Pengelompokan Topik:** Diskusi publik terpolarisasi ke dalam beberapa topik utama (seperti performa pemain, taktik pelatih, dan jadwal pertandingan).
    3. **Pengaruh Karakter Teks:** Jumlah kata dan hashtag memiliki korelasi yang sangat kecil terhadap jumlah likes pada postingan.
    """)
    
    st.subheader("⚠️ 2 Keterbatasan Analisis / Metode:")
    st.markdown("""
    1. **Cakupan Sampel Data:** Pengambilan sampel data terbatas pada periode crawling tertentu.
    2. **Kompleksitas Variabel Engagement:** Model regresi linear sederhana belum mempertimbangkan variabel gambar, video, jam tayang, serta jumlah pengikut akun.
    """)