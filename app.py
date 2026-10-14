import streamlit as st
import pandas as pd
import os

from regresi import train_and_evaluate, plot_actual_vs_pred, predict_likes

st.set_page_config(
    page_title="Dashboard Analisis Data X: Timnas Indonesia", 
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ Dashboard Analisis Data X: Timnas Indonesia")

st.sidebar.header("📌 Navigasi Menu")

menu_options = ["Home", "Data", "Sentimen", "Clustering", "Regresi", "Temuan & Keterbatasan"]
choice = st.sidebar.selectbox("Pilih Halaman:", menu_options)

@st.cache_data
def load_data():
    if os.path.exists('dataset_timnas.csv'):
        return pd.read_csv('dataset_timnas.csv')
    return None

df = load_data()

if choice == "Home":
    st.header("Beranda & Info Proyek")
    st.markdown("""
    Selamat datang di **Dashboard Analisis Data X (Twitter) mengenai Timnas Indonesia**.
    
    Aplikasi ini mengintegrasikan seluruh tahapan analisis data:
    1. **Data:** Pratinjau dataset hasil crawling & preprocessing.
    2. **Sentimen:** Distribusi sentimen dan evaluasi manual.
    3. **Clustering:** Pengelompokan topik teks menggunakan K-Means.
    4. **Regresi:** Estimasi likes berdasarkan jumlah kata dan hashtag.
    
    *Gunakan menu dropdown di sebelah kiri untuk berpindah halaman.*
    """)

elif choice == "Data":
    st.header("Pratinjau Dataset Utama")
    if df is not None:
        st.write(f"**Total Postingan:** {len(df)} data")
        st.dataframe(df)
        
        required_cols = ['text', 'created_at', 'like_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            st.success("Semua kolom wajib ('text', 'created_at', 'like_count') tersedia.")
        else:
            st.warning(f"Kolom wajib belum lengkap: {missing_cols}")
    else:
        st.error("Dataset 'dataset_timnas.csv' belum ditemukan. Harap unggah dataset dari Anggota 1.")

elif choice == "Sentimen":
    st.header("Analisis Sentimen")
    if os.path.exists('evaluasi_sentimen.csv'):
        df_sentimen = pd.read_csv('evaluasi_sentimen.csv')
        
        if 'sentiment' in df_sentimen.columns:
            st.subheader("Distribusi Sentimen")
            st.bar_chart(df_sentimen['sentiment'].value_counts())
            
        st.subheader("Tabel Evaluasi Manual (20 Sampel Data)")
        st.dataframe(df_sentimen)
    else:
        st.info("File 'evaluasi_sentimen.csv' milik Anggota 2 belum diunggah.")

elif choice == "Clustering":
    st.header("Clustering Data Teks (K-Means)")
    if os.path.exists('hasil_cluster.csv'):
        df_cluster = pd.read_csv('hasil_cluster.csv')
        st.subheader("Hasil Pengelompokan Topik")
        st.dataframe(df_cluster)
    else:
        st.info("File 'hasil_cluster.csv' milik Anggota 3 belum diunggah.")

elif choice == "Regresi":
    st.header("Regresi Linear: Estimasi Jumlah Likes")
    
    if df is not None and 'like_count' in df.columns and 'text' in df.columns:
        model, mae, r2, y_test, y_pred = train_and_evaluate(df)
        
        col1, col2 = st.columns(2)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        st.subheader("Grafik Perbandingan Likes Aktual vs Prediksi")
        st.pyplot(plot_actual_vs_pred(y_test, y_pred))
        
        st.markdown("---")
        st.subheader("Formulir Estimasi Likes")
        with st.form("prediction_form"):
            user_words = st.number_input("Jumlah Kata (Word Count)", min_value=1, value=10)
            user_hashtags = st.number_input("Jumlah Hashtag", min_value=0, value=1)
            submit_button = st.form_submit_button("Hitung Estimasi Likes")
            
            if submit_button:
                pred = predict_likes(model, user_words, user_hashtags)
                st.success(f"Estimasi Jumlah Likes: **{max(0, pred):.1f}**")
                if pred < 0:
                    st.warning("Estimasi bernilai negatif menunjukkan keterbatasan model regresi linear.")

        with st.expander("Penjelasan Evaluasi & Keterbatasan Model"):
            st.write("""
            - **MAE:** Menunjukkan rata-rata selisih mutlak antara jumlah likes aktual dengan hasil prediksi.
            - **R²:** Menunjukkan seberapa besar variansi likes yang dapat dijelaskan oleh jumlah kata dan hashtag.
            - **Keterbatasan:** Jumlah likes di X lebih banyak dipengaruhi oleh faktor visual, waktu posting, dan jangkauan akun.
            """)
    else:
        st.error("Dataset 'dataset_timnas.csv' atau kolom 'like_count' belum siap.")

elif choice == "Temuan & Keterbatasan":
    st.header("Temuan Utama & Keterbatasan Proyek")
    
    st.subheader("3 Temuan Utama:")
    st.markdown("""
    1. Kebanyakan pembicaraan mengenai Timnas Indonesia didominasi oleh sentimen positif.
    2. Pembahasan publik terbagi ke dalam 3 kelompok topik utama.
    3. Jumlah kata dan hashtag memberikan pengaruh minim terhadap jumlah likes.
    """)
    
    st.subheader("2 Keterbatasan Data/Metode:")
    st.markdown("""
    1. Pengambilan sampel data terbatas pada rentang waktu tertentu.
    2. Model regresi linear belum memperhitungkan variabel pendukung seperti jenis media dan waktu publikasi.
    """)