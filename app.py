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

st.sidebar.header("Navigasi")
menu_options = ["Data", "Sentimen", "Clustering", "Regresi"]
choice = st.sidebar.selectbox("Pilih Menu:", menu_options)

@st.cache_data
def load_data():
    if os.path.exists('dataset_timnas.csv'):
        return pd.read_csv('dataset_timnas.csv')
    return None

df = load_data()

if choice == "Data":
    st.header("Pratinjau Dataset")
    if df is not None:
        st.write(f"Total Postingan: {len(df)}")
        st.dataframe(df)
    else:
        st.error("Dataset 'dataset_timnas.csv' belum tersedia.")

elif choice == "Sentimen":
    st.header("Analisis Sentimen")
    if os.path.exists('evaluasi_sentimen.csv'):
        st.dataframe(pd.read_csv('evaluasi_sentimen.csv'))
    else:
        st.info("File 'evaluasi_sentimen.csv' belum diunggah.")

elif choice == "Clustering":
    st.header("Clustering Data Teks")
    if os.path.exists('hasil_cluster.csv'):
        st.dataframe(pd.read_csv('hasil_cluster.csv'))
    else:
        st.info("File 'hasil_cluster.csv' belum diunggah.")

elif choice == "Regresi":
    st.header("Regresi Linear: Estimasi Jumlah Likes")
    
    if df is not None and 'like_count' in df.columns:
        model, mae, r2, y_test, y_pred = train_and_evaluate(df)
        
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        st.subheader("Grafik Likes Aktual vs Prediksi")
        st.pyplot(plot_actual_vs_pred(y_test, y_pred))
        
        st.subheader("Estimasi Likes")
        with st.form("prediction_form"):
            user_words = st.number_input("Jumlah Kata", min_value=1, value=10)
            user_hashtags = st.number_input("Jumlah Hashtag", min_value=0, value=1)
            submit_button = st.form_submit_button("Hitung Estimasi")
            
            if submit_button:
                pred = predict_likes(model, user_words, user_hashtags)
                st.success(f"Estimasi Jumlah Likes: {max(0, pred):.1f}")
                if pred < 0:
                    st.warning("Estimasi bernilai negatif menunjukkan keterbatasan model regresi linear.")

        with st.expander("Evaluasi & Keterbatasan Model"):
            st.write("""
            - MAE menunjukkan rata-rata selisih mutlak antara jumlah likes aktual dan hasil prediksi.
            - R² menunjukkan seberapa besar variasi likes yang dapat dijelaskan oleh jumlah kata dan hashtag.
            - Keterbatasan: Jumlah likes di X lebih banyak dipengaruhi oleh faktor visual, waktu posting, dan jangkauan akun.
            """)
    else:
        st.error("File 'dataset_timnas.csv' atau kolom 'like_count' belum siap.")

st.markdown("---")
st.subheader("Temuan Utama & Keterbatasan Proyek")
st.markdown("""
**3 Temuan Utama:**
1. Kebanyakan pembicaraan mengenai Timnas Indonesia didominasi oleh sentimen positif.
2. Pembahasan publik terbagi ke dalam 3 kelompok utama.
3. Jumlah kata dan hashtag hanya memberikan pengaruh kecil terhadap jumlah likes.

**2 Keterbatasan Data/Metode:**
1. Pengambilan data hanya mencakup sampel pada rentang waktu tertentu.
2. Model regresi linear belum memperhitungkan variabel pendukung seperti tipe media dan waktu publikasi.
""")