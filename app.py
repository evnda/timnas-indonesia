import streamlit as st
import pandas as pd
import os
import sys
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul buatan anggota kelompok
from regresi import train_and_evaluate, plot_actual_vs_pred, predict_likes
from wordcloud import WordCloud

SENTIMEN_PATH = os.path.join(
    os.path.dirname(__file__),
    'Timnas_Indonesia',
    'analisis-sentiment',
    'sentimen.py',
)
sentimen_spec = importlib.util.spec_from_file_location('analisis_sentimen', SENTIMEN_PATH)
sentimen_module = importlib.util.module_from_spec(sentimen_spec) if sentimen_spec else None
if sentimen_spec and sentimen_spec.loader and sentimen_module:
    sys.modules[sentimen_spec.name] = sentimen_module
    sentimen_spec.loader.exec_module(sentimen_module)

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
    if sentimen_module:
        output_path = os.path.join(os.path.dirname(SENTIMEN_PATH), 'output')
        sentimen_module.halaman_sentimen(out_dir=output_path)
    else:
        st.error(f"Modul analisis sentimen tidak ditemukan: {SENTIMEN_PATH}")

# 4. HALAMAN CLUSTERING (K-Means)

elif choice == "Clustering":
    st.header("Pengelompokan Pembahasan Teks (K-Means Clustering)")

    # Membaca file hasil clustering
    cluster_path = "hasil_cluster.csv"
    if not os.path.exists(cluster_path) and os.path.exists(
        "Timnas_Indonesia/hasil_cluster.csv"
    ):
        cluster_path = "Timnas_Indonesia/hasil_cluster.csv"

    if os.path.exists(cluster_path):
        df_cluster = pd.read_csv(cluster_path)

        if "cluster" not in df_cluster.columns:
            st.error(
                "Kolom 'cluster' tidak ditemukan dalam file 'hasil_cluster.csv'."
            )
        else:
            # Pastikan label cluster berupa angka murni
            df_cluster["cluster"] = pd.to_numeric(
                df_cluster["cluster"], errors="coerce"
            )
            df_cluster = df_cluster.dropna(subset=["cluster"]).copy()
            df_cluster["cluster"] = df_cluster["cluster"].astype(int)

            if df_cluster.empty:
                st.warning("Tidak ada data cluster yang valid.")
            else:
                # 1. Grafik Distribusi Cluster
                st.subheader("📊 Distribusi Jumlah Data per Cluster")
                cluster_counts = (
                    df_cluster["cluster"].value_counts().sort_index()
                )

                fig, ax = plt.subplots(figsize=(7, 3.5))
                sns.barplot(
                    x=cluster_counts.index.astype(str),
                    y=cluster_counts.values,
                    color="#d95f8d",
                    ax=ax,
                )
                ax.set_xlabel("ID Cluster")
                ax.set_ylabel("Jumlah Postingan")
                ax.set_title("Distribusi Postingan per Cluster")
                st.pyplot(fig)
                plt.close(fig)

                # 2. Interaktif WordCloud per Cluster
                st.divider()
                st.subheader("☁️ Word Cloud per Cluster")

                daftar_cluster = sorted(df_cluster["cluster"].unique().tolist())
                cluster_terpilih = st.selectbox(
                    "Pilih cluster yang ingin ditampilkan:",
                    options=daftar_cluster,
                    format_func=lambda x: f"Cluster {x + 1}",
                )

                df_pilihan = df_cluster[
                    df_cluster["cluster"] == cluster_terpilih
                ].copy()
                st.write(
                    f"Jumlah postingan pada Cluster {cluster_terpilih + 1}: "
                    f"**{len(df_pilihan)} postingan**"
                )

                # Menentukan kolom teks
                if "text" in df_pilihan.columns:
                    kolom_teks = "text"
                elif "text_raw" in df_pilihan.columns:
                    kolom_teks = "text_raw"
                else:
                    kolom_teks = None

                if kolom_teks is None:
                    st.warning(
                        "Kolom 'text' atau 'text_raw' tidak ditemukan dalam hasil clustering."
                    )
                else:
                    teks_cluster = (
                        df_pilihan[kolom_teks]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )
                    teks_cluster = teks_cluster[teks_cluster != ""]
                    teks_gabungan = " ".join(teks_cluster.tolist())

                    if not teks_gabungan:
                        st.warning(
                            "Tidak ada teks yang dapat digunakan untuk membuat Word Cloud."
                        )
                    else:
                        stopwords_indonesia = {
                            "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk",
                            "dengan", "pada", "dalam", "karena", "jadi", "akan", "ada",
                            "atau", "juga", "sudah", "belum", "masih", "saat", "saja",
                            "lebih", "bisa", "aku", "saya", "kamu", "dia", "mereka",
                            "kita", "kami", "nya", "lah", "kan", "nih", "sih", "aja",
                            "yg", "dgn", "buat", "kalau", "kalo", "tapi", "namun",
                            "the", "and", "is", "are", "of", "to", "in", "on", "for",
                            "with", "at", "by", "from",
                        }

                        try:
                            wordcloud = WordCloud(
                                width=1200,
                                height=600,
                                background_color="white",
                                stopwords=stopwords_indonesia,
                                max_words=100,
                                collocations=False,
                                random_state=42,
                            ).generate(teks_gabungan)

                            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                            ax_wc.imshow(
                                wordcloud, interpolation="bilinear"
                            )
                            ax_wc.axis("off")
                            ax_wc.set_title(
                                f"Word Cloud Cluster {cluster_terpilih + 1}"
                            )

                            st.pyplot(fig_wc)
                            plt.close(fig_wc)

                        except ValueError:
                            st.warning(
                                "Word Cloud tidak dapat dibuat. Periksa apakah teks cluster mengandung kata yang dapat diproses."
                            )

                # 3. Kata Dominan Cluster
                st.divider()
                st.subheader("🔤 Kata Dominan Cluster")
                if "kata_dominan" in df_cluster.columns:
                    kata_cluster = (
                        df_pilihan["kata_dominan"].dropna().astype(str)
                    )
                    if not kata_cluster.empty:
                        st.write(kata_cluster.iloc[0])

                # 4. Tabel Hasil Clustering
                st.divider()
                st.subheader("📑 Tabel Hasil Clustering")
                st.dataframe(df_cluster, use_container_width=True)

    else:
        st.info(
            "File 'hasil_cluster.csv' dari Anggota 3 belum ditemukan. Periksa lokasi file tersebut."
        )

# 5. HALAMAN REGRESI (Bagian Anggota 4)

elif choice == "Regresi":
    st.header("Regresi Linear: Estimasi Jumlah Likes")
    
    # Memastikan dataset utama dan kolom wajib tersedia
    if df is not None and 'like_count' in df.columns and 'text' in df.columns:
        # Melatih dan mengevaluasi model regresi dari regresi.py
        model, mae, r2, y_test, y_pred = train_and_evaluate(df)
        
        # 1. Menampilkan Metrik Evaluasi Utama
        col1, col2 = st.columns(2)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        
        # 2. Grafik Perbandingan Likes
        st.subheader("📈 Grafik Perbandingan Likes Aktual vs Prediksi")
        st.pyplot(plot_actual_vs_pred(y_test, y_pred))
        
        st.markdown("---")
        
        # 3. Formulir Estimasi Likes Interaktif
        st.subheader("🧮 Formulir Interaktif Estimasi Likes")
        with st.form("prediction_form"):
            user_words = st.number_input("Input Jumlah Kata (Word Count):", min_value=1, value=12)
            user_hashtags = st.number_input("Input Jumlah Hashtag:", min_value=0, value=2)
            submit_button = st.form_submit_button("Hitung Estimasi Likes")
            
            if submit_button:
                pred = predict_likes(model, user_words, user_hashtags)
                
                # Penanganan hasil estimasi negatif sesuai ketentuan tugas
                if pred < 0:
                    st.warning(f"Hasil Estimasi Mentah: **{pred:.2f}** (Secara Fisik Dibulatkan ke **0 Likes**)")
                    st.info("💡 **Penjelasan Estimasi Negatif:** Model regresi linear dapat menghasilkan nilai negatif secara matematis. Dalam konteks jumlah likes, nilai negatif ini tidak bermakna fisik dan menunjukkan keterbatasan garis linieritas model.")
                else:
                    st.success(f"Estimasi Jumlah Likes: **{pred:.1f} Likes**")

        st.markdown("---")
        
        # 4. Interpretasi Resmi Hasil Model Regresi (Wajib Dipresentasikan)
        st.subheader("📌 Interpretasi & Evaluasi Model Regresi")
        st.markdown(f"""
        * **Makna Nilai MAE ({mae:.2f}):** Secara rata-rata, prediksi model memiliki simpangan atau kesalahan sebesar **{mae:.2f} likes** dari jumlah likes aktual pada data uji.
        * **Makna Nilai R² ({r2:.4f}):** Variabel `word_count` dan `hashtag_count` hanya mampu menjelaskan sekitar **{max(0, r2*100):.2f}%** dari variasi jumlah likes (sisanya dipengaruhi faktor luar).
        * **Keterbatasan Model:** Jumlah likes di platform X (Twitter) tidak hanya bergantung pada panjang teks atau jumlah hashtag, melainkan lebih banyak dipengaruhi oleh **isi materi visual (foto/video), reputasi/jangkauan akun, waktu unggah (prime time), serta algoritma jangkauan platform**.
        """)
    else:
        st.error("Dataset 'dataset_timnas.csv' atau kolom wajib ('text', 'like_count') belum siap untuk diolah pada modul regresi.")

# 6. HALAMAN TEMUAN & KETERBATASAN

elif choice == "Temuan & Keterbatasan":
    st.header("Temuan Utama Analisis & Keterbatasan Proyek")
    
    st.subheader("💡 3 Temuan Utama Analisis Data:")
    st.markdown("""
    1. **Dominasi Sentimen:** Perbincangan masyarakat di X mengenai Timnas Indonesia didominasi oleh respons netral.
    2. **Pengelompokan Topik:** Diskusi publik terpolarisasi ke dalam beberapa topik utama (seperti performa pemain, taktik pelatih, dan jadwal pertandingan).
    3. **Pengaruh Karakter Teks:** Jumlah kata dan hashtag tidak memiliki korelasi terhadap jumlah likes pada postingan.
    """)
    
    st.subheader("⚠️ 2 Keterbatasan Analisis / Metode:")
    st.markdown("""
    1. **Cakupan Sampel Data:** Pengambilan sampel data terbatas pada periode crawling tertentu.
    2. **Kompleksitas Variabel Engagement:** Model regresi linear sederhana belum mempertimbangkan variabel gambar, video, jam tayang, serta jumlah pengikut akun.
    """)