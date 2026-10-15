from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ==========================================
# 1. MEMBACA HASIL CLUSTERING
# ==========================================

folder = Path(__file__).resolve().parent
file_cluster = folder / "hasil_cluster.csv"

df = pd.read_csv(file_cluster)

print("=" * 50)
print("WORD CLOUD CLUSTERING TIMNAS INDONESIA")
print("=" * 50)

print("Jumlah postingan:", len(df))


# ==========================================
# 2. MEMERIKSA KOLOM DATA
# ==========================================

if "cluster" not in df.columns:
    raise ValueError("Kolom 'cluster' tidak ditemukan.")

if "text" in df.columns:
    kolom_teks = "text"
elif "text_raw" in df.columns:
    kolom_teks = "text_raw"
else:
    raise ValueError(
        "Kolom 'text' atau 'text_raw' tidak ditemukan."
    )

df[kolom_teks] = df[kolom_teks].fillna("").astype(str)


# ==========================================
# 3. STOPWORDS
# ==========================================

stopwords_indonesia = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "untuk", "dengan", "pada", "dalam", "karena",
    "jadi", "akan", "ada", "atau", "juga", "sudah",
    "belum", "masih", "saat", "saja", "lebih", "bisa",
    "aku", "saya", "kamu", "dia", "mereka", "kita",
    "kami", "nya", "lah", "kan", "nih", "sih", "aja",
    "yg", "dgn", "buat", "tentang", "ketika", "kalau",
    "kalo", "tapi", "namun", "the", "and", "is", "are",
    "of", "to", "in", "on", "for", "with", "at", "by",
    "from"
}


# ==========================================
# 4. MENYIAPKAN FOLDER OUTPUT
# ==========================================

folder_output = folder / "wordcloud"
folder_output.mkdir(exist_ok=True)

jumlah_cluster = sorted(df["cluster"].dropna().unique())

print("\nJumlah cluster:", len(jumlah_cluster))


# ==========================================
# 5. MEMBUAT WORD CLOUD PER CLUSTER
# ==========================================

for cluster_id in jumlah_cluster:

    data_cluster = df[df["cluster"] == cluster_id]

    teks_cluster = " ".join(
        data_cluster[kolom_teks]
        .astype(str)
        .tolist()
    )

    if not teks_cluster.strip():
        print(f"Cluster {cluster_id} tidak memiliki teks.")
        continue

    try:
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            stopwords=stopwords_indonesia,
            max_words=100,
            collocations=False,
            random_state=42
        ).generate(teks_cluster)

    except ValueError:
        print(
            f"Cluster {cluster_id} tidak memiliki "
            "kata yang dapat divisualisasikan."
        )
        continue

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Word Cloud Cluster {cluster_id + 1}",
        fontsize=18
    )

    fig.tight_layout()

    file_gambar = (
        folder_output /
        f"wordcloud_cluster_{cluster_id + 1}.png"
    )

    fig.savefig(
        file_gambar,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"Word Cloud tersimpan: {file_gambar}")


# ==========================================
# 6. SELESAI
# ==========================================

print("\nProses pembuatan Word Cloud selesai.")
print("Lokasi hasil:", folder_output)