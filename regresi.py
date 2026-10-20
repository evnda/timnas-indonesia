import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

FITUR = ['word_count', 'hashtag_count']


RAW_PATHS = [
    os.path.join(os.path.dirname(__file__), 'Timnas_Indonesia', 'dataset_timnas_raw.csv'),
    os.path.join(os.path.dirname(__file__), 'dataset_timnas_raw.csv'),
]


def tambah_engagement(df):
    """Tambahkan retweet_count & reply_count dari dataset mentah (jika ada).

    Dicocokkan lewat teks asli (text_raw = full_text). Jika kolom sudah ada
    atau file mentah tidak ditemukan, data dikembalikan apa adanya.
    """
    df = df.copy()
    if 'retweet_count' in df.columns and 'reply_count' in df.columns:
        return df
    if 'text_raw' not in df.columns:
        return df
    for path in RAW_PATHS:
        if os.path.exists(path):
            raw = pd.read_csv(path, encoding='utf-8-sig')
            if not {'full_text', 'retweet_count', 'reply_count'} <= set(raw.columns):
                continue
            raw = raw.assign(_key=raw['full_text'].astype(str).str.strip())
            raw = raw.drop_duplicates('_key')[['_key', 'retweet_count', 'reply_count']]
            df['_key'] = df['text_raw'].astype(str).str.strip()
            df = df.merge(raw, on='_key', how='left').drop(columns='_key')
            break
    return df


def prepare_data(df):
    """Buat fitur dan target. Target di-log agar hubungan lebih linear."""
    df = tambah_engagement(df)  # sudah berupa salinan, df asli aman

    # Hashtag harus dihitung dari teks MENTAH. Kolom 'text' sudah dibersihkan,
    # sehingga tanda '#' sudah hilang dan hasilnya selalu 0.
    kolom_hashtag = 'text_raw' if 'text_raw' in df.columns else 'text'

    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df[kolom_hashtag].apply(lambda x: str(x).count('#'))

    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
    df = df.dropna(subset=['like_count']).copy()
    df['like_count'] = df['like_count'].clip(lower=0)

    # log1p menangani nilai 0 dan meredam outlier (likes sangat miring ke kanan)
    df['log_like'] = np.log1p(df['like_count'])
    if 'retweet_count' in df.columns:
        df['log_retweet'] = np.log1p(pd.to_numeric(df['retweet_count'], errors='coerce'))
    return df


def train_and_evaluate(df):
    df = prepare_data(df)

    X = df[FITUR]
    y = df['log_like']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # R² pada skala log (skala di mana model linear dilatih)
    r2 = r2_score(y_test, y_pred)
    # MAE dikembalikan ke skala likes asli agar mudah dibaca
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred).clip(min=0))

    return model, mae, r2, y_test, y_pred


def plot_actual_vs_pred(y_test, y_pred):
    """Aktual vs prediksi pada skala log1p(likes)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='blue', alpha=0.7)

    batas = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(batas, batas, 'r--', lw=2, label='Prediksi Ideal')

    ax.set_xlabel('Aktual log(1 + Likes)')
    ax.set_ylabel('Prediksi log(1 + Likes)')
    ax.set_title('Perbandingan Likes Aktual vs Prediksi (skala log)')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    """Heatmap korelasi Pearson fitur regresi & engagement."""
    df = prepare_data(df)
    kolom = ['word_count', 'hashtag_count', 'like_count']
    for k in ['retweet_count', 'reply_count']:
        if k in df.columns:
            kolom.append(k)
    korelasi = df[kolom].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        korelasi, annot=True, fmt='.3f', cmap='coolwarm',
        vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title('Heatmap Korelasi Fitur Regresi & Engagement')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    return fig


def plot_engagement(df):
    """Analisis tambahan: hubungan antar-engagement (BUKAN model prediksi)."""
    df = prepare_data(df)
    if 'log_retweet' not in df.columns:
        return None
    df = df.dropna(subset=['log_retweet'])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(data=df, x='log_retweet', y='log_like', ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    r = df['log_retweet'].corr(df['log_like'])
    ax.set_title(f'Retweet vs Likes (skala log), r = {r:.2f}')
    ax.set_xlabel('log(1 + Retweet)')
    ax.set_ylabel('log(1 + Likes)')
    fig.tight_layout()
    return fig


def plot_linearitas(df):
    """Sebelum vs sesudah transformasi log: bukti hubungan jadi lebih linear."""
    df = prepare_data(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.regplot(data=df, x='word_count', y='like_count', ax=axes[0],
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0].set_title('Sebelum: Likes asli (tidak linear)')
    axes[0].set_xlabel('Jumlah Kata')
    axes[0].set_ylabel('Likes')

    sns.regplot(data=df, x='word_count', y='log_like', ax=axes[1],
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[1].set_title('Sesudah: log(1 + Likes)')
    axes[1].set_xlabel('Jumlah Kata')
    axes[1].set_ylabel('log(1 + Likes)')

    fig.tight_layout()
    return fig


def predict_likes(model, word_count, hashtag_count):
    """Prediksi likes dalam skala asli (tidak pernah negatif)."""
    input_data = pd.DataFrame([[word_count, hashtag_count]], columns=FITUR)
    log_pred = model.predict(input_data)[0]
    return max(0.0, float(np.expm1(log_pred)))