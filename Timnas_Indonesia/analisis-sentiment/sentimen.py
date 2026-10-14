
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

URUTAN = ['positive', 'neutral', 'negative']
EMOJI = {'positive': '😃', 'neutral': '😐', 'negative': '😡'}
PEMBAHASAN_MISMATCH = [
    {
        'id': 230,
        'judul': 'Teks sangat pendek dan minimalis',
        'teks': 'selamat untuk timnas garuda indonesia garuda indonesia',
        'manual': 'Positive',
        'prediksi': 'Neutral',
        'penyebab': ('Kalimat singkat dan repetitif. Meskipun memuat kata "selamat", '
                     'model transformer cenderung memprediksi netral karena minimnya '
                     'variasi struktur narasi atau indikator emosi kuat.'),
    },
    {
        'id': 157,
        'judul': 'Penggunaan bahasa daerah dan kata gaul',
        'teks': ('kade mang aya nu nyamber teu nasionalis masih cicing d indo teu '
                 'ngaledek pemain padahal cuma ngebandingin hasil wajar bae emang timnas '
                 'na bae bapuk main d kandang menang adu finalty 11vs10 pula artinya '
                 'taktik herdman mentok'),
        'manual': 'Neutral',
        'prediksi': 'Negative',
        'penyebab': ('Campuran bahasa Sunda seperti "kade mang" dan "cicing d indo", '
                     'serta kata gaul seperti "bapuk" dan "mentok", dapat membuat model '
                     'menangkap kata bernada negatif secara terpisah dari konteks diskusi.'),
    },
    {
        'id': 209,
        'judul': 'Sarkasme dan kata ambigu',
        'teks': ('untung hanya persatuan sepak bola surabaya bukan persatuan sepak bola '
                 'indonesia surabaya lagi'),
        'manual': 'Neutral',
        'prediksi': 'Positive',
        'penyebab': ('Struktur kalimat mengandung sindiran dan permainan kata nama '
                     'organisasi. Kata "untung" dapat diasosiasikan model dengan rasa '
                     'syukur, meskipun konteksnya bukan sentimen positif.'),
    },
    {
        'id': 9,
        'judul': 'Typos, perpanjangan karakter, dan code-mixing',
        'teks': ('yang suka hujat timnas indonesia plisss nonton ini gada yang isntant '
                 'we grow up step by stepppppp pov pemain dan pelatih grow uppp suporter '
                 'plis timnas johnherdman indonesia garuda'),
        'manual': 'Positive',
        'prediksi': 'Neutral',
        'penyebab': ('Karakter berulang, kata tidak baku, serta campuran bahasa Indonesia '
                     'dan Inggris dapat menyamarkan pesan dukungan optimis sehingga '
                     'diprediksi netral.'),
    },
]


@st.cache_data
def _baca_csv(path):
    return pd.read_csv(path)


@st.cache_data
def _baca_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def halaman_sentimen(out_dir='output'):
    st.subheader('Analisis Sentimen Postingan Twitter (X)')

    wajib = ['ringkasan_sentimen.csv', 'hasil_sentimen.csv', 'meta_sentimen.json']
    hilang = [f for f in wajib if not os.path.exists(os.path.join(out_dir, f))]
    if hilang:
        st.error(f'File hasil belum ditemukan di folder "{out_dir}": {", ".join(hilang)}. '
                 'Jalankan notebook analisis dulu atau cek path folder output.')
        return

    ringkasan = _baca_csv(os.path.join(out_dir, 'ringkasan_sentimen.csv'))
    hasil = _baca_csv(os.path.join(out_dir, 'hasil_sentimen.csv'))
    meta = _baca_json(os.path.join(out_dir, 'meta_sentimen.json'))

    st.caption(f'Keyword: {meta.get("keyword", "-")} | Metode: {meta.get("metode", "-")} | '
               f'Jumlah data: {meta.get("jumlah_data", len(hasil))} | '
               f'Diproses: {meta.get("diproses_pada", "-")}')

    # --- Jumlah dan persentase ---
    st.write('### Jumlah dan Persentase Sentimen')
    kolom = st.columns(3)
    data = ringkasan.set_index('sentimen')
    for k, s in zip(kolom, URUTAN):
        if s in data.index:
            k.metric(f'{s.capitalize()} {EMOJI[s]}', int(data.loc[s, 'jumlah']),
                     f'{data.loc[s, "persentase"]}%', delta_color='off')
    st.dataframe(ringkasan, hide_index=True)

    # --- Grafik distribusi ---
    st.write('### Distribusi Sentimen')
    df_grafik = ringkasan.copy()
    df_grafik['sentimen_label'] = df_grafik['sentimen'].astype(str).str.capitalize()
    df_grafik['label_teks'] = df_grafik['jumlah'].astype(str) + ' (' + df_grafik['persentase'].astype(str) + '%)'

    color_scale = alt.Scale(
        domain=['Positive', 'Neutral', 'Negative'],
        range=['#2e9e5b', '#8a94a6', '#d64545']
    )

    bars = alt.Chart(df_grafik).mark_bar(stroke='black', strokeWidth=1, opacity=0.85).encode(
        x=alt.X('sentimen_label:N', sort=['Positive', 'Neutral', 'Negative'], title='Kategori Sentimen', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('jumlah:Q', title='Jumlah Postingan (Tweets)', scale=alt.Scale(domain=[0, float(df_grafik['jumlah'].max()) * 1.25])),
        color=alt.Color('sentimen_label:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('sentimen_label:N', title='Sentimen'),
            alt.Tooltip('jumlah:Q', title='Jumlah'),
            alt.Tooltip('persentase:Q', title='Persentase (%)')
        ]
    )

    text = alt.Chart(df_grafik).mark_text(dy=-10, fontSize=11, fontWeight='bold').encode(
        x=alt.X('sentimen_label:N', sort=['Positive', 'Neutral', 'Negative']),
        y=alt.Y('jumlah:Q'),
        text='label_teks:N'
    )

    chart = (bars + text).properties(
        title=alt.TitleParams(f'Distribusi Sentimen Publik Terhadap Timnas Indonesia (N={len(hasil)})', fontSize=14, anchor='middle')
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Tren harian (jika ada kolom tanggal) ---
    if 'tanggal' in hasil.columns and hasil['tanggal'].notna().any():
        st.write('### Tren Harian per Sentimen')
        tren = (hasil.dropna(subset=['tanggal'])
                .groupby(['tanggal', 'sentiment']).size().unstack(fill_value=0)
                .reindex(columns=URUTAN, fill_value=0))
        st.line_chart(tren)

    # --- Contoh postingan ---
    st.write('### Contoh Postingan per Sentimen')
    pilih = st.selectbox('Pilih sentimen', URUTAN)
    contoh = hasil[hasil['sentiment'] == pilih]
    if len(contoh):
        st.dataframe(contoh[['text', 'sentiment', 'confidence']]
                     .sample(min(10, len(contoh)), random_state=1), hide_index=True)
    else:
        st.info('Tidak ada postingan pada kategori ini.')

    # --- Evaluasi manual ---
    st.write('### Evaluasi: Label Manual vs Prediksi Sistem')
    f_eval = os.path.join(out_dir, 'ringkasan_sentimen.json')
    f_tabel = os.path.join(out_dir, 'evaluasi_sentimen.csv')
    ev = meta.get('evaluasi_manual')
    if ev is None and os.path.exists(f_eval):
        ev = _baca_json(f_eval)

    if ev is not None:
        total_sampel = ev.get('total_sampel', ev.get('jumlah_sampel', '-'))
        jumlah_sesuai = ev.get('jumlah_sesuai', '-')
        persen = ev.get('persen_kesesuaian', '-')
        c1, c2 = st.columns(2)
        c1.metric('Kesesuaian', f'{persen}%')
        c2.metric('Sesuai / Sampel', f'{jumlah_sesuai} / {total_sampel}')
        st.info('**Catatan Evaluasi:** Hasil ini bersifat indikatif. Dengan 20 sampel, '
                'setiap postingan mewakili bobot 5%. Angka kesesuaian memberi gambaran '
                'terbatas tentang performa model, bukan akurasi mutlak pada seluruh '
                'percakapan.')

        if os.path.exists(f_tabel):
            tabel = _baca_csv(f_tabel)
            if {'id_post', 'prediksi'}.issubset(tabel.columns) and {'id_post', 'sentiment'}.issubset(hasil.columns):
                prediksi_dari_hasil = hasil.set_index('id_post')['sentiment']
                tabel['prediksi'] = tabel['prediksi'].fillna('').astype(object)
                prediksi_kosong = tabel['prediksi'].astype(str).str.strip().eq('')
                tabel.loc[prediksi_kosong, 'prediksi'] = tabel.loc[prediksi_kosong, 'id_post'].map(prediksi_dari_hasil)

            kolom_sampel = [c for c in ['teks', 'label_manual', 'prediksi'] if c in tabel.columns]
            st.write('**Semua sampel**')
            st.dataframe(tabel[kolom_sampel], hide_index=True)

            if 'sesuai' in tabel.columns:
                beda = tabel[~tabel['sesuai'].fillna(False)]
            elif {'label_manual', 'prediksi'}.issubset(tabel.columns):
                prediksi_tersedia = tabel['prediksi'].fillna('').astype(str).str.strip().ne('')
                sesuai = (tabel['label_manual'].fillna('').astype(str).str.lower()
                          == tabel['prediksi'].fillna('').astype(str).str.lower())
                beda = tabel[~sesuai & prediksi_tersedia]
                if not prediksi_tersedia.any():
                    beda = None
            else:
                beda = None

            if beda is not None:
                st.write(f'**Contoh ketidaksesuaian ({len(beda)})**')
                kolom_beda = [c for c in ['teks', 'label_manual', 'prediksi', 'confidence']
                              if c in beda.columns]
                st.dataframe(beda[kolom_beda], hide_index=True)
            else:
                st.info('Kolom prediksi pada CSV evaluasi belum diisi.')

        st.write('### Pembahasan Ketidaksesuaian (Analisis Mismatch)')
        for item in PEMBAHASAN_MISMATCH:
            with st.expander(f'{item["judul"]} (ID Post: {item["id"]})'):
                st.markdown(f'**Teks:** "{item["teks"]}"')
                st.markdown(f'**Label Manual:** {item["manual"]}  \n'
                            f'**Prediksi Sistem:** {item["prediksi"]}')
                st.markdown(f'**Penyebab:** {item["penyebab"]}')
    else:
        st.info('Hasil evaluasi manual belum tersedia.')


if __name__ == '__main__':
    st.set_page_config(page_title='Analisis Sentimen', layout='wide')
    st.title('Sentiment Analysis NLP App')
    halaman_sentimen()
