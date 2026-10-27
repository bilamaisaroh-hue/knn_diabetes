import math
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SPK KNN Diabetes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Dataset contoh untuk demonstrasi pembelajaran, bukan data pasien nyata.
DATA = [
    {"Usia": 25, "BMI": 22.1, "Glukosa": 88,  "Tekanan darah": 110, "Riwayat keluarga": 0, "Status": 0},
    {"Usia": 31, "BMI": 24.5, "Glukosa": 95,  "Tekanan darah": 118, "Riwayat keluarga": 0, "Status": 0},
    {"Usia": 45, "BMI": 31.2, "Glukosa": 145, "Tekanan darah": 135, "Riwayat keluarga": 1, "Status": 1},
    {"Usia": 52, "BMI": 33.0, "Glukosa": 165, "Tekanan darah": 142, "Riwayat keluarga": 1, "Status": 1},
    {"Usia": 36, "BMI": 27.1, "Glukosa": 115, "Tekanan darah": 125, "Riwayat keluarga": 0, "Status": 0},
    {"Usia": 60, "BMI": 35.2, "Glukosa": 180, "Tekanan darah": 150, "Riwayat keluarga": 1, "Status": 1},
    {"Usia": 42, "BMI": 28.5, "Glukosa": 130, "Tekanan darah": 128, "Riwayat keluarga": 1, "Status": 0},
    {"Usia": 29, "BMI": 23.0, "Glukosa": 92,  "Tekanan darah": 112, "Riwayat keluarga": 0, "Status": 0},
    {"Usia": 55, "BMI": 30.8, "Glukosa": 158, "Tekanan darah": 145, "Riwayat keluarga": 1, "Status": 1},
    {"Usia": 48, "BMI": 29.4, "Glukosa": 140, "Tekanan darah": 138, "Riwayat keluarga": 1, "Status": 1},
]

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f2f6ff 0%, #f8faff 45%, #ffffff 100%);
        color: #273247;
    }
    .block-container { max-width: 900px; padding-top: 2.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #202b42; }
    .hero {
        padding: 1.55rem 1.7rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #e2eaff, #f0f4ff);
        border: 1px solid #d7e1f6;
        margin-bottom: 1.25rem;
    }
    .eyebrow {
        color: #4e6da8; font-size: .82rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .45rem;
    }
    .hero p { color: #66738a; margin-bottom: 0; line-height: 1.65; }
    div[data-testid="stForm"], div[data-testid="stExpander"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #dfe6f2;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 28px rgba(50,75,120,.05);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; border: 0; border-radius: 12px;
        background: #526fd0; color: white; font-weight: 700;
        padding: .72rem 1rem; min-height: 3rem;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #405bb8; color: white; border: 0;
    }
    .result-box {
        border-radius: 16px; padding: 1.1rem 1.2rem; margin: .8rem 0;
        border: 1px solid #dce5f2; background: #fff;
    }
    .safe { background: #eaf8ef; border-color: #c8ebd4; color: #176b3b; }
    .risk { background: #fff0f0; border-color: #ffd4d4; color: #a72b38; }
    .small-note { color: #6f7b8e; font-size: .86rem; line-height: 1.6; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e1e8f3; padding: .8rem;
        border-radius: 14px;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Sistem Pendukung Keputusan</div>
  <h1 style="margin:0 0 .55rem 0;">🩺 Prediksi Diabetes dengan KNN</h1>
  <p>Masukkan data contoh pasien untuk melihat cara kerja metode
  <b>K-Nearest Neighbor</b>. Tampilan ini ditujukan untuk pembelajaran.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1. Input Data Pasien")
st.caption("Isi data berikut, lalu tekan tombol untuk menjalankan perhitungan KNN.")

with st.form("form_pasien"):
    c1, c2 = st.columns(2)
    with c1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.5, step=0.1, format="%.1f")
        glukosa = st.number_input("Glukosa", min_value=20, max_value=500, value=130, step=1)
    with c2:
        tekanan = st.number_input("Tekanan darah", min_value=40, max_value=250, value=128, step=1)
        riwayat_label = st.selectbox("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"])
        k = st.selectbox("Nilai K (jumlah tetangga)", [3, 5, 7, 9], index=0)
    submitted = st.form_submit_button("⚙️ Proses Prediksi KNN", use_container_width=True)

def normalize(value, low, high):
    return (value - low) / (high - low)

def distance(patient, row):
    ranges = {
        "Usia": (1, 120), "BMI": (10, 70), "Glukosa": (20, 500),
        "Tekanan darah": (40, 250), "Riwayat keluarga": (0, 1)
    }
    total = 0.0
    for key, value in patient.items():
        lo, hi = ranges[key]
        a = normalize(value, lo, hi)
        b = normalize(row[key], lo, hi)
        total += (a - b) ** 2
    return math.sqrt(total)

if submitted:
    patient = {
        "Usia": int(usia),
        "BMI": float(bmi),
        "Glukosa": int(glukosa),
        "Tekanan darah": int(tekanan),
        "Riwayat keluarga": 1 if riwayat_label == "Ada" else 0,
    }
    ranked = sorted(
        [{**row, "Jarak": distance(patient, row)} for row in DATA],
        key=lambda x: x["Jarak"]
    )
    neighbors = ranked[:k]
    positives = sum(row["Status"] == 1 for row in neighbors)
    negatives = k - positives
    prediction = 1 if positives > negatives else 0

    st.subheader("2. Hasil Prediksi")
    result_class = "risk" if prediction == 1 else "safe"
    result_text = "Kelompok berisiko diabetes" if prediction == 1 else "Kelompok tidak berisiko diabetes"
    st.markdown(
        f'<div class="result-box {result_class}"><h3 style="margin:0 0 .35rem 0;">{result_text}</h3>'
        f'<div>Hasil model demonstrasi berdasarkan suara terbanyak dari {k} tetangga terdekat.</div></div>',
        unsafe_allow_html=True
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Tetangga K", k)
    m2.metric("Label diabetes", positives)
    m3.metric("Label tidak diabetes", negatives)
    st.markdown("**Tetangga terdekat yang digunakan**")
    neighbor_table = pd.DataFrame(neighbors)
    neighbor_table["Status"] = neighbor_table["Status"].map({0: "Tidak diabetes", 1: "Diabetes"})
    neighbor_table["Jarak"] = neighbor_table["Jarak"].map(lambda x: round(x, 4))
    st.dataframe(
        neighbor_table[["Usia", "BMI", "Glukosa", "Tekanan darah", "Riwayat keluarga", "Status", "Jarak"]],
        use_container_width=True, hide_index=True
    )
    st.markdown(
        '<p class="small-note">Penting: hasil ini hanya simulasi pembelajaran dengan dataset kecil buatan. '
        'Ini bukan alat diagnosis atau penilaian medis. Untuk pemeriksaan kesehatan, konsultasikan dengan tenaga kesehatan.</p>',
        unsafe_allow_html=True
    )

with st.expander("Lihat Dataset Acuan (10 data contoh)"):
    df = pd.DataFrame(DATA)
    df["Riwayat keluarga"] = df["Riwayat keluarga"].map({0: "Tidak ada", 1: "Ada"})
    df["Status"] = df["Status"].map({0: "Tidak diabetes", 1: "Diabetes"})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Dataset ini dibuat khusus sebagai contoh program dan bukan data pasien nyata.")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#7a879a;font-size:.85rem;">SPK • Metode K-Nearest Neighbor (KNN) · Demo pembelajaran</div>',
    unsafe_allow_html=True
)
