import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# === Load semua aset (model, scaler, encoder) ===
base_path = Path('.')

def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_pickle(base_path / 'final_model.pkl')
scaler = load_pickle(base_path / 'scaler.pkl')
ordinal_maps = load_pickle(base_path / 'encoders/ordinal_mappings.pkl')

# Load label encoders
encoders = {
    'Gender': load_pickle(base_path / 'encoders/Gender_encoder.pkl'),
    'HighCalorieFood': load_pickle(base_path / 'encoders/HighCalorieFood_encoder.pkl'),
    'CalorieMonitoring': load_pickle(base_path / 'encoders/CalorieMonitoring_encoder.pkl'),
    'FamilyHistoryOverweight': load_pickle(base_path / 'encoders/FamilyHistoryOverweight_encoder.pkl'),
    'Transportation': load_pickle(base_path / 'encoders/Transportation_encoder.pkl'),
    'ObesityLevel': load_pickle(base_path / 'encoders/ObesityLevel_encoder.pkl'),
}

# ✅ Ganti scaler.feature_names_in_ dengan kolom hardcoded dari data training
scaled_columns = [
    'Age', 'Height', 'Weight',
    'VegetableConsumption', 'MealFrequency',
    'WaterIntake', 'PhysicalActivity', 'TechnologyUse'
]

# === Streamlit UI ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("💡 Prediksi Tingkat Obesitas")

with st.form("form_prediksi"):
    st.header("📋 Masukkan Informasi Anda")

    umur = st.number_input("Usia", 1, 100, step=1)
    jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['Gender'].classes_)
    tinggi_cm = st.number_input("Tinggi Badan (cm)", 100.0, 250.0, step=1.0)
    tinggi_m = tinggi_cm / 100
    berat = st.number_input("Berat Badan (kg)", 20.0, 200.0, step=1.0)

    konsumsi_alkohol = st.selectbox("Konsumsi Alkohol", list(ordinal_maps['AlcoholConsumption'].keys()))
    makanan_tinggi_kalori = st.selectbox("Sering Makan Tinggi Kalori?", encoders['HighCalorieFood'].classes_)
    konsumsi_sayur = st.selectbox("Frekuensi Makan Sayur", [1, 2, 3])
    frekuensi_makan = st.selectbox("Frekuensi Makan per Hari", [1, 2, 3, 4])
    pemantauan_kalori = st.selectbox("Pantau Kalori Harian?", encoders['CalorieMonitoring'].classes_)
    air_putih = st.selectbox("Minum Air Putih (gelas per hari)", list(range(1, 9)))
    riwayat_keluarga = st.selectbox("Ada Riwayat Kegemukan di Keluarga?", encoders['FamilyHistoryOverweight'].classes_)
    olahraga = st.selectbox("Frekuensi Olahraga per Minggu", list(range(0, 5)))
    waktu_gadget = st.selectbox("Durasi Gunakan Teknologi (jam/hari)", list(range(0, 5)))
    konsumsi_snack = st.selectbox("Kebiasaan Ngemil", list(ordinal_maps['SnackConsumption'].keys()))
    transportasi = st.selectbox("Jenis Transportasi", encoders['Transportation'].classes_)

    prediksi_button = st.form_submit_button("🔍 Prediksi")

# === Proses Prediksi ===
if prediksi_button:
    # Bentuk DataFrame input
    input_df = pd.DataFrame([{
        'Age': umur,
        'Gender': jenis_kelamin,
        'Height': tinggi_m,
        'Weight': berat,
        'AlcoholConsumption': konsumsi_alkohol,
        'HighCalorieFood': makanan_tinggi_kalori,
        'VegetableConsumption': konsumsi_sayur,
        'MealFrequency': frekuensi_makan,
        'CalorieMonitoring': pemantauan_kalori,
        'WaterIntake': air_putih,
        'FamilyHistoryOverweight': riwayat_keluarga,
        'PhysicalActivity': olahraga,
        'TechnologyUse': waktu_gadget,
        'SnackConsumption': konsumsi_snack,
        'Transportation': transportasi
    }])

    # Mapping ordinal
    for col in ['AlcoholConsumption', 'SnackConsumption']:
        input_df[col] = input_df[col].map(ordinal_maps[col])

    # Label Encoding
    for col in ['Gender', 'HighCalorieFood', 'CalorieMonitoring', 'FamilyHistoryOverweight', 'Transportation']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Validasi kolom input vs scaler
    missing_cols = [col for col in scaled_columns if col not in input_df.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di input: {missing_cols}")
        st.stop()

    # Transformasi fitur numerik
    scaled_data = scaler.transform(input_df[scaled_columns])

    # Gabung dengan ordinal features
    fitur_akhir = np.hstack([
        scaled_data,
        input_df[['AlcoholConsumption', 'SnackConsumption']].values
    ])

    # Prediksi dan tampilkan hasil
    hasil_prediksi = model.predict(fitur_akhir)[0]
    label_hasil = encoders['ObesityLevel'].inverse_transform([hasil_prediksi])[0]
    st.success(f"Hasil Prediksi Tingkat Obesitas: **{label_hasil}** 🎯")
