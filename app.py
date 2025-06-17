import streamlit as st
import pandas as pd
import joblib

# === Load Model & Preprocessing ===
model = joblib.load("model_ObesityDataSet.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
encoder_target = joblib.load("encoder_target.pkl")

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🎯 Prediksi Obesitas Berdasarkan Gaya Hidup")
st.markdown("Isi data di bawah ini untuk memprediksi tingkat obesitas berdasarkan gaya hidupmu:")

# === Form Input ===
with st.form("form_obesitas"):
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Usia", min_value=1, max_value=100, value=25)
    height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, value=65.0)
    family = st.selectbox("Riwayat Keluarga Overweight", ["yes", "no"])
    favc = st.selectbox("Sering Konsumsi Makanan Kalori Tinggi?", ["yes", "no"])
    fcvc = st.slider("Frekuensi Konsumsi Sayur (0–3)", 0.0, 3.0, 2.0)
    ncp = st.slider("Jumlah Makan per Hari (0–3)", 0.0, 3.0, 3.0)
    caec = st.selectbox("Ngemil di luar jam makan?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Merokok?", ["yes", "no"])
    ch2o = st.slider("Minum Air Putih per Hari (0–3)", 0.0, 3.0, 2.0)
    scc = st.selectbox("Konsultasi Gizi?", ["yes", "no"])
    faf = st.slider("Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0)
    tue = st.slider("Waktu Layar (TV/Gadget) (0–3)", 0.0, 3.0, 1.0)
    calc = st.selectbox("Konsumsi Alkohol?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Jenis Transportasi", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
    
    submitted = st.form_submit_button("🔍 Prediksi")

# === Validasi & Prediksi ===
if submitted:
    if height <= 0 or weight <= 0:
        st.error("❌ Tinggi dan Berat Badan tidak boleh nol atau negatif.")
    elif age <= 0:
        st.error("❌ Usia tidak boleh nol atau negatif.")
    else:
        df_input = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }])

        # Encode kolom kategorikal
        for col in encoders:
            df_input[col] = encoders[col].transform(df_input[col])

        # ✅ Tambahkan baris ini untuk mengurutkan kolom sesuai saat training
        df_input = df_input[scaler.feature_names_in_]

        # Scaling & Prediksi
        input_scaled = scaler.transform(df_input)
        pred_numeric = model.predict(input_scaled)[0]
        pred_label = encoder_target.inverse_transform([pred_numeric])[0]

        st.success(f"✅ Prediksi Tingkat Obesitas Anda: **{pred_label}**")
