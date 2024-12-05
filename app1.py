import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kategori Teks dengan SVM",
    page_icon="📘",
    layout="wide"
)

# Memuat model SVM dan TF-IDF Vectorizer yang sudah disimpan
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Logo Aplikasi", use_column_width=True)  # Ganti URL dengan logo Anda
    st.title("Tentang Aplikasi")
    st.write(
        """
        Aplikasi ini menggunakan model **Support Vector Machine (SVM)** untuk
        memprediksi kategori teks berdasarkan input pengguna. Model ini 
        telah dilatih menggunakan data yang relevan.
        """
    )
    st.write("Masukkan teks Anda di kolom utama untuk memulai prediksi!")

# Header aplikasi
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prediksi Kategori Teks</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Aplikasi berbasis SVM untuk klasifikasi teks</p>", unsafe_allow_html=True)

# Membuat input teks untuk prediksi
st.markdown("---")
st.markdown("### **Masukkan teks di bawah ini untuk mendapatkan kategori:**")
input_text = st.text_area("", placeholder="Ketikkan teks Anda di sini...")

# Menambahkan tombol untuk memprediksi
if st.button("🔍 Prediksi"):
    if input_text.strip():
        # Transformasi TF-IDF pada input teks
        input_vector = vectorizer.transform([input_text])

        # Melakukan prediksi
        prediction = svm_model.predict(input_vector)

        # Menampilkan hasil prediksi
        st.markdown("<h3 style='color: #4CAF50;'>Hasil Prediksi:</h3>", unsafe_allow_html=True)
        st.success(f"📌 **Kategori Teks: {prediction[0]}**")
    else:
        st.warning("⚠️ Silakan masukkan teks untuk melakukan prediksi.")

# Footer
st.markdown("---")
st.markdown(
    """
    <footer style="text-align: center;">
        Dibuat dengan ❤️ oleh <strong>Tim Anda</strong> | 2024
    </footer>
    """,
    unsafe_allow_html=True
)
