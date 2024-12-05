import streamlit as st
#import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


import joblib  # Tambahkan di awal kode
svm_model = joblib.load('svm_model.pkl')  # Memuat model SVM
vectorizer = joblib.load('vectorizer.pkl')  # Memuat TF-IDF Vectorizer


# Menampilkan judul aplikasi
st.title("Prediksi Kategori Teks dengan SVM ")

# Membuat input teks untuk prediksi
input_text = st.text_area("Masukkan teks untuk prediksi", "")

# Menambahkan tombol untuk memprediksi
if st.button("Prediksi"):
    if input_text:
        # Melakukan transformasi TF-IDF pada input teks
        input_vector = vectorizer.transform([input_text])

        # Melakukan prediksi
        prediction = svm_model.predict(input_vector)

        # Menampilkan hasil prediksi
        st.write(f"Prediksi Kategori: {prediction[0]}")

    else:
        st.warning("Silakan masukkan teks untuk prediksi.")
