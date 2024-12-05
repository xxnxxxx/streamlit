import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kategori Teks SVM",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memuat model SVM dan TF-IDF Vectorizer
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200", caption="Logo Aplikasi", use_column_width=True)  # Ganti URL gambar jika ada logo
    st.title("Navigasi")
    st.write("💡 **Aplikasi ini mendukung prediksi kategori teks menggunakan model SVM yang telah dilatih.**")
    st.markdown("---")
    st.subheader("Cara Penggunaan:")
    st.write("""
    1. Masukkan teks di area input di halaman utama.
    2. Klik tombol "Prediksi" untuk melihat hasilnya.
    3. Hasil kategori akan ditampilkan di bawahnya.
    """)
    st.markdown("---")
    st.caption("📢 **Catatan**: Aplikasi ini hanya untuk demo dan bukan aplikasi final.")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Prediksi Kategori Teks</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Aplikasi berbasis SVM untuk klasifikasi teks.</p>", unsafe_allow_html=True)

# Input teks
st.markdown("### Masukkan teks untuk mendapatkan prediksi:")
input_text = st.text_area("Teks Anda:", placeholder="Ketikkan teks di sini...", height=150)

# Hasil prediksi
if st.button("🔍 Prediksi"):
    if input_text.strip():
        # Transformasi teks menggunakan vectorizer
        input_vector = vectorizer.transform([input_text])

        # Prediksi
        prediction = svm_model.predict(input_vector)

        # Menampilkan hasil prediksi
        st.markdown("### 🔹 **Hasil Prediksi**")
        st.success(f"📌 Kategori Teks: **{prediction[0]}**")
    else:
        st.error("⚠️ Harap masukkan teks terlebih dahulu.")

# Footer dengan visual tambahan
st.markdown("---")
cols = st.columns([1, 2, 1])
with cols[1]:
    st.image("https://via.placeholder.com/600x100", caption="Terima Kasih telah menggunakan aplikasi ini!", use_column_width=True)
st.markdown(
    """
    <footer style="text-align: center; color: gray; font-size: small;">
        Dibuat dengan ❤️ oleh Tim Anda | © 2024
    </footer>
    """,
    unsafe_allow_html=True
)
