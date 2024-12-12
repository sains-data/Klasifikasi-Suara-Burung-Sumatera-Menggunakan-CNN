import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        model_files_exist = all([os.path.exists(os.path.join(dest_folder, file)) for file in output_files])
        if model_files_exist:
            return False  # Model sudah ada, tidak perlu mengunduh ulang

        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]

        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

        api = KaggleApi()
        api.authenticate()

        os.makedirs(dest_folder, exist_ok=True)
        for output_file in output_files:
            api.kernels_output(kernel_name, path=dest_folder, force=True)
        return True
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh model: {str(e)}")
        return None

kernel_name = "evanaryaputra28/dl-tb"
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

download_status = download_model_from_kaggle(kernel_name, output_files, dest_folder)

melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")

if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model MFCC: {str(e)}")

# Fungsi untuk memproses MFCC menjadi gambar 64x64x3
def preprocess_mfcc(mfcc):
    # Convert array to image using Pillow
    mfcc_image = Image.fromarray(mfcc)
    mfcc_image = mfcc_image.resize((64, 64))  # Resize gambar menjadi 64x64
    mfcc_resized = np.array(mfcc_image)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)  # Tambahkan channel
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)  # Ubah grayscale menjadi RGB
    return mfcc_resized

# Fungsi untuk memproses Melspectrogram menjadi gambar 64x64x3
def preprocess_melspec(melspec):
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_image = Image.fromarray(melspec_db)
    melspec_image = melspec_image.resize((64, 64))  # Resize gambar menjadi 64x64
    melspec_resized = np.array(melspec_image)
    melspec_resized = np.expand_dims(melspec_resized, axis=-1)  # Tambahkan channel
    melspec_resized = np.repeat(melspec_resized, 3, axis=-1)  # Ubah grayscale menjadi RGB
    return melspec_resized

# Fungsi untuk menampilkan spektrum
def plot_spectrogram(data, sr, title, y_axis, x_axis):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, sr=sr, x_axis=x_axis, y_axis=y_axis, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

st.title("Deep Learning in Audio: Klasifikasi Suara Burung di Indonesia Bagian Barat 🦜")
st.markdown("""
    **Selamat datang di aplikasi klasifikasi suara burung menggunakan Deep Learning!**
    Unggah file audio dalam format MP3 atau WAV, dan aplikasi akan menghasilkan spektrum
    MFCC dan Melspectrogram serta prediksi model.
""")

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    if st.button("Prediksi Kelas Burung"):
        with st.spinner("Memproses..."):
            try:
                y, sr = librosa.load(temp_file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

                st.subheader("Spektrum MFCC")
                plot_spectrogram(mfcc, sr, "MFCC", y_axis="mel", x_axis="time")

                st.subheader("Spektrum Melspectrogram")
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                plot_spectrogram(melspec_db, sr, "Melspectrogram", y_axis="mel", x_axis="time")

                mfcc_image = preprocess_mfcc(mfcc)
                melspec_image = preprocess_melspec(melspec)

                mfcc_image = np.expand_dims(mfcc_image, axis=0)
                melspec_image = np.expand_dims(melspec_image, axis=0)

                mfcc_result = mfcc_model.predict(mfcc_image)
                melspec_result = melspec_model.predict(melspec_image)

                mfcc_pred_class = np.argmax(mfcc_result, axis=1)[0]
                melspec_pred_class = np.argmax(melspec_result, axis=1)[0]
                mfcc_accuracy = np.max(mfcc_result)
                melspec_accuracy = np.max(melspec_result)

                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_pred_class} dengan akurasi {mfcc_accuracy * 100:.2f}%")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_pred_class} dengan akurasi {melspec_accuracy * 100:.2f}%")
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning
    </p>
""", unsafe_allow_html=True)
