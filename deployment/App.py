import os
import json
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import warnings
import time

# Menyembunyikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CSS untuk styling halaman

# Fungsi untuk menampilkan splash screen
def splash_screen():
    st.markdown("""
    <style>
        /* Full-screen splash screen */
        .splash-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: #2E2E2E; /* Latar belakang abu-abu gelap */
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .splash-logo {
            width: 350px; /* Ukuran logo */
            height: auto;
            border: 5px solid white; /* Border putih */
            border-radius: 15px; /* Ujung melengkung */
            animation: fadeIn 3s ease-in-out; /* Animasi fade-in */
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    <div class="splash-screen">
        <img src="https://raw.githubusercontent.com/Lion0510/deployment/main/images/LogoApp.jpeg" alt="Logo" class="splash-logo">
    </div>
    """, unsafe_allow_html=True)

# Tampilkan splash screen
splash_screen()
time.sleep(3)  # Biarkan splash screen selama 3 detik

# Hapus splash screen dengan mengganti CSS
st.markdown("""
<style>
    .splash-screen {
        display: none; /* Hilangkan splash screen */
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        .stApp {
            background-image: url('https://raw.githubusercontent.com/Lion0510/deployment/main/images/bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        /* Container Header */
        .header-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap; /* Agar responsif */
        }
        .logo {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 120px;
            height: 120px;
            object-fit: contain;
        }
        /* Box Header Judul */
        .header-box {
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            width: 90%;
            max-width: 800px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .header-box h1 {
            font-size: 2.5em;
            margin: 0;
            text-align: center; /* Posisikan teks di tengah */
            line-height: 1.2;
        }
        .header-box p {
            font-size: 1.2em;
            margin-top: 5px;
        }
        /* Konten Utama */
        .content-section {
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px auto;
            width: 90%;
            max-width: 800px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .footer {
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Kamus deskripsi kelas burung
BIRD_CLASSES = {
    0: {
        "name": "Pitta sordida",
        "description": "Burung ini terkenal dengan bulu-bulunya yang warna-warni, seperti hijau, biru, dan kuning. Pitta Sayap Hitam hidup di hutan-hutan tropis dan suka mencari makan di tanah, biasanya berupa serangga kecil dan cacing.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/main/images/Pitta_sordida.jpg'
    },
    1: {
        "name": "Dryocopus javensis",
        "description": "Burung pelatuk ini memiliki bulu hitam dengan warna merah mencolok di kepalanya. Ia menggunakan paruhnya yang kuat untuk mematuk batang pohon, mencari serangga, atau membuat sarang.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/refs/heads/main/images/%20Dryocopus_javensis.jpg'
    },
    2: {
        "name": "Caprimulgus macrurus",
        "description": "Burung ini aktif di malam hari dan memiliki bulu yang menyerupai warna kulit kayu, sehingga mudah berkamuflase. Kangkok Malam Besar memakan serangga dan sering ditemukan di area terbuka dekat hutan.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/main/images/Caprimulgus_macrurus.jpg'
    },
    3: {
        "name": "Pnoepyga pusilla",
        "description": "Burung kecil ini hampir tidak memiliki ekor dan sering bersembunyi di semak-semak. Suaranya sangat nyaring meskipun ukurannya kecil. Mereka makan serangga kecil dan hidup di daerah pegunungan.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/main/images/Pnoepyga_pusilla.jpg'
    },
    4: {
        "name": "Anthipes solitaris",
        "description": "Kacer Soliter adalah burung kecil yang suka berada di dekat aliran sungai. Bulunya berwarna abu-abu dan putih dengan suara kicauan yang lembut. Ia sering makan serangga kecil.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/main/images/Anthipes_solitaris.jpg'
    },
    5: {
        "name": "Buceros rhinoceros",
        "description": "Enggang Badak adalah burung besar dengan paruh besar yang melengkung dan tanduk di atasnya. Burung ini adalah simbol keberagaman hutan tropis dan sering ditemukan di Kalimantan dan Sumatra. Mereka memakan buah-buahan, serangga, dan bahkan hewan kecil.",
        "image": 'https://raw.githubusercontent.com/Lion0510/deployment/main/images/Buceros_rhinoceros.jpg'
    }
}

# Fungsi untuk mendapatkan informasi kelas berdasarkan prediksi
def get_bird_info(pred_class):
    if pred_class in BIRD_CLASSES:
        return BIRD_CLASSES[pred_class]
    else:
        return {"name": "Unknown", "description": "Deskripsi tidak tersedia.", "image": None}


# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        model_files_exist = all([os.path.exists(os.path.join(dest_folder, file)) for file in output_files])
        if model_files_exist:
            return False  # Model sudah ada, tidak perlu mengunduh ulang

        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KEY"]

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

kernel_name = "evanaryaputra28/tubes-dll"
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

download_status = download_model_from_kaggle(kernel_name, output_files, dest_folder)

# Path untuk model yang disimpan
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
    mfcc_image = Image.fromarray(mfcc)
    mfcc_image = mfcc_image.resize((64, 64))
    mfcc_resized = np.array(mfcc_image)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)
    return mfcc_resized

# Fungsi untuk memproses Melspectrogram menjadi gambar 64x64x3
def preprocess_melspec(melspec):
    # Convert to decibel scale
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # Normalize the spectrogram
    melspec_normalized = (melspec_db - np.min(melspec_db)) / (np.max(melspec_db) - np.min(melspec_db))
    
    # Resize to 64x64
    melspec_image = Image.fromarray(melspec_normalized * 255).convert('RGB')
    melspec_resized = melspec_image.resize((64, 64))
    melspec_array = np.array(melspec_resized)
    
    # Expand dimensions to match model input shape: (1, 64, 64, 3)
    melspec_processed = np.expand_dims(melspec_array, axis=0)
    
    return melspec_processed

# Fungsi untuk menampilkan spektrum
def plot_spectrogram(data, sr, title, y_axis, x_axis):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, sr=sr, x_axis=x_axis, y_axis=y_axis, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def preprocess_melspec(melspec):
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_image = Image.fromarray(melspec_db).resize((64, 64))  # Resize ke (64, 64)
    melspec_resized = np.array(melspec_image)

    # Ubah grayscale menjadi 3 channel (RGB)
    if len(melspec_resized.shape) == 2:
        melspec_resized = np.expand_dims(melspec_resized, axis=-1)
        melspec_resized = np.repeat(melspec_resized, 3, axis=-1)

    melspec_resized = np.expand_dims(melspec_resized, axis=0)  # Tambahkan batch dimension
    return melspec_resized

# Tambahkan CSS ke aplikasi
#add_custom_css()

# Header dengan logo dan judul
st.markdown("""
<div class="header-content">
    <img src="https://raw.githubusercontent.com/Lion0510/deployment/main/images/Logo2.jpg" alt="Logo Fakultas Sains" class="logo">
    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA" class="logo">
    <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi" class="logo">
</div>
<div class="header-box">
    <h1>Tweetify</h1>
    <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.bird-image {
    width: 100%; 
    max-width: 800px; 
    height: auto; 
    border-radius: 15px; 
    box-shadow: 0 15px 25px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.bird-image:hover {
    transform: scale(1.05);
    box-shadow: 0 20px 35px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# Konten Aplikasi
st.markdown("""
<section class="content-section">
    <h1>Klasifikasi Suara</h1>
    <p>Burung yang termasuk dalam klasifikasi ini adalah:</p>
    <ul>
        <li>Pitta sordida</li>
        <li>Dryocopus javensis</li>
        <li>Caprimulgus macrurus</li>
        <li>Pnoepyga pusilla</li>
        <li>Anthipes solitaris</li>
        <li>Buceros rhinoceros</li>
    </ul>
""", unsafe_allow_html=True)

#upload audio
# Header untuk upload file dengan background hitam transparan
st.markdown("""
<div style='background-color: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); text-align: center;'>
    <h3 style='color: white; margin-bottom: 10px;'>Unggah File Audio</h3>
    <p style='color: white;'>Pilih file audio dalam format MP3 atau WAV untuk memulai klasifikasi suara burung.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Gaya untuk file uploader */
        .stFileUploader {
            background-color: rgba(0, 0, 0, 0.5); /* Latar belakang semi transparan */
            color: white; /* Teks putih untuk file */
            font-size: 16px; /* Ukuran font yang lebih besar */
            border-radius: 10px; /* Sudut membulat */
            padding: 10px; /* Padding agar file tidak terlalu rapat */
        }

        /* Gaya untuk teks nama file yang diunggah */
        .stFileUploader > div {
            font-size: 18px; /* Ukuran font untuk nama file yang diunggah */
            color: white; /* Teks putih untuk nama file */
        }

        /* Gaya untuk ukuran file yang diunggah */
        .stFileUploader > div > div {
            font-size: 16px; /* Ukuran font untuk ukuran file */
            color: white; /* Warna teks putih untuk ukuran file */
        }
    </style>
""", unsafe_allow_html=True)

# Upload file audio
uploaded_audio = st.file_uploader("Unggah file audio (MP3/WAV)", type=["mp3", "wav"], label_visibility="hidden")

if uploaded_audio is not None:
    # Tampilkan audio
    st.audio(uploaded_audio, format="audio/mp3")
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    with st.spinner("Memproses..."):
        try:
            # Load audio sekali saja di awal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                y, sr = librosa.load(temp_file_path, sr=None)  # Muat file audio

            # Menghasilkan MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Tampilkan spektrum MFCC
            st.markdown("""
            <div style='background-color: rgba(0, 0, 0, 0.6); padding: 10px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Spektrum MFCC</h3>
            </div>
            """, unsafe_allow_html=True)
            plot_spectrogram(mfcc, sr, "MFCC", y_axis="mel", x_axis="time")

            # Menghasilkan MelSpectrogram
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            melspec_db = librosa.power_to_db(melspec, ref=np.max)

            # Tampilkan spektrum Melspectrogram
            st.markdown("""
            <div style='background-color: rgba(0, 0, 0, 0.6); padding: 10px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Spektrum Melspectrogram</h3>
            </div>
            """, unsafe_allow_html=True)
            plot_spectrogram(melspec_db, sr, "Melspectrogram", y_axis="mel", x_axis="time")

            # Preproses Melspectrogram untuk model
            melspec_resized = preprocess_melspec(melspec)

            # Prediksi menggunakan model
            predictions = melspec_model.predict(melspec_resized)[0]

            # Dapatkan top 3 kelas berdasarkan probabilitas tertinggi
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_probabilities = predictions[top_3_indices]

            # Tampilkan hasil prediksi top 3
            st.markdown("""
            <div style='background-color: rgba(0, 0, 0, 0.8); padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);'>
                <h3 style='color: white; text-align: center; margin-bottom: 10px;'>Hasil Prediksi Top 3</h3>
            </div>
            """, unsafe_allow_html=True)

            for idx, (class_idx, probability) in enumerate(zip(top_3_indices, top_3_probabilities), 1):
                bird_info = get_bird_info(class_idx)
                prediction_percentage = probability * 100
                
                st.markdown(f"""
                <div style='background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;'>
                    <p style='color: white; font-size: 18px;'><strong>Peringkat {idx}:</strong></p>
                    <p style='color: white;'><strong>Kelas:</strong> {class_idx}</p>
                    <p style='color: white; font-size: 20px;'><strong>Nama Burung:</strong> {bird_info['name']}</p>
                    <p style='color: white; font-size: 18px;'><strong>Akurasi:</strong> {prediction_percentage:.2f}%</p>
                    <img src="{bird_info['image']}" alt="{bird_info['name']}" style='width: 40%; max-width: 500px; height: auto; border-radius: 10px; margin-top: 10px;'>
                    <p style='color: white; font-style: italic; margin-top: 10px;'>{bird_info.get('description', 'Deskripsi tidak tersedia.')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error saat memproses audio: {str(e)}")

        
# Footer
st.markdown("""
<div class="footer">
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning</p>
</div>
""", unsafe_allow_html=True)
