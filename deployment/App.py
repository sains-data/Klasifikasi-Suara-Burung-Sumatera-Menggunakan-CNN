import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import gdown
from io import BytesIO
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Google Drive file URLs for models
melspec_model_url = 'https://drive.google.com/uc?id=192VGvINbZKOyjhGioyBhjfd2alGe6ATM'
mfcc_model_url = 'https://drive.google.com/uc?id=1aRBAt6bHVMW3t6QwbLHzCPn3fQuqd71h'

# File path to save the models
melspec_model_path = 'melspec_model.h5'
mfcc_model_path = 'mfcc_model.h5'

# Download models from Google Drive if not already downloaded
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.info(f"Downloading model from {model_url}...")
        gdown.download(model_url, model_path, quiet=False)
        st.success(f"Model downloaded and saved to {model_path}")
    else:
        st.info(f"Model already exists at {model_path}, skipping download.")

# Download the models
download_model(melspec_model_url, melspec_model_path)
download_model(mfcc_model_url, mfcc_model_path)

# Load models
try:
    melspec_model = load_model(melspec_model_path)  # Load the Mel-spectrogram model
    mfcc_model = load_model(mfcc_model_path)  # Load the MFCC model
except Exception as e:
    st.error(f"Error loading models: {e}")

# Title of the app
st.title("West Indonesia Birds Audio Classifier 🦜")

# Introduction
st.markdown("""**Selamat datang di aplikasi Klasifikasi Suara Burung!** Aplikasi ini akan mengklasifikasikan suara burung berdasarkan file audio yang diunggah. Cukup unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung!""")

# File upload section
uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Display file details
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Process the audio file
    audio_bytes = uploaded_audio.read()
    with BytesIO(audio_bytes) as audio_buffer:
        # Load audio using librosa
        y, sr = librosa.load(audio_buffer, sr=None)  # Keep original sample rate

    # Extract Mel-spectrogram and MFCC features
    st.subheader("Mel-spectrogram dan MFCC Extracted Features")

    # Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Resize Mel-spectrogram to a fixed width (e.g., 500)
    mel_db_resized = librosa.util.fix_length(mel_db, size=500, axis=-1)  # Resize width to 500

    # Plot Mel-spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_db_resized, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel-spectrogram')
    st.pyplot(fig)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Plot MFCC
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    st.pyplot(fig)

    # Prepare features for prediction
    mel_spectrogram = mel_db_resized[..., np.newaxis]  # Add channel dimension
    mfcc = mfcc.T  # Transpose MFCC to match input shape

    # Normalize features
    mel_spectrogram = mel_spectrogram / np.max(mel_spectrogram)
    mfcc = mfcc / np.max(mfcc)

    # Reshape Mel-spectrogram to have 3 channels (required by the model)
    mel_spectrogram = np.repeat(mel_spectrogram, 3, axis=-1)  # Repeat channels 3 times to match the model input

    # Flatten the mel_spectrogram input to match the expected input shape for the model (9216)
    mel_spectrogram_flattened = mel_spectrogram.flatten().reshape(1, -1)  # Flatten into a single vector

    # Predict using the models
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            try:
                # Reshape for model input
                mel_spectrogram_input = mel_spectrogram_flattened  # Use the flattened mel spectrogram
                mfcc_input = np.expand_dims(mfcc, axis=0)  # Add batch dimension

                # Predict using the models (Melspec and MFCC models)
                melspec_pred = melspec_model.predict(mel_spectrogram_input)
                mfcc_pred = mfcc_model.predict(mfcc_input)

                # Decode predictions
                melspec_pred_class = np.argmax(melspec_pred, axis=1)[0]
                mfcc_pred_class = np.argmax(mfcc_pred, axis=1)[0]

                # Display results
                st.subheader("Hasil Prediksi:")
                st.write(f"**Melspec Model Prediksi:** Kelas {melspec_pred_class}")
                st.write(f"**MFCC Model Prediksi:** Kelas {mfcc_pred_class}")
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.markdown("""<hr><p style="text-align:center; font-size:12px; color:#555;">Aplikasi ini dibangun menggunakan Streamlit dan TensorFlow. Dataset burung Indonesia diambil dari Kaggle.</p><p style="text-align:center; font-size:12px; color:#555;">Desain oleh <strong>AI Model</strong>.</p>""", unsafe_allow_html=True)
