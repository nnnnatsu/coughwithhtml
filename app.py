import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model #type: ignore
from utils.preprocess import preprocess_audio
from utils.recorder import record_audio
from utils.denoise import denoise_audio
import shutil
import soundfile as sf
from utils.output import output
import pandas as pd

className = ['B, C, M, P']
# Load your trained model
def load_models():
    MP = load_model(r'D:\kikis_projects\cough sound detection\MODELS\USE_model2\MODEL_P01_lstm98.h5')
    MC = load_model(r'D:\kikis_projects\cough sound detection\MODELS\USE_model2\MODEL_C01_lstm94.h5')
    MB = load_model(r'D:\kikis_projects\cough sound detection\MODELS\USE_model2\MODEL_B01_lstm98.h5')
    MBP = load_model(r'D:\kikis_projects\cough sound detection\MODELS\USE_model2\MODEL_BP_lstm75.h5')
    return MP, MB, MC, MBP

MP, MB, MC, MBP = load_models()
# MP.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# MC.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# MB.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# MPB.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Temporary directory for saving audio files
temp_dir = 'temp_audio'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Set up the Streamlit app
st.title("Audio Classification AI")
st.write("Record an audio sample or upload an audio file to classify it.")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose input method", ["Record Audio", "Upload Audio"])

# Function to handle audio processing
def process_audio(audio_path):
    denoised_segments, sr = denoise_audio(audio_path, temp_dir)
    for i, segment in enumerate(denoised_segments):
        segment_path = os.path.join(temp_dir, f'segment_{i+1}.wav')
        sf.write(segment_path, segment, sr)
        st.audio(segment_path, format='audio/wav')

    # Preprocess the denoised audio
    features = preprocess_audio(audio_path)
    return features

def calPred(prediction):
    percent1 = round(prediction[0][0] * 100,2)
    percent2 = round(prediction[0][1] * 100,2)
    type = np.argmax(prediction, axis=1)
    return percent1, percent2, type

def get_predictions(data, models):
    MP, MB, MC, MBP = models
    
    # Predict with MP, MB, MC
    pred_MP = MP.predict(data)
    pred_MB = MB.predict(data)
    pred_MC = MC.predict(data)

    # Calculate confidence levels and values
    MP_percent1, MP_percent2, MP_type = calPred(pred_MP)
    print(f'MP: {MP_percent1}, {MP_percent2}, {MP_type}')
    MB_percent1, MB_percent2, MB_type = calPred(pred_MB)
    print(f'MB: {MB_percent1}, {MB_percent2}, {MB_type}')
    MC_percent1, MC_percent2, MC_type = calPred(pred_MC)
    print(f'MC: {MC_percent1}, {MC_percent2}, {MC_type}')

    # Initialize MBP prediction and confidence level
    pred_MBP, MBP_type = None, None
    
    # Check if MP and MB both predicted 1 and their confidence levels are close
    if MP_type == 1 and MB_type == 1 and abs(MP_type - MB_type) <= 10:
        # Predict with MBP
        pred_MBP = MBP.predict(data)
        MBP_percent1, MBP_percent2, MBP_type = calPred(pred_MBP)
        print(f'MBP: {MBP_percent1}, {MBP_percent2}, {MBP_type}')
    
    return {
        'MP': {'prediction': pred_MP, 'confidence': MP_type},
        'MB': {'prediction': pred_MB, 'confidence': MB_type},
        'MC': {'prediction': pred_MC, 'confidence': MC_type},
        'MBP': {'prediction': pred_MBP, 'confidence': MBP_type}
    }



# Record Audio
if option == "Record Audio":
    if st.button("Record Audio"):
        audio_path = os.path.join(temp_dir, 'recorded_audio.wav')
        record_audio(audio_path)
        data = process_audio(audio_path)
         # Get model predictions
        predictions = get_predictions(data, (MP, MB, MC, MBP))
        
        # Display results
        st.write("MP Prediction:", predictions['MP']['prediction'])
        st.write("MP Confidence:", predictions['MP']['confidence'])
        
        st.write("MB Prediction:", predictions['MB']['prediction'])
        st.write("MB Confidence:", predictions['MB']['confidence'])
        
        st.write("MC Prediction:", predictions['MC']['prediction'])
        st.write("MC Confidence:", predictions['MC']['confidence'])
        
        if predictions['MBP']['prediction'] is not None:
            st.write("MBP Prediction:", predictions['MBP']['prediction'])
            st.write("MBP Confidence:", predictions['MBP']['confidence'])
            
         # Clean up temporary files
        os.remove(audio_path)
        for segment_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, segment_file))


# Upload Audio
elif option == "Upload Audio":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.read())
        data = process_audio(audio_path)

        # Clean up temporary files
        os.remove(audio_path)
        for segment_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, segment_file))

# Cleanup temporary directory on app exit
def cleanup_temp_dir():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

cleanup_temp_dir()
