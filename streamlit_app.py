import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
import base64
from streamlit.components.v1 import html

# Function to load and return model
@st.cache_data(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the input data
def preprocess_input(audio_data, num_mfcc=13, n_fft=2048, hop_length=512, expected_time_steps=120):
    # Save audio data to a temporary file
    with io.BytesIO(audio_data) as f:
        audio, sr = sf.read(f)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Normalize MFCCs
    mfccs_normalized = mfccs.T  # Transpose to have shape (time_steps, num_mfcc)
    
    # Adjust number of time steps to match expected shape
    if mfccs_normalized.shape[0] < expected_time_steps:
        # Pad MFCCs if fewer time steps
        mfccs_normalized = np.pad(mfccs_normalized, ((0, expected_time_steps - mfccs_normalized.shape[0]), (0, 0)))
    elif mfccs_normalized.shape[0] > expected_time_steps:
        # Truncate MFCCs if more time steps
        mfccs_normalized = mfccs_normalized[:expected_time_steps, :]
    
    # Add an additional dimension for compatibility with Conv2D input shape
    mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]  # Shape (1, expected_time_steps, num_mfcc, 1)
    
    return mfccs_reshaped

# Function to create a download link for the audio file
def get_download_link(data, filename, text):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# HTML for recording
html_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Application</title>
    <!-- Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
        }
        header {
            font-family: 'Poppins', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background-color: #f0f0f0;
        }
        h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 3em;
        }
        .left_text{
            align-items: left;
        }
        h2 {
            font-family: 'Poppins', sans-serif;
            font-weight: 100;
            font-size: 1.5em;
        }
        .transparent{
            opacity: 0.8;
        }
        .custom-button {
            padding: 10px 20px;
            margin: 10px;
            font-size: 1.2em;
            color: rgb(0, 0, 0);
            border: 1.5px solid rgb(0, 0, 0);
            cursor: pointer;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .custom-button-round {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            border: 1.5px solid rgb(0, 0, 0);
            color: rgb(0, 0, 0);
            font-size: 1.2em;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            text-decoration: none;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .custom-button:hover {
            background-color: #dcdcdc;
            color: #000000;
        }
        .custom-button-round.active {
            background-color: #007BFF;
            color: #ffffff;
        }
        .custom-button-round:hover {
            background-color: #dcdcdc;
            color: #000000;
        }
        #content {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #e0e0e0;
        }
    </style>
</head>
<body>
    <section id="home">
        <header class="text-center">
            <h1>Cough Sound Diagnose</h1>
            <h2 class="transparent"><i>Covid-19, Pneumonia, and Bronchitis using Artificial Intelligence</i></h2>
            <button onclick="scrollToSection()" class="custom-button"><u>click here to start&nbsp;</u>
                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor" class="bi bi-arrow-down-right-square" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M15 2a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1zM0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2zm5.854 3.146a.5.5 0 1 0-.708.708L9.243 9.95H6.475a.5.5 0 1 0 0 1h3.975a.5.5 0 0 0 .5-.5V6.475a.5.5 0 1 0-1 0v2.768z"/>
                </svg>
            </button>
        </header>
    </section>
    <section id="content">
        <div>
            <h1 style="margin-top: -220px;">Record your <br>cough here</h1>
        </div>
        <div>
            <div><audio id="audioPlayback" controls style="margin-top: 0px;"></audio></div>
            <button id="recordButton" class="custom-button-round" style="margin-top: 200px;" onclick="toggleRecording()"><svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor" class="bi bi-mic" viewBox="0 0 16 16">
                <path d="M3.5 6.5A.5.5 0 0 1 4 7v1a4 4 0 0 0 8 0V7a.5.5 0 0 1 1 0v1a5 5 0 0 1-4.5 4.975V15h3a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1h3v-2.025A5 5 0 0 1 3 8V7a.5.5 0 0 1 .5-.5"/>
                <path d="M10 8a2 2 0 1 1-4 0V3a2 2 0 1 1 4 0zM8 0a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V3a3 3 0 0 0-3-3"/>
            </svg></button>
        </div>
    </section>
    <!-- Bootstrap JS, Popper.js, and jQuery -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script>
        let isRecording = false;
        let mediaRecorder;
        let recordedChunks = [];

        function toggleRecording() {
            if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById("recordButton").classList.remove("active");
            } else {
                startRecording();
            }
        }

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => {
                        if (event.data.size > 0) {
                            recordedChunks.push(event.data);
                        }
                    };
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        document.getElementById("audioPlayback").src = audioUrl;

                        // Send audio to Streamlit
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64data = reader.result.split(',')[1];
                            const audioData = Uint8Array.from(atob(base64data), c => c.charCodeAt(0));
                            Streamlit.setComponentValue(audioData);
                        };
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById("recordButton").classList.add("active");
                })
                .catch(error => {
                    console.error('Error accessing media devices.', error);
                });
        }

        function scrollToSection() {
            document.getElementById('content').scrollIntoView({ behavior: 'smooth' });
        }
