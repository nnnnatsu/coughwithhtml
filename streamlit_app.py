import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
import base64
from streamlit.components.v1 import html as st_html

# Function to load and return model
@st.cache(allow_output_mutation=True)
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
    mfccs = librosa.util.normalize(mfccs)
    
    # Adjust number of time steps to match expected shape
    if mfccs.shape[1] < expected_time_steps:
        # Pad MFCCs if fewer time steps
        mfccs = np.pad(mfccs, ((0, 0), (0, expected_time_steps - mfccs.shape[1])))
    elif mfccs.shape[1] > expected_time_steps:
        # Truncate MFCCs if more time steps
        mfccs = mfccs[:, :expected_time_steps]
    
    # Add an additional dimension for compatibility with Conv2D input shape
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Shape (1, num_mfcc, expected_time_steps, 1)
    
    return mfccs

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
    <title>Cough Sound Diagnose</title>
    <!-- Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
        }
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .card {
            background: #ffffff;
            box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            padding: 40px;
            width: 400px;
            text-align: center;
        }
        h1 {
            font-weight: bold;
            margin-bottom: 20px;
        }
        .button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin-top: 20px;
            cursor: pointer;
            border-radius: 6px;
        }
        .button:hover {
            background-color: #0056b3;
        }
        audio {
            width: 100%;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Cough Sound Diagnose</h1>
            <button id="recordButton" class="button" onclick="toggleRecording()">Start Recording</button>
            <audio id="audioPlayback" controls style="display: none;"></audio>
        </div>
    </div>
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
                document.getElementById("recordButton").innerText = "Start Recording";
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
                        document.getElementById("audioPlayback").style.display = "block";

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
                    document.getElementById("recordButton").innerText = "Stop Recording";
                })
                .catch(error => {
                    console.error('Error accessing media devices.', error);
                });
        }
    </script>
</body>
</html>
"""

# Load the model
model = load_model("MODEL_CNN.h5")

# Set up Streamlit layout
st.title("Cough Sound Diagnosis")
st.markdown("Record your cough sound below:")

# Embed the HTML string
st_html(html_string)

# Audio input from the custom component
audio_data = st.experimental_get_query_params().get("audio", None)

if audio_data:
    # Convert the audio data back to bytes
    audio_data = bytes(audio_data)

    # Preprocess the audio data
    preprocessed_input = preprocess_input(audio_data)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    predicted_class = np.argmax(prediction)

    # Map predicted class to condition
    condition_mapping = {0: 'Normal', 1: 'COVID-19', 2: 'Pneumonia', 3: 'Bronchitis'}
    condition = condition_mapping.get(predicted_class, 'Unknown')

    # Display the prediction
    st.write(f"Predicted Condition: {condition}")

    # Create a download link for the recorded audio
    download_link = get_download_link(audio_data, "recorded_cough.wav", "Download recorded cough sound")
    st.markdown(download_link, unsafe_allow_html=True)
else:
    st.write("No audio data received.")
