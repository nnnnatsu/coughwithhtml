import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io

# Function to load and return model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the input data
def preprocess_input(audio_data, num_mfcc=13, n_fft=2048, hop_length=512, expected_time_steps=120):
    with io.BytesIO(audio_data) as f:
        audio, sr = sf.read(f)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_normalized = mfccs.T

    if mfccs_normalized.shape[0] < expected_time_steps:
        mfccs_normalized = np.pad(mfccs_normalized, ((0, expected_time_steps - mfccs_normalized.shape[0]), (0, 0)))
    elif mfccs_normalized.shape[0] > expected_time_steps:
        mfccs_normalized = mfccs_normalized[:expected_time_steps, :]
    
    mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]
    
    return mfccs_reshaped

# Function to display consent popup
def show_consent_popup():
    st.markdown("""
    <style>
        .popup-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
        }
        .popup-content {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            max-width: 600px;
            text-align: left;
        }
        .popup-content h2 {
            margin-top: 0;
        }
        .popup-buttons {
            display: flex;
            justify-content: flex-end;
        }
        .popup-buttons button {
            margin-left: 10px;
        }
    </style>
    <div class="popup-container">
        <div class="popup-content">
            <div id="en" style="display: block;">
                <h2>Medical Information Details</h2>
                <p>This application is designed to analyze users' cough sounds to assess the risk of pneumonia, bronchitis, and COVID-19.</p>
                <p><strong>Limitations of Information:</strong></p>
                <ul>
                    <li>This analysis cannot replace the examination and diagnosis by a healthcare professional.</li>
                    <li>The results provided are only preliminary predictions and may not always be accurate.</li>
                    <li>Users should consult a doctor or medical expert for an accurate diagnosis.</li>
                </ul>
                <p><strong>Additional Information:</strong></p>
                <ul>
                    <li>The recorded sounds will not be saved or stored in our system.</li>
                    <li>This application does not collect any personally identifiable information, such as name, address, or other personal data.</li>
                    <li>The data used for analysis will be automatically deleted after the analysis is complete.</li>
                </ul>
                <p><strong>Privacy Rights:</strong></p>
                <ul>
                    <li>This application respects and complies with data protection principles as required by applicable law.</li>
                    <li>All data used for analysis will be kept confidential and will not be disclosed to third parties without the user's consent.</li>
                </ul>
                <h2>Consent Agreement</h2>
                <p>By using this application, you agree to the following terms:</p>
                <ul>
                    <li><strong>Voluntary Participation:</strong> Your use of this application is voluntary. There is no coercion involved.</li>
                    <li><strong>Data Usage:</strong> The application will use your cough sound for analysis without saving or storing any data. All data will be automatically deleted after the analysis is complete.</li>
                    <li><strong>Accuracy of Information:</strong> The information derived from the analysis is only a preliminary prediction and cannot replace a diagnosis by a healthcare professional.</li>
                    <li><strong>Data Protection:</strong> Your information will not be used for identification or shared with third parties without your consent.</li>
                </ul>
                <div class="popup-buttons">
                    <button onclick="acceptConsent()">I accept and understand the above terms</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        function acceptConsent() {
            window.parent.postMessage({type: 'acceptConsent'}, '*');
        }
    </script>
    """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    if 'consent_given' not in st.session_state:
        st.session_state.consent_given = False

    if not st.session_state.consent_given:
        show_consent_popup()
    else:
        st.title('Audio Classification App')
        st.write('Upload an audio file and get predictions!')

        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format='audio/wav')

            processed_data = preprocess_input(audio_data)
            model_path = 'MODEL_CNN.h5'
            model = load_model(model_path)

            if processed_data is not None:
                prediction = model.predict(processed_data)
                st.write('Prediction:')
                st.write(prediction)

st.script_run_ctx.add_callback(lambda msg: st.session_state.update({'consent_given': msg.get('type') == 'acceptConsent'}))

if __name__ == '__main__':
    main()
