import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io

# Function to load and return model
@st.cache_resource(allow_output_mutation=True)
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

# Main function to run the Streamlit app
def main():
    st.title('Audio Classification App')

    # Check for consent
    if 'consent' not in st.session_state:
        st.session_state.consent = False

    if not st.session_state.consent:
        st.markdown("""
            ## Medical Information and Consent
            This application is designed to analyze users' cough sounds to assess the risk of pneumonia, bronchitis, and COVID-19.

            **Limitations of Information:**
            - This analysis cannot replace the examination and diagnosis by a healthcare professional.
            - The results provided are only preliminary predictions and may not always be accurate.
            - Users should consult a doctor or medical expert for an accurate diagnosis.

            **Additional Information:**
            - The recorded sounds will not be saved or stored in our system.
            - This application does not collect any personally identifiable information, such as name, address, or other personal data.
            - The data used for analysis will be automatically deleted after the analysis is complete.

            **Privacy Rights:**
            - This application respects and complies with data protection principles as required by applicable law.
            - All data used for analysis will be kept confidential and will not be disclosed to third parties without the user's consent.

            ### Consent Agreement
            By using this application, you agree to the following terms:
            - **Voluntary Participation:** Your use of this application is voluntary. There is no coercion involved.
            - **Data Usage:** The application will use your cough sound for analysis without saving or storing any data. All data will be automatically deleted after the analysis is complete.
            - **Accuracy of Information:** The information derived from the analysis is only a preliminary prediction and cannot replace a diagnosis by a healthcare professional.
            - **Data Protection:** Your information will not be used for identification or shared with third parties without your consent.
        """)
        if st.button('I accept and understand the above terms'):
            st.session_state.consent = True
            st.experimental_rerun()

    if st.session_state.consent:
        st.write('Upload an audio file and get predictions!')
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format='audio/wav')  # Display the audio file

            # Preprocess the audio data
            processed_data = preprocess_input(audio_data)

            # Load the model
            model_path = 'MODEL_CNN.h5'  # Replace with your model path
            model = load_model(model_path)

            # Make prediction
            if processed_data is not None:
                prediction = model.predict(processed_data)
                st.write('Prediction:')
                st.write(prediction)  # Display the prediction results

if __name__ == '__main__':
    main()
