import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="output.wav", duration=10, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file
    print("Recording complete.")
    return filename
