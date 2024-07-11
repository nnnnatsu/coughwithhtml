import os
import librosa
import numpy as np
import soundfile as sf

def find_peaks(y, sr, threshold=0.5, min_distance=1000):
    peaks = np.where(np.abs(y) > threshold)[0]
    
    filtered_peaks = []
    last_peak = -min_distance
    for peak in peaks:
        if peak - last_peak >= min_distance:
            filtered_peaks.append(peak)
            last_peak = peak
    
    return filtered_peaks

def extract_segment(y, sr, peak_index, segment_duration_ms):
    segment_length = int(segment_duration_ms * sr / 1000)
    half_segment = segment_length // 2

    start_index = max(0, peak_index - half_segment)
    end_index = min(len(y), peak_index + half_segment)

    segment = y[start_index:end_index]
    if len(segment) < segment_length:
        padding = np.zeros(segment_length - len(segment))
        segment = np.concatenate((segment, padding))
    
    return segment

def denoise_audio(input_path, output_dir, threshold_factor=0.85, min_distance_sec=1.0, segment_duration_ms=1000):
    y, sr = librosa.load(input_path, sr=None)

    threshold = threshold_factor * max(np.abs(y))
    min_distance = int(min_distance_sec * sr)
    peaks = find_peaks(y, sr, threshold, min_distance)

    segments = []
    for i, peak_index in enumerate(peaks):
        segment = extract_segment(y, sr, peak_index, segment_duration_ms)
        segments.append(segment)

        if output_dir:
            output_path = os.path.join(output_dir, f'segment_{i+1}.wav')
            sf.write(output_path, segment, sr)
            print(f"Segment {i+1} saved to {output_path}")
    print('\naudio was denoised\n')

    return segments, sr

# Example usage:
# input_path = 'path/to/audio/file.wav'
# output_dir = 'path/to/output/directory'
# segments, sr = denoise_audio(input_path, output_dir)
