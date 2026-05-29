import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks

print("Analysis started")
# Load audio file
audio, sample_rate = librosa.load(
    "Never gonna give you up - Rick Astley.mp3",
    sr=None
)

harmonic, percussive = librosa.effects.hpss(audio)

print("Audio loaded")

# Frame the audio
frame_size = 2048
hop_size = 512

frames = []

for i in range(0, len(audio) - frame_size, hop_size):
    frame = audio[i:i + frame_size]
    frames.append(frame)

frames = np.array(frames)

print("Audio split into frames")

# Apply windowing by using a hanning function, which is the standart in music analysis
window = np.hanning(frame_size)
windowed_frames = frames * window

# Fourier Transform
fft_frames = np.fft.rfft(windowed_frames)

print("Frames windowed")
bin_to_hz = sample_rate / frame_size  # ~10.7 Hz per bin

def get_harmonic_peaks(peaks, magnitudes, bin_to_hz, tolerance=0.10, min_harmonics=3):
    """
    Among all detected peaks, find the subset that forms a harmonic series.
    Returns indices of peaks that are part of the strongest harmonic series found.
    """
    best_harmonic_peaks = set()
    best_score = 0

    for candidate in peaks:
        f0 = candidate * bin_to_hz
        if f0 < 50:  # skip DC and infrasound
            continue

        matched = []
        for n in range(1, 17):  # check up to 16 harmonics
            expected_bin = (f0 * n) / bin_to_hz
            if expected_bin >= len(magnitudes):
                break
            window = max(int(expected_bin * tolerance), 2)
            lo = int(expected_bin - window)
            hi = int(expected_bin + window)
            nearby = [p for p in peaks if lo <= p <= hi]
            if nearby:
                best = max(nearby, key=lambda p: magnitudes[p])
                matched.append(best)

        if len(matched) >= min_harmonics and len(matched) > best_score:
            best_score = len(matched)
            best_harmonic_peaks = set(matched)

    return best_harmonic_peaks

filtered_frames = []

for frame in fft_frames:
    magnitudes = np.abs(frame)
    noise_floor = np.median(magnitudes)
    threshold = noise_floor * 50

    peaks, props = find_peaks(magnitudes, height=threshold, distance=5, prominence=noise_floor * 2)
    harmonic_peaks = get_harmonic_peaks(peaks, magnitudes, bin_to_hz)
    # build spread: each peak index plus its neighbours
    spread = set()
    for p in harmonic_peaks:
        spread.add(p - 1)
        spread.add(p)
        spread.add(p + 1)

    # zero out all bins not in spread
    filtered_frame = frame.copy()
    mask = np.isin(np.arange(len(frame)), list(spread))
    filtered_frame[~mask] = 0

    filtered_frames.append(filtered_frame)

filtered_frames = np.array(filtered_frames)

cutoff_bin = int(8000 / bin_to_hz)  # convert 1kHz to bin index
low_cutoff_bin = int(50 / bin_to_hz)

double_filtered_frames = []

for frame in filtered_frames:
    filtered_frame = frame.copy()
    filtered_frame[:low_cutoff_bin] = 0
    filtered_frame[cutoff_bin:] = 0
    double_filtered_frames.append(filtered_frame)

double_filtered_frames = np.array(double_filtered_frames)

print("Frames filtered")

# Create spectrogram
magnitude = np.abs(fft_frames)

magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))



# Inverse FFT
reconstructed_frames = np.fft.irfft(double_filtered_frames)

# Rebuild audio
output = np.zeros(len(audio))

for i, frame in enumerate(reconstructed_frames):
    start = i * hop_size
    output[start:start + frame_size] += frame

#start creating new frames with longer period

# Frame the audio
frame_size = 2048*20
hop_size = 512*20
bin_to_hz = sample_rate/frame_size
frames = []

for i in range(0, len(output) - frame_size, hop_size):
    frame = output[i:i + frame_size]
    frames.append(frame)

frames = np.array(frames)

# Apply windowing by using a hanning function, which is the standart in music analysis
window = np.hanning(frame_size)
windowed_frames = frames * window

# Fourier Transform
fft_frames = np.fft.rfft(windowed_frames)

triple_filtered_frames = []
low_cutoff_bin = int(10 / bin_to_hz)
high_cutoff_bin = int(8000 / bin_to_hz)
for frame in fft_frames:
    filtered_frame = frame.copy()
    filtered_frame[:low_cutoff_bin] = 0
    filtered_frame[high_cutoff_bin:] = 0
    triple_filtered_frames.append(filtered_frame)

triple_filtered_frames = np.array(triple_filtered_frames)

# Inverse FFT
reconstructed_frames = np.fft.irfft(triple_filtered_frames)

# Rebuild audio
output = []
output = np.zeros(len(audio))

for i, frame in enumerate(reconstructed_frames):
    start = i * hop_size
    output[start:start + frame_size] += frame

# Save reconstructed audio
sf.write("wint_test_filtered.wav", output, sample_rate)
print("Reconstructed audio saved as reconstructed.wav")