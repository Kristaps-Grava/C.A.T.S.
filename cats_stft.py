import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf




# Load audio file
audio, sample_rate = librosa.load("Never gonna give you up - Rick Astley.mp3", sr=None)



print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Duration:", len(audio) / sample_rate, "seconds")





plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title("Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()





#Framing the audio
frame_size = 2048
hop_size = 512

frames = []

for i in range(0, len(audio) - frame_size, hop_size):
    frame = audio[i:i + frame_size]
    frames.append(frame)

frames = np.array(frames)

print("Frames shape:", frames.shape)











#applying the window function
window = np.hanning(frame_size)

windowed_frames = frames * window



#Applying the actual FFT to each frame
fft_frames = np.fft.rfft(windowed_frames)




#Creating a spectrogram by taking the magnitude of the FFT for each frame
plt.figure(figsize=(12, 8))

# Waveform
plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title("Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

# Spectrogram
plt.subplot(2, 1, 2)

plt.imshow(
    20 * np.log10(magnitude.T + 1e-8),
    origin='lower',
    aspect='auto',
    cmap='magma'
)

plt.title("Spectrogram")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")
plt.colorbar(label="dB")

plt.tight_layout()
plt.show()





#Inverse FFT
reconstructed_frames = np.fft.irfft(fft_frames)






#Rebuild 
output = np.zeros(len(audio))

for i, frame in enumerate(reconstructed_frames):
    start = i * hop_size
    output[start:start + frame_size] += frame




#Save reconstructed audio
sf.write("reconstructed.wav", output, sample_rate)
