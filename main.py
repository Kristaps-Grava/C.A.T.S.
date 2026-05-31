import numpy as np
import librosa
import soundfile as sf


class AudioCleaner:
  def __init__(self, audio_input_file, audio_output_file):
    self.audio_input_file = audio_input_file
    self.audio_output_file = audio_output_file

  def load_audio(self):
    self.audio, self.sample_rate = librosa.load(self.audio_input_file, sr=None)

  def save_audio(self):
    sf.write(self.audio_output_file, self.audio, self.sample_rate)

  def split_audio_to_frames(self):
    self.frame_size, self.hop_size = 2048, 512

    self.frames = []
    for i in range(0, len(self.audio) - self.frame_size, self.hop_size):
        frame = self.audio[i:i + self.frame_size]
        self.frames.append(frame)

    self.frames = np.array(self.frames)

  def window_frames(self):
    window = np.hanning(self.frame_size)
    self.frames = self.frames * window

  def filter_frames(self):
    bin_to_hz = self.sample_rate/self.frame_size
    low_cutoff_bin = int(80 / bin_to_hz)
    high_cutoff_bin = int(4000 / bin_to_hz)

    # removing all frequencies that don't exceed a certain threshold
    noise_floor = np.median(self.fft_frames)
    threshold = noise_floor * 5
    noise_mask = self.fft_frames < threshold
    self.fft_frames[noise_mask] *= 0.05

    # voice
    vocal_band = self.fft_frames[low_cutoff_bin:high_cutoff_bin, :]
    raw_frame_energies = np.sum(vocal_band, axis=0)

    smoothing_window = 15
    smoothed_energies = np.convolve(raw_frame_energies, np.ones(smoothing_window) / smoothing_window, mode='same')

    print(f"Loudest frame:  {np.max(smoothed_energies)}")
    print(f"Median frame:   {np.median(smoothed_energies)}")
    print(f"Quietest frame: {np.min(smoothed_energies)}")

    speech_threshold = 160
    silence_frames  = smoothed_energies < speech_threshold
    self.fft_frames[:, silence_frames] *= 0.1

    self.fft_frames[:low_cutoff_bin, :] = 0
    self.fft_frames[high_cutoff_bin:, :] = 0

    self.fft_frames =  self.fft_frames * np.exp(1j*self.fft_frame_phases)

  def reconstruct_audio(self):
    self.frames = np.fft.irfft(self.fft_frames.T)

    window = np.hanning(self.frame_size)
    self.frames *= window

    output_length = (len(self.frames)-1)*self.hop_size+self.frame_size
    reconstructed = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i, frame in enumerate(self.frames):
      start = i*self.hop_size
      reconstructed[start:start + self.frame_size] += frame
      window_sum[start:start + self.frame_size] += window ** 2

    reconstructed /= np.maximum(window_sum, 1e-8)

    self.audio = librosa.resample(reconstructed, orig_sr=self.sample_rate, target_sr=self.sample_rate)

  def run(self):
    self.load_audio()
    self.split_audio_to_frames()
    self.window_frames()

    fft = np.fft.rfft(self.frames).T
    self.fft_frames = np.abs(fft)
    self.fft_frame_phases = np.angle(fft)

    self.filter_frames()
    self.reconstruct_audio()
    self.save_audio()


if __name__ == '__main__':
    input_file = "Never gonna give you up - Rick Astley.mp3"
    output_file = "output.wav"

    cleaner = AudioCleaner(input_file, output_file)
    cleaner.run()