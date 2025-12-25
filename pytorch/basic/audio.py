import torch
import torchaudio
import torchaudio.transforms as T

# 假设 waveform 和 sr 来自 torchaudio.load
# waveform: (channels, num_frames), sr: int
original_sr = 16000
resample_sr = 8000

# 创建一个虚拟波形
waveform_16k = torch.randn(1, original_sr * 2)  # 2秒的随机噪声

resampler = T.Resample(orig_freq=original_sr, new_freq=resample_sr)
resampled_waveform = resampler(waveform_16k)

print(f"Original shape: {waveform_16k.shape}")
print(f"Resampled shape: {resampled_waveform.shape}")


# waveform: (channels, num_frames)
n_fft = 400
hop_length = 160
win_length = n_fft  # 通常等于 n_fft

# 创建一个虚拟波形
waveform_mono = torch.randn(1, 16000)  # 1秒单声道

spectrogram_transform = T.Spectrogram(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    power=2.0,  # 计算功率谱, None 表示幅度谱
)
spec = spectrogram_transform(waveform_mono)  # Output: (channel, freq_bins, time_frames)

print(f"Spectrogram shape: {spec.shape}")  # (1, n_fft/2 + 1, num_frames)
