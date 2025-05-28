# utils/audio_tools.py

import torchaudio
import matplotlib.pyplot as plt
import torch

def plot_spectrogram(
    waveform,
    sample_rate,
    title="Mel Spectrogram",
    n_fft=1024,
    win_length=None,
    hop_length=None,
    n_mels=128,
    f_min=0.0,
    f_max=None,
    top_db=None
):
    """
    Plots a Mel spectrogram, but with explicit FFT parameters so that
    you know exactly how many freq bins you're getting.
    """
    if f_max is None:
        f_max = sample_rate / 2

    # If you don’t pass win_length or hop_length they default to n_fft and n_fft//4 respectively
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length or n_fft,
        hop_length=hop_length or n_fft // 4,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    mel_spec = mel_transform(torch.tensor(waveform))

    # convert to decibels, but don’t floor (top_db=None)
    db_spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(mel_spec)

    plt.figure(figsize=(10, 4))
    plt.imshow(db_spec.numpy(), aspect="auto", origin="lower")
    plt.title(title + f"  (n_fft={n_fft}, n_mels={n_mels})")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    return plt