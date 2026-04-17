import librosa
import numpy as np

def load_and_preprocess_audio(
    audio_path,
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
    max_duration=5.0
):
    """
    Loads audio, trims/pads to max_duration, returns log-mel spectrogram.
    Output shape: (1, time, n_mels)
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)
    max_len = int(sample_rate * max_duration)

    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # normalize to mean=0, std=1
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    # shape (1, time, n_mels)
    log_mel = np.expand_dims(log_mel.T, axis=0)
    return log_mel