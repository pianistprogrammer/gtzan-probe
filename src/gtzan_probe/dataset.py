"""Shared dataset and audio-loading utilities."""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from .config import SR, N_MELS, N_FFT, HOP, FIXED_W, DURATION, SEG_S


# ── SpecAugment helpers ──────────────────────────────────────────────────────
def _freq_mask(S, max_width=12):
    """Zero out a random horizontal band (frequency masking)."""
    n_mels = S.shape[0]
    w = np.random.randint(1, max_width + 1)
    f0 = np.random.randint(0, n_mels - w)
    S[f0 : f0 + w, :] = 0.0
    return S


def _time_mask(S, max_width=16):
    """Zero out a random vertical band (time masking)."""
    n_frames = S.shape[1]
    w = np.random.randint(1, min(max_width + 1, n_frames))
    t0 = np.random.randint(0, n_frames - w)
    S[:, t0 : t0 + w] = 0.0
    return S


class GTZANDataset(Dataset):
    """Loads GTZAN clips as fixed-size mel spectrogram tensors."""

    def __init__(self, meta_df, label_encoder, segment_s=SEG_S, augment=False,
                 segments_per_clip=1):
        self.records = meta_df.reset_index(drop=True)
        self.le = label_encoder
        self.seg_len = int(segment_s * SR)
        self.augment = augment
        self.segments_per_clip = segments_per_clip if augment else 1
        # Pre-load audio once to avoid repeated disk I/O
        self._audio_cache = {}

    def __len__(self):
        return len(self.records) * self.segments_per_clip

    def _load_audio(self, filepath):
        if filepath not in self._audio_cache:
            try:
                y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
            except Exception:
                y = np.zeros(int(SR * DURATION))
            self._audio_cache[filepath] = y
        return self._audio_cache[filepath]

    def __getitem__(self, idx):
        real_idx = idx % len(self.records)
        row = self.records.iloc[real_idx]
        label = self.le.transform([row.genre])[0]

        y = self._load_audio(row.filepath).copy()

        # Random segment for augmentation
        if self.augment and len(y) > self.seg_len:
            start = np.random.randint(0, len(y) - self.seg_len)
            y = y[start : start + self.seg_len]
        else:
            if len(y) >= self.seg_len:
                y = y[: self.seg_len]
            else:
                y = np.pad(y, (0, self.seg_len - len(y)))

        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

        # Resize time axis to fixed width
        if S_db.shape[1] != FIXED_W:
            from skimage.transform import resize
            S_db = resize(S_db, (N_MELS, FIXED_W), anti_aliasing=True)

        # SpecAugment (frequency + time masking)
        if self.augment:
            S_db = _freq_mask(S_db.copy())
            S_db = _time_mask(S_db)

        tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)
        return tensor, label, row.filename


class GTZANMultiCropDataset(Dataset):
    """Test-time dataset: yields ALL non-overlapping segments per clip for aggregation."""

    def __init__(self, meta_df, label_encoder, segment_s=SEG_S):
        self.le = label_encoder
        self.seg_len = int(segment_s * SR)
        self.items = []  # (filepath, start_sample, genre, filename, clip_idx)
        for clip_idx, (_, row) in enumerate(meta_df.iterrows()):
            try:
                y, _ = librosa.load(row.filepath, sr=SR, duration=DURATION)
            except Exception:
                y = np.zeros(int(SR * DURATION))
            n_segs = max(1, len(y) // self.seg_len)
            for s in range(n_segs):
                self.items.append((y, s * self.seg_len, row.genre, row.filename, clip_idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        y_full, start, genre, filename, clip_idx = self.items[idx]
        label = self.le.transform([genre])[0]
        y = y_full[start : start + self.seg_len]
        if len(y) < self.seg_len:
            y = np.pad(y, (0, self.seg_len - len(y)))
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        if S_db.shape[1] != FIXED_W:
            from skimage.transform import resize
            S_db = resize(S_db, (N_MELS, FIXED_W), anti_aliasing=True)
        tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
        return tensor, label, filename, clip_idx


def load_spectrogram(filepath, start_s=0.0):
    """Load a single clip as a normalised mel spectrogram tensor (1, 1, 128, W)."""
    y, _ = librosa.load(filepath, sr=SR, offset=start_s, duration=SEG_S)
    if len(y) < int(SEG_S * SR):
        y = np.pad(y, (0, int(SEG_S * SR) - len(y)))

    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    if S_db.shape[1] != FIXED_W:
        from skimage.transform import resize
        S_db = resize(S_db, (N_MELS, FIXED_W), anti_aliasing=True)

    return S_db, torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
