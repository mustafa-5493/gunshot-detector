import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def audio_to_melspectrogram(file_path):
    """Load audio and convert to mel-spectrogram tensor."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Pad if shorter than DURATION
    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX
    )

    # Convert to log scale (dB) — more perceptually meaningful
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    return mel_db.astype(np.float32)


def preprocess_and_save():
    """Convert all UrbanSound8K audio to mel-spectrograms and save as .npy files."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    metadata_path = os.path.join(DATA_RAW, "metadata", "UrbanSound8K.csv")
    df = pd.read_csv(metadata_path)

    print(f"Processing {len(df)} audio files...")
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fold       = row["fold"]
        filename   = row["slice_file_name"]
        class_id   = row["classID"]
        label      = 1 if class_id == GUNSHOT_CLASS_ID else 0

        audio_path = os.path.join(DATA_RAW, "audio", f"fold{fold}", filename)
        save_name  = f"fold{fold}_{filename.replace('.wav', '')}_label{label}.npy"
        save_path  = os.path.join(DATA_PROCESSED, save_name)

        if os.path.exists(save_path):
            continue  # already processed

        mel = audio_to_melspectrogram(audio_path)
        if mel is None:
            skipped += 1
            continue

        np.save(save_path, mel)

    print(f"Done. Skipped {skipped} files.")


class UrbanSoundDataset(Dataset):
    """PyTorch Dataset for preprocessed mel-spectrogram .npy files."""

    def __init__(self, split="train"):
        """
        split: "train" uses folds 1-9, "test" uses fold 10
        """
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.samples = []  # list of (npy_path, label)

        for fname in os.listdir(DATA_PROCESSED):
            if not fname.endswith(".npy"):
                continue

            # Extract fold number from filename
            fold_num = int(fname.split("_")[0].replace("fold", ""))
            label    = int(fname.split("_label")[-1].replace(".npy", ""))

            if split == "test" and fold_num == TEST_FOLD:
                self.samples.append((os.path.join(DATA_PROCESSED, fname), label))
            elif split == "train" and fold_num != TEST_FOLD:
                self.samples.append((os.path.join(DATA_PROCESSED, fname), label))

        print(f"[{split.upper()}] {len(self.samples)} samples | "
              f"Gunshots: {sum(1 for _, l in self.samples if l == 1)} | "
              f"Non-gunshots: {sum(1 for _, l in self.samples if l == 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)                          # (N_MELS, time)
        mel = torch.tensor(mel).unsqueeze(0)         # (1, N_MELS, time)
        return mel, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    print("Preprocessing UrbanSound8K...")
    preprocess_and_save()
    print("\nVerifying datasets...")
    train_ds = UrbanSoundDataset(split="train")
    test_ds  = UrbanSoundDataset(split="test")
    mel, label = train_ds[0]
    print(f"Sample shape: {mel.shape} | Label: {label.item()}")