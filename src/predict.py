"""
Single audio file inference — GunshotCNN
Usage:
    python src/predict.py --file path/to/audio.wav
    python src/predict.py --file path/to/audio.wav --threshold 0.7
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.model import GunshotCNN


def load_model():
    checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint), (
        f"No checkpoint found at {checkpoint}. Run train.py first."
    )
    model = GunshotCNN()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_audio(file_path):
    """Load a .wav file and convert to mel-spectrogram tensor."""
    assert os.path.exists(file_path), f"Audio file not found: {file_path}"

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

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
        fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Shape: (1, 1, N_MELS, time) — batch of 1
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def predict(model, tensor, threshold=0.5):
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)
    gunshot_prob = probs[0, 1].item()
    predicted    = gunshot_prob >= threshold
    return predicted, gunshot_prob


def main():
    parser = argparse.ArgumentParser(description="GunshotCNN — single file inference")
    parser.add_argument("--file",      type=str,   required=True,  help="Path to .wav audio file")
    parser.add_argument("--threshold", type=float, default=0.5,    help="Detection threshold (default: 0.5)")
    args = parser.parse_args()

    print(f"\n[config] Device: {DEVICE}")
    print(f"[input]  File: {args.file}")
    print(f"[config] Threshold: {args.threshold}\n")

    model  = load_model()
    tensor = preprocess_audio(args.file)

    is_gunshot, confidence = predict(model, tensor, threshold=args.threshold)

    print("=" * 45)
    if is_gunshot:
        print("  ⚠  GUNSHOT DETECTED")
    else:
        print("  ✓  No threat detected")
    print(f"  Confidence : {confidence:.4f} ({confidence*100:.1f}%)")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {DEVICE}")
    print("=" * 45)


if __name__ == "__main__":
    main()