import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.dataset import UrbanSoundDataset
from src.model import GunshotCNN


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for mels, labels in loader:
        mels = mels.to(DEVICE, non_blocking=True)

        with autocast(enabled=USE_AMP):
            logits = model(mels)

        probs  = torch.softmax(logits, dim=1)[:, 1]  # probability of gunshot
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Gunshot", "Gunshot"],
        yticklabels=["Non-Gunshot", "Gunshot"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Gunshot Detector")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"Confusion matrix saved to {path}")


def plot_roc_curve(labels, probs):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc          = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Gunshot Detector")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    print(f"ROC curve saved to {path}")
    return auc


def main():
    checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint), f"No checkpoint found at {checkpoint}"

    model = GunshotCNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    print(f"Loaded checkpoint: {checkpoint}")

    test_ds     = UrbanSoundDataset(split="test")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    labels, preds, probs = get_predictions(model, test_loader)

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(
        labels, preds,
        target_names=["Non-Gunshot", "Gunshot"],
        digits=4
    ))

    auc = plot_roc_curve(labels, probs)
    print(f"ROC-AUC Score: {auc:.4f}")

    plot_confusion_matrix(labels, preds)


if __name__ == "__main__":
    main()