import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.dataset import UrbanSoundDataset
from src.model import GunshotCNN, count_parameters


def get_class_weights(dataset):
    """Compute inverse-frequency class weights to handle class imbalance."""
    labels = [label for _, label in dataset.samples]
    n_total    = len(labels)
    n_gunshot  = sum(labels)
    n_other    = n_total - n_gunshot
    # Weight inversely proportional to frequency
    w_other    = n_total / (2 * n_other)
    w_gunshot  = n_total / (2 * n_gunshot)
    return torch.tensor([w_other, w_gunshot], dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (mels, labels) in enumerate(loader):
        mels   = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            logits = model(mels)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * mels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += mels.size(0)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for mels, labels in loader:
        mels   = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(enabled=USE_AMP):
            logits = model(mels)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * mels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += mels.size(0)

    return total_loss / total, correct / total


def save_plots(train_losses, val_losses, train_accs, val_accs):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Training curves saved to {path}")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = UrbanSoundDataset(split="train")
    test_ds  = UrbanSoundDataset(split="test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GunshotCNN().to(DEVICE)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # ── Class weights (handles imbalance) ─────────────────────────────────────
    class_weights = get_class_weights(train_ds)
    print(f"Class weights — Non-gunshot: {class_weights[0]:.3f} | "
          f"Gunshot: {class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    scaler = GradScaler(enabled=USE_AMP)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...\n")
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s | Peak VRAM: {vram_mb:.0f}MB")

        if SAVE_BEST_ONLY and val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    save_plots(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()