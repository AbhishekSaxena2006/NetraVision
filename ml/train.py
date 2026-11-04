import os
import sys
import time
import json
import argparse
import pandas as pd

# Ensure standard output handles UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import config
from dataset import get_dataloaders
from model import get_model
from utils import compute_class_weights, calculate_qwk, EarlyStopping

METRICS_JSON_PATH = os.path.join(config.OUTPUT_DIR, "training_metrics.json")
METRICS_CSV_PATH = os.path.join(config.OUTPUT_DIR, "training_history.csv")

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        # Live progress every 25 batches
        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            current_acc = np.mean(np.array(all_preds) == np.array(all_targets))
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d}] Batch [{batch_idx+1:03d}/{total_batches:03d}] "
                  f"| Loss: {batch_loss:.4f} | Running Acc: {current_acc*100:.1f}%", flush=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    epoch_qwk = calculate_qwk(all_targets, all_preds)
    return epoch_loss, epoch_acc, epoch_qwk

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    val_qwk = calculate_qwk(all_targets, all_preds)
    return val_loss, val_acc, val_qwk

def save_metrics_history(history):
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    pd.DataFrame(history).to_csv(METRICS_CSV_PATH, index=False)

def run_training(epochs=config.NUM_EPOCHS, dry_run=False, resume=False):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    if device.type == "cpu":
        # Utilize multiple CPU threads for acceleration
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"[INFO] Using compute device: CPU with {num_threads} parallel worker threads")
    else:
        print(f"[INFO] Using compute device: CUDA GPU ({torch.cuda.get_device_name(0)})")
        print(f"[INFO] Dedicated VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

    print("[DATA] Preparing stratified train and validation data loaders...")
    train_loader, val_loader, train_df, val_df = get_dataloaders()
    print(f"       Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    print(f"       Batch size: {config.BATCH_SIZE} | Total train batches per epoch: {len(train_loader)}")

    # Compute class weights to handle severe class imbalance
    class_weights = compute_class_weights(train_df['diagnosis'].values, num_classes=config.NUM_CLASSES).to(device)
    print(f"[WEIGHTS] Computed class weights for imbalance: {np.round(class_weights.cpu().numpy(), 2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = get_model(num_classes=config.NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=5, mode='max')

    best_qwk = -1.0
    history = []

    if dry_run:
        print("\n[TEST] Running 1-step dry run test to verify pipeline forward & backward pass...")
        images, targets = next(iter(train_loader))
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        print(f"[SUCCESS] Pipeline test passed! Loss on sample batch: {loss.item():.4f}")
        return

    start_epoch = 1
    if resume and os.path.exists(config.MODEL_SAVE_PATH):
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        best_qwk = float(checkpoint.get('val_qwk', -1.0))
        early_stopping.best_score = best_qwk
        if os.path.exists(METRICS_JSON_PATH):
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        for _ in range(1, start_epoch):
            scheduler.step()
        print(f"[RESUME] Loaded checkpoint from Epoch {start_epoch - 1} with Best Val QWK: {best_qwk:.4f}")
        print(f"[RESUME] Starting from Epoch {start_epoch} (Current LR: {scheduler.get_last_lr()[0]:.6f})")

    print(f"\n[TRAIN] Starting training from Epoch {start_epoch} to {epochs} with Early Stopping (patience=5)...")
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        print(f"\n--- Epoch {epoch:02d}/{epochs:02d} ---")
        train_loss, train_acc, train_qwk = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time

        is_best = val_qwk > best_qwk
        if is_best:
            best_qwk = float(val_qwk)
            torch.save({
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'val_qwk': float(val_qwk),
                'val_acc': float(val_acc),
                'class_names': config.CLASS_NAMES
            }, config.MODEL_SAVE_PATH)
            save_msg = f" * Saved new best model checkpoint to {config.MODEL_SAVE_PATH}"
        else:
            save_msg = ""

        # Record metrics
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "train_qwk": round(train_qwk, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_qwk": round(val_qwk, 4),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": round(elapsed, 1),
            "is_best": is_best
        }
        history.append(epoch_record)
        save_metrics_history(history)

        print(f"[Epoch {epoch:02d} Summary] ({elapsed:.1f}s) "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% QWK: {train_qwk:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% QWK: {val_qwk:.3f}{save_msg}", flush=True)

        if early_stopping(val_qwk):
            print(f"\n[STOP] Early stopping triggered at epoch {epoch}! (No QWK improvement in 5 epochs)")
            break

    print(f"\n[DONE] Training complete!")
    print(f"[SAVE] Best Model Checkpoint: {config.MODEL_SAVE_PATH} (Best Val QWK: {best_qwk:.4f})")
    print(f"[SAVE] Metrics JSON: {METRICS_JSON_PATH}")
    print(f"[SAVE] Metrics CSV:  {METRICS_CSV_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Perform a quick 1-batch validation check without full training")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing best_model.pth checkpoint")
    args = parser.parse_args()

    run_training(epochs=args.epochs, dry_run=args.dry_run, resume=args.resume)
