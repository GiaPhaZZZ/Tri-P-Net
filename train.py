import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data.create_mel_dataset import process_gtzan
from build_dataset.build_dataset import get_samples, MelNPYDataset 
from build_dataset.seed import set_seed

from models.dual_pooling_fusion import DualMelFusion

import warnings
warnings.filterwarnings("ignore")


# CONFIG
DATASET_PATH = "./genres_original"
PROCESSED_PATH = "./gtzan_mel_3s"
CHECKPOINT_DIR = "./checkpoints"

EPOCHS = 1
BATCH_SIZE = 16
NUM_CLASSES = 10
NUM_WORKERS = 2
PATIENCE = 30
SEED = 42


# SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Only process dataset once
if not os.path.exists(PROCESSED_PATH):
    process_gtzan(DATASET_PATH, PROCESSED_PATH)

# DATA
# ==============================

train_samples, class_map = get_samples(PROCESSED_PATH, "train")
val_samples, _ = get_samples(PROCESSED_PATH, "valid")

train_ds = MelNPYDataset(train_samples, train=True)
val_ds = MelNPYDataset(val_samples, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# MODEL
# ==============================
model = DualMelFusion(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4 )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)


# TRAINING LOOP
# ===============================
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "epoch_times": []
}

def train():
    
    history = { "train_loss": [], "train_acc": [],
                "val_loss": [], "val_acc": [],
                "epoch_times": []}
    
    best_val_loss = float("inf")
    COUNTER = 0
    
    total_start = time.time()
    best_epoch_idx = 0
    time_to_best_model = 0.0
    
    print(f"Starting training on {device} for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        
        if device.type == "cuda": torch.cuda.synchronize()
        epoch_start = time.time()

        # ===================== TRAIN =====================
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for mel, mel_comp, y in train_loader:

            mel = mel.to(device, non_blocking=True)
            mel_comp = mel_comp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(mel, mel_comp)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)

            train_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # ===================== VALIDATION =====================
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for mel, mel_comp, y in val_loader:

                mel = mel.to(device, non_blocking=True)
                mel_comp = mel_comp.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                out = model(mel, mel_comp)
                loss = criterion(out, y)

                preds = out.argmax(dim=1)

                val_loss += loss.item() * y.size(0)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = correct / total
        

        # ===================== TIMING =====================
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_duration)

        avg_epoch_time = np.mean(history["epoch_times"])

        # ===================== SAVE BEST =====================
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_idx = epoch
            time_to_best_model = time.time() - epoch_start

            save_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "class_map": class_map
            }, save_path)
            
            COUNTER = 0
            marker = "*"
        else:
            COUNTER += 1

        # ===================== LOG =====================
        print(f"Epoch {epoch:03d} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
            f"{epoch_duration:.2f}s (Avg: {avg_epoch_time:.2f}s) {marker}")

        scheduler.step(val_loss)

        if COUNTER >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        # ===================== END OF TRAINING =====================
        history_path = os.path.join(CHECKPOINT_DIR, "history.json")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        print(f"Training history saved to {history_path}")
        
if __name__ == "__main__":
    train()