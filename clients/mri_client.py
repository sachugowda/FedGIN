import torch
import flwr as fl
from dataset import OrganMaskDataset, custom_collate
from sampler import AdaptiveDrainSampler
from torch.utils.data import DataLoader
from model import UNet
from gin import apply_gin
from utiles import CombinedLoss, dice_coefficient_2d, compute_3d_dice
import os
import json
from datetime import datetime
import time
import numpy as np
from fl_experiment_config import (
    USE_GIN, CHECKPOINT_DIR, MRI_METRICS_FILE, SERVER_PORT,
    WANDB_PROJECT, MODEL_CONFIG, TRAIN_CONFIG, TRAIN_FOLDER, VAL_MRI_FOLDER
)

class MRIClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modality = "MRI"
        self.model = UNet(
            input_channels=MODEL_CONFIG["input_channels"], 
            n_classes=MODEL_CONFIG["n_classes"]
        ).to(self.device)
        self.train_dataset = OrganMaskDataset(TRAIN_FOLDER, modality="MRI")
        self.val_dataset = OrganMaskDataset(VAL_MRI_FOLDER, modality="MRI")

        # Create directories for saving metrics
        self.metrics_dir = CHECKPOINT_DIR
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize metrics history
        self.metrics_history = []
        self.current_round = 0

        self.use_gin = USE_GIN
        train_sampler = AdaptiveDrainSampler(self.train_dataset, batch_size=MODEL_CONFIG["batch_size"])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )

        self.criterion = CombinedLoss(alpha=0.5).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=MODEL_CONFIG["learning_rate"], 
            weight_decay=MODEL_CONFIG["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=3, 
            factor=0.5, 
            min_lr=1e-5
        )

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict)

    def save_metrics(self, metrics):
        def to_python_type(val):
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            return val
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics['round'] = self.current_round
        metrics['timestamp'] = timestamp
        metrics_clean = {k: to_python_type(v) for k, v in metrics.items()}
        self.metrics_history.append(metrics_clean)
        # Save all metrics to a single file
        metrics_file = MRI_METRICS_FILE
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def fit(self, parameters, config):
        if self.current_round == 0:
            gin_status = "ENABLED" if self.use_gin else "DISABLED"
            print(f"[MRI Client] Training: GIN is {gin_status}")
        self.current_round += 1  # Increment round at the start of fit
        self.set_parameters(parameters)
        self.model.train()

        optimizer = self.optimizer
        loss_fn = self.criterion
        scaler = torch.amp.GradScaler()

        epoch_train_loss, epoch_train_dice = 0.0, 0.0
        num_batches = 0

        for epoch in range(config.get("local_epochs", 1)):
            for inputs, masks, paths in self.train_loader:
                inputs, masks = inputs.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
                masks = (masks > 0).float()

                # === GIN Augmentation (if enabled) ===
                if self.use_gin:
                    with torch.no_grad():
                        inputs = apply_gin(inputs)
                    inputs = inputs.to(self.device, dtype=next(self.model.parameters()).dtype)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += loss.item()
                with torch.no_grad():
                    dice_score_2d = dice_coefficient_2d(outputs, masks)
                    epoch_train_dice += dice_score_2d
                num_batches += 1

        epoch_train_loss /= num_batches
        epoch_train_dice /= num_batches

        # Validation (3D Dice and Loss)
        val_loss = self.validate_loss(self.model, self.val_loader, loss_fn)
        val_dice = compute_3d_dice(self.model, self.val_loader, self.device, dataset_name=f"Validation {self.modality}")

        # Step the scheduler based on validation loss
        self.scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save training metrics (rounded for easier post-processing)
        metrics = {
            'train_loss': round(epoch_train_loss, 4),
            'train_dice': round(epoch_train_dice, 4),
            'val_dice': round(val_dice, 4)  # Keep this for local tracking
        }
        self.save_metrics(metrics)

        print(f"[MRI Client] Round {self.current_round} training complete. Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}, Val Dice: {val_dice:.4f}")

        return self.get_parameters({}), len(self.train_dataset), {
            "modality": self.modality,
            "dice": float(val_dice),
            "train_dice": float(epoch_train_dice),
            "train_loss": float(epoch_train_loss),
            "learning_rate": float(current_lr)
        }

    def validate_loss(self, model, val_loader, loss_fn):
        model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for inputs, masks, _ in val_loader:
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_dice = compute_3d_dice(self.model, self.val_loader, self.device, dataset_name=f"Validation {self.modality}")
        print(f"[{self.modality} Client] Evaluate (sent to server): Val Dice: {val_dice:.4f}")
        return float(val_dice), int(len(self.val_dataset)), {"modality": self.modality, "dice": float(val_dice)}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=f"localhost:{SERVER_PORT}", client=MRIClient())
