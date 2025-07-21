from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, parameters_to_ndarrays
import json
import os
from datetime import datetime
import wandb
import torch
import numpy as np
from typing import List, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes
from clients.fl_experiment_config import (
    CHECKPOINT_DIR, SERVER_METRICS_FILE, WANDB_SERVER_NAME, 
    SERVER_PORT, WANDB_PROJECT, TRAIN_CONFIG, MODEL_CONFIG
)

from clients.model import UNet 



CHECKPOINT_DIR = CHECKPOINT_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class MetricsCallback:
    def __init__(self):
        self.metrics_history = []
        self.best_avg_dice = 0.0
        self.best_model_round = -1
        self.best_model_path = None
        self.final_model_path = None
        self.logs_dir = CHECKPOINT_DIR
        # Initialize wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_SERVER_NAME,
            resume="allow",
            config={
                "architecture": MODEL_CONFIG["architecture"],
                "dataset": MODEL_CONFIG["dataset"],
                "modalities": ["CT", "MRI"],
                "num_rounds": TRAIN_CONFIG["num_rounds"],
                "min_clients": TRAIN_CONFIG["min_clients"]
            }
        )

    def on_aggregate_fit_metrics(self, server_round, metrics):
        if metrics:
            fit_results = metrics.get("fit_results", [])
            client_metrics = []
            for _, fit_res in fit_results:
                if hasattr(fit_res, 'metrics'):
                    client_metrics.append(fit_res.metrics)

            ct_metrics = next((m for m in client_metrics if m.get("modality", "").lower() == "ct"), {})
            mri_metrics = next((m for m in client_metrics if m.get("modality", "").lower() == "mri"), {})

            ct_train_dice = float(ct_metrics.get("train_dice", 0.0))
            mri_train_dice = float(mri_metrics.get("train_dice", 0.0))
            ct_train_loss = float(ct_metrics.get("train_loss", 0.0))
            mri_train_loss = float(mri_metrics.get("train_loss", 0.0))
            ct_lr = float(ct_metrics.get("learning_rate", 0.0))
            mri_lr = float(mri_metrics.get("learning_rate", 0.0))

            print("\n=== Aggregated Training Metrics ===")
            print(f"Round {server_round}")
            print(f"Number of clients: {len(fit_results)}")
            print(f"CT Train Dice: {ct_train_dice:.4f}, MRI Train Dice: {mri_train_dice:.4f}")
            print(f"CT Train Loss: {ct_train_loss:.4f}, MRI Train Loss: {mri_train_loss:.4f}")
            print(f"CT LR: {ct_lr:.6f}, MRI LR: {mri_lr:.6f}")
            print("===============================\n")

            # Store training metrics for later logging with validation metrics
            self.current_round_metrics = {
                # Training Dice (1 graph, 2 lines)
                "Training Dice/ct": ct_train_dice,
                "Training Dice/mri": mri_train_dice,
                
                # Training Loss (1 graph, 2 lines)
                "Training Loss/ct": ct_train_loss,
                "Training Loss/mri": mri_train_loss,
                
                # Learning Rate (1 graph, 2 lines)
                "Learning Rate/ct": ct_lr,
                "Learning Rate/mri": mri_lr
            }
            
            # Don't log here, will log together with validation metrics
            # wandb.log(current_round_metrics)

    def on_aggregate_evaluate_metrics(self, server_round, metrics, global_model=None):
        if metrics:
            client_metrics = metrics.get("client_metrics", [])
            ct_metrics = next((m for m in client_metrics if m.get("modality", "").lower() == "ct"), None)
            mri_metrics = next((m for m in client_metrics if m.get("modality", "").lower() == "mri"), None)

            ct_val_dice = ct_metrics.get("dice", 0) if ct_metrics else 0
            mri_val_dice = mri_metrics.get("dice", 0) if mri_metrics else 0

            print(f"\n=== Round {server_round} Results ===")
            print(f"CT Client - Val Dice: {ct_val_dice:.4f}")
            print(f"MRI Client - Val Dice: {mri_val_dice:.4f}")
            avg_dice = (ct_val_dice + mri_val_dice) / 2
            print(f"Average Val Dice (CT + MRI): {avg_dice:.4f}")
            print(f"Best Model Round: {self.best_model_round if self.best_model_round != -1 else 'N/A'} (Dice: {self.best_avg_dice:.4f})")
            print("===================\n")

            # Save metrics
            round_metrics = {
                "round": server_round,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "ct_val_dice": float(ct_val_dice),
                "mri_val_dice": float(mri_val_dice),
                "avg_val_dice": float(avg_dice)
            }
            self.metrics_history.append(round_metrics)
            
            # Save all metrics to a single file
            metrics_file = SERVER_METRICS_FILE
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)

            # Log all metrics together with the correct round number
            wandb_metrics = {
                **self.current_round_metrics,  # Include training metrics
                "Validation CT Dice": ct_val_dice,
                "Validation MRI Dice": mri_val_dice,
                "Average Val Dice": avg_dice
            }
            wandb.log(wandb_metrics)

            # Save best model
            if global_model is not None and avg_dice > self.best_avg_dice:
                self.best_avg_dice = avg_dice
                self.best_model_round = server_round
                # Always save a new best model with the round number, keep all bests
                best_model_path = os.path.join(self.logs_dir, f"best_model_round_{server_round}.pth")
                torch.save(global_model.state_dict(), best_model_path)
                print(f"[Server] New best model saved at round {server_round} with avg val dice: {avg_dice:.4f}")

    def save_final_model(self, global_model, final_round):
        if global_model is not None:
            self.final_model_path = os.path.join(self.logs_dir, f"final_model_round_{final_round}.pth")
            torch.save(global_model.state_dict(), self.final_model_path)
            print(f"[Server] Final model saved at {self.final_model_path}")

    def __del__(self):
        try:
            wandb.finish()
        except Exception:
            pass

# Define metric aggregation function
def simple_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients."""
    # Extract client metrics
    client_metrics = []
    for num_examples, m in metrics:
        if "modality" in m and "dice" in m:  # Only include metrics with both
            client_metrics.append({
                "dice": m["dice"],
                "modality": m["modality"]
            })
    
    # Get CT and MRI dice scores
    ct_dice = next((m["dice"] for m in client_metrics if m["modality"].lower() == "ct"), 0.0)
    mri_dice = next((m["dice"] for m in client_metrics if m["modality"].lower() == "mri"), 0.0)
    
    # Simple average of CT and MRI dice scores
    avg_dice = (ct_dice + mri_dice) / 2 if (ct_dice > 0 or mri_dice > 0) else 0.0
    
    # Round all dice scores to 4 decimal places for consistency with clients
    ct_dice = round(ct_dice, 4)
    mri_dice = round(mri_dice, 4)
    avg_dice = round(avg_dice, 4)
    
    return {
        "dice": avg_dice,
        "client_metrics": [
            {"modality": "CT", "dice": ct_dice},
            {"modality": "MRI", "dice": mri_dice}
        ]
    }

# Initialize metrics callback
metrics_callback = MetricsCallback()

class CustomFedAvg(FedAvg):
    def __init__(self, metrics_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_callback = metrics_callback
        self.global_model = None  # Will hold the latest global model

    def aggregate_fit(self, rnd, results, failures):
        aggregated_fit = super().aggregate_fit(rnd, results, failures)
        if aggregated_fit is not None:
            parameters, _ = aggregated_fit
            self.global_model = self.parameters_to_model(parameters)
            
            # Collect metrics from all clients
            metrics = {
                "fit_results": results,
                "num_clients": len(results)
            }
            self.metrics_callback.on_aggregate_fit_metrics(rnd, metrics)
            
        return aggregated_fit

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ):
        # Aggregate as usual
        aggregated_eval = super().aggregate_evaluate(rnd, results, failures)
        # Collect per-client metrics
        client_metrics = []
        for client, eval_res in results:
            print(f"[DEBUG] Client {getattr(client, 'cid', 'unknown')} metrics: {eval_res.metrics}")
            metrics = eval_res.metrics if hasattr(eval_res, 'metrics') else {}
            client_metrics.append({
                "client_id": getattr(client, "cid", "unknown"),
                **metrics
            })
        # Add to metrics dict
        if aggregated_eval is not None:
            loss, metrics = aggregated_eval
            metrics["client_metrics"] = client_metrics
            self.metrics_callback.on_aggregate_evaluate_metrics(rnd, metrics, self.global_model)
        return aggregated_eval

    def parameters_to_model(self, parameters):
        model = UNet(input_channels=1, n_classes=1)
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
        model.load_state_dict(state_dict)
        return model

# Define strategy
strategy = CustomFedAvg(
    metrics_callback=metrics_callback,
    evaluate_metrics_aggregation_fn=simple_average,
    min_fit_clients=2,  # Both CT and MRI clients
    min_evaluate_clients=2,
    min_available_clients=2,
)


# Define config
config = ServerConfig(num_rounds=TRAIN_CONFIG["num_rounds"])

app = ServerApp(
    config=config,
    strategy=strategy,
)

if __name__ == "__main__":
    from flwr.server import start_server
    start_server(
        server_address=f"0.0.0.0:{SERVER_PORT}",
        config=config,
        strategy=strategy,
    )
    # Save the final model after all rounds
    metrics_callback.save_final_model(strategy.global_model, final_round=config.num_rounds)