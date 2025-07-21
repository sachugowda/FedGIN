import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, focal_weight=0.25, pos_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(self, y_pred, y_true):
        # Focal loss component
        bce_per_element = F.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction='none', pos_weight=self.pos_weight.to(y_pred.device)
        )
        pt = torch.exp(-bce_per_element)
        focal_loss = (self.focal_weight * (1 - pt) ** self.gamma * bce_per_element).mean()

        # Dice loss component
        y_pred_sigmoid = torch.sigmoid(y_pred)
        intersection = (y_pred_sigmoid * y_true).sum(dim=(1,2,3))
        union = y_pred_sigmoid.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
        dice_loss = 1 - (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss

def dice_coefficient_2d(pred, true, smooth=1e-6):
    """2D Dice score for training batches"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * true).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + true.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def dice_coefficient_3d(pred, true, smooth=1e-6):
    """3D Dice score for validation volumes"""
    pred = (pred > 0.5).astype(np.float32)
    true = (true > 0.5).astype(np.float32)
    intersection = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true)
    dice = (2. * intersection + smooth) / (union + smooth)
    return min(max(dice, 0.0), 1.0)  # Ensure dice is between 0 and 1

def aggregate_volumetric_dice(model, data_loader, device):
    """Compute 3D Dice score per patient volume"""
    model.eval()
    patient_volumes = {}

    with torch.no_grad():
        for x, y, paths in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            for i in range(x.size(0)):
                patient_id = paths[i].split('_')[1]  # Adjust based on your naming convention
                slice_idx = int(paths[i].split('_')[-1].replace('.npy', ''))
                
                if patient_id not in patient_volumes:
                    patient_volumes[patient_id] = {
                        'pred': [],
                        'true': [],
                        'slices': []
                    }
                
                pred_slice = torch.sigmoid(outputs[i]).cpu().squeeze().numpy()
                true_slice = y[i].cpu().squeeze().numpy()
                
                patient_volumes[patient_id]['pred'].append((slice_idx, pred_slice))
                patient_volumes[patient_id]['true'].append((slice_idx, true_slice))
    
    # Compute 3D Dice per patient
    dice_scores = {}
    for pid, data in patient_volumes.items():
        # Sort slices by index
        sorted_pred = np.stack([p[1] for p in sorted(data['pred'], key=lambda x: x[0])])
        sorted_true = np.stack([t[1] for t in sorted(data['true'], key=lambda x: x[0])])
        dice_scores[pid] = dice_coefficient_3d(sorted_pred, sorted_true)
    
    return dice_scores

def compute_3d_dice(model, data_loader, device, dataset_name="Validation"):
    model.eval()
    patient_volumes = {}

    with torch.no_grad():
        for x, y, paths in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            for i in range(x.size(0)):
                patient_id = paths[i].split('_')[1]  # Adjust as needed
                slice_idx = int(paths[i].split('_')[-1].replace('.npy', ''))
                if patient_id not in patient_volumes:
                    patient_volumes[patient_id] = {'pred': [], 'true': []}
                pred_slice = torch.sigmoid(outputs[i]).cpu().squeeze().numpy()
                true_slice = y[i].cpu().squeeze().numpy()
                patient_volumes[patient_id]['pred'].append((slice_idx, pred_slice))
                patient_volumes[patient_id]['true'].append((slice_idx, true_slice))

    dice_scores = {}
    for pid, data in patient_volumes.items():
        sorted_pred = np.stack([p[1] for p in sorted(data['pred'], key=lambda x: x[0])])
        sorted_true = np.stack([t[1] for t in sorted(data['true'], key=lambda x: x[0])])
        dice_scores[pid] = dice_coefficient_3d(sorted_pred, sorted_true)

    avg_dice_score = np.mean(list(dice_scores.values())) if dice_scores else 0.0
    print(f"\n--- {dataset_name} 3D Dice Scores ---")
    print(f"Average {dataset_name} 3D Dice Score: {avg_dice_score:.4f}")
    return avg_dice_score