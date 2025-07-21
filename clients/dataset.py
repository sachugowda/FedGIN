import os
import numpy as np
import torch
from torch.utils.data import Dataset
from fl_experiment_config import SEED

# ======================== Reproducibility ======================== #
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================== Dataset Loader (Modality-Specific) ======================== #
class OrganMaskDataset(Dataset):
    def __init__(self, folder_path, modality="CT"):
        assert modality in ["CT", "MRI"], "Modality must be either 'CT' or 'MRI'"
        self.modality = modality
        self.folder_path = folder_path
        self.file_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith('.npy') and f.startswith(modality)
        ]

        self.organ_slices = []
        self.non_organ_slices = []

        for idx, file_path in enumerate(self.file_paths):
            data = np.load(file_path, allow_pickle=True).item()
            mask = data['mask']
            if np.any(mask):
                self.organ_slices.append(idx)
            else:
                self.non_organ_slices.append(idx)

        def percent(n, total): return 100.0 * n / total if total > 0 else 0.0
        print(f"✅ Dataset loaded from: {folder_path} | Modality: {modality}")
        print(f"Total slices: {len(self.file_paths)}")
        print(f"  Organ slices: {len(self.organ_slices)} ({percent(len(self.organ_slices), len(self.file_paths)):.2f}%)")
        print(f"  Non-Organ slices: {len(self.non_organ_slices)} ({percent(len(self.non_organ_slices), len(self.file_paths)):.2f}%)")
        print("---------------------------\n")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            data = np.load(self.file_paths[idx], allow_pickle=True).item()
            img_tensor = torch.tensor(data['image'].astype(np.float32)).unsqueeze(0)
            mask_tensor = torch.tensor(data['mask'].astype(np.float32)).unsqueeze(0)
            return img_tensor, mask_tensor, self.file_paths[idx]
        except Exception as e:
            print(f"[Dataset Error] File: {self.file_paths[idx]} | Error: {e}")
            raise

# ======================== Collate Function ======================== #
def custom_collate(batch):
    imgs, masks, paths = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), paths