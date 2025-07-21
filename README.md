# FedGIN: Federated Learning with Dynamic Global Intensity Non-linear Augmentation for Organ Segmentation using Multi-modal Images

> **Abstract:**
> Medical image segmentation plays a crucial role in AI-assisted diagnostics, surgical planning, and treatment monitoring. Accurate and robust segmentation models are essential for enabling reliable, data-driven clinical decision making across diverse imaging modalities. Given the inherent variability in image characteristics across modalities, developing a unified model capable of generalizing effectively to multiple modalities would be highly beneficial. This model could streamline clinical workflows and reduce the need for modality-specific training. However, real-world deployment faces major challenges, including data scarcity, domain shift between modalities (e.g., CT vs. MRI), and privacy restrictions that prevent data sharing. To address these issues, we propose FedGIN, a Federated Learning (FL) framework that enables multimodal organ segmentation without sharing raw patient data. Our method integrates a lightweight Global Intensity Non-linear (GIN) augmentation module that harmonizes modality-specific intensity distributions during local training. We evaluated FedGIN using two types of datasets: a *limited dataset* and a *complete dataset*. In the limited dataset scenario, the model was initially trained using only MRI data, and CT data was added to assess its performance improvements. In the complete dataset scenario, both MRI and CT data were fully utilized for training on all clients. In the limited-data scenario, FedGIN achieved a 12–18% improvement in 3D Dice scores on MRI test cases compared to FL without GIN and consistently outperformed local baselines. In the complete dataset scenario, FedGIN demonstrated near-centralized performance, with a 30% Dice score improvement over the MRI-only baseline and a 10% improvement over the CT-only baseline, highlighting its strong cross-modality generalization under privacy constraints.
>
> **This project is related to this work, which is accepted at the MICCAI 2025 DeCaF workshop.**

## Project Structure

```
FL-ALLDATA/
└── FL/
    ├── server.py
    ├── clients/
    │   ├── ct_client.py
    │   ├── mri_client.py
    │   └── ...
    ├── data/
    │   └── gb/
    │       ├── train/
    │       ├── val_ct/
    │       └── val_mri/
    ├── requirements.txt
    └── ...
```

## Setup

1. **Clone the repository** (if not already):
   ```bash
   git clone <your-repo-url>
   cd FL-ALLDATA/FL
   ```

2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare the data**  
   Place your data in the `data/gb/train`, `data/gb/val_ct`, and `data/gb/val_mri` folders. The expected format is the same as used in the code (see `OrganMaskDataset` in `clients/dataset.py`).

4. **Configure experiment settings**  
   Edit `clients/fl_experiment_config.py` to set the organ, data ratio, GIN usage, and other parameters as needed.

## Dataset

This project uses multi-modal medical image data for organ segmentation, specifically CT and MRI slices. 

- **Training Data:** Totalsegmentator dataset
- **Testing Data:** AMOS 22 dataset

The dataset is organized as follows:

```
FL-ALLDATA/FL/data/gb/
    ├── train/      # Training data (CT and MRI)
    ├── val_ct/     # Validation data for CT
    └── val_mri/    # Validation data for MRI
```

### Data Format

- Each folder contains `.npy` files, named by modality and slice (e.g., `CT_s0586_slice_1.npy`, `MRI_s0168_slice_1.npy`).
- Each `.npy` file is a Python dictionary with at least two keys:
  - `'image'`: a 2D numpy array (the image slice)
  - `'mask'`: a 2D numpy array (the corresponding segmentation mask)

Example for loading a file:
```python
import numpy as np
data = np.load('CT_s0586_slice_1.npy', allow_pickle=True).item()
image = data['image']  # shape: (H, W)
mask = data['mask']    # shape: (H, W)
```

- The `train/` folder contains both CT and MRI slices, with filenames starting with `CT_` or `MRI_`.
- The `val_ct/` and `val_mri/` folders contain validation slices for CT and MRI, respectively.

### Customization

- You can add your own data by following the same `.npy` file structure and placing them in the appropriate folders.
- The dataset loader (`OrganMaskDataset` in `clients/dataset.py`) will automatically detect and load all valid files.

## Running the Project

### 1. Start the Federated Server

```bash
python server.py
```
This will start the Flower server and wait for clients to connect.

### 2. Start the Clients

Open two separate terminals (or run in the background):

- **CT Client:**
  ```bash
  python clients/ct_client.py
  ```

- **MRI Client:**
  ```bash
  python clients/mri_client.py
  ```

Clients will connect to the server and begin federated training.

### 3. Monitoring

- Training and validation metrics are logged to Weights & Biases (wandb).
- Metrics are also saved locally in the `checkpoints/` directory.

## Configuration

All experiment and path settings are in `clients/fl_experiment_config.py`.  
You can change the organ, data ratio, GIN usage, and other hyperparameters there.

## Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies.

## Pushing to GitHub

1. Initialize a git repository (if not already):
   ```bash
   git init
   git remote add origin <your-repo-url>
   ```

2. Add and commit your files:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```
// ... existing code ...

## Acknowledgement

This project uses the Global Intensity Non-linear (GIN) augmentation method from [cheng-01037/Causality-Medical-Image-Domain-Generalization](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization/).


3. Push to GitHub:
   ```bash
   git push -u origin main
   ``` 
