import torch
import wandb
import random
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import CATransformer
from data import get_ppg2ecg_datasets


# -------------------------------
# Utils
# -------------------------------
def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_deterministic(31)


# -------------------------------
# Train function
# -------------------------------
def train_catransformer(config):

    device = config["device"]
    batch_size = config["batch_size"]
    n_epoch = config["n_epoch"]
    lr = config["lr"]
    PATH = config["PATH"]
    dataset = config["dataset"]

    wandb.init(
        project="CATransformer",
        entity="CATransformer",
        mode="offline",
        config=config
    )

    # Dataset
    dataset_train, _ = get_ppg2ecg_datasets()
    dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = CATransformer(
        input_length=512,
        in_channels=1,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        k_cycles=2
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=n_epoch
    )

    # Loss (CATransformer paper uses L1 / L2)
    criterion = nn.L1Loss()

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(n_epoch):

        print(f"\n========== Epoch {epoch} ==========\n")
        model.train()

        pbar = tqdm(dataloader)
        for ppg, ecg_gt, _ in pbar:

            ppg = ppg.float().to(device)        # [B, 1, 512]
            ecg_gt = ecg_gt.float().to(device)  # [B, 1, 512]

            optimizer.zero_grad()

            ecg_pred = model(ppg)
            loss = criterion(ecg_pred, ecg_gt)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"L1 loss: {loss.item():.4f}")

            wandb.log({
                "train_loss": loss.item()
            })

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                f"{PATH}/{dataset}/catransformer_epoch_{epoch}.pth"
            )


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    config = {
        "n_epoch": 500,
        "batch_size": 128,
        "lr": 1e-4,
        "device": "cuda:0",
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 8,
        "ff_dim": 512,
        "PATH": "/data/user/RCFM/saved",
        "dataset": "PTBXL"
    }

    train_catransformer(config)
