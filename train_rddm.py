import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from data import get_ppg2ecg_datasets, get_ecg2ecg_datasets
import torch.nn as nn
from metrics import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

set_deterministic(31)

def train_rddm(config):

    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    alpha1 = config["alpha1"]
    alpha2 = config["alpha2"]
    PATH = config["PATH"]
    dataset = config["dataset"]

    wandb.init(
        project="RDDM",
        entity="RDDM",
        id="rddm_mmecg_v1",
        mode="offline",
        config=config
    )

    dataset_train, _ = get_ppg2ecg_datasets()

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT
    )

    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    rddm.to(device)

    optim = torch.optim.AdamW([*rddm.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters()], lr=1e-4)

    # rddm = nn.DataParallel(rddm)
    # Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    # Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    scheduler = CosineAnnealingLR(optim, T_max=100) # CosineAnnealingLRWarmup
    for i in range(n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n\n")

        rddm.train()
        Conditioning_network1.train()
        Conditioning_network2.train()
        pbar = tqdm(dataloader)

        for y12_ecg, x_ecg, ecg_roi in pbar:
            
            ## Train Diffusion
            optim.zero_grad()
            x_ecg = x_ecg.float().to(device)
            y12_ecg = y12_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            ecg_conditions1 = Conditioning_network1(x_ecg, drop_prob=cond_mask)
            ecg_conditions2 = Conditioning_network2(x_ecg, drop_prob=cond_mask)

            ddpm_loss, region_loss = rddm(x=y12_ecg, cond1=ecg_conditions1, cond2=ecg_conditions2, patch_labels=ecg_roi)

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            
            loss = ddpm_loss + region_loss

            loss.mean().backward()
            
            optim.step()

            pbar.set_description(f"loss: {loss.mean().item():.4f}")

            wandb.log({
                "DDPM_loss": ddpm_loss.mean().item(),
                "Region_loss": region_loss.mean().item(),
            })

        scheduler.step()

        if (i+1) % 20 == 0:
            # if use torch.parallel, save the state_dict of the module
            if isinstance(rddm, nn.DataParallel):
                # save the state_dict of the module
                torch.save(rddm.module.state_dict(), f"{PATH}/{dataset}/rddm_main_network_epoch_{i}.pth")
                torch.save(Conditioning_network1.module.state_dict(), f"{PATH}/{dataset}/ConditionNet1_epoch_{i}.pth")
                torch.save(Conditioning_network2.module.state_dict(), f"{PATH}/{dataset}/ConditionNet2_epoch_{i}.pth")

            else:
                torch.save(rddm.state_dict(), f"{PATH}/{dataset}/rddm_main_network_epoch_{i}.pth")
                torch.save(Conditioning_network1.state_dict(), f"{PATH}/{dataset}/ConditionNet1_epoch_{i}.pth")
                torch.save(Conditioning_network2.state_dict(), f"{PATH}/{dataset}/ConditionNet2_epoch_{i}.pth")

                
if __name__ == "__main__":

    config = {
        "n_epoch": 1000,
        "batch_size": 128,
        "nT":50,
        "device": "cuda:2",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "PATH": "/data/user/RCFM/saved",
        "dataset": "PTBXL"
    }

    train_rddm(config)