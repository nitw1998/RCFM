import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os

from model import DiffusionUNetCrossAttention, ConditionNet
from data import get_ppg2ecg_datasets, get_ecg2ecg_datasets
from conditional_flow_matcher import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from optimal_transport import OTPlanSampler, wasserstein

import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader


def set_deterministic(seed):
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


class RegionAwareFlowMatching(nn.Module):
    def __init__(self, flow_model, flow_matcher_type="vp", criterion=nn.MSELoss(), sigma=0.1):
        super(RegionAwareFlowMatching, self).__init__()
        self.flow_model = flow_model
        self.criterion = criterion
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=0.05)

        # Support different flow matchers
        if flow_matcher_type == "conditional":
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "target":
            self.flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "sb":
            self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "vp":
            self.flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"Unknown flow matcher type: {flow_matcher_type}")

    def create_masked_velocity(self, x, y, patch_labels):
        """Apply region mask to velocity field"""
        v_target = x - y
        mask = (patch_labels == 1).float()
        masked_v_target = v_target * mask
        return masked_v_target, mask

    def forward(self, x=None, y=None, cond=None, mode="train", patch_labels=None):
        if mode == "train":
            # Sample trajectory points and velocity
            t, x_t, v_target = self.flow_matcher.sample_location_and_conditional_flow(x, y)
            
            # Apply region-aware masking if patch_labels provided
            if patch_labels is not None:
                v_pred = self.flow_model(x_t, cond, t.to(x.device))
                # Masked loss
                masked_v_target, mask = self.create_masked_velocity(x, y, patch_labels)
                masked_v_pred = v_pred * mask
                loss_masked = self.criterion(masked_v_pred, masked_v_target)
                # Full loss（flow matching 原始）
                v_target = y - x  # 注意此处是 flow direction：x → y，v = y - x
                loss_all = self.criterion(v_pred, v_target)
                # 加权合并
                alpha = 0.1  # 可以自由设置，例如 0.7 表示更关注 mask 区域
                return alpha * loss_masked + (1 - alpha) * loss_all

            else:
                v_pred = self.flow_model(x_t, cond, t.to(x.device))
                return self.criterion(v_pred, v_target)

        elif mode == "sample":
            # Start from noise
            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            window_size = y.shape[-1] if y is not None else 512
            
            x = torch.randn(n_sample, 1, window_size).to(device)
            steps = 50
            dt = 1.0 / steps
            
            # Integrate from t=0 to t=1 (noise to data)
            for i in range(steps):
                t = torch.tensor(i / steps).to(x.device)  # 0 to 1
                v = self.flow_model(x, cond, t.repeat(x.size(0)))
                x = x + v * dt  # Forward integration
                
            return x



def train_flowmatching(config):
    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    PATH = config["PATH"]
    warmup_epochs = config.get("warmup_epochs", 10)
    flow_matcher_type = config.get("flow_matcher_type", "conditional")
    use_region_aware = config.get("use_region_aware", True)
    dataset = config["dataset"]
    wandb.init(
        project="RCFM",
        entity="RCFM",
        id="rcfm_vp_v1",
        mode="offline",
        config=config
    )

    dataset_train, _ = get_ppg2ecg_datasets()
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    flow_model = RegionAwareFlowMatching(
        flow_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        criterion=nn.MSELoss(),
        sigma=0.1,
        flow_matcher_type=flow_matcher_type
    ).to(device)

    condition_net = ConditionNet().to(device)

    optim = torch.optim.AdamW(
        list(flow_model.parameters()) + list(condition_net.parameters()),
        lr=1e-4
    )

    total_epochs = n_epoch
    main_epochs = total_epochs - warmup_epochs

    warmup_scheduler = LinearLR(optim, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optim, T_max=main_epochs)
    scheduler = SequentialLR(optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    for i in range(n_epoch):
        print(f"\n********** Epoch - {i} **********\n")

        flow_model.train()
        condition_net.train()
        pbar = tqdm(dataloader)
        
        epoch_losses = []

        for step, (y12_ecg, x_ecg, ecg_roi) in enumerate(pbar):
            x_ecg = x_ecg.float().to(device)  # This should be the conditioning signal
            y12_ecg = y12_ecg.float().to(device)  # This should be the target data
            ecg_roi = ecg_roi.float().to(device)

            optim.zero_grad()
            ecg_conditions = condition_net(x_ecg, drop_prob=cond_mask)

            # Note: In flow matching, we typically go from noise to data
            # So x should be target data, y should be noise (or source)
            if use_region_aware:
                loss = flow_model(
                    x=y12_ecg,  # target data
                    y=torch.randn_like(y12_ecg),  # source noise
                    cond=ecg_conditions, 
                    patch_labels=ecg_roi, 
                    mode="train"
                )
            else:
                loss = flow_model(
                    x=y12_ecg,  # target data
                    y=torch.randn_like(y12_ecg),  # source noise
                    cond=ecg_conditions, 
                    mode="train"
                )
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(flow_model.parameters()) + list(condition_net.parameters()), 
                max_norm=1.0
            )
            
            optim.step()
            
            epoch_losses.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f}")
            
            wandb.log({
                "FlowMatching_loss": loss.item(),
                "lr": optim.param_groups[0]['lr'],
                "step": i * len(dataloader) + step
            })

            # Visualize flow trajectories for first batch every 10 epochs
            if step == 0 and i % 10 == 0:
                with torch.no_grad():
                    noise_sample = torch.randn_like(y12_ecg[:1])
                    t_vis = torch.tensor([0.5]).to(device)  # middle of trajectory
                    t_expanded = t_vis.repeat(1, 1, 1)
                    x_t_vis = t_expanded * y12_ecg[:1] + (1 - t_expanded) * noise_sample
                    
                    # plot_dir = os.path.join(PATH, "plots")
                    # os.makedirs(plot_dir, exist_ok=True)
                    # plot_path = os.path.join(plot_dir, f"epoch{i}_flow_match.png")
                    # plot_ecg_traces(y12_ecg, noise_sample, x_t_vis, save_path=plot_path)

        # Log epoch statistics
        avg_loss = np.mean(epoch_losses)
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch": i
        })
        
        scheduler.step()

        # Save model checkpoints
        if (i+1) % 20 == 0:
            torch.save(flow_model.state_dict(), f"{PATH}/{dataset}/rcfm_vp01_{flow_matcher_type}_epoch{i}.pth")
            torch.save(condition_net.state_dict(), f"{PATH}/{dataset}/condition_rcfm_vp01_{flow_matcher_type}_epoch{i}.pth")

        print(f"Epoch {i} completed. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    set_deterministic(31)

    config = {
        "n_epoch": 1000,
        "batch_size": 256,
        "nT": 50,
        "device": "cuda:6",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "PATH": "./saved/",
        "warmup_epochs": 10,
        "flow_matcher_type": "vp",  # 可选: conditional / target / sb / vp
        "use_region_aware": True,  # 是否使用区域感知
        "dataset": "MIMIC-AFib"
    }

    train_flowmatching(config)