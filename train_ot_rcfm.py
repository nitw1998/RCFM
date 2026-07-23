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
from data import get_datasets
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
        v_target = x - y
        mask = (patch_labels == 1).float()
        masked_v_target = v_target * mask
        return masked_v_target, mask

    def forward(self, x=None, y=None, cond=None, mode="train", patch_labels=None):
        if mode == "train":
            t, x_t, v_target = self.flow_matcher.sample_location_and_conditional_flow(x, y)
            v_pred = self.flow_model(x_t, cond, t.to(x.device))
            return self.criterion(v_pred, v_target)

        elif mode == "sample":
            x = torch.randn_like(y)
            steps = 100
            for i in range(steps):
                t = torch.tensor(1.0 - i / steps).to(x.device)
                v = self.flow_model(x, cond, t.repeat(x.size(0)))
                x = x + v / steps
            return x


def plot_ecg_traces(x, y, xt, save_path=None):
    x, y, xt = x[0].detach().cpu().numpy(), y[0].detach().cpu().numpy(), xt[0].detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(x[0], label="x0", alpha=0.8)
    plt.plot(y[0], label="x1", alpha=0.8)
    plt.plot(xt[0], label="x_t", linestyle="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def train_flowmatching(config):
    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    PATH = config["PATH"]
    warmup_epochs = config.get("warmup_epochs", 10)
    flow_matcher_type = config.get("flow_matcher_type", "conditional")

    wandb.init(
        project="INSERT PROJECT NAME HERE",
        entity="INSERT ENTITY HERE",
        id="INSERT ID HERE",
        mode="offline",
        config=config
    )

    dataset_train, _ = get_datasets()
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

        for step, (y12_ecg, x_ecg, ecg_roi) in enumerate(pbar):
            x_ecg = x_ecg.float().to(device)
            y12_ecg = y12_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            optim.zero_grad()
            ecg_conditions = condition_net(x_ecg, drop_prob=cond_mask)

            loss = flow_model(x=y12_ecg, y=x_ecg, cond=ecg_conditions, patch_labels=ecg_roi, mode="train")
            loss.mean().backward()
            optim.step()

            pbar.set_description(f"loss: {loss.mean().item():.4f}")
            wandb.log({
                "FlowMatching_loss": loss.mean().item(),
                "lr": optim.param_groups[0]['lr']
            })

            # 可视化前 1 个 batch 的流轨迹
            # if step == 0 and i % 10 == 0:
            #     t, xt, ut = flow_model.flow_matcher.sample_location_and_conditional_flow(y12_ecg, x_ecg)
            #     plot_dir = os.path.join(PATH, "plots")
            #     os.makedirs(plot_dir, exist_ok=True)
            #     plot_path = os.path.join(plot_dir, f"epoch{i}_flow_match.png")
            #     plot_ecg_traces(y12_ecg, x_ecg, xt, save_path=plot_path)

        scheduler.step()

        if i % 5 == 0:
            torch.save(flow_model.state_dict(), f"{PATH}/flow_model_vp_epoch{i}.pth")
            torch.save(condition_net.state_dict(), f"{PATH}/condition_net_vp_epoch{i}.pth")


if __name__ == "__main__":
    set_deterministic(31)

    config = {
        "n_epoch": 20,
        "batch_size": 128,
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "PATH": "E:\\repo\\RDDM-main\\RegionFlowMatching\\saved",
        "warmup_epochs": 10,
        "flow_matcher_type": "vp"  # 可选: conditional / target / sb / vp
    }

    train_flowmatching(config)
