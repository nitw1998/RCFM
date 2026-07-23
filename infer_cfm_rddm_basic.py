import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from data import get_ppg2ecg_datasets, get_ecg2ecg_datasets

import torch.nn as nn

# -------------------- MinimalFlowMatching (CFM) --------------------
class MinimalFlowMatching(nn.Module):
    def __init__(self, flow_model, criterion=nn.MSELoss()):
        super(MinimalFlowMatching, self).__init__()
        self.flow_model = flow_model
        self.criterion = criterion

    def forward(self, x=None, y=None, cond=None, window_size=None, mode="train", steps=50):
        if mode == "train":
            t = torch.rand(x.size(0), device=x.device)
            t_expanded = t.view(-1, 1, 1)
            x_t = (1 - t_expanded) * y + t_expanded * x
            v_target = x - y
            v_pred = self.flow_model(x_t, cond, t)
            return self.criterion(v_pred, v_target)
        elif mode == "sample":
            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            window_size = y.shape[-1] if y is not None else (window_size or 512)
            x = torch.randn(n_sample, 1, window_size, device=device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((n_sample,), i / steps, device=device)
                v = self.flow_model(x, cond, t)
                x = x + v * dt
            return x

# -------------------- Robust checkpoint finder --------------------
def find_latest_checkpoint(dir_path, prefix):
    """
    Support:
      - {prefix}_epoch_123.pth
      - {prefix}_epoch123.pth
    Return newest path, or None if not found.
    """
    patterns = [
        os.path.join(dir_path, f"{prefix}_epoch_*.pth"),
        os.path.join(dir_path, f"{prefix}_epoch*.pth"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    def extract_epoch(p):
        m = re.search(r"_epoch_?(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    parsed = [(extract_epoch(p), p) for p in candidates]
    parsed = [pp for pp in parsed if pp[0] >= 0]
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    return parsed[-1][1]

# -------------------- Paper-style plotting --------------------
def plot_four_vertical(input_sig, gt, pred_cfm, pred_rddm, fs=None, normalize=False,
                       save_dir="./figs", basename="comparison_four_vertical",
                       figsize=(7.0, 10.0), dpi=400):
    """
    input_sig, gt, pred_cfm, pred_rddm: 1D numpy arrays with same length
    fs: sampling rate (Hz). If None, x-axis is sample index.
    normalize: if True, apply per-trace min-max scaling (useful for PPG vs ECG visual comparison)
    """
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(len(gt)) if fs is None else np.arange(len(gt)) / float(fs)

    def mm(x):
        x = x.astype(np.float64)
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn + 1e-8)

    a_in  = mm(input_sig) if normalize else input_sig
    a_gt  = mm(gt) if normalize else gt
    a_cfm = mm(pred_cfm) if normalize else pred_cfm
    a_rddm= mm(pred_rddm) if normalize else pred_rddm

    # Paper aesthetics
    plt.rcParams.update({
        "font.size": 15,
        "font.family": "serif",          # fallback-safe serif
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    lw = 1.2

    axes[0].plot(x, a_in, linewidth=lw)
    axes[0].set_title("Input RCG Signal", fontsize=15)
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(x, a_gt, linewidth=lw)
    axes[1].set_title("Ground Truth ECG", fontsize=15)
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(x, a_cfm, linewidth=lw)
    axes[2].set_title("RCFM Prediction", fontsize=15)
    axes[2].set_ylabel("Amplitude")

    axes[3].plot(x, a_rddm, linewidth=lw)
    axes[3].set_title("RDDM Prediction", fontsize=15)
    axes[3].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (s)" if fs is not None else "Sample Index")

    png_path = os.path.join(save_dir, f"{basename}.png")
    pdf_path = os.path.join(save_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", transparent=True)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")
    plt.close(fig)

# -------------------- Inference & comparison --------------------
@torch.no_grad()
def infer_and_compare(
    dataset="WESAD",
    data_root_ckpt="/data/user/RCFM/saved",
    pick_index=18,
    manual_epoch_cfm=None,
    manual_epoch_rddm=None,
    sample_steps=50,
    num_workers=2,
    batch_size=32,
    fs=None,                 # sampling rate if you want seconds on x-axis
    normalize=False,         # normalize traces for visual comparison
    save_dir="./figs",
    basename="comparison_four_vertical"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ---- Data ----
    print("[Data] Loading datasets...")
    if dataset in ["ICBEB", "PTBXL"]:
        _, dataset_test = get_ecg2ecg_datasets()
    else:
        _, dataset_test = get_ppg2ecg_datasets()

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_y, all_x, _ = next(iter(test_loader))
    if dataset in ["ICBEB", "PTBXL"]:
        print(all_y.shape)
        y_target = all_y[pick_index:pick_index+1].float().to(device)
        x_cond_in = all_x[pick_index:pick_index+1].float().to(device)
        L = y_target.shape[-1]
    else:
        y_target = all_y[pick_index:pick_index+1].float().to(device)
        x_cond_in = all_x[pick_index:pick_index+1].float().to(device)
        L = y_target.shape[-1]
    seq_len_for_model = L
    print(f"[Info] Sample index = {pick_index}, length = {L}")

    # ---- CFM ----
    flow_model = DiffusionUNetCrossAttention(seq_len_for_model, 1, device, num_heads=4).to(device)
    condition_net_cfm = ConditionNet().to(device)
    cfm = MinimalFlowMatching(flow_model=flow_model).to(device)

    ckpt_dir = os.path.join(data_root_ckpt, dataset)
    cfm_prefix = "minimal_cfm"
    cond_prefix = "condition_net"

    if manual_epoch_cfm is not None:
        cfm_ckpt = os.path.join(ckpt_dir, f"{cfm_prefix}_epoch_{manual_epoch_cfm}.pth")
        cond_ckpt = os.path.join(ckpt_dir, f"{cond_prefix}_epoch_{manual_epoch_cfm}.pth")
    else:
        cfm_ckpt = find_latest_checkpoint(ckpt_dir, cfm_prefix)
        cond_ckpt = find_latest_checkpoint(ckpt_dir, cond_prefix)

    missing = []
    if cfm_ckpt is None:  missing.append(f"{cfm_prefix}_epoch*.pth")
    if cond_ckpt is None: missing.append(f"{cond_prefix}_epoch*.pth")
    if missing:
        raise FileNotFoundError(
            "CFM checkpoints not found under:\n"
            f"  {ckpt_dir}\nMissing patterns:\n  - " + "\n  - ".join(missing)
        )

    print(f"[Load] CFM:  {os.path.basename(cfm_ckpt)}")
    print(f"[Load] Cond: {os.path.basename(cond_ckpt)}")
    cfm.load_state_dict(torch.load(cfm_ckpt, map_location=device))
    condition_net_cfm.load_state_dict(torch.load(cond_ckpt, map_location=device))
    cfm.eval(); condition_net_cfm.eval()

    # ---- RDDM ----
    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(seq_len_for_model, 1, device, num_heads=8),
        region_model=DiffusionUNetCrossAttention(seq_len_for_model, 1, device, num_heads=8),
        betas=(1e-4, 0.2),
        n_T=50
    ).to(device)
    condition_net1 = ConditionNet().to(device)
    condition_net2 = ConditionNet().to(device)

    rddm_prefix = "rddm_main_network"
    cond1_prefix = "ConditionNet1"
    cond2_prefix = "ConditionNet2"

    if manual_epoch_rddm is not None:
        rddm_ckpt = os.path.join(ckpt_dir, f"{rddm_prefix}_epoch{manual_epoch_rddm}.pth")
        cond1_ckpt = os.path.join(ckpt_dir, f"{cond1_prefix}_epoch{manual_epoch_rddm}.pth")
        cond2_ckpt = os.path.join(ckpt_dir, f"{cond2_prefix}_epoch{manual_epoch_rddm}.pth")
    else:
        rddm_ckpt = find_latest_checkpoint(ckpt_dir, rddm_prefix)
        cond1_ckpt = find_latest_checkpoint(ckpt_dir, cond1_prefix)
        cond2_ckpt = find_latest_checkpoint(ckpt_dir, cond2_prefix)

    missing = []
    if rddm_ckpt is None:  missing.append(f"{rddm_prefix}_epoch*.pth")
    if cond1_ckpt is None: missing.append(f"{cond1_prefix}_epoch*.pth")
    if cond2_ckpt is None: missing.append(f"{cond2_prefix}_epoch*.pth")
    if missing:
        raise FileNotFoundError(
            "RDDM checkpoints not found under:\n"
            f"  {ckpt_dir}\nMissing patterns:\n  - " + "\n  - ".join(missing)
        )

    print(f"[Load] RDDM:  {os.path.basename(rddm_ckpt)}")
    print(f"[Load] Cond1: {os.path.basename(cond1_ckpt)}")
    print(f"[Load] Cond2: {os.path.basename(cond2_ckpt)}")
    rddm.load_state_dict(torch.load(rddm_ckpt, map_location=device))
    condition_net1.load_state_dict(torch.load(cond1_ckpt, map_location=device))
    condition_net2.load_state_dict(torch.load(cond2_ckpt, map_location=device))
    rddm.eval(); condition_net1.eval(); condition_net2.eval()

    # ---- Inference ----
    conds_cfm = condition_net_cfm(x_cond_in)
    y_pred_cfm = cfm(y=y_target, cond=conds_cfm, mode="sample", steps=sample_steps)

    cond1 = condition_net1(x_cond_in)
    cond2 = condition_net2(x_cond_in)

    # NOTE: If your RDDM.sample signature differs, adjust this line accordingly.
    # Expect shape [1, 1, L]. If tuple returned, first element is the signal.
    out_rddm = rddm.forward(cond1=cond1, cond2=cond2, mode='sample', window_size=512)
    if isinstance(out_rddm, tuple):
        y_pred_rddm = out_rddm[0]
    else:
        y_pred_rddm = out_rddm

    # ---- To numpy ----
    input_np = x_cond_in.squeeze().detach().cpu().numpy()
    gt_np    = y_target.squeeze().detach().cpu().numpy()
    cfm_np   = y_pred_cfm.squeeze().detach().cpu().numpy()
    rddm_np  = y_pred_rddm.squeeze().detach().cpu().numpy()

    # ---- Plot & Save (four vertically stacked subplots) ----
    plot_four_vertical(
        input_sig=input_np, gt=gt_np, pred_cfm=cfm_np, pred_rddm=rddm_np,
        fs=fs, normalize=normalize, save_dir=save_dir, basename=basename
    )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    infer_and_compare(
        dataset="mmECG",
        data_root_ckpt="/data/user/RCFM/saved",
        pick_index=25,
        manual_epoch_cfm=879,     # e.g., 80
        manual_epoch_rddm=None,    # e.g., 20
        sample_steps=50,
        fs=None,                   # e.g., 256 if you want seconds on x-axis
        normalize=False,           # True if you want min-max per-trace scaling
        save_dir="./figs",
        basename="mmECG_cmp"
    )
