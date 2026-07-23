# std_vis.py
import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import get_ecg2ecg_datasets, get_ppg2ecg_datasets
from diffusion import load_pretrained_DPM
from metrics import *
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def visualize_xt_distance_to_target(model, condition_fn, target_signal, model_type="RCFM", steps=50, device="cuda", save_path=None):
    model.eval()
    target_signal = target_signal.to(device)
    B = target_signal.shape[0]
    xt = torch.randn_like(target_signal).to(device)
    distances = []

    for i in range(steps):
        t = torch.tensor([i / steps] * B, device=device)

        if model_type == "RCFM":
            cond = condition_fn(target_signal)
            v = model(xt, cond=cond, t=t)
            xt = xt + v * (1.0 / steps)
        if model_type == "CFM":
            cond = condition_fn(target_signal)
            v = model(xt, cond=cond, t=t)
            xt = xt + v * (1.0 / steps)
        elif model_type == "RDDM":
            cond1 = condition_fn[0](target_signal)
            cond2 = condition_fn[1](target_signal)
            v = model(xt, cond1=cond1, cond2=cond2, t=t, mode="velocity")
            xt = xt + v * (1.0 / steps)
        else:
            raise ValueError("model_type must be 'RCFM' or 'RDDM'")

        dist = F.mse_loss(xt, target_signal).item()
        distances.append(dist)

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, steps), distances, label=f"{model_type} Distance")
    plt.xlabel("Time step t")
    plt.ylabel("MSE to Target ECG")
    plt.title(f"{model_type} Trajectory Convergence")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def run_eval():
    config = {
        "batch_size": 256,
        "nT": 50,
        "device": "cuda",
        "window_size": 4,
        "eval_datasets": ["mmECG"],
        "model_name": "CFM",
    }

    window_size = config["window_size"]
    EVAL_DATASETS = config["eval_datasets"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    PATH = f"/data/user/RCFM/saved/{EVAL_DATASETS[0]}/"
    device = config["device"]
    model_name = config["model_name"]

    set_deterministic(31)

    if "PTBXL" in EVAL_DATASETS:
        _, dataset_test = get_ecg2ecg_datasets(datasets=EVAL_DATASETS, window_size=window_size)
    else:
        _, dataset_test = get_ppg2ecg_datasets(datasets=EVAL_DATASETS, window_size=window_size)

    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    if model_name == "RDDM":
        dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(PATH=PATH, nT=nT, type=model_name, device=device)
        dpm.eval()
        Conditioning_network1.eval()
        Conditioning_network2.eval()

        for i, (y12_ecg, x_ecg, ecg_roi) in enumerate(testloader):
            if i == 0:
                visualize_xt_distance_to_target(
                    model=dpm,
                    condition_fn=(Conditioning_network1, Conditioning_network2),
                    target_signal=y12_ecg[:1].float(),
                    model_type="RDDM",
                    steps=50,
                    device=device,
                    save_path=f"/data/user/RCFM/plots/RDDM_traj_distance.png"
                )
            break

    elif model_name == "CFM":
        flow_model, condition_net, _ = load_pretrained_DPM(PATH=PATH, nT=nT, type=model_name, device=device)
        flow_model.eval()
        condition_net.eval()

        for i, (y12_ecg, x_ecg, ecg_roi) in enumerate(testloader):
            if i == 0:
                visualize_xt_distance_to_target(
                    model=flow_model,
                    condition_fn=condition_net,
                    target_signal=y12_ecg[:1].float(),
                    model_type="CFM",
                    steps=50,
                    device=device,
                    save_path=f"/data/user/RCFM/plots/CFM_traj_distance.png"
                )
            break

if __name__ == "__main__":
    run_eval()
