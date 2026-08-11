import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data import get_ecg2ecg_datasets, get_ppg2ecg_datasets
import torch.nn as nn
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

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def eval_diffusion( window_size, EVAL_DATASETS, nT=10, batch_size=64, PATH=None, model_name="RCFM", device="cuda"):

    if "PTBXL" not in EVAL_DATASETS and "ICBEB" not in EVAL_DATASETS:

        _, dataset_test = get_ppg2ecg_datasets(datasets=EVAL_DATASETS, window_size=window_size)
    
    else:
        _, dataset_test = get_ecg2ecg_datasets(datasets=EVAL_DATASETS, window_size=window_size)

    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    if model_name == "RDDM":

        dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
            PATH=PATH,
            nT=nT,
            type=model_name,
            device=device
        )
        
        # dpm = nn.DataParallel(dpm)
        # Conditioning_network1 = nn.DataParallel(Conditioning_network1)
        # Conditioning_network2 = nn.DataParallel(Conditioning_network2)

        dpm.eval()
        Conditioning_network1.eval()
        Conditioning_network2.eval()

        with torch.no_grad():

            fd_list = []
            fake_ecgs = np.zeros((1, 128*window_size))
            real_ecgs = np.zeros((1, 128*window_size))
            real_ppgs = np.zeros((1, 128*window_size))

            for y12_ecg, x_ecg in tqdm(testloader):

                x_ecg = x_ecg.float().to(device)
                y12_ecg = y12_ecg.float().to(device)

                generated_windows = []

                for ppg_window in torch.split(x_ecg, 128*4, dim=-1):
                    
                    if ppg_window.shape[-1] != 128*4:
                        
                        ppg_window = F.pad(ppg_window, (0, 128*4 - ppg_window.shape[-1]), "constant", 0)

                    ppg_conditions1 = Conditioning_network1(ppg_window)
                    ppg_conditions2 = Conditioning_network2(ppg_window)

                    xh = dpm(
                        cond1=ppg_conditions1, 
                        cond2=ppg_conditions2, 
                        mode="sample", 
                        window_size=128*4
                    )
                    
                    generated_windows.append(xh.cpu().numpy())

                xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window_size]

                fd = calculate_FD(y12_ecg, torch.from_numpy(xh).to(device))

                fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128*window_size)))
                real_ecgs = np.concatenate((real_ecgs, y12_ecg.reshape(-1, 128*window_size).cpu().numpy()))
                real_ppgs = np.concatenate((real_ppgs, x_ecg.reshape(-1, 128*window_size).cpu().numpy()))
                fd_list.append(fd)


                # plot one example
                if True:
                    PICPATH = f"/data/user/RCFM/plots"
                    plt.figure(figsize=(12, 6))
                    plt.plot(real_ppgs[7].reshape(-1), label='Input ECG', alpha=0.7)
                    plt.plot(real_ecgs[7].reshape(-1), label='Real ECG', alpha=0.7)
                    plt.plot(fake_ecgs[7].reshape(-1), label='Generated ECG', alpha=0.7)
                    plt.title('Input vs Real vs Generated ECG')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.savefig(f"{PICPATH}/{model_name}_ecg2ecg_icbeb_p2.png")
                    plt.close()

            mae_hr_ecg, rmse_score = evaluation_pipeline(real_ecgs[1:], fake_ecgs[1:])

            tracked_metrics = {
                "RMSE_score": rmse_score,
                "MAE_HR_ECG": mae_hr_ecg,
                "FD": sum(fd_list) / len(fd_list),
            }

            return tracked_metrics

    elif model_name == "RCFM" or model_name == "CFM":
        flow_model, condition_net, _ = load_pretrained_DPM(
            PATH=PATH,
            nT=nT,
            type=model_name,
            device=device
        )
        # dpm = nn.DataParallel(flow_model)
        # condition_net = nn.DataParallel(condition_net)

        flow_model.eval().to(device)
        condition_net.eval().to(device)


        with torch.no_grad():

            fd_list = []
            fake_ecgs = np.zeros((1, 128*window_size))
            real_ecgs = np.zeros((1, 128*window_size))
            real_ppgs = np.zeros((1, 128*window_size))

            for y12_ecg, x_ecg in tqdm(testloader):

                x_ecg = x_ecg.float().to(device)
                y12_ecg = y12_ecg.float().to(device)

                generated_windows = []

                for ppg_window in torch.split(x_ecg, 128*4, dim=-1):
                    
                    if ppg_window.shape[-1] != 128*4:
                        
                        ppg_window = F.pad(ppg_window, (0, 128*4 - ppg_window.shape[-1]), "constant", 0)

                    ppg_conditions1 = condition_net(ppg_window)

                    xh = flow_model(
                        cond=ppg_conditions1, 
                        mode="sample", 
                        window_size=128*4,
                      
                    )
                    
                    generated_windows.append(xh.cpu().numpy())

                xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window_size]

                fd = calculate_FD(y12_ecg, torch.from_numpy(xh).to(device))

                fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128*window_size)))
                real_ecgs = np.concatenate((real_ecgs, y12_ecg.reshape(-1, 128*window_size).cpu().numpy()))
                real_ppgs = np.concatenate((real_ppgs, x_ecg.reshape(-1, 128*window_size).cpu().numpy()))
                fd_list.append(fd)

                # plot one example
                if True:
                    PICPATH = f"/data/user/RCFM/plots"
                    plt.figure(figsize=(12, 6))
                    plt.plot(real_ppgs[1].reshape(-1), label='Input PPG', alpha=0.7)
                    plt.plot(real_ecgs[1].reshape(-1), label='Real ECG', alpha=0.7)
                    plt.plot(fake_ecgs[1].reshape(-1), label='Generated ECG', alpha=0.7)
                    plt.title('Input vs Real vs Generated ECG')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.savefig(f"{PICPATH}/{model_name}_ppg2ecg_comparison_p1.png")
                    plt.close()


            mae_hr_ecg, rmse_score = evaluation_pipeline(real_ecgs[1:], fake_ecgs[1:])

            tracked_metrics = {
                "RMSE_score": rmse_score,
                "MAE_HR_ECG": mae_hr_ecg,
                "FD": sum(fd_list) / len(fd_list),
            }

            return tracked_metrics

if __name__ == "__main__":

    config = {
        "batch_size": 256,
        "nT": 50,
        "device": "cuda",
        "window_size": 4, # Seconds
        "eval_datasets": ["MIMIC-AFib"]
    }

    # TABLE 1 results
    print("\n******* Standard evaluation (Table 1) results *******")
    for dataset_name in ["MIMIC-AFib"]: #"PTBXL" "MIMIC-AFib" "ICBEB"

        tracked_metrics = eval_diffusion(
            window_size=4,
            EVAL_DATASETS=[dataset_name],
            nT=50,
            PATH=f"/data/user/RCFM/saved/{dataset_name}/",
            model_name="CFM",
            device="cuda:1"
        )
        print(f"\n{dataset_name}: RMSE is {tracked_metrics['RMSE_score']}, MAE_HR is {tracked_metrics['MAE_HR_ECG']}, FD is {tracked_metrics['FD']}")
        print("-"*1000)

    # # TABLE 2 results
    # print("\n******* Heart Rate estimation (Table 2) results *******")
    # for dataset_name in ["WESAD", "DALIA"]:
        
    #     tracked_metrics = eval_diffusion(
    #         window_size=8,
    #         EVAL_DATASETS=[dataset_name],
    #         nT=10,
    #     )
    #     print(f"\n{dataset_name}: Mean Absolute Error (BPM) is {tracked_metrics['MAE_HR_ECG']}")
    #     print("-"*1000)
