import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import DiffusionUNetCrossAttention, ConditionNet
from data import get_ppg2ecg_datasets, get_ecg2ecg_datasets


class MinimalFlowMatching(nn.Module):
    """
    Minimal Flow Matching implementation for debugging
    Uses the simplest possible flow matching formulation
    """
    def __init__(self, flow_model, criterion=nn.MSELoss()):
        super(MinimalFlowMatching, self).__init__()
        self.flow_model = flow_model
        self.criterion = criterion

    def forward(self, x=None, y=None, cond=None, window_size=None,mode="train"):
        if mode == "train":
            # Simple linear interpolation flow matching
            # t ~ Uniform(0, 1)
            t = torch.rand(x.size(0), device=x.device)
            
            # Linear interpolation: x_t = (1-t)*y + t*x
            t_expanded = t.view(-1, 1, 1)  # Shape: [batch, 1, 1]
            x_t = (1 - t_expanded) * y + t_expanded * x
            
            # Velocity target: dx/dt = x - y (constant velocity)
            v_target = x - y
            
            # Predict velocity
            v_pred = self.flow_model(x_t, cond, t)
            
            return self.criterion(v_pred, v_target)

        elif mode == "sample":
            # Start from noise
            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            window_size = y.shape[-1] if y is not None else 512
            
            # Initialize with noise
            x = torch.randn(n_sample, 1, window_size).to(device)
            
            # Euler integration
            steps = 50  # Fewer steps for debugging
            dt = 1.0 / steps
            
            for i in range(steps):
                t = torch.full((n_sample,), i / steps, device=device)
                v = self.flow_model(x, cond, t)
                x = x + v * dt
                
            return x


def train_minimal_cfm(dataset = "MIMIC-AFib"):
    """Minimal training loop for debugging"""
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Small config for debugging
    config = {
        "batch_size": 256,
        "learning_rate": 1e-4,
        "n_epochs": 1000,
    }
    
    # Load data
    print("Loading datasets...")
    dataset_train, _ = get_ppg2ecg_datasets() # for ECG-PPG datasets
    dataloader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    
    # Models
    print("Initializing models...")
    flow_model = DiffusionUNetCrossAttention(512, 1, device, num_heads=4).to(device)
    condition_net = ConditionNet().to(device)
    
    cfm = MinimalFlowMatching(flow_model=flow_model).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(cfm.parameters()) + list(condition_net.parameters()),
        lr=config["learning_rate"]
    )
    
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        cfm.train()
        condition_net.train()
        
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (y12_ecg, x_ecg, ecg_roi) in enumerate(pbar):
            # Move to device
            x_ecg = x_ecg.float().to(device)
            y12_ecg = y12_ecg.float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get conditions
            conditions = condition_net(x_ecg)
            
            # Generate noise
            noise = torch.randn_like(y12_ecg)
            
            # Flow matching loss
            loss = cfm(x=y12_ecg, y=noise, cond=conditions, mode="train")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(cfm.parameters()) + list(condition_net.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Logging
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Debug info for first few batches
            if batch_idx < 3 and epoch == 0:
                print(f"\nBatch {batch_idx} debug:")
                print(f"  Input shape: {x_ecg.shape}")
                print(f"  Target shape: {y12_ecg.shape}")
                print(f"  Noise shape: {noise.shape}")
                print(f"  Loss: {loss.item():.4f}")
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print("  WARNING: NaN/Inf detected!")
                    break
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
        
        #save model state
        if (epoch+1) % 20 == 0:
            PATH= f"/data/user/RCFM/saved/{dataset}"
            torch.save(cfm.state_dict(), f"{PATH}/minimal_cfm_epoch_{epoch}.pth")
            torch.save(condition_net.state_dict(), f"{PATH}/condition_net_epoch_{epoch}.pth")

        # Simple sampling test every few epochs
        if epoch % 5 == 0:
            print("Testing sampling...")
            cfm.eval()
            condition_net.eval()
            
            with torch.no_grad():
                # Use a small batch for testing
                test_batch = next(iter(dataloader))
                y12_test, x_test, _ = test_batch
                x_test = x_test.float().to(device)[:4]
                y12_test = y12_test.float().to(device)[:4]
                
                test_conditions = condition_net(x_test)
                try:
                    samples = cfm(y=y12_test, cond=test_conditions, mode="sample")
                    print(f"  Sampling successful! Shape: {samples.shape}")
                    print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

                except Exception as e:
                    print(f"  Sampling failed: {e}")
    
    print("Training completed!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_minimal_cfm(dataset="MIMIC-AFib")
