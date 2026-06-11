import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.fft as fft
import torch.fft
import math
from model import FNO2d, ODEPairDataset, plot_losses, spectral_loss, total_loss
from ANI import N0

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A(nn.Module):
    def __init__(self, Nx=256, Ny=512, Lx=1.0, Ly=2.0, Re=1e4, device='cuda:3'):
        super().__init__()
        self.Nx = Nx  # Now represents the fine-grid Nx
        self.Ny = Ny  # Now represents the fine-grid Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re  # Reynolds number
        self.device = device

        # Grid spacing for the original (fine) resolution
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Cache coefficients for the original (fine) grid
        self._cache_coefficients()

    def _get_spectral_operators(self, Nx, Ny, Lx, Ly, device):
        dx = Lx / Nx
        dy = Ly / Ny
        # Wavenumbers for FFT. torch.fft.fftfreq handles the correct ordering for FFT.
        kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=dx).to(device)
        ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=dy).to(device)
        
        # Create 2D wavenumber grids using meshgrid for broadcasting operations
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        # Laplacian operator in Fourier space: -(k_x^2 + k_y^2)
        laplacian_k = -(Kx**2 + Ky**2)
        laplacian_k[0, 0] = 0.0

        # Denominator for projection operator and stream function calculation: |k|^2 = k_x^2 + k_y^2
        denom = (Kx**2 + Ky**2)
        
        # Create a safe version of the denominator to avoid division by zero at (0,0) wavenumber.
        denom_safe = denom.clone()
        denom_safe[0, 0] = 1.0 

        # Projection operator (P) components for ensuring incompressibility (∇ · u = 0).
        # P = I - (k k^T) / |k|^2
        # P_xx = 1 - kx^2 / |k|^2
        # P_yy = 1 - ky^2 / |k|^2
        # P_xy = -kx ky / |k|^2
        # P_yx = -ky kx / |k|^2
        Pkx_x = 1.0 - Kx * Kx / denom_safe
        Pky_y = 1.0 - Ky * Ky / denom_safe
        Pkx_y = -Kx * Ky / denom_safe
        Pky_x = -Ky * Kx / denom_safe

        Pkx_x[0,0] = 0.0
        Pky_y[0,0] = 0.0
        Pkx_y[0,0] = 0.0
        Pky_x[0,0] = 0.0

        return Kx, Ky, laplacian_k, Pkx_x, Pky_y, Pkx_y, Pky_x, denom_safe

    def _cache_coefficients(self):
        (self.Kx_fine, self.Ky_fine, self.laplacian_k_fine, 
         self.Pkx_x_fine, self.Pky_y_fine, self.Pkx_y_fine, self.Pky_x_fine, self.denom_safe_fine) = \
            self._get_spectral_operators(self.Nx, self.Ny, self.Lx, self.Ly, self.device)

    def _compute_derivatives_spectral(self, field_real, Kx, Ky):
        field_k = torch.fft.fft2(field_real)
        
        # Derivative in Fourier space: d/dx -> i * kx * F(field)
        dfdx_k = 1j * Kx.unsqueeze(0) * field_k
        dfdx_real = torch.fft.ifft2(dfdx_k).real

        # Derivative in Fourier space: d/dy -> i * ky * F(field)
        dfdy_k = 1j * Ky.unsqueeze(0) * field_k
        dfdy_real = torch.fft.ifft2(dfdy_k).real
        
        return dfdx_real, dfdy_real

    def _apply_incompressibility_projection(self, u_k, v_k, Pkx_x, Pky_y, Pkx_y, Pky_x):
        # Apply the projection matrix: [P_xx P_yx; P_xy P_yy] * [u_k; v_k]
        u_x_k_proj = Pkx_x.unsqueeze(0) * u_k + Pky_x.unsqueeze(0) * v_k
        u_y_k_proj = Pkx_y.unsqueeze(0) * u_k + Pky_y.unsqueeze(0) * v_k
        return u_x_k_proj, u_y_k_proj

    def _vorticity_to_velocity_spectral(self, omega_real, Kx, Ky, denom_safe):
        # FFT vorticity to Fourier space
        omega_k = torch.fft.fft2(omega_real)

        # Solve for stream function in Fourier space: ψ_k = -ω_k / (k_x^2 + k_y^2)
        # denom_safe handles the (0,0) mode, ensuring stability.
        psi_k = -omega_k / denom_safe.unsqueeze(0)
        # The mean stream function is arbitrary, so set its (0,0) mode to zero.
        # This also ensures the (0,0) mode of psi_k is 0 after division by denom_safe[0,0]=1.0.
        # psi_k[:, 0, 0] = 0.0 

        # Convert stream function to velocity components in Fourier space:
        # u_k = -i k_y ψ_k
        # v_k = i k_x ψ_k
        u_x_k = -1j * Ky.unsqueeze(0) * psi_k
        u_y_k = 1j * Kx.unsqueeze(0) * psi_k

        # Inverse FFT to get real-space velocity components
        u_x_real = torch.fft.ifft2(u_x_k).real
        u_y_real = torch.fft.ifft2(u_y_k).real

        return u_x_real, u_y_real

    def _velocity_to_vorticity_spectral(self, u_x_real, u_y_real, Kx, Ky):
        # Compute derivatives: ∂u/∂y and ∂v/∂x
        _, dudy = self._compute_derivatives_spectral(u_x_real, Kx, Ky)
        dvdx, _ = self._compute_derivatives_spectral(u_y_real, Kx, Ky)

        # Calculate vorticity: ω = ∂v/∂x - ∂u/∂y
        omega_real = dvdx - dudy
        return omega_real

import re
import glob
import os

def find_best_checkpoint(current_limit, model_dir="models"):
    if not os.path.exists(model_dir):
        return None
    
    pattern = re.compile(r"best_model_fno_(\d+)_final_loadbefore\.pth")
    
    existing_sizes = []
    for f in os.listdir(model_dir):
        match = pattern.search(f)
        if match:
            size = int(match.group(1))
            if size < current_limit:
                existing_sizes.append(size)
    
    if not existing_sizes:
        return None
    
    best_prev_size = max(existing_sizes)
    return os.path.join(model_dir, f"best_model_fno_{best_prev_size}_final_loadbefore.pth")

if __name__ == "__main__":
    batch_size = 8
    # data_dir = "../dataset"

    parser = argparse.ArgumentParser(description="Training script for datasets.")

    parser.add_argument(
        "--train_limit", 
        type=int, 
        default=4000, 
        help="Number of samples to limit in the training dataset (default: 4000)"
    )
    parser.add_argument(
        "--val_limit", 
        type=int, 
        default=1000, 
        help="Number of samples to limit in the validation dataset (default: 1000)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../dataset/", 
        help="Path to the directory containing .pt files (default: ../dataset)"
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    train_dataset = ODEPairDataset(
        os.path.join(data_dir, "train_input.pt"),
        os.path.join(data_dir, "train_output.pt"),
        limit=args.train_limit,
    )
    val_dataset = ODEPairDataset(
        os.path.join(data_dir, "val_input.pt"),
        os.path.join(data_dir, "val_output.pt"),
        limit=args.val_limit,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = FNO2d(modes1=32, modes2=32, width=64, in_channels=4, out_channels=1).to(device)
    if args.train_limit == 40:
        model.load_state_dict(torch.load('models/best_model_fno_8000.pth', map_location=device))
    else:
        checkpoint = find_best_checkpoint(args.train_limit)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    dt    = torch.tensor(0.01).to(device)

    A_model = A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, Re=1e4).to(device)

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 100
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-6)

    best_val_loss = float('inf')

    train_loss_lists = []
    val_loss_lists = []

    # Create directories for saving loss files if they don't exist
    os.makedirs('logs', exist_ok=True)
    train_loss_file = open(os.path.join('logs', f"train_loss_{args.train_limit}.txt"), "w")
    val_loss_file = open(os.path.join('logs', f"val_loss_{args.train_limit}.txt"), "w")

    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Unpack inputs, targets, and epsilons from the DataLoader
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"[train] Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            optimizer.zero_grad()
            inputs = inputs.to(device)
            B, _, Nx, Ny = inputs.shape
            nu_layer = torch.full((B, 1, Nx, Ny), 1e-4, device=inputs.device, dtype=inputs.dtype)
            targets = targets.to(device)
            omega_input = A_model._velocity_to_vorticity_spectral(
                inputs[:, 0], inputs[:, 1], A_model.Kx_fine, A_model.Ky_fine
            )
            omega_input = omega_input.unsqueeze(1)
            combined_input = torch.cat([
                omega_input,      
                inputs,         
                nu_layer      
            ], dim=1)

            preds = model(combined_input, dt)
            u, v = A_model._vorticity_to_velocity_spectral(preds.squeeze(1), A_model.Kx_fine, A_model.Ky_fine, A_model.denom_safe_fine)
            preds1 = torch.cat([u.unsqueeze(1), v.unsqueeze(1)], dim=1)

            loss = total_loss(preds1, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_lists.append(train_loss)
        train_loss_file.write(f"{train_loss:.5e}\n")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Unpack inputs, targets, and epsilons from the DataLoader
            # for inputs, targets in val_loader:
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                print(f"[val] Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(val_loader)}", end='\r')
                inputs = inputs.to(device)
                B, _, Nx, Ny = inputs.shape
                nu_layer = torch.full((B, 1, Nx, Ny), 1e-4, device=inputs.device, dtype=inputs.dtype)
                targets = targets.to(device)
                omega_input = A_model._velocity_to_vorticity_spectral(
                    inputs[:, 0], inputs[:, 1], A_model.Kx_fine, A_model.Ky_fine
                )
                omega_input = omega_input.unsqueeze(1)
                combined_input = torch.cat([
                    omega_input,      
                    inputs,         
                    nu_layer      
                ], dim=1)

                preds = model(combined_input, dt)
                u, v = A_model._vorticity_to_velocity_spectral(preds.squeeze(1), A_model.Kx_fine, A_model.Ky_fine, A_model.denom_safe_fine)
                preds1 = torch.cat([u.unsqueeze(1), v.unsqueeze(1)], dim=1)
                val_loss += total_loss(preds1, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create directory for saving models if it doesn't exist
            os.makedirs('.', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', f"best_model_fno_{args.train_limit}_final_loadbefore.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")
