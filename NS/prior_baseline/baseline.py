from ANI import N0, ANIBASE, ResidualBlockWithT, MLP
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch.fft as fft
import torch.fft
import math


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A(N0):
    def __init__(self, Nx=256, Ny=512, Lx=1.0, Ly=2.0, Re=1e4, device='cuda:2'):
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

    def single_step(self, u: torch.Tensor, dts: float = None, dt_max: float = 1e-3) -> torch.Tensor:
        """
        Solves ω_t = μ Δω using ETDRK4 method (only linear diffusion).
        u: [B, 1, Nx, Ny] - vorticity field
        returns: [B, 1, Nx, Ny] - advanced vorticity field
        """
        if u.ndim != 4 or u.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, Nx, Ny], got {u.shape}")
        
        # Time step size
        dt = dt_max if dts is None else dts
        mu = 1.0 / self.Re
        B, _, Nx, Ny = u.shape
        
        # Compute exponential factor: exp(μ Δ t * Δ)
        L = mu * self.laplacian_k_fine  # shape [Nx, Ny]
        exp_Ldt = torch.exp(L * dt).to(u.device)  # shape [Nx, Ny]

        # FFT
        u_real = u.squeeze(1)  # [B, Nx, Ny]
        u_k = torch.fft.fft2(u_real)  # [B, Nx, Ny]

        # ETD step: multiply in Fourier space
        u_k_next = u_k * exp_Ldt.unsqueeze(0)  # broadcast over batch

        # iFFT back
        u_next_real = torch.fft.ifft2(u_k_next).real  # [B, Nx, Ny]
        return u_next_real.unsqueeze(1)  # [B, 1, Nx, Ny]

class CNEXTUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_width=64, cond_dim=1, n_blocks=4):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_width = base_width
        self.norm = nn.InstanceNorm2d(base_width, affine=False)

        # FiLM 参数预测器：输出 gamma/beta 共 2×width×n_blocks
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 2 * base_width * n_blocks)
        )

        # 编码器 blocks（Conv → InstanceNorm → FiLM → GELU）
        self.encoder = nn.ModuleList()
        for _ in range(n_blocks):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, base_width, 3, padding=1),
                nn.InstanceNorm2d(base_width, affine=False),
            ))
            in_channels = base_width  # only first uses in_channels=4

        # 中间处理
        self.middle = nn.Sequential(
            self.make_res_block(base_width),
            self.make_res_block(base_width),
            self.make_res_block(base_width)
        )

        # 解码器（可选添加更多层）
        self.decoder = nn.Sequential(
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 1)
        )

    def make_res_block(self, width):
        return nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1)
        )

    def apply_film(self, x, gamma, beta):
        # x: [B, C, H, W], gamma/beta: [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x)
        return gamma * x + beta

    def forward(self, x, cond):
        B = x.shape[0]
        if cond.dim() == 0:
            cond = cond.expand(B, 1)  # [B, 1]
        elif cond.dim() == 1:
            cond = cond.unsqueeze(-1)  # [B] → [B, 1]
        film_params = self.film_mlp(cond)  # [B, 2 * C * n_blocks]
        film_params = film_params.view(B, self.n_blocks, 2, self.base_width)

        # 编码阶段，每层都用 FiLM
        for i, block in enumerate(self.encoder):
            x = block(x)  # Conv + InstanceNorm
            gamma = film_params[:, i, 0]  # [B, C]
            beta = film_params[:, i, 1]   # [B, C]
            x = self.apply_film(x, gamma, beta)
            x = F.gelu(x)

        # 中间层
        for res_block in self.middle:
            x = x + res_block(x)

        # 解码器
        x = self.decoder(x)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.complex128,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):    
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, film_input=1, num_layers=3):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.num_layers = num_layers

        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for (x, y)

        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(width, affine=False) for _ in range(num_layers)
        ])

        # MLP FiLM 输出 gamma/beta： [num_layers, 2, width]
        self.film_mlp = nn.Sequential(
            nn.Linear(film_input, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 2 * width * num_layers)
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, x):
        B, H, W, _ = x.shape
        gridx = torch.linspace(0, 1, W+1, device=x.device)[:-1]
        gridy = torch.linspace(0, 1, H+1, device=x.device)[:-1]
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing="xy")
        grid = torch.stack((gridx, gridy), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def apply_film(self, x, gamma, beta):
        return gamma * x + beta

    def forward(self, x, dts=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x)
        x = torch.cat([x, grid], dim=-1)  # [B, H, W, C+2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.pad(x, [0, self.padding, 0, self.padding], mode='circular')

        # FiLM 调制参数
        if dts is None:
            gamma_beta = torch.cat([
                torch.ones(B, self.num_layers, self.width, device=x.device),
                torch.zeros(B, self.num_layers, self.width, device=x.device)
            ], dim=-1)  # [B, num_layers, 2*width]
        else:
            if dts.dim() == 0:
                dts = dts.repeat(B, 1)
            elif dts.dim() == 1:
                dts = dts.view(B, -1)
            gamma_beta = self.film_mlp(dts)  # [B, 2*width*num_layers]
            gamma_beta = gamma_beta.view(B, self.num_layers, 2, self.width)  # [B, L, 2, width]

        for i in range(self.num_layers):
            x_res = x
            x = self.norms[i](x)

            gamma = gamma_beta[:, i, 0].unsqueeze(-1).unsqueeze(-1)  # [B, width, 1, 1]
            beta = gamma_beta[:, i, 1].unsqueeze(-1).unsqueeze(-1)
            x = self.apply_film(x, gamma, beta)

            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x)
            x = F.gelu(x1 + x2) + x_res

        x = x[..., :-self.padding, :-self.padding]  # 去掉 padding
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # [B, H, W, out_channels]
        return x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

class NavierStokes(nn.Module):
    def __init__(self, N0_SCHEME: N0, modes1=16, modes2=16, width=64, dt=0.1, device=device):
        super(NavierStokes, self).__init__()
        self.N0_SCHEME = N0_SCHEME
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.dt    = dt
        self.mu    = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(1.0, dtype=torch.float64, device=device)

        self.fno = FNO2d(modes1=modes1, modes2=modes2, width=width, film_input=1, in_channels=3, out_channels=1).to(device)
        # self.fno   = CNN2d_FiLM(in_channels=2, out_channels=2, width=width, film_input=1, n_layers=2).to(device)
        # self.fno   = FiLMUNet(in_channels=4, out_channels=2, base_width=width, cond_dim=1).to(device)
        # self.fno = CNEXTUNet(in_channels=3, out_channels=2, base_width=width, cond_dim=1).to(device)

        self.dt_tensor = torch.tensor(dt, dtype=torch.float64, device=device)
        # self.register_buffer("dt_tensor", torch.tensor(dt/2, dtype=torch.float64))
        self.Kx = self.N0_SCHEME.Kx_fine.unsqueeze(0).unsqueeze(0)  # [1, 1, Nx, Ny]
        self.Ky = self.N0_SCHEME.Ky_fine.unsqueeze(0).unsqueeze(0)  # [1, 1, Nx, Ny]

        self.Lx = 1.0
        self.Ly = 1.0
        self.Nx = N0_SCHEME.Kx_fine.shape[0]
        self.Ny = N0_SCHEME.Kx_fine.shape[1]
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        # 固定的归一化网格
        y_idx = torch.arange(self.Ny, device=device).view(1, self.Ny, 1).expand(1, self.Ny, self.Nx)
        x_idx = torch.arange(self.Nx, device=device).view(1, 1, self.Nx).expand(1, self.Ny, self.Nx)
        # self.register_buffer("x_phys", x_idx * self.dx)
        # self.register_buffer("y_phys", y_idx * self.dy)
        self.x_phys = x_idx * self.dx
        self.y_phys = y_idx * self.dy

    def add_coords(self, x):
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return torch.cat([x, grid], dim=1)
    
    def modify_input(self, x):
        return (x - self.mu) / self.sigma
    
    def modify_output(self, x):
        return self.sigma * x + self.mu
    
    def get_mu_and_sigma(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, device=device)

    def project_to_div_free(self, u):
        u_x_k = torch.fft.fft2(u[:, 0])
        u_y_k = torch.fft.fft2(u[:, 1])

        u_x_k_proj, u_y_k_proj = self.N0_SCHEME._apply_incompressibility_projection(
            u_x_k, u_y_k,
            self.N0_SCHEME.Pkx_x_fine, self.N0_SCHEME.Pky_y_fine,
            self.N0_SCHEME.Pkx_y_fine, self.N0_SCHEME.Pky_x_fine
        )

        u_x_real = torch.fft.ifft2(u_x_k_proj).real
        u_y_real = torch.fft.ifft2(u_y_k_proj).real
        return torch.stack([u_x_real, u_y_real], dim=1)

    def forward(self, x):
        # A: Convert velocity to vorticity
        x = x.to(self.dt_tensor.device)

        omega = self.N0_SCHEME._velocity_to_vorticity_spectral(
            x[:, 0], x[:, 1],
            self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine
        )  # [B, H, W]
        omega1 = omega.unsqueeze(1)
        omega1 = self.N0_SCHEME.single_step(omega1, dts=self.dt).squeeze(1)
        mean_old = omega.mean(dim=(-2, -1), keepdim=True)
        u, v = self.N0_SCHEME._vorticity_to_velocity_spectral(
            omega, self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine, self.N0_SCHEME.denom_safe_fine
        )
        omega_features = torch.stack([omega, u, v], dim=1)  # [B, H, W, 3]
        delta = self.fno(omega_features, self.dt_tensor)  # [B, 3, H, W]
        residual = self.dt_tensor * delta
        omega_new = omega1 + residual.squeeze(1)
        mean_new = omega_new.mean(dim=(-2, -1), keepdim=True)
        omega = omega_new - mean_new + mean_old
        
        u, v = self.N0_SCHEME._vorticity_to_velocity_spectral(
            omega, self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine, self.N0_SCHEME.denom_safe_fine
        )
        
        # Return the final velocity field
        return torch.stack([u, v], dim=1)  # [B, 2, H, W]

    def predict(self, x):
        return self.forward(x)

class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None, limit=8000):
        # Load all three files
        raw_inputs  = torch.load(input_file)
        raw_outputs = torch.load(output_file)

        # Limit data to 2000 samples if they exist, otherwise take all
        num_samples_to_load = min(limit, raw_inputs.shape[0])

        self.inputs = (raw_inputs[:num_samples_to_load]).to(torch.float64)
        self.outputs = (raw_outputs[:num_samples_to_load]).to(torch.float64)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Return inputs, outputs, and epsilons
        return self.inputs[idx], self.outputs[idx]
    
def plot_losses(losses, losses_val):
    plt.figure()
    plt.plot(losses, label="Train")
    plt.plot(losses_val, label="Validation")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig("loss_plot_small.png")
    plt.close()

def spectral_loss(pred, target, model, alpha=0.1):
    """
    pred, target: [B, C, H, W], real tensors
    alpha: 高频权重放大因子
    """
    u_pred = pred[:, 0, :, :]
    v_pred = pred[:, 1, :, :]
    u_target = target[:, 0, :, :]
    v_target = target[:, 1, :, :]

    omega_pred = model._velocity_to_vorticity_spectral(u_pred, v_pred, model.Kx_fine, model.Ky_fine)
    omega_target = model._velocity_to_vorticity_spectral(u_target, v_target, model.Kx_fine, model.Ky_fine)

    return F.mse_loss(omega_pred, omega_target)


def total_loss(pred, target, model, λ_spatial=1.0, λ_spectral=1.0, alpha=0.1):
    # spatial = F.mse_loss(pred, target)
    # spectral = spectral_loss(pred, target, model, alpha)
    # return λ_spatial * spatial + λ_spectral * spectral
    return F.mse_loss(pred, target)

import re
import glob
import os

def find_best_checkpoint(current_limit, model_dir="models"):
    if not os.path.exists(model_dir):
        return None
    
    pattern = re.compile(r"best_model_fno_(\d+)_loadbefore\.pth")
    
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
    return os.path.join(model_dir, f"best_model_fno_{best_prev_size}_loadbefore.pth")


if __name__ == "__main__":
    # data_dir = "../dataset" # Or "../dataset" if that's where your data is

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
        default="../dataset", 
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

    batch_size   = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    A_model = A()

    model = NavierStokes(N0_SCHEME=A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, device=device), modes1=32, modes2=32, width=64, dt=0.01).to(device)
    # if args.train_limit != 40:
    #     checkpoint_path = find_best_checkpoint(args.train_limit)
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # 多卡并行
    # model = torch.compile(model, mode="max-autotune") # Uncomment if you want to use torch.compile

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 100
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-5)

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
            targets = targets.to(device)
            preds = model(inputs)
            # loss = criterion(preds, targets)
            loss = total_loss(preds, targets, A_model)
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
                targets = targets.to(device)
                preds = model(inputs)
                val_loss += total_loss(preds, targets, A_model).item() * inputs.size(0)
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
            torch.save(model.state_dict(), os.path.join('models', f"best_model_fno_{args.train_limit}_loadbefore.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")
