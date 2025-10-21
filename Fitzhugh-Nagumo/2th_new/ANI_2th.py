from ANI import N0, ANIBASE, ResidualBlockWithT, MLP
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(555)
np.random.seed(555)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cdouble))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cdouble))

    #Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = torch.einsum("bixy,ioxy->boxy",a,b)
        return op

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(4, self.width) # input channel is 4: (U, V, x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class FNO_ETDRK4(nn.Module):
    def __init__(self, modes1, modes2, width, N=128, Du=1e-2, Dv=1e-2, device='cuda'):
        super(FNO_ETDRK4, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.N = N # Spatial resolution
        self.Du = Du # Diffusion coefficient for u
        self.Dv = Dv # Diffusion coefficient for v
        self.device = device

        self.fno = FNO2d(modes1, modes2, width)

        self._init_fft_grid()
        self.etdrk4_coeff_cache = {}

    def _init_fft_grid(self):
        k_x = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(k_x, k_y, indexing='ij')
        self.K2 = -(KX**2 + KY**2) # [N, N]

        # Prepare grid for FNO input
        grid_x = torch.linspace(0, 1, self.N+1, device=self.device)[:-1]
        grid_y = torch.linspace(0, 1, self.N+1, device=self.device)[:-1]
        # Stack as [N, N, 2]
        self.grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1) # [N, N, 2]

    def _compute_etdrk4_coeffs(self, dt_val):
        # dt_val should be a scalar
        L_u = self.Du * self.K2 # Linear part for u [N, N]
        L_v = self.Dv * self.K2 # Linear part for v [N, N]

        # Calculate coefficients for u
        E_u = torch.exp(dt_val * L_u) # [N, N]
        E2_u = torch.exp(dt_val * L_u / 2) # [N, N]
        Q_u, f1_u, f2_u, f3_u = self._phi_functions(dt_val, L_u) # All [N, N]

        # Calculate coefficients for v
        E_v = torch.exp(dt_val * L_v) # [N, N]
        E2_v = torch.exp(dt_val * L_v / 2) # [N, N]
        Q_v, f1_v, f2_v, f3_v = self._phi_functions(dt_val, L_v) # All [N, N]

        # Stack coefficients along the channel dimension (dim=0)
        # Resulting shapes: [2, N, N]
        E = torch.stack([E_u, E_v], dim=0)     # [2, N, N]
        E2 = torch.stack([E2_u, E2_v], dim=0)  # [2, N, N]
        Q = torch.stack([Q_u, Q_v], dim=0)     # [2, N, N]
        f1 = torch.stack([f1_u, f1_v], dim=0)  # [2, N, N]
        f2 = torch.stack([f2_u, f2_v], dim=0)  # [2, N, N]
        f3 = torch.stack([f3_u, f3_v], dim=0)  # [2, N, N]

        return E, E2, Q, f1, f2, f3

    def _phi_functions(self, dt, L):
        # Helper function to compute the phi functions for ETDRK4
        # L: [N, N]
        # Returns Q, f1, f2, f3 for a single component (u or v)

        M = 64 # Number of points for contour integral
        j = torch.arange(1, M + 1, device=self.device)
        r = torch.exp(2j * torch.pi * (j - 0.5) / M) # [M]
        # Reshape L to [N, N, 1] for broadcasting with r, then to [N*N, 1] for easier complex ops
        LR = dt * L.unsqueeze(-1) + r # [N, N, M]

        # Ensure that division by zero (LR=0) is handled
        epsilon = 1e-10 # Small epsilon to avoid division by zero
        LR_safe = LR + epsilon * (LR == 0).float() # Add epsilon only where LR is zero

        exp_LR = torch.exp(LR)
        exp_LR_half = torch.exp(LR / 2)

        Q_integral = torch.mean((exp_LR_half - 1) / LR_safe, dim=-1).real
        f1_integral = torch.mean((-4 - LR_safe + exp_LR * (4 - 3 * LR_safe + LR_safe**2)) / (LR_safe**3), dim=-1).real
        f2_integral = torch.mean((2 + LR_safe + exp_LR * (-2 + LR_safe)) / (LR_safe**3), dim=-1).real
        f3_integral = torch.mean((-4 - 3 * LR_safe - LR_safe**2 + exp_LR * (4 - LR_safe)) / (LR_safe**3), dim=-1).real

        Q = dt * Q_integral
        f1 = dt * f1_integral
        f2 = dt * f2_integral
        f3 = dt * f3_integral

        return Q, f1, f2, f3


    def forward(self, x, dts):
        # x: [B, N, N, 2] (u and v as channels)
        # dts: [B] (dt value for each sample in the batch)
        if isinstance(dts, torch.Tensor) and len(dts.shape) == 0:
            dts = dts.repeat(x.shape[0])

        u = x[:,:,:,0:2].permute(0,3,1,2)
        # grid = x[:,:,:,2:]
        grid = self.grid.repeat(x.shape[0], 1, 1, 1).to(x.device) # [B, N, N, 2]
        x = torch.cat([x, grid], dim=-1) # [B, N, N, 4]

        B, C, H, W = u.shape # C will be 2 (for u, v)

        # Get unique dt values and their inverse indices
        unique_dts, inverse_indices = torch.unique(dts, return_inverse=True)

        # Prepare coefficients for the batch
        # Coefficients will be [B, C, H, W]
        batch_E = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_E2 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_Q = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f1 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f2 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f3 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)

        for i, dt_val in enumerate(unique_dts):
            dt_val_item = dt_val.item() # Convert tensor scalar to Python float for dict key
            if dt_val_item not in self.etdrk4_coeff_cache:
                # _compute_etdrk4_coeffs returns [C, H, W]
                E, E2, Q, f1, f2, f3 = self._compute_etdrk4_coeffs(dt_val)
                self.etdrk4_coeff_cache[dt_val_item] = (E, E2, Q, f1, f2, f3)
            else:
                E, E2, Q, f1, f2, f3 = self.etdrk4_coeff_cache[dt_val_item]

            # Assign coefficients to the correct batch indices
            indices_in_batch = (inverse_indices == i).nonzero(as_tuple=True)[0]
            # E, E2, ... are [C, H, W]. Need to expand them to [1, C, H, W] before assigning
            # and then broadcast them to the batch indices.
            batch_E[indices_in_batch] = E.unsqueeze(0)
            batch_E2[indices_in_batch] = E2.unsqueeze(0)
            batch_Q[indices_in_batch] = Q.unsqueeze(0)
            batch_f1[indices_in_batch] = f1.unsqueeze(0)
            batch_f2[indices_in_batch] = f2.unsqueeze(0)
            batch_f3[indices_in_batch] = f3.unsqueeze(0)


        
        # stage 1
        u_hat = torch.fft.fft2(u)
        nu = (self.fno(x)).permute(0,3,1,2)
        nu_hat = torch.fft.fft2(nu)
        
        a_hat = batch_E2 * u_hat + batch_Q * nu_hat
        a = torch.fft.ifft2(a_hat).real
        
        # stage 2
        au = torch.cat((a.permute(0,2,3,1), grid), dim=-1)
        na = (self.fno(au)).permute(0,3,1,2)
        na_hat = torch.fft.fft2(na)
        
        b_hat = batch_E2 * u_hat + batch_Q * na_hat
        b = torch.fft.ifft2(b_hat).real
        
        # stage 3
        bu = torch.cat((b.permute(0,2,3,1), grid), dim=-1)
        nb = (self.fno(bu)).permute(0,3,1,2)
        nb_hat = torch.fft.fft2(nb)
        
        c_hat = batch_E2 * a_hat + batch_Q * (2*nb_hat - nu_hat)
        c = torch.fft.ifft2(c_hat).real
        
        # stage 4
        cu = torch.cat((c.permute(0,2,3,1), grid), dim=-1)
        nc = (self.fno(cu)).permute(0,3,1,2)
        nc_hat = torch.fft.fft2(nc)
        
        u_hat = batch_E * u_hat + batch_f1 * nu_hat + 2 * batch_f2 * (na_hat + nb_hat) + batch_f3 * nc_hat
        u = torch.fft.ifft2(u_hat).real
        
        return u.permute(0,2,3,1)
    
class FiLM(nn.Module):
    def __init__(self, width, hidden_dim=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(width, affine=False, dtype=torch.float64)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden_dim, width, dtype=torch.float64)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden_dim, width, dtype=torch.float64)
        )

    def forward(self, x, eps):
        # x: [B, C, H, W], eps: [B]
        x = self.norm(x)
        eps = eps.view(-1, 1).to(dtype=torch.float64)
        gamma = self.gamma_mlp(eps).view(-1, x.size(1), 1, 1)
        beta = self.beta_mlp(eps).view(-1, x.size(1), 1, 1)
        return gamma * x + beta

    
class FNO2d_FiLM(nn.Module):
    def __init__(self, modes1, modes2, width, N=128):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(4, width, dtype=torch.float64)  # [u,v,x,y]

        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.film1 = FiLM(width)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.film2 = FiLM(width)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.film3 = FiLM(width)

        self.fc1 = nn.Linear(width, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 2, dtype=torch.float64)

        x = torch.linspace(0, 1, N+1, dtype=torch.float64)[:-1]
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1)

    def forward(self, x, eps):
        # x: [B, N, N, 2]
        grid = self.grid.to(x.device).to(dtype=torch.float64)
        grid = grid.repeat(x.size(0), 1, 1, 1)
        x    = torch.cat((x, grid), dim=-1)  # [B, N, N, 4]
        x = self.fc0(x).permute(0, 3, 1, 2)  # [B, C, H, W], float64

        x1 = self.conv1(x)
        x1 = self.film1(x1, eps)
        x1 = F.gelu(x1)

        x2 = self.conv2(x1)
        x2 = self.film2(x2, eps)
        x2 = F.gelu(x1 + x2)

        x3 = self.conv3(x2)
        x3 = self.film3(x3, eps)
        x = F.gelu(x2 + x3)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class FiLMBlock(nn.Module):
    def __init__(self, channels, film_input_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.film = nn.Sequential(
            nn.Linear(film_input_dim, channels * 2)
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, eps):
        # x: [B, C, H, W], eps: [B] or [B, 1]
        B, C, H, W = x.shape
        if eps.ndim == 1:
            eps = eps.view(B, 1)

        x = self.norm(x)  # InstanceNorm
        gamma_beta = self.film(eps)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        x = gamma * x + beta  # FiLM
        x = self.conv(F.gelu(x))
        return x

class ResNet_FiLM(nn.Module):
    def __init__(self, in_channels=2, hidden=32, out_channels=2, film_input_dim=3, n_blocks=3):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            FiLMBlock(hidden, film_input_dim) for _ in range(n_blocks)
        ])

        self.decoder = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x, eps):
        # x: [B, H, W, 2], eps: [B] or [B, 1]
        x = x.permute(0, 3, 1, 2)  # -> [B, 2, H, W]
        x = self.encoder(x)       # -> [B, hidden, H, W]
        for block in self.blocks:
            x = x + block(x, eps)  # 残差连接
        x = self.decoder(x)
        return x.permute(0, 2, 3, 1)  # -> [B, H, W, 2]

class CNEXTUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_width=32, cond_dim=1, n_blocks=2):
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
        x = x.permute(0, 3, 1, 2)
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
        return x.permute(0, 2, 3, 1)  # -> [B, H, W, 2]

class RK4_solver(nn.Module):
    def __init__(self, modes1=8, modes2=8, width=40, dt=1e-2, device='cuda'):
        super().__init__()
        # self.model = FNO2d_FiLM(modes1, modes2, width)
        # self.model = PDEModelFiLMResNet()
        self.model = ResNet_FiLM()
        # self.model   = CNEXTUNet()
        self.dt = dt
        self.device = device

    def forward(self, u, eps=None, dt=None):
        """
        u: [B, N, N, C]   (float64)
        eps: [B] or scalar
        return: u_{n+1}, shape = [B, N, N, C]
        """
        if dt is None:
            dt = self.dt

        # k1
        k1 = self.model(u, eps)

        # k2
        u2 = u + dt/2 * k1
        k2 = self.model(u2, eps)

        # k3
        u3 = u + dt/2 * k2
        k3 = self.model(u3, eps)

        # k4
        u4 = u + dt * k3
        k4 = self.model(u4, eps)

        # Combine
        u_next = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return u_next


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.relu = nn.GELU()
        self.rescale = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.rescale(x)
        x = self.norm(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)



class ResNetWithFiLM(nn.Module):
    def __init__(self, channels=2, width=64, n_blocks=4):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, width, 1)
        self.blocks = nn.Sequential(*[SimpleResBlock(width, width) for _ in range(n_blocks)])
        self.film = FiLM(width)
        self.out_proj = nn.Conv2d(width, channels, 1)

    def forward(self, x, eps):
        # x: [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.in_proj(x)
        x = self.film(x, eps)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x.permute(0, 2, 3, 1)  # back to [B, H, W, C]


class FHN_Dataset(Dataset):
    def __init__(self, data_dir, split, device='cpu'):
        self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs_all.npy'))[:2000]
        self.outputs = np.load(os.path.join(data_dir, f'{split}_outputs_all.npy'))[:2000]
        self.dt = np.load(os.path.join(data_dir, f'{split}_eps_all.npy'))[:2000]
        self.device = device

        # Ensure data is float32 and move to device if applicable later
        # It's often better to convert to tensor and move to device in __getitem__ or during DataLoader iteration
        # to manage memory efficiently, especially for large datasets.

        # Check for consistency
        assert len(self.inputs) == len(self.outputs) == len(self.dt), "Mismatch in dataset sizes!"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors and move to device
        input_tensor = torch.from_numpy(self.inputs[idx]).to(self.device)
        output_tensor = torch.from_numpy(self.outputs[idx]).to(self.device)
        dt_tensor = torch.tensor(self.dt[idx], dtype=torch.float64).to(self.device) # dt is called eps in this context

        return input_tensor, output_tensor, dt_tensor
import hashlib
class ETDRK4Cache:
    def __init__(self):
        self.cache = {}

    def _hash(self, K2, dt):
        K2_np = K2.detach().cpu().numpy()
        m = hashlib.sha256()
        m.update(K2_np.tobytes())
        m.update(str(dt).encode())
        return m.hexdigest()

    def get(self, K2, dt):
        key = self._hash(K2, dt)
        return self.cache.get(key, None)

    def set(self, K2, dt, E, E2, Q, f1, f2, f3):
        key = self._hash(K2, dt)
        self.cache[key] = (E, E2, Q, f1, f2, f3)

class FitzHughNagumo2D(torch.nn.Module):
    def __init__(self, N=128, device='cuda'):
        super().__init__()
        self.N = N
        self.device = device
        self._init_fft_grid()
        self.etdrk4_cache = ETDRK4Cache()

    def _init_fft_grid(self):
        N = self.N
        k = torch.fft.fftfreq(N, d=1.0 / N, device=self.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(k, k, indexing='ij')
        self.K2 = -(KX ** 2 + KY ** 2)

    def _load_or_precompute_etdrk4(self, dt, D):
        cached = self.etdrk4_cache.get(D * self.K2, dt)
        if cached is not None:
            return cached

        K2 = self.K2
        L = D * K2
        E = torch.exp(dt * L)
        E2 = torch.exp(dt * L / 2)

        M = 64
        j = torch.arange(1, M + 1, device=self.device)
        r = torch.exp(2j * torch.pi * (j - 0.5) / M)
        LR = dt * L.unsqueeze(-1) + r  # [N, N, M]

        Q = dt * torch.mean((torch.exp(LR / 2) - 1) / LR, dim=-1).real
        f1 = dt * torch.mean(
            (-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR ** 2)) / (LR ** 3), dim=-1
        ).real
        f2 = dt * torch.mean(
            (2 + LR + torch.exp(LR) * (-2 + LR)) / (LR ** 3), dim=-1
        ).real
        f3 = dt * torch.mean(
            (-4 - 3 * LR - LR ** 2 + torch.exp(LR) * (4 - LR)) / (LR ** 3), dim=-1
        ).real

        self.etdrk4_cache.set(L, dt, E, E2, Q, f1, f2, f3)
        return E, E2, Q, f1, f2, f3

    def solve(self, u0, v0, dt=1e-3, t_end=1.0, 
            Du=1e-2, Dv=1e-2, a=0.7, b=0.8, eps=0.01,
            return_all=False, save_interval=10):
        """
        解 FitzHugh-Nagumo 方程组，输入初始 u0, v0，均为 [B, N, N]
        支持每个样本使用不同的 eps ∈ [B]（可为标量或张量）
        """
        B, N = u0.shape[0], self.N
        u = u0.to(self.device)
        v = v0.to(self.device)
        u_hat = torch.fft.fft2(u)
        v_hat = torch.fft.fft2(v)
        t = 0.0

        # ETDRK4 系数预计算
        Eu, Eu2, Qu, f1u, f2u, f3u = self._load_or_precompute_etdrk4(dt, Du)
        Ev, Ev2, Qv, f1v, f2v, f3v = self._load_or_precompute_etdrk4(dt, Dv)

        # 支持 batch-wise eps
        eps = torch.tensor(eps, device=self.device, dtype=u.dtype)
        if eps.ndim == 0:
            eps = eps.expand(B)          # 标量 -> 扩展为 [B]
        eps = eps.view(B, 1, 1)          # [B, 1, 1] 以便广播到 [B, N, N]

        if return_all:
            history_u = [u.detach().cpu().clone()]
            history_v = [v.detach().cpu().clone()]

        while t < t_end:
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real

            Nu = u - u**3 / 3 - v
            Nv = eps * (u + a - b * v)

            Nu_hat = torch.fft.fft2(Nu)
            Nv_hat = torch.fft.fft2(Nv)

            ua = Eu2 * u_hat + Qu * Nu_hat
            va = Ev2 * v_hat + Qv * Nv_hat
            ua_real = torch.fft.ifft2(ua).real
            va_real = torch.fft.ifft2(va).real
            Na_hat = torch.fft.fft2(ua_real - ua_real**3 / 3 - va_real)
            Nb_hat = torch.fft.fft2(eps * (ua_real + a - b * va_real))

            ub = Eu2 * u_hat + Qu * Na_hat
            vb = Ev2 * v_hat + Qv * Nb_hat
            ub_real = torch.fft.ifft2(ub).real
            vb_real = torch.fft.ifft2(vb).real
            Nb2_hat = torch.fft.fft2(ub_real - ub_real**3 / 3 - vb_real)
            Nc2_hat = torch.fft.fft2(eps * (ub_real + a - b * vb_real))

            uc = Eu2 * u_hat + Qu * (2 * Nb2_hat - Nu_hat)
            vc = Ev2 * v_hat + Qv * (2 * Nc2_hat - Nv_hat)
            uc_real = torch.fft.ifft2(uc).real
            vc_real = torch.fft.ifft2(vc).real
            Nc_hat = torch.fft.fft2(uc_real - uc_real**3 / 3 - vc_real)
            Nd_hat = torch.fft.fft2(eps * (uc_real + a - b * vc_real))

            u_hat = Eu * u_hat + f1u * Nu_hat + 2 * f2u * (Na_hat + Nb2_hat) + f3u * Nc_hat
            v_hat = Ev * v_hat + f1v * Nv_hat + 2 * f2v * (Nb_hat + Nc2_hat) + f3v * Nd_hat
            t += dt

            if return_all and int(t / dt) % save_interval == 0:
                history_u.append(torch.fft.ifft2(u_hat).real.detach().cpu())
                history_v.append(torch.fft.ifft2(v_hat).real.detach().cpu())

        if return_all:
            return torch.stack(history_u, dim=0), torch.stack(history_v, dim=0)  # [T, B, N, N]
        else:
            return torch.fft.ifft2(u_hat).real, torch.fft.ifft2(v_hat).real


class A(N0):
    def __init__(self):
        super().__init__()
        self.model = FNO_ETDRK4(modes1=8, modes2=8, width=40, N=128).to(device)
        # load pth model
        state_dict = torch.load("../basemodel/best_model.pth", map_location=device)
        self.model.load_state_dict(state_dict)
        # model 参数不可导
        for param in self.model.parameters():
            param.requires_grad = False

        # 评估模式
        self.model.eval()
        # self.solver = FitzHughNagumo2D(N=128, device=device)

    def single_step(self, u: torch.Tensor, parameters=None, dts=None, **kwargs) -> torch.Tensor:
        return self.model(u, dts)
        # return self.solver.solve(u[..., 0], u[..., 1], t_end=0.1, eps=0.1)
        # u, v = self.solver.solve(u[..., 0], u[..., 1], t_end=0.005, eps=0.01)
        # return torch.stack([u, v], dim=-1)
    
class AllenCahn(nn.Module):
    def __init__(self, N0_SCHEME: N0, modes1=16, modes2=16, width=64, dt=0.01):
        super(AllenCahn, self).__init__()
        self.N0_SCHEME = N0_SCHEME
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.dt    = dt
        self.mu    = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(1.0, dtype=torch.float64, device=device)

        # self.fno   = FNO2d_FiLM(modes1=modes1, modes2=modes2, width=width).to(device)
        self.fno   = RK4_solver(modes1=modes1, modes2=modes2, width=width).to(device)
        # self.fno     = ResNetWithFiLM().to(device)
        # self.fno     = PDEModelFiLMResNet().to(device)

        self.dt_tensor = torch.tensor(dt/2, dtype=torch.float64, device=device)
    
    def modify_input(self, x):
        return (x - self.mu) / self.sigma
    
    def modify_output(self, x):
        return self.sigma * x + self.mu
    
    def get_mu_and_sigma(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, device=device)

    def forward(self, x, eps):
        x = self.N0_SCHEME.single_step(x, dts=self.dt_tensor)
        x = self.fno(x, eps)
        x = self.N0_SCHEME.single_step(x, dts=self.dt_tensor)
        return x
    
    def predict(self, x, eps):
        x = self.N0_SCHEME.single_step(x, dts=self.dt_tensor)
        x = self.fno(x, eps)
        x = self.N0_SCHEME.single_step(x, dts=self.dt_tensor)
        return x

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

if __name__ == "__main__":
    # Ensure the 'data' directory exists and contains the generated files
    # For this example, I'll assume the files are in a 'data' directory
    # relative to where this script is run.
    # Adjust paths if your data is in a different location.
    data_dir = "../dataset/data" # Or "../dataset" if that's where your data is

    train_dataset = FHN_Dataset(
        data_dir, 'train_fhn', device=device
    )
    val_dataset   = FHN_Dataset(
        data_dir, 'val_fhn', device=device
    )

    batch_size   = 24
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    model = AllenCahn(N0_SCHEME=A(), modes1=8, modes2=8, width=20, dt=0.01).to(device)
    # checkpoint = torch.load("models/best_model.pth", map_location=device)
    # model.load_state_dict(checkpoint)
    # model = torch.compile(model, mode="max-autotune") # Uncomment if you want to use torch.compile

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )

    epochs = 500
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-5)

    best_val_loss = float('inf')

    train_loss_lists = []
    val_loss_lists = []

    # Create directories for saving loss files if they don't exist
    os.makedirs('logs', exist_ok=True)
    train_loss_file = open(os.path.join('logs', "train_loss.txt"), "w")
    val_loss_file = open(os.path.join('logs', "val_loss.txt"), "w")

    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Unpack inputs, targets, and epsilons from the DataLoader
        for batch_idx, (inputs, targets, epsilons) in enumerate(train_loader):
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            inputs, targets, epsilons = inputs.to(device), targets.to(device), epsilons.to(device)
            optimizer.zero_grad()
            # Pass epsilons to the model's forward method
            preds = model(inputs, epsilons)
            loss = criterion(preds, targets)
            # clip gradient
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for inputs, targets, epsilons in val_loader:
                # Pass epsilons to the model's forward method
                inputs, targets, epsilons = inputs.to(device), targets.to(device), epsilons.to(device)
                preds = model(inputs, epsilons)
                val_loss += criterion(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create directory for saving models if it doesn't exist
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', "best_model.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")