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
# from kan import KAN
from pit import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

mesh_ltt = torch.linspace(0, 1, 128+1)[:-1].reshape(-1, 1).to(device)

def plot_cons(u, x, save_path):
    # convert u to pris
    rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
    v = mom / rho
    kinetic = 0.5 * v**2 * rho
    internal = E - kinetic
    p = (1.4 - 1) * internal
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(x, rho)
    axs[0, 0].set_title('Density')
    axs[0, 1].plot(x, v)
    axs[0, 1].set_title('Velocity')
    axs[1, 0].plot(x, p)
    axs[1, 0].set_title('Pressure')
    axs[1, 1].plot(x, internal)
    axs[1, 1].set_title('Internal Energy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# class A(N0):
#     def __init__(self, mu=0.0, sigma=1.0, gamma=1.4, dx=1/128, dt=0.025, Ng=3):
#         super(A, self).__init__()
#         self.mu = mu
#         self.sigma = sigma
#         self.gamma = gamma
#         self.dx = dx
#         self.dt = dt
#         self.Ng = Ng  # 边界层的厚度

#     def flux(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         rho = torch.clamp(rho, min=1e-6)
#         v = mom / rho
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal

#         f1 = mom
#         f2 = mom * v + p
#         f3 = v * (E + p)

#         return torch.stack([f1, f2, f3], dim=-1)

#     def compute_alpha(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         v = mom / rho
#         rho = torch.clamp(rho, min=1e-6)
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal
#         c = torch.sqrt(self.gamma * p / rho + 1e-8)
#         return torch.abs(v) + c

#     def compute_rhs(self, u: torch.Tensor) -> torch.Tensor:
#         """
#         Lax-Friedrichs 空间离散形式 du/dt = RHS(u)
#         u: [B, Nx + 2*Ng, 3]
#         return: [B, Nx, 3] — 只返回物理域
#         """
#         self.update_boundaries(u)

#         Ng = self.Ng
#         u_center = u[:, Ng:-Ng, :]
#         u_m1 = u[:, Ng-1:-Ng-1, :]
#         u_p1 = u[:, Ng+1:-Ng+1, :]

#         f_p1 = self.flux(u_p1)
#         f_m1 = self.flux(u_m1)

#         rhs = - (f_p1 - f_m1) / (2 * self.dx) + (u_p1 - 2 * u_center + u_m1) / (2 * self.dx)
#         return rhs


#     def single_step(self, u: torch.Tensor, parameters=None, dts=None, mu=None, sigma=None) -> torch.Tensor:
#         """
#         Lax-Friedrichs 单步推进
#         u: [batch, Nx, 3] → 自动 pad → [batch, Nx+2*Ng, 3]
#         返回: [batch, Nx, 3]
#         """
#         Ng = self.Ng
#         u = F.pad(u, pad=(0, 0, Ng, Ng), mode="replicate")  # pad 维度顺序为 (last_dim...)，即 (3方向, x方向)

#         self.update_boundaries(u)

#         u_center = u[:, Ng:-Ng, :]           # [B, Nx, 3]
#         u_m1 = u[:, Ng-1:-Ng-1, :]           # [B, Nx, 3]
#         u_p1 = u[:, Ng+1:-Ng+1, :]           # [B, Nx, 3]

#         f_p1 = self.flux(u_p1)
#         f_m1 = self.flux(u_m1)

#         u_new = 0.5 * (u_p1 + u_m1) - (dts / (2 * self.dx)) * (f_p1 - f_m1)

#         return u_new  # 直接返回 physical domain

#     def update_boundaries(self, u: torch.Tensor):
#         """
#         更新边界值
#         """
#         Ng = self.Ng
#         Nx = u.shape[1] - 2 * Ng  # 假设 u 的形状为 [batch, Nx+2*Ng, 3]

#         # 左边界
#         u[:, 0:Ng, :] = u[:, Ng, :].unsqueeze(1).repeat(1, Ng, 1)

#         # 右边界
#         u[:, Nx+Ng:Nx+2*Ng, :] = u[:, Nx+Ng-1, :].unsqueeze(1).repeat(1, Ng, 1)

#     def Cons2Pri(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         v = mom / rho
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal
#         return torch.stack([rho, v, p], dim=-1)
    
#     def Pri2Cons(self, u):
#         rho, v, p = u[..., 0], u[..., 1], u[..., 2]
#         mom = rho * v
#         internal = p / (self.gamma - 1)
#         kinetic = 0.5 * v**2 * rho
#         E = internal + kinetic
#         return torch.stack([rho, mom, E], dim=-1)

class A(N0):
    def __init__(self, mu=0.0, sigma=1.0, gamma=1.4, dx=1/128, dt=0.025, Ng=2):
        super(A, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.dx = dx
        self.dt = dt
        # Ng 至少需要 2（因为我们用到 i-1 和 i+1）
        self.Ng = max(int(Ng), 2)

    def flux(self, u):
        rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
        rho = torch.clamp(rho, min=1e-6)
        v = mom / rho
        kinetic = 0.5 * v**2 * rho
        internal = torch.clamp(E - kinetic, min=1e-8)
        p = (self.gamma - 1) * internal

        f1 = mom
        f2 = mom * v + p
        f3 = v * (E + p)
        return torch.stack([f1, f2, f3], dim=-1)

    def compute_alpha(self, u):
        """ 局部最大特征速度 |v| + c，用于 LF 人工黏性 """
        rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
        rho = torch.clamp(rho, min=1e-6)
        v = mom / rho
        kinetic = 0.5 * v**2 * rho
        internal = torch.clamp(E - kinetic, min=1e-8)
        p = (self.gamma - 1) * internal
        c = torch.sqrt(self.gamma * p / rho + 1e-12)
        return torch.abs(v) + c  # [B, N(?),]

    def minmod(self, a, b):
        cond = (a*b > 0)
        return torch.where(cond, torch.sign(a) * torch.min(torch.abs(a), torch.abs(b)), torch.zeros_like(a))

    def compute_rhs(self, u: torch.Tensor) -> torch.Tensor:
        B, Nx, _ = u.shape
        Ng = self.Ng
        dx = self.dx

        # pad with ghost cells
        u_pad = F.pad(u, (0, 0, Ng, Ng), mode="replicate")
        self.update_boundaries(u_pad)

        # slopes (limited)
        du_forward = u_pad[:, 2:] - u_pad[:, 1:-1]   # u_{i+1}-u_i
        du_backward = u_pad[:, 1:-1] - u_pad[:, :-2] # u_i - u_{i-1}
        slope = self.minmod(du_backward, du_forward) # [B, Nx+2Ng-2, 3]

        # reconstruct left/right states at interfaces
        uL = u_pad[:, 1:-2] + 0.5 * slope[:, :-1]   # left state of i+1/2
        uR = u_pad[:, 2:-1] - 0.5 * slope[:, 1:]    # right state of i+1/2

        # flux at interfaces
        fL = self.flux(uL)
        fR = self.flux(uR)
        amax = torch.maximum(self.compute_alpha(uL), self.compute_alpha(uR))
        flux_face = 0.5 * (fL + fR) - 0.5 * amax[..., None] * (uR - uL)

        # conservative update
        rhs = -(flux_face[:, 1:] - flux_face[:, :-1]) / dx
        return rhs

    def single_step(self, u: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        """
        u:   [B, Nx, 3]
        dts: [B] 或 [B,1,1]
        返回: [B, Nx, 3]
        """
        if dts.dim() == 1:
            dts_exp = dts.view(-1, 1, 1)
        elif dts.dim() == 3 and dts.shape[1:] == (1, 1):
            dts_exp = dts
        else:
            raise ValueError(f"dts 形状不对: {dts.shape}")

        rhs1 = self.compute_rhs(u)
        u1 = u + dts_exp * rhs1

        rhs2 = self.compute_rhs(u1)
        u_new = 0.5 * (u + u1 + dts_exp * rhs2)
        return u_new

    # --------------------- 边界条件（ghost points 更新） ---------------------
    def update_boundaries(self, u_pad: torch.Tensor):
        """
        简单复制型边界（可按需替换为 periodic/壁面/入流出流）
        u_pad: [B, Nx+2Ng, 3]
        """
        Ng = self.Ng
        Nx = u_pad.shape[1] - 2 * Ng  # 物理点数

        # 左侧 Ng 个 ghost points 等于第一个物理点
        u_pad[:, 0:Ng, :] = u_pad[:, Ng:Ng+1, :].expand(-1, Ng, -1)
        # 右侧 Ng 个 ghost points 等于最后一个物理点
        u_pad[:, Nx+Ng:, :] = u_pad[:, Nx+Ng-1:Nx+Ng, :].expand(-1, Ng, -1)

    # --------------------- Cons/Prim 变换（与原版一致） ---------------------
    def Cons2Pri(self, u):
        rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
        rho = torch.clamp(rho, min=1e-6)
        v = mom / rho
        kinetic = 0.5 * v**2 * rho
        internal = torch.clamp(E - kinetic, min=1e-8)
        p = (self.gamma - 1) * internal
        return torch.stack([rho, v, p], dim=-1)

    def Pri2Cons(self, u):
        rho, v, p = u[..., 0], u[..., 1], u[..., 2]
        mom = rho * v
        internal = p / (self.gamma - 1)
        kinetic = 0.5 * v**2 * rho
        E = internal + kinetic
        return torch.stack([rho, mom, E], dim=-1)


# class A(N0):
#     def __init__(self, mu=0.0, sigma=1.0, gamma=1.4, dx=1/128, dt=0.025, Ng=3):
#         super(A, self).__init__()
#         self.mu = mu
#         self.sigma = sigma
#         self.gamma = gamma
#         self.dx = dx
#         self.dt = dt
#         self.Ng = Ng  # 边界层的厚度

#     def flux(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         rho = torch.clamp(rho, min=1e-6)
#         v = mom / rho
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal

#         f1 = mom
#         f2 = mom * v + p
#         f3 = v * (E + p)

#         return torch.stack([f1, f2, f3], dim=-1)

#     def compute_alpha(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         v = mom / rho
#         rho = torch.clamp(rho, min=1e-6)
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal
#         c = torch.sqrt(self.gamma * p / rho + 1e-8)
#         return torch.abs(v) + c

#     def rusanov_flux(self, u_L, u_R):
#         """
#         Rusanov 数值通量 (Local Lax-Friedrichs flux)。
#         F(u_L, u_R) = 0.5 * (f(u_L) + f(u_R)) - 0.5 * alpha_max * (u_R - u_L)
#         其中 alpha_max 是界面处的最大特征速度。
#         """
#         f_L = self.flux(u_L)
#         f_R = self.flux(u_R)

#         # 计算左右状态的 alpha 并取最大值
#         alpha_L = self.compute_alpha(u_L)
#         alpha_R = self.compute_alpha(u_R)
#         # 确保 alpha_max 在所有组件上都应用
#         alpha_max = torch.max(alpha_L, alpha_R).unsqueeze(-1)

#         return 0.5 * (f_L + f_R) - 0.5 * alpha_max * (u_R - u_L)

#     def minmod_limiter(self, a, b):
#         """
#         Minmod 限制器函数：
#         minmod(a, b) = s * min(|a|, |b|) 如果 sgn(a) == sgn(b)
#                      = 0                  否则
#         这里 s 是 a 和 b 的符号。
#         """
#         # 创建一个掩码，指示 a 和 b 是否同号
#         same_sign_mask = torch.sign(a) == torch.sign(b)

#         # 计算 min(|a|, |b|)
#         min_abs = torch.min(torch.abs(a), torch.abs(b))

#         # 结合符号和同号掩码
#         result = torch.where(same_sign_mask, torch.sign(a) * min_abs, torch.zeros_like(a))
#         return result

#     # Note: These reconstruction functions will now operate on slices, not single cells
#     # They are still called by compute_rhs, but compute_rhs passes slices.
#     def reconstruct_left_state(self, u_prev, u_curr, u_next):
#         """
#         MUSCL 重构界面左侧状态 (u_i+1/2^L)
#         u_prev: u_{i-1} slice
#         u_curr: u_i slice
#         u_next: u_{i+1} slice
#         返回 u_i+1/2^L (slice)
#         """
#         delta_minus = u_curr - u_prev
#         delta_plus = u_next - u_curr
#         phi = self.minmod_limiter(delta_minus, delta_plus)
#         return u_curr + 0.5 * phi

#     def reconstruct_right_state(self, u_prev, u_curr, u_next):
#         """
#         MUSCL 重构界面右侧状态 (u_i-1/2^R)
#         u_prev: u_{i-1} slice
#         u_curr: u_i slice
#         u_next: u_{i+1} slice
#         返回 u_i-1/2^R (slice)
#         """
#         delta_minus = u_curr - u_prev
#         delta_plus = u_next - u_curr
#         phi = self.minmod_limiter(delta_minus, delta_plus)
#         return u_curr - 0.5 * phi

#     def compute_rhs(self, u: torch.Tensor) -> torch.Tensor:
#         """
#         TVD MUSCL 空间离散形式 du/dt = RHS(u)
#         u: [B, Nx + 2*Ng, 3] (包含边界层)
#         return: [B, Nx, 3] — 只返回物理域
#         """
#         Ng = self.Ng
#         # Physical domain is from Ng to -Ng (exclusive)
#         # This will be [B, Nx, 3]
        
#         # Extract slices needed for reconstruction
#         # For u_i-2 to u_i+2, we need slices of length Nx
#         # Example: u_i corresponds to u[:, Ng:-Ng, :]
        
#         # u_i-2 (for cell i from Ng to Nx_total-Ng-1)
#         # The slice needs to correspond to the physical domain.
#         # If the physical domain is from index Ng to Nx_total - Ng - 1
#         # Then for u_i-2, the indices would be Ng-2 to Nx_total - Ng - 2
#         u_i_minus_2 = u[:, Ng-2 : -Ng-2, :] # [B, Nx, 3]

#         # u_i-1
#         u_i_minus_1 = u[:, Ng-1 : -Ng-1, :] # [B, Nx, 3]

#         # u_i (center cell values for physical domain)
#         u_i = u[:, Ng : -Ng, :] # [B, Nx, 3]

#         # u_i+1
#         u_i_plus_1 = u[:, Ng+1 : -Ng+1, :] # [B, Nx, 3]

#         # u_i+2
#         u_i_plus_2 = u[:, Ng+2 : -Ng+2, :] # [B, Nx, 3]

#         # Compute reconstructed states for F_{i+1/2} (cell i's right interface)
#         # u_L_at_ip1_half uses u_i-1, u_i, u_i+1
#         u_L_at_ip1_half = self.reconstruct_left_state(u_i_minus_1, u_i, u_i_plus_1)
#         # u_R_at_ip1_half uses u_i, u_i+1, u_i+2
#         u_R_at_ip1_half = self.reconstruct_right_state(u_i, u_i_plus_1, u_i_plus_2)

#         # Compute reconstructed states for F_{i-1/2} (cell i's left interface)
#         # u_L_at_im1_half uses u_i-2, u_i-1, u_i
#         u_L_at_im1_half = self.reconstruct_left_state(u_i_minus_2, u_i_minus_1, u_i)
#         # u_R_at_im1_half uses u_i-1, u_i, u_i+1
#         u_R_at_im1_half = self.reconstruct_right_state(u_i_minus_1, u_i, u_i_plus_1)

#         # Compute numerical fluxes for all interfaces at once
#         F_ip1_half = self.rusanov_flux(u_L_at_ip1_half, u_R_at_ip1_half)
#         F_im1_half = self.rusanov_flux(u_L_at_im1_half, u_R_at_im1_half)

#         # Calculate RHS for the entire physical domain
#         rhs = - (F_ip1_half - F_im1_half) / self.dx
        
#         return rhs

#     def single_step(self, u: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
#         Ng = self.Ng
#         u_padded = F.pad(u, pad=(0, 0, Ng, Ng), mode="replicate")
#         self.update_boundaries(u_padded)

#         if dts.dim() == 1:
#             dts_expanded = dts.unsqueeze(-1).unsqueeze(-1)
#         elif dts.dim() == 3 and dts.shape[1:] == (1, 1):
#             dts_expanded = dts
#         else:
#             raise ValueError(f"dts 形状不对: {dts.shape}")

#         # 阶段 1
#         rhs1 = self.compute_rhs(u_padded)
#         u1 = u + dts_expanded * rhs1
#         u1_padded = F.pad(u1, pad=(0, 0, Ng, Ng), mode="replicate")
#         self.update_boundaries(u1_padded)

#         # 阶段 2
#         rhs2 = self.compute_rhs(u1_padded)
#         u_new = 0.5 * u + 0.5 * (u1 + dts_expanded * rhs2)

#         return u_new

#     def update_boundaries(self, u: torch.Tensor):
#         """
#         更新边界值 (Simple replication for ghost cells)
#         """
#         Ng = self.Ng
#         Nx = u.shape[1] - 2 * Ng # 假设 u 的形状为 [batch, Nx+2*Ng, 3]

#         # 左边界 (ghost cells 镜像第一个物理单元)
#         u[:, 0:Ng, :] = u[:, Ng, :].unsqueeze(1).repeat(1, Ng, 1)

#         # 右边界 (ghost cells 镜像最后一个物理单元)
#         u[:, Nx+Ng:Nx+2*Ng, :] = u[:, Nx+Ng-1, :].unsqueeze(1).repeat(1, Ng, 1)

#     def Cons2Pri(self, u):
#         rho, mom, E = u[..., 0], u[..., 1], u[..., 2]
#         v = mom / rho
#         kinetic = 0.5 * v**2 * rho
#         internal = E - kinetic
#         internal = torch.clamp(internal, min=1e-8)
#         p = (self.gamma - 1) * internal
#         return torch.stack([rho, v, p], dim=-1)

#     def Pri2Cons(self, u):
#         rho, v, p = u[..., 0], u[..., 1], u[..., 2]
#         mom = rho * v
#         internal = p / (self.gamma - 1)
#         kinetic = 0.5 * v**2 * rho
#         E = internal + kinetic
#         return torch.stack([rho, mom, E], dim=-1)

class LocallyConnected1D(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.output_size = input_size  # same padding assumed
        self.weight = nn.Parameter(torch.randn(
            self.output_size, out_channels, in_channels, kernel_size
        ))  # [N_out, C_out, C_in, K]
        if bias:
            self.bias = nn.Parameter(torch.randn(self.output_size, out_channels))
        else:
            self.bias = None

        self.pad = kernel_size // 2

    def forward(self, x):
        """
        x: [B, C_in, N]
        returns: [B, C_out, N]
        """
        B, C_in, N = x.shape
        x_padded = F.pad(x, (self.pad, self.pad), mode='replicate')  # [B, C_in, N+2p]

        out = []
        for i in range(self.output_size):
            x_slice = x_padded[:, :, i:i + self.kernel_size]  # [B, C_in, K]
            w = self.weight[i]  # [C_out, C_in, K]
            # einsum over input: [B, C_in, K] and weight: [C_out, C_in, K]
            out_i = torch.einsum('bck,ock->bo', x_slice, w)
            if self.bias is not None:
                out_i += self.bias[i]
            out.append(out_i)  # [B, C_out]

        out = torch.stack(out, dim=-1)  # [B, C_out, N]
        return out


class CNN1D_FiLM(nn.Module):
    def __init__(self, width=64, layers=2, input_dim=3, output_dim=3):
        super().__init__()
        self.width = width
        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_size = 128  # 网格大小固定

        # 输入映射
        self.fc0 = nn.Linear(input_dim, width)

        # 单一局部卷积核 (kernel size = 5)
        self.kernel_size = 5
        self.padding = self.kernel_size // 2
        self.convs = nn.ModuleList([
            LocallyConnected1D(width, width, input_size=self.input_size, kernel_size=self.kernel_size)
            for _ in range(layers)
        ])

        # FiLM 参数生成器
        self.instancenorm = nn.InstanceNorm1d(width, affine=False)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2 * width * layers)
        )

        # 非共享通道混合器
        self.channel_mixer = nn.ModuleList([
            nn.Sequential(
                LocallyConnected1D(width, width, input_size=self.input_size, kernel_size=1),
                nn.GELU(),
                LocallyConnected1D(width, width, input_size=self.input_size, kernel_size=1)
            ) for _ in range(layers)
        ])

        # 输出映射
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def apply_film(self, x, gamma, beta):
        # x: [B, C, N], gamma/beta: [B, C, 1]
        return gamma * x + beta

    def forward(self, x, dt):
        """
        x:  [B, N, input_dim]   — 输入为守恒量 [ρ, ρu, E]
        dt: [B] or [B, 1]
        return: [B, N, output_dim]
        """
        B, N, _ = x.shape
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)

        # FiLM 参数生成
        film_params = self.film_mlp(dt)  # [B, 2*width*layers]
        film_params = film_params.view(B, self.width, 2, self.layers)
        gammas = film_params[:, :, 0, :]  # [B, C, L]
        betas  = film_params[:, :, 1, :]  # [B, C, L]

        # 初始映射并转置
        x = self.fc0(x)       # [B, N, C]
        x = x.permute(0, 2, 1)  # [B, C, N]

        for i in range(self.layers):
            x = self.instancenorm(x)
            gamma = gammas[:, :, i].unsqueeze(-1)  # [B, C, 1]
            beta = betas[:, :, i].unsqueeze(-1)
            x = self.apply_film(x, gamma, beta)

            # 卷积 + 通道混合 + 残差
            x_conv = self.convs[i](F.pad(x, (self.padding, self.padding), mode='replicate'))
            x_mixed = self.channel_mixer[i](x)
            x = F.gelu(x_conv + x_mixed)

        # 映射回输出
        x = x.permute(0, 2, 1)  # [B, N, C]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP1D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, layers=8, output_dim=3):
        super().__init__()
        self.input_dim = input_dim + 1
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        # 输入层：线性 + 阶跃激活（Hardtanh）
        self.initial_proj = nn.Linear(self.input_dim, hidden_dim)
        # self.input_activation = nn.Hardtanh(0.0, 1.0)  # 输入先clip近似阶跃
        self.input_activation = nn.Tanh()  # 输入先clip近似阶跃

        # dt → FiLM: 每层生成 gamma, beta
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim)
            ) for _ in range(layers)
        ])

        # InstanceNorm（不含 affine）
        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(hidden_dim, affine=False) for _ in range(layers)
        ])

        # 每层的 MLP 块（Linear → GELU → Linear）
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(layers)
        ])

        # 局部感知卷积
        # self.local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.local_conv = LocallyConnected1D(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            input_size=128,           # 例如 N=128
            kernel_size=5,
            bias=True
        )


        # 输出层 + 阶跃激活
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Hardtanh(0.0, 1.0)  # 输出强制 clip 到 [0, 1]

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.local_conv.weight, mode='fan_in', nonlinearity='relu')
        if self.local_conv.bias is not None:
            nn.init.zeros_(self.local_conv.bias)

        for layer in self.mlps:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(module.bias)

        for film in self.film_layers:
            for module in film:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        nn.init.kaiming_normal_(self.initial_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.initial_proj.bias)
        nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, dt):
        """
        x:  [B, N, 3]        -- 输入守恒量
        dt: [B] or [B, 1]    -- 时间步
        """
        B, N, _ = x.shape
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)  # [B, 1]

        x_pos = torch.linspace(0, 1, N, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        x = torch.cat([x, x_pos], dim=-1)  # [B, N, 3+1] = [B, N, 4]

        # 输入层 + 激活
        x = self.initial_proj(x)       # [B, N, hidden_dim]
        x = self.input_activation(x)   # clip to [0, 1]

        # 多层 ResNet 块
        for i in range(self.layers):
            x_perm = x.permute(0, 2, 1)                          # [B, hidden_dim, N]
            x_norm = self.norms[i](x_perm).permute(0, 2, 1)     # [B, N, hidden_dim]

            gamma_beta = self.film_layers[i](dt)                # [B, 2*hidden_dim]
            gamma, beta = gamma_beta.chunk(2, dim=-1)           # [B, hidden_dim] each
            gamma = gamma.unsqueeze(1)                          # [B, 1, hidden_dim]
            beta = beta.unsqueeze(1)                            # [B, 1, hidden_dim]

            x_film = gamma * x_norm + beta                      # FiLM 调制
            x = x + self.mlps[i](x_film)                        # ResNet 加法

        # 输出层 + clip（模拟阶跃）
        x = self.output_proj(x)             # [B, N, output_dim]
        return x

class FiLM_MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim * 2)  # output gamma and beta
        )

    def forward(self, x):
        out = self.film(x)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)  # shape: [B, 1, hid_dim]


class pit_sod(pit_fixed):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):
        super(pit_sod, self).__init__(space_dim,  
                                  in_dim, 
                                  out_dim, 
                                  hid_dim,
                                  n_head, 
                                  n_blocks,
                                  mesh_ltt,
                                  en_loc, 
                                  de_loc)
        self.film_mlp = FiLM_MLP(1, self.hid_dim, self.hid_dim)
        self.instancenorm = nn.InstanceNorm1d(self.hid_dim, affine=False)


    def forward(self, mesh_in, func_in, mesh_out, dts):
        '''
        func_in: (B, L, in_dim)
        mesh_in: (L, space_dim)
        mesh_out: (L_out, space_dim)
        dts: (B,)
        '''
        # 拼接 mesh + 输入函数
        func_in  = torch.cat((torch.tile(mesh_in.unsqueeze(0), [func_in.shape[0], 1, 1]), func_in), -1)
        func_ltt = self.encoder(mesh_in, func_in, self.mesh_ltt)  # (B, L_latent, hid_dim)

        # InstanceNorm -> FiLM调制
        func_ltt = func_ltt.permute(0, 2, 1)  # [B, C, L]
        func_ltt = self.instancenorm(func_ltt)
        func_ltt = func_ltt.permute(0, 2, 1)  # [B, L, C]

        gamma, beta = self.film_mlp(dts.unsqueeze(-1))  # [B, 1, hid_dim]
        func_ltt = gamma * func_ltt + beta              # [B, L, hid_dim]

        # processor & decoder
        func_ltt = self.processor(func_ltt, self.mesh_ltt)
        func_out = self.decoder(self.mesh_ltt, func_ltt, mesh_out)
        return func_out

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.fc(x)
        out = self.ln(out)
        out = self.act(out)
        return out + x  # 残差连接

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim) # Add LayerNorm here

    def forward(self, x):
        # x: [B, N_faces, dim]
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # [B, N, D]
        attn = (q @ k.transpose(-2, -1)) / (D ** 0.5)  # [B, N, N]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, N, D]
        out = self.proj(out)
        out = self.norm(out) # Apply LayerNorm after projection
        return out

class FluxNet(nn.Module):
    def __init__(self, hidden_dim=128, layers=4):
        super().__init__()
        self.fc_in = nn.Linear(15, hidden_dim)  # ← 6 → 9 输入
        self.attn = SelfAttention(hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(layers - 2)]
        )
        self.fc_out = nn.Linear(hidden_dim, 3)

    def forward(self, input_net):  # input_net: [B, N, 9]
        out = self.fc_in(input_net)
        out = self.attn(out) + out  # residual attention
        out = self.blocks(out)
        out = self.fc_out(out)
        return out



class Euler(nn.Module):
    def __init__(self, N0_SCHEME: N0, modes=32, width=128, layers=4):
        super(Euler, self).__init__()
        self.N0_SCHEME = N0_SCHEME
        self.modes = modes
        self.width = width
        self.mu    = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(1.0, dtype=torch.float64, device=device)
        self.Ng    = 3
        self.dx    = 1.0/128.0

        # self.fno   = MLP1D(input_dim=6, hidden_dim=self.width, layers=layers, output_dim=3)
        self.flux_net = FluxNet(hidden_dim=self.width, layers=layers)

    
    def modify_input(self, x):
        return (x - self.mu) / self.sigma
    
    def modify_output(self, x):
        return self.sigma * x + self.mu
    
    def get_mu_and_sigma(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, device=device)

    def apply_boundary(self, u):
        B, N, C = u.shape
        Ng = self.Ng
        u_ext = torch.zeros(B, N + 2*Ng, C, dtype=u.dtype, device=u.device)

        # 物理区域赋值
        u_ext[:, Ng:Ng+N, :] = u

        # 边界条件，复制边界值
        # 左边界
        u_ext[:, :Ng, :] = u[:, :1, :].expand(B, Ng, C)
        # 右边界
        u_ext[:, Ng+N:, :] = u[:, -1:, :].expand(B, Ng, C)

        return u_ext

    def compute_flux_net_only(self, u_ext):
        uLL = u_ext[:, :-3, :]
        uL  = u_ext[:, 1:-2, :]  # [B, N+2*Ng-1, 3]
        uR   = u_ext[:, 2:-1, :]
        uRR  = u_ext[:, 3:, :]   # [B, N+2*Ng-1, 3]
        grad = (uR - uL).abs() # [B, N+2*Ng-1, 3]

        input_net = torch.cat([uLL, uL, uR, uRR, grad], dim=-1)  # [B, N+2*Ng-1, 9]

        delta_flux = self.flux_net(input_net) # [B, N+2*Ng-1, 3]
        return delta_flux


    def compute_net_rhs(self, u):
        u_ext = self.apply_boundary(u)  # 加边界
        delta_flux = self.compute_flux_net_only(u_ext)  # [B, N+2*Ng-1, 3]

        # 取中心网格面通量 (对应物理单元+1)
        flux_center = delta_flux[:, 1:1 + u.shape[1] + 1, :]  # [B, N+1, 3]

        # 差分计算残差通量导数
        rhs = - (flux_center[:, 1:] - flux_center[:, :-1]) / self.dx  # [B, N, 3]

        return rhs

    def net_ssprk3(self, u, dt):
        # Stage 1
        L1 = self.compute_net_rhs(u)
        u1 = u + dt * L1
        # Stage 2
        L2 = self.compute_net_rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * L2)
        # Stage 3
        L3 = self.compute_net_rhs(u2)
        u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L3)
        return u3

    def block_tau(self, u: torch.Tensor, dts):
        u = self.N0_SCHEME.single_step(u, dts=dts[:,0,0]/2)
        u = self.net_ssprk3(u, dts)
        u = self.N0_SCHEME.single_step(u, dts=dts[:,0,0]/2)
        return u

    def forward(self, x):
        u, dts = x[:, :, :-1], x[:, :, -1:]
        dts = dts[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        k1 = self.block_tau(u, dts)
        return k1

    def predict(self, x):
        return self.forward(x)

class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None):
        raw_inputs  = np.load(input_file)[:100000, :]
        raw_outputs = np.load(output_file)[:100000, :]

        self.inputs = torch.from_numpy(raw_inputs).to(device)
        self.outputs = torch.from_numpy(raw_outputs).to(device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
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

def spatial_derivative_l1(preds, targets):
    """
    preds, targets: [B, N, C] tensors
    Return: scalar loss ||∂x preds - ∂x targets||_1
    """
    dx = 1.0 / preds.shape[1]  # 假设空间区间为 [0, 1]
    dudx_pred = preds[:, 1:, :] - preds[:, :-1, :]
    dudx_target = targets[:, 1:, :] - targets[:, :-1, :]
    return torch.mean(torch.abs(dudx_pred - dudx_target)) / dx

def spatial_derivative_l1(preds, targets):
    """
    preds, targets: [B, N, C] tensors
    Return: scalar loss ||∂x preds - ∂x targets||_1
    """
    dx = 1.0 / preds.shape[1]  # 假设空间区间为 [0, 1]
    dudx_pred = preds[:, 1:, :] - preds[:, :-1, :]
    dudx_target = targets[:, 1:, :] - targets[:, :-1, :]
    return torch.mean(torch.abs(dudx_pred - dudx_target)) / dx

def detect_oscillations(preds, window_size=5, threshold_factor=1.5, use_std=True):
    B, N, C = preds.shape
    if window_size < 3:
        window_size = 3 # Ensure minimum size for second derivative

    # 计算离散二阶差分
    second_diff = preds[:, 2:, :] - 2 * preds[:, 1:-1, :] + preds[:, :-2, :]
    abs_second_diff = torch.abs(second_diff) # [B, N-2, C]

    # 将二阶差分结果pad到原始N维度，边界填充0
    padded_abs_second_diff = F.pad(abs_second_diff, (0,0,1,1), 'constant', 0) # [B, N, C]

    flat_osc_metric = padded_abs_second_diff[padded_abs_second_diff != 0]

    if flat_osc_metric.numel() == 0:
        return torch.zeros_like(preds, dtype=torch.float32)

    if use_std:
        mean_val = torch.mean(flat_osc_metric)
        std_val = torch.std(flat_osc_metric)
        threshold = mean_val + threshold_factor * std_val
    else:
        threshold = threshold_factor * torch.max(flat_osc_metric)
        if threshold == 0:
            threshold = 1e-6

    oscillation_mask = (padded_abs_second_diff > threshold).float()
    return oscillation_mask

def conservative_to_primitive(U, gamma=1.4):
    """
    U: Tensor of shape [B, N, 3], where last dim is [rho, rho*u, rho*E]
    Returns: Tensor of shape [B, N, 3], [rho, u, p]
    """
    rho = U[..., 0]     # [B, N]
    rho_u = U[..., 1]
    rho_E = U[..., 2]

    u = rho_u / rho
    E = rho_E / rho
    kinetic = 0.5 * u ** 2
    p = (gamma - 1.0) * rho * (E - kinetic)

    W = torch.stack([rho, u, p], dim=-1)  # [B, N, 3]
    return W


# === 完整的损失计算 ===
def calculate_total_loss(preds, targets,
                         lambda_data=1.0,
                         lambda_tv_osc=0.1,
                         lambda_grad_osc=0.05,
                         osc_detection_params=None):
    if osc_detection_params is None:
        osc_detection_params = {'window_size': 5, 'threshold_factor': 1.5, 'use_std': True}

    preds = conservative_to_primitive(preds, gamma=1.4)
    targets = conservative_to_primitive(targets, gamma=1.4)

    # 1. 数据保真损失 (L2 Loss)
    loss_data = F.mse_loss(preds, targets)

    # 2. 震荡区域检测 (对预测值进行检测，并分离梯度)
    oscillation_mask = detect_oscillations(
        preds.detach(), 
        window_size=osc_detection_params['window_size'],
        threshold_factor=osc_detection_params['threshold_factor'],
        use_std=osc_detection_params['use_std']
    )

    total_loss = lambda_data * loss_data

    # 3. 选择性惩罚项 (只在检测到的震荡区域应用)
    if lambda_tv_osc > 0 or lambda_grad_osc > 0:
        # 计算预测值的空间一阶差分
        diff_preds = preds[:, 1:, :] - preds[:, :-1, :] # [B, N-1, C]

        # 匹配 oscillation_mask 到 diff_preds 的维度
        # 简单取前后点的mask均值作为该“边”的mask
        oscillation_mask_diff = (oscillation_mask[:, :-1, :] + oscillation_mask[:, 1:, :]) / 2.0
        # 确保mask值在0到1之间，如果是硬二值mask，则只是 0.0, 0.5, 1.0

        if lambda_tv_osc > 0:
            # 震荡区域的TV损失 (L1 梯度)
            loss_tv_selective = torch.mean(torch.abs(diff_preds) * oscillation_mask_diff)
            total_loss += lambda_tv_osc * loss_tv_selective

        if lambda_grad_osc > 0:
            # 震荡区域的梯度L2损失 (L2 梯度)
            loss_grad_selective = torch.mean((diff_preds ** 2) * oscillation_mask_diff)
            total_loss += lambda_grad_osc * loss_grad_selective

    return total_loss

def total_variation_loss(u_phys):
    B, N, C = u_phys.shape
    return (u_phys[:, 1:, :] - u_phys[:, :-1, :]).abs().sum() / (B * (N - 1) * C)

if __name__ == "__main__":
    # # 首先加载 train dataset，并提取 mu, sigma
    train_dataset = ODEPairDataset("../dataset/train_input_best.npy", "../dataset/train_output_best.npy")
    val_dataset   = ODEPairDataset("../dataset/val_input_best.npy", "../dataset/val_output_best.npy")

    # 后面继续创建 dataloader 和训练模型...

    batch_size   = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    model = Euler(N0_SCHEME=A(), modes=32, width=128, layers=8).to(device)
    # model = torch.compile(model, mode="max-autotune")

    # 损失和优化器
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 1000
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-5)

    best_val_loss = float('inf')

    train_loss_lists = []
    val_loss_lists = []

    train_loss_file = open("train_loss_test.txt", "w")
    val_loss_file = open("val_loss_test.txt", "w")

    lambda_reg = 1e-3
    lambda_tvd = 1e-1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            optimizer.zero_grad()
            preds = model(inputs)
            preds = conservative_to_primitive(preds, gamma=1.4)
            targets = conservative_to_primitive(targets, gamma=1.4)
            mse_loss = criterion(preds, targets)
            reg_loss = spatial_derivative_l1(preds, targets)
            tvd_loss = total_variation_loss(preds)
            loss = mse_loss + lambda_reg * reg_loss + lambda_tvd * tvd_loss
            # loss = calculate_total_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_lists.append(train_loss)
        train_loss_file.write(f"{train_loss:.5e}\n")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                preds = conservative_to_primitive(preds, gamma=1.4)
                targets = conservative_to_primitive(targets, gamma=1.4)
                val_loss += criterion(preds, targets).item() * inputs.size(0) + \
                    lambda_reg * spatial_derivative_l1(preds, targets).item() * inputs.size(0) + \
                    lambda_tvd * total_variation_loss(preds).item() * inputs.size(0)
                # val_loss += calculate_total_loss(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_test_best.pth")
            print("Saved best model!")
    

    plot_losses(train_loss_lists, val_loss_lists)
