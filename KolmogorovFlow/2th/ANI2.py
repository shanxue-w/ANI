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
import torch.fft as fft
import torch.fft
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A(N0):
    def __init__(self, Nx=256, Ny=512, Lx=1.0, Ly=2.0, Re=1e4, Cs=0.18, device='cuda:1'):
        super().__init__()
        self.Nx = Nx  # Now represents the fine-grid Nx
        self.Ny = Ny  # Now represents the fine-grid Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re  # Reynolds number
        self.device = device
        self.mu = 1.0 / Re  # Viscosity
        self.Cs = Cs  # Smagorinsky constant
        self.dt_step = 5e-4

        # Grid spacing for the original (fine) resolution
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.delta = math.sqrt(self.dx**2 + self.dy**2)
        self.kappa = 2
        self.delta_tilde = self.kappa * self.delta
        x       = torch.linspace(0, Lx, Nx+1, device=device)[:-1]
        y       = torch.linspace(0, Ly, Ny+1, device=device)[:-1]
        self.x, self.y = torch.meshgrid(x, y, indexing='ij')
        # external force -(16\pi)^3 sin(16 \pi y) 10^{-4}
        self.F = -8 * math.pi * torch.cos(8 * math.pi * self.y) * 1e-1
        # self.F = 0
        self.F_fft = fft.fft2(self.F)

        self._cached_dt_coeffs = None

        # Cache coefficients for the original (fine) grid
        self._cache_coefficients()
        self._cache_dealiasing_mask()
        self.gaussian_filter_kernel = self._create_gaussian_kernel()


    def _create_gaussian_kernel(self):
        # 计算波数的平方 K_sq = kx^2 + ky^2
        K_sq = self.Kx_fine**2 + self.Ky_fine**2
        
        # 计算高斯核 G_hat = exp(-(K^2 * delta_tilde^2) / 24)
        kernel = torch.exp(-K_sq * (self.delta_tilde**2) / 24.0)
        return kernel

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

    def _cache_dealiasing_mask(self):
        """
        2/3 de-aliasing mask (Orszag-style), built from *linear wavenumber indices*,
        similar to your CuPy implementation:
            kc = (2/3) * (n/2)
            mask = (KX^2 + KY^2) < 1.4 * kc^2
        For Nx != Ny we use an elliptical form.
        """
        Nx_f = self.Nx 
        Ny_f = self.Ny

        # "linear" wavenumber indices (integers like ..., -2, -1, 0, 1, 2, ...)
        # This matches cp.fft.fftfreq(n, 1/n) logic.
        kx_lin = torch.fft.fftfreq(Nx_f, d=1.0 / Nx_f, device=self.device)  # shape [Nx_f]
        ky_lin = torch.fft.fftfreq(Ny_f, d=1.0 / Ny_f, device=self.device)  # shape [Ny_f]
        KX, KY = torch.meshgrid(kx_lin, ky_lin, indexing="ij")              # [Nx_f, Ny_f]

        # 2/3 rule cutoff in each direction: kc = (2/3)*(n/2) = n/3
        kc_x = (2.0 / 3.0) * (Nx_f / 2.0)   # = Nx_f/3
        kc_y = (2.0 / 3.0) * (Ny_f / 2.0)   # = Ny_f/3

        # CuPy version used: (KX^2 + KY^2) < 1.4 * kc^2
        # For rectangular grids, use an ellipse: (KX/kc_x)^2 + (KY/kc_y)^2 < 1.4
        alpha = 1.4
        dealias_mask = (KX / kc_x) ** 2 + (KY / kc_y) ** 2 < alpha

        self.dealias_mask = dealias_mask  # bool [Nx_f, Ny_f]

    def _compute_derivatives_x(self, field_real, Kx):
        # Compute the derivative in x direction using the Fourier transform
        field_k = torch.fft.fft2(field_real)
        dfdx_k = 1j * Kx.unsqueeze(0) * field_k
        dfdx_real = torch.fft.ifft2(dfdx_k).real
        return dfdx_real

    def _compute_derivatives_y(self, field_real, Ky):
        # Compute the derivative in y direction using the Fourier transform
        field_k = torch.fft.fft2(field_real)
        dfdy_k = 1j * Ky.unsqueeze(0) * field_k
        dfdy_real = torch.fft.ifft2(dfdy_k).real
        return dfdy_real

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
    
    def _lap(self, w):
        lap_w_fft = self.laplacian_k_fine * torch.fft.fft2(w)
        return torch.fft.ifft2(lap_w_fft).real

    # def _s_mag(self, u, v, dwdx, dwdy):
    #     dudx, dudy = self._compute_derivatives_spectral(u, self.Kx_fine, self.Ky_fine)
    #     dvdx, dvdy = self._compute_derivatives_spectral(v, self.Kx_fine, self.Ky_fine)
    #     S11, S22   = dudx, dvdy
    #     S12        = 0.5 * (dudy + dvdx)
    #     S2         = 2*(S11**2 + S22**2) + 4*S12**2
    #     return torch.sqrt(dwdx**2 + dwdy**2), S11, S22, S12
    
    # def _eddy_viscosity(self, u, v, dwdx, dwdy):
    #     S, S11, S22, S12 = self._s_mag(u, v, dwdx, dwdy)
    #     return (self.Cs * self.delta)**3 * S, S11, S22, S12

    def _s_mag(self, u, v):
        dudx, dudy = self._compute_derivatives_spectral(u, self.Kx_fine, self.Ky_fine)
        dvdx, dvdy = self._compute_derivatives_spectral(v, self.Kx_fine, self.Ky_fine)
        S11, S22   = dudx, dvdy
        S12        = 0.5 * (dudy + dvdx)
        S2         = 2*(S11**2 + S22**2) + 4*S12**2
        return torch.sqrt(S2), S11, S22, S12

    def _test_filter(self, f):
        # 1. 将场 f 变换到谱空间
        f_hat = torch.fft.fft2(f)
        # 2. 在谱空间中应用预先计算好的高斯核
        f_hat_filtered = f_hat * self.gaussian_filter_kernel
        # 3. 将滤波后的结果逆变换回物理空间
        f_filtered = torch.fft.ifft2(f_hat_filtered)
        
        return f_filtered.real

    def _calculate_Cs_delta_squared(self, u, v, omega):
        # 1. 批量 FFT + 滤波
        f_stack = torch.stack([u, v, omega])
        f_hat_filtered = torch.fft.fft2(f_stack) * self.gaussian_filter_kernel.unsqueeze(0)
        u_t, v_t, omega_t = torch.fft.ifft2(f_hat_filtered).real

        # 2. 对流项
        dwdx, dwdy = self._compute_derivatives_spectral(omega, self.Kx_fine, self.Ky_fine)
        dwdx_t, dwdy_t = self._compute_derivatives_spectral(omega_t, self.Kx_fine, self.Ky_fine)
        J = u * dwdx + v * dwdy
        J_t = u_t * dwdx_t + v_t * dwdy_t
        J_tf = torch.fft.ifft2(torch.fft.fft2(J) * self.gaussian_filter_kernel).real

        # 3. S 和 lap(omega)
        S_mag = self._s_mag(u, v)[0]
        S_mag_t = self._s_mag(u_t, v_t)[0]
        lap_omega = self._lap(omega)
        lap_omega_t = torch.fft.ifft2(torch.fft.fft2(lap_omega) * self.gaussian_filter_kernel).real

        # 4. 最终 Cs_delta_sq
        H = J_t - J_tf
        M = self.kappa**2 * S_mag_t * lap_omega_t - torch.fft.ifft2(torch.fft.fft2(S_mag * lap_omega) * self.gaussian_filter_kernel).real
        Cs_delta_sq = torch.mean(H*M) / (torch.mean(M*M) + 1e-12)
        return Cs_delta_sq
    
    def _eddy_viscosity(self, u, v, omega):
        S, S11, S22, S12 = self._s_mag(u, v)
        return (self.Cs * self.delta)**2 * S, S11, S22, S12
    # def _eddy_viscosity(self, u, v, omega):
    #     """
    #     计算最终的涡粘性场 nu_t。
    #     """
    #     S_mag, S11, S22, S12 = self._s_mag(u, v)
    #     Cs_delta_squared_value = self._calculate_Cs_delta_squared(u, v, omega)
    #     nu_t = Cs_delta_squared_value * S_mag
    #     return nu_t, S11, S22, S12

    def rhs_fft(self, omega):
        u, v = self._vorticity_to_velocity_spectral(omega, self.Kx_fine, self.Ky_fine, self.denom_safe_fine)
        dwdx, dwdy = self._compute_derivatives_spectral(omega, self.Kx_fine, self.Ky_fine)
        nu_t, S11, S22, S12 = self._eddy_viscosity(u, v, omega)

        term_1 = - (u * dwdx + v * dwdy)
        term_1_fft = torch.fft.fft2(term_1)
        # term_2 is nu_t \Delta w -> we need term_2_fft
        # term_2_fft = nu_t * self.laplacian_k_fine * omega
        dwdx2 = self._compute_derivatives_x(dwdx, self.Kx_fine)
        dwdy2 = self._compute_derivatives_y(dwdy, self.Ky_fine)
        term_2 = nu_t * (dwdx2 + dwdy2)
        # qx = nu_t * dwdx
        # qy = nu_t * dwdy
        # # divergence of flux
        # dqxdx = self._compute_derivatives_x(qx, self.Kx_fine)
        # dqydy = self._compute_derivatives_y(qy, self.Ky_fine)
        # term_2 = dqxdx + dqydy
        term_2_fft = torch.fft.fft2(term_2)

        return (term_1_fft + term_2_fft + self.F_fft) * self.dealias_mask.unsqueeze(0)


    def _compute_etdrk4_coefficients(self, dt):
        """
        Computes the ETDRK4 coefficients based on the linear term (Laplacian)
        and the time step dt using a numerical integration method (M=64).
        """
        L = self.laplacian_k_fine / self.Re

        # The linear operator in Fourier space is L = ν * ∇^2
        z = dt * L.to(torch.complex128)
        
        # Prepare tensors for the coefficients with complex128 dtype
        # phi1 = torch.zeros_like(z, dtype=torch.complex128)
        # phi2 = torch.zeros_like(z, dtype=torch.complex128)
        # phi3 = torch.zeros_like(z, dtype=torch.complex128)

        z_nz = z.unsqueeze(-1)

        E = torch.exp(z_nz.squeeze(-1))
        E2 = torch.exp(0.5 * z_nz.squeeze(-1))            

        # Use M=64 for numerical integration
        M = 64

        r = torch.exp(1j * np.pi * (torch.arange(1, M+1, dtype=torch.float64) - 0.5) / M).to(self.device)
        LR = z_nz + r.unsqueeze(0)

        Q  = torch.mean( (torch.exp(LR / 2.0)-1.0) / LR, dim=-1)
        A_coeffs = torch.mean( (-4.0-LR+torch.exp(LR) * (4.0-3.0*LR+LR**2)) / LR**3, dim=-1)
        B_coeffs = torch.mean( (2.0+LR+torch.exp(LR)*(-2.0+LR)) / LR**3, dim=-1)
        C_coeffs = torch.mean( (-4-3*LR - LR**2 + torch.exp(LR)*(4-LR)) / LR**3, dim=-1)

        return E, E2, Q, A_coeffs, B_coeffs, C_coeffs

    def _etdrk4_step(self, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs a single step of the ETDRK4 time-stepping scheme for vorticity.
        """
        # Compute the ETDRK4 coefficients for this time step
        # E, E2, phi1, phi2, phi3, A_coeffs, B_coeffs, C_coeffs = self._compute_etdrk4_coefficients(dt)
        # E, E2, phi1, _, _, A_coeffs, B_coeffs, C_coeffs = self._compute_etdrk4_coefficients(dt)
        if math.isclose(dt, self.dt_step):
            if self._cached_dt_coeffs is None:
                self._cached_dt_coeffs = self._compute_etdrk4_coefficients(dt)
            E, E2, phi1, A_coeffs, B_coeffs, C_coeffs = self._cached_dt_coeffs
        else:
            E, E2, phi1, A_coeffs, B_coeffs, C_coeffs = self._compute_etdrk4_coefficients(dt)
        
        # === Stage 1 ===
        omega_real = omega.squeeze(1)
        omega_k_n = torch.fft.fft2(omega_real)
        N0_k = self.rhs_fft(omega_real)

        # === Stage 2 ===
        omega_a_k = E2.unsqueeze(0) * omega_k_n + dt * phi1.unsqueeze(0) * N0_k
        omega_a_real = torch.fft.ifft2(omega_a_k).real
        Na_k = self.rhs_fft(omega_a_real)

        # === Stage 3 ===
        omega_b_k = E2.unsqueeze(0) * omega_k_n + dt * phi1.unsqueeze(0) * Na_k
        omega_b_real = torch.fft.ifft2(omega_b_k).real
        Nb_k = self.rhs_fft(omega_b_real)

        # === Stage 4 ===
        omega_c_k = E2.unsqueeze(0) * omega_a_k + dt * phi1.unsqueeze(0) * (2*Nb_k - N0_k)
        omega_c_real = torch.fft.ifft2(omega_c_k).real
        Nc_k = self.rhs_fft(omega_c_real)

        # === Final Update Step ===
        omega_k_new = (
            E.unsqueeze(0) * omega_k_n +
            dt * (A_coeffs.unsqueeze(0) * N0_k +
                  2.0 * B_coeffs.unsqueeze(0) * (Na_k + Nb_k) +
                  C_coeffs.unsqueeze(0) * Nc_k)
        )
        omega_new_real = torch.fft.ifft2(omega_k_new).real
        return omega_new_real.unsqueeze(1)
    
    def single_step(self, u: torch.Tensor, dts: float = None, dt_max: float = 5e-4) -> torch.Tensor:
        remain_time = dts 
        while remain_time > 0:
            dt = min(dt_max, remain_time)
            u = self._etdrk4_step(u, dt)
            remain_time -= dt
        return u
    #     if u.ndim != 4 or u.shape[1] != 1:
    #         raise ValueError(f"Expected input shape [B, 1, Nx, Ny], got {u.shape}")
        
    #     # Time step size
    #     dt = dt_max if dts is None else dts
    #     mu = 1.0 / self.Re
    #     B, _, Nx, Ny = u.shape
        
    #     # Compute exponential factor: exp(μ Δ t * Δ)
    #     L = mu * self.laplacian_k_fine  # shape [Nx, Ny]
    #     exp_Ldt = torch.exp(L * dt).to(u.device)  # shape [Nx, Ny]

    #     # FFT
    #     u_real = u.squeeze(1)  # [B, Nx, Ny]
    #     u_k = torch.fft.fft2(u_real)  # [B, Nx, Ny]

    #     # ETD step: multiply in Fourier space
    #     u_k_next = u_k * exp_Ldt.unsqueeze(0)  # broadcast over batch

    #     # iFFT back
    #     u_next_real = torch.fft.ifft2(u_k_next).real  # [B, Nx, Ny]
    #     return u_next_real.unsqueeze(1)  # [B, 1, Nx, Ny]

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}


class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"
    ):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class Upsample(nn.Module):
    r"""Upsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2, padding_mode="circular",
            ),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    r"""Downsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2, padding_mode="circular",
            ),
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, n_spatial_dims, cond_dim=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode="circular",
        )  # depthwise conv
        self.norm = LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 2 * dim)
        )

        self.film_norm = nn.InstanceNorm2d(dim, affine=False)


    def forward(self, x, cond):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)
        x = self.pwconv1(x)

        x = self.act(x)
        x = self.pwconv2(x)
        # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])

        # FiLM
        gamma_beta = self.film_mlp(cond)  # (B, 2*dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # reshape gamma/beta to (B, C, 1, 1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)

        x = self.film_norm(x)
        x = gamma * x + beta

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(dim_in, n_spatial_dims)
                for _ in range(depth)
            ]
        )

    def forward(self, x, cond):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.resample(x)
        return x


class UNetConvNext(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        # super().__init__(n_spatial_dims, spatial_resolution)
        self.n_spatial_dims = n_spatial_dims
        # self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing
        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1, padding_mode="circular",
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1, padding_mode="circular",
        )
        for i in range(stages):
            encoder.append(
                Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                )
            )
            decoder.append(
                Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x, cond):
        x = self.in_proj(x)
        skips = []
        for i, enc in enumerate(self.encoder):
            skips.append(x)
            x = enc(x, cond)  # 直接传 x 和 cond
        x = self.neck(x, cond)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x, cond)
        x = self.out_proj(x)
        return x

class CNEXTUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_width=64, cond_dim=1, n_blocks=4):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_width = base_width
        self.norm = nn.InstanceNorm2d(base_width, affine=False)

        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 2 * base_width * n_blocks)
        )

        self.encoder_convs = nn.ModuleList()
        curr_in = in_channels
        for _ in range(n_blocks):
            self.encoder_convs.append(
                nn.Conv2d(curr_in, base_width, 3, padding=1, padding_mode='circular')
            )
            curr_in = base_width

        self.middle = nn.Sequential(*[self.make_res_block(base_width) for _ in range(3)])

        self.decoder = nn.Sequential(
            nn.Conv2d(base_width, base_width, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(base_width, base_width, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 1, padding_mode='circular')
        )

    def make_res_block(self, width):
        return nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode='circular')
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
            cond = cond.view(1).expand(B, 1)
        elif cond.dim() == 1:
            cond = cond.unsqueeze(-1)
            if cond.shape[0] == 1:
                cond = cond.expand(B, 1)
        params = self.film_mlp(cond).view(B, self.n_blocks, 2, self.base_width)
        gammas = params[:, :, 0, :].permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)
        betas = params[:, :, 1, :].permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)

        for i in range(self.n_blocks):
            x = self.encoder_convs[i](x)
            x = self.norm(x) 
            x = gammas[i] * x + betas[i] 
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

class VecFiLM2d(nn.Module):
    def __init__(self, cond_dim: int, width: int, hidden: int = 128):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * width),
        )
        # start from identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # self.norm = nn.InstanceNorm1d(width, affine=False)

    def forward(self, x, cond):
        B, C, _, _ = x.shape
        gb = self.mlp(cond)                    
        gamma, beta = gb.split(self.width, -1)
        gamma = gamma.view(B, C, 1, 1)        
        beta  = beta.view(B, C, 1, 1)      
        return (1.0 + gamma) * x + beta

class FNO2d(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int, d_in: int, cond_dim: int, film_hidden: int = 128, blocks=4, Nx_fixed=128, Ny_fixed=128, Lx=1.0, Ly=1.0, out_dim: int = 1):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.blocks = blocks
        self.fc0 = nn.Linear(d_in + 4, self.width)

        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(blocks)])
        self.ws    = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(blocks)])
        self.films = nn.ModuleList([VecFiLM2d(cond_dim, width, film_hidden) for _ in range(blocks)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

        g = self.get_grid(Nx_fixed, Ny_fixed, device="cpu", Lx=Lx, Ly=Ly)
        self.register_buffer("grid_fixed", g)

    def forward(self, x, cond):
        # grid = self.get_grid(x.shape, x.device)
        # grid = periodic_grid_feats_2d(x.shape[-3], x.shape[-2], x.device)
        # grid = grid.expand(x.shape[0], -1, -1, -1)
        grid = self.grid_fixed.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for n in range(self.blocks-1):
            x = self.convs[n](x) + self.ws[n](x)
            x = self.films[n](x, cond)
            x = F.gelu(x)

        x = self.convs[-1](x) + self.ws[-1](x)
        x = self.films[-1](x, cond)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    def get_grid(self, Nx=128, Ny=128, Lx=1.0, Ly=1.0, device="cpu"):
        x = torch.linspace(0.0, Lx, Nx + 1, device=device)[:-1]
        y = torch.linspace(0.0, Ly, Ny + 1, device=device)[:-1]

        X, Y = torch.meshgrid(x, y, indexing="ij")
        feats = torch.stack([
            torch.sin(2 * math.pi * X / Lx),
            torch.cos(2 * math.pi * X / Lx),
            torch.sin(2 * math.pi * Y / Ly),
            torch.cos(2 * math.pi * Y / Ly),
        ], dim=-1).unsqueeze(0)

        return feats


class NavierStokes(nn.Module):
    def __init__(self, N0_SCHEME: N0, modes1=16, modes2=16, width=64, dt=0.1, device=device):
        super(NavierStokes, self).__init__()
        self.N0_SCHEME = N0_SCHEME
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.dt    = dt / 2
        self.mu    = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(1.0, dtype=torch.float64, device=device)

        # self.fno = FNO2d(modes1=modes1, modes2=modes2, width=width, cond_dim=1, d_in=3, out_dim=1).to(device)
        # self.fno   = CNN2d_FiLM(in_channels=2, out_channels=2, width=width, film_input=1, n_layers=2).to(device)
        # self.fno   = FiLMUNet(in_channels=4, out_channels=2, base_width=width, cond_dim=1).to(device)
        self.fno = CNEXTUNet(in_channels=5, out_channels=1, base_width=width, cond_dim=1).to(device)
        self.fno = torch.compile(self.fno, mode="max-autotune")

        # self.fno  = UNetConvNext(5, 1, width)

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

        x       = torch.linspace(0, self.Lx, self.Nx+1, device=device)[:-1]
        y       = torch.linspace(0, self.Ly, self.Ny+1, device=device)[:-1]
        self.x_grid, self.y_grid = torch.meshgrid(x, y, indexing='ij')
        
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
        omega = x.to(self.dt_tensor.device)
        B, C, H, W = omega.shape
        # self.x_grid [H, W] -> [B, H, W]
        x_grid = self.x_grid.unsqueeze(0).repeat(B, 1, 1)
        y_grid = self.y_grid.unsqueeze(0).repeat(B, 1, 1)
        # omega = self.N0_SCHEME._velocity_to_vorticity_spectral(
        #     x[:, 0], x[:, 1],
        #     self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine
        # )  # [B, H, W]
        
        # B: Time-step the vorticity
        # Perform a standard single-step update on the vorticity field
        # omega = omega.unsqueeze(1)
        omega = self.N0_SCHEME.single_step(omega, dts=self.dt).squeeze(1)

        mean_old = omega.mean(dim=(-2, -1), keepdim=True)
        # Convert the updated vorticity back to velocity
        u, v = self.N0_SCHEME._vorticity_to_velocity_spectral(
            omega, self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine, self.N0_SCHEME.denom_safe_fine
        )
        # Use the neural network to predict a residual displacement (delta_x, delta_y)
        omega_features = torch.stack([omega, u, v, x_grid, y_grid], dim=1)  # [B, H, W, 3]
        # omega_features = torch.stack([omega, u, v], dim=-1)
        # The neural network outputs a 2-channel residual displacement field
        # print(self.dt_tensor.shape, self.fno(omega_features, self.dt_tensor).shape)
        # cond_input = self.dt_tensor.view(1, -1).expand(B, -1)
        delta = self.dt_tensor * self.fno(omega_features, self.dt_tensor)  # [B, 3, H, W]
        omega_new = omega + delta.squeeze(1)
        mean_new = omega_new.mean(dim=(-2, -1), keepdim=True)
        omega = omega_new - mean_new + mean_old

        # Perform a second standard single-step update
        # omega = omega.unsqueeze(1)
        # omega = self.N0_SCHEME.single_step(omega, dts=self.dt).squeeze(1)
        
        # Convert the final vorticity back to velocity
        # u, v = self.N0_SCHEME._vorticity_to_velocity_spectral(
        #     omega, self.N0_SCHEME.Kx_fine, self.N0_SCHEME.Ky_fine, self.N0_SCHEME.denom_safe_fine
        # )
        
        # # Return the final velocity field
        # return torch.stack([u, v], dim=1)  # [B, 2, H, W]

        return self.N0_SCHEME.single_step(omega, dts=self.dt)

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
    # u_pred = pred[:, 0, :, :]
    # v_pred = pred[:, 1, :, :]
    # u_target = target[:, 0, :, :]
    # v_target = target[:, 1, :, :]

    # omega_pred = model._velocity_to_vorticity_spectral(u_pred, v_pred, model.Kx_fine, model.Ky_fine)
    # omega_target = model._velocity_to_vorticity_spectral(u_target, v_target, model.Kx_fine, model.Ky_fine)

    return F.mse_loss(pred, target)


def total_loss(pred, target, model, λ_spatial=100.0, λ_vorticity=1.0, λ_spectrum=0.1, alpha_high=2.0):
    """
    pred, target: [B, 1, H, W]  (u, v)
    model: 需要有 _velocity_to_vorticity_spectral 方法
    λ_spatial, λ_vorticity, λ_spectrum: 三种损失的权重
    α_high: 高频放大系数（>1 增强高频的约束）
    """
    # === 1. 空间域损失（L2）
    # spatial_loss = F.mse_loss(pred, target)

    # # === 2. 涡量域损失（保持涡结构）
    # u_pred, v_pred = pred[:, 0], pred[:, 1]
    # u_true, v_true = target[:, 0], target[:, 1]
    # ω_pred = model._velocity_to_vorticity_spectral(u_pred, v_pred, model.Kx_fine, model.Ky_fine)
    # ω_true = model._velocity_to_vorticity_spectral(u_true, v_true, model.Kx_fine, model.Ky_fine)
    # vorticity_loss = F.l1_loss(ω_pred, ω_true)
    # return vorticity_loss
    return F.l1_loss(pred, target)
    # # === 总损失
    # return λ_spatial * spatial_loss + λ_vorticity * vorticity_loss
    # return F.mse_loss(pred, target)


if __name__ == "__main__":
    data_dir = "../dataset" # Or "../dataset" if that's where your data is

    train_dataset = ODEPairDataset(
        os.path.join(data_dir, "train_input_new.pt"),
        os.path.join(data_dir, "train_output_new.pt"),
        limit=8000,
    )
    val_dataset   = ODEPairDataset(
        os.path.join(data_dir, "test_input_new.pt"),
        os.path.join(data_dir, "test_output_new.pt"),
        limit=2000,
    )

    batch_size   = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    A_model = A(Nx=128, Ny=128, Lx=1.0, Ly=1.0)

    model = NavierStokes(N0_SCHEME=A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device), modes1=32, modes2=32, width=32, dt=5e-3, device=device)
    # 多卡并行
    # model = torch.compile(model, mode="max-autotune") # Uncomment if you want to use torch.compile

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 10
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
            torch.save(model.state_dict(), os.path.join('models', "best_model_cnextunet_32.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")