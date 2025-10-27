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
import matplotlib.colors as mcolors
from matplotlib import cm

import os 
os.makedirs("../../results/NS", exist_ok=True)


plt.rcParams.update({
    "font.size": 14,        
    "axes.labelsize": 14,  
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14, 
    "legend.fontsize": 14, 
})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)
class A(N0):
    def __init__(self, Nx=256, Ny=512, Lx=1.0, Ly=2.0, Re=1e4, Cs=0.18, device='cuda:0'):
        super().__init__()
        self.Nx = Nx  # Now represents the fine-grid Nx
        self.Ny = Ny  # Now represents the fine-grid Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re  # Reynolds number
        self.device = device
        self.mu = 1.0 / Re  # Viscosity
        self.Cs = Cs  # Smagorinsky constant

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

        self._cached_dt_coeffs = {}

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
        # Wavenumbers for FFT. torch.fft.fftfreq handBaseline the correct ordering for FFT.
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
        Creates a 2/3 de-aliasing mask in Fourier space.
        """
        # The 2/3 rule truncates modes beyond 2/3 of the maximum wavenumber.
        # This corresponds to indices beyond 2/3 of the total grid size.
        cutoff_x = int(2.0 * self.Nx / 3.0)
        cutoff_y = int(2.0 * self.Ny / 3.0)

        # Create a 1D mask for the x and y dimensions
        mask_x = torch.zeros(self.Nx, dtype=torch.bool, device=self.device)
        mask_y = torch.zeros(self.Ny, dtype=torch.bool, device=self.device)

        # Set mask to True for modes to be kept.
        # In a standard FFT, the modes are ordered from 0 to N/2-1, then -N/2 to -1.
        # So we keep modes from 0 to cutoff and -cutoff to -1.
        mask_x[:cutoff_x] = True
        mask_x[self.Nx - cutoff_x:] = True
        mask_y[:cutoff_y] = True
        mask_y[self.Ny - cutoff_y:] = True

        # Combine the 1D masks into a 2D mask for the full grid
        self.dealias_mask = mask_x.unsqueeze(1) & mask_y.unsqueeze(0)

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
        # denom_safe handBaseline the (0,0) mode, ensuring stability.
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
        f_hat = torch.fft.fft2(f)
        f_hat_filtered = f_hat * self.gaussian_filter_kernel
        f_filtered = torch.fft.ifft2(f_hat_filtered)
        
        return f_filtered.real

    def _calculate_Cs_delta_squared(self, u, v, omega):
        f_stack = torch.stack([u, v, omega])
        f_hat_filtered = torch.fft.fft2(f_stack) * self.gaussian_filter_kernel.unsqueeze(0)
        u_t, v_t, omega_t = torch.fft.ifft2(f_hat_filtered).real

        dwdx, dwdy = self._compute_derivatives_spectral(omega, self.Kx_fine, self.Ky_fine)
        dwdx_t, dwdy_t = self._compute_derivatives_spectral(omega_t, self.Kx_fine, self.Ky_fine)
        J = u * dwdx + v * dwdy
        J_t = u_t * dwdx_t + v_t * dwdy_t
        J_tf = torch.fft.ifft2(torch.fft.fft2(J) * self.gaussian_filter_kernel).real

        S_mag = self._s_mag(u, v)[0]
        S_mag_t = self._s_mag(u_t, v_t)[0]
        lap_omega = self._lap(omega)
        lap_omega_t = torch.fft.ifft2(torch.fft.fft2(lap_omega) * self.gaussian_filter_kernel).real

        H = J_t - J_tf
        M = self.kappa**2 * S_mag_t * lap_omega_t - torch.fft.ifft2(torch.fft.fft2(S_mag * lap_omega) * self.gaussian_filter_kernel).real
        Cs_delta_sq = torch.mean(H*M) / (torch.mean(M*M) + 1e-12)
        return Cs_delta_sq
    
    def _eddy_viscosity(self, u, v, omega):
        S, S11, S22, S12 = self._s_mag(u, v)
        return (self.Cs * self.delta)**2 * S, S11, S22, S12

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
        
        # Use a small value epsilon to handle the z=0 case
        eps = 1e-12
        is_zero = (torch.abs(z) < eps)
        
        # Prepare tensors for the coefficients with complex128 dtype
        E = torch.zeros_like(z, dtype=torch.complex128)
        E2 = torch.zeros_like(z, dtype=torch.complex128)
        phi1 = torch.zeros_like(z, dtype=torch.complex128)
        phi2 = torch.zeros_like(z, dtype=torch.complex128)
        phi3 = torch.zeros_like(z, dtype=torch.complex128)

        A_coeffs = torch.zeros_like(z, dtype=torch.complex128)
        B_coeffs = torch.zeros_like(z, dtype=torch.complex128)
        C_coeffs = torch.zeros_like(z, dtype=torch.complex128)

        # Handle the special case where z=0
        if is_zero.any():
            E[is_zero] = 1.0
            E2[is_zero] = 1.0
            phi1[is_zero] = 1.0
            phi2[is_zero] = 0.5
            phi3[is_zero] = 1.0 / 6.0
            A_coeffs[is_zero] = 1.0 - 1.5 * 0.5 + 0.5 * (1.0 / 6.0)
            B_coeffs[is_zero] = 0.5 - 1.0 / 6.0
            C_coeffs[is_zero] = 0.5 * (1.0 / 6.0)

        # Handle non-zero z with numerical integration
        if (~is_zero).any():
            z_nz = z[~is_zero].unsqueeze(-1)
            
            # Use M=64 for numerical integration
            M = 64
            
            # Use a circular contour in the complex plane
            theta = torch.arange(M, dtype=torch.float64, device=z.device) * (2.0 * math.pi / M)
            w_points = z_nz + torch.exp(1j * theta).unsqueeze(0)
            
            # The integrands for the phi functions
            # Note: The original paper uses a different contour, but this one is also common
            phi1_integrand = (torch.exp(w_points) - 1.0) / w_points
            phi2_integrand = (torch.exp(w_points) - 1.0 - w_points) / (w_points**2)
            phi3_integrand = (torch.exp(w_points) - 1.0 - w_points - w_points**2 / 2.0) / (w_points**3)
            
            # Sum the contributions and average
            phi1_nz = torch.mean(phi1_integrand, dim=1)
            phi2_nz = torch.mean(phi2_integrand, dim=1)
            phi3_nz = torch.mean(phi3_integrand, dim=1)

            E_nz = torch.exp(z_nz.squeeze(-1))
            E2_nz = torch.exp(0.5 * z_nz.squeeze(-1))

            # The final coefficients are derived from the phi functions
            A_coeffs_nz = phi1_nz - 1.5 * phi2_nz + 0.5 * phi3_nz
            B_coeffs_nz = phi2_nz - phi3_nz
            C_coeffs_nz = 0.5 * phi3_nz

            E[~is_zero] = E_nz
            E2[~is_zero] = E2_nz
            phi1[~is_zero] = phi1_nz
            phi2[~is_zero] = phi2_nz
            phi3[~is_zero] = phi3_nz
            A_coeffs[~is_zero] = A_coeffs_nz
            B_coeffs[~is_zero] = B_coeffs_nz
            C_coeffs[~is_zero] = C_coeffs_nz
        
        return E, E2, phi1, phi2, phi3, A_coeffs, B_coeffs, C_coeffs
    
    def _etdrk4_step(self, omega: torch.Tensor, dt: float) -> torch.Tensor:
        if not hasattr(self, "_cached_dt_coeffs"):
            self._cached_dt_coeffs = {}

        # 查找缓存
        if dt in self._cached_dt_coeffs:
            E, E2, phi1, A_coeffs, B_coeffs, C_coeffs = self._cached_dt_coeffs[dt]
        else:
            E, E2, phi1, _, _, A_coeffs, B_coeffs, C_coeffs = self._compute_etdrk4_coefficients(dt)
            self._cached_dt_coeffs[dt] = (E, E2, phi1, A_coeffs, B_coeffs, C_coeffs)

        # === Stage 1 ===
        omega_real = omega.squeeze(1)
        omega_k_n = torch.fft.fft2(omega_real)
        N0_k = self.rhs_fft(omega_real)

        # === Stage 2 ===
        omega_a_k = E2.unsqueeze(0) * omega_k_n + 0.5 * dt * phi1.unsqueeze(0) * N0_k
        omega_a_real = torch.fft.ifft2(omega_a_k).real
        Na_k = self.rhs_fft(omega_a_real)

        # === Stage 3 ===
        omega_b_k = E2.unsqueeze(0) * omega_k_n + 0.5 * dt * phi1.unsqueeze(0) * Na_k
        omega_b_real = torch.fft.ifft2(omega_b_k).real
        Nb_k = self.rhs_fft(omega_b_real)

        # === Stage 4 ===
        omega_c_k = E2.unsqueeze(0) * omega_k_n + 0.5 * dt * phi1.unsqueeze(0) * (2*Nb_k - Na_k)
        omega_c_real = torch.fft.ifft2(omega_c_k).real
        Nc_k = self.rhs_fft(omega_c_real)

        # === Final Update Step ===
        omega_k_new = (
            E.unsqueeze(0) * omega_k_n +
            dt * (A_coeffs.unsqueeze(0) * N0_k +
                  B_coeffs.unsqueeze(0) * (Na_k + Nb_k) +
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
    

u_true_t199 = np.load('2th/u_199_true.npy')
v_true_t199 = np.load('2th/v_199_true.npy')
u_2th_t199 = np.load('2th/u_199_2th.npy')
v_2th_t199 = np.load('2th/v_199_2th.npy')
u_4th_t199 = np.load('4th/u_199_4th.npy')
v_4th_t199 = np.load('4th/v_199_4th.npy')
u_Baseline_t199 = np.load('baseline/u_199_base.npy')
v_Baseline_t199 = np.load('baseline/v_199_base.npy')

u_true_t299 = np.load('2th/u_299_true.npy')
v_true_t299 = np.load('2th/v_299_true.npy')
u_2th_t299 = np.load('2th/u_299_2th.npy')
v_2th_t299 = np.load('2th/v_299_2th.npy')
u_4th_t299 = np.load('4th/u_299_4th.npy')
v_4th_t299 = np.load('4th/v_299_4th.npy')
u_Baseline_t299 = np.load('baseline/u_299_base.npy')
v_Baseline_t299 = np.load('baseline/v_299_base.npy')

u_true_t349 = np.load('2th/u_349_true.npy')
v_true_t349 = np.load('2th/v_349_true.npy')
u_2th_t349 = np.load('2th/u_349_2th.npy')
v_2th_t349 = np.load('2th/v_349_2th.npy')
u_4th_t349 = np.load('4th/u_349_4th.npy')
v_4th_t349 = np.load('4th/v_349_4th.npy')
u_Baseline_t349 = np.load('baseline/u_349_base.npy')
v_Baseline_t349 = np.load('baseline/v_349_base.npy')

x = np.linspace(0, 1, 256+1)[:-1]
y = np.linspace(0, 1, 256+1)[:-1]
X, Y = np.meshgrid(x, y, indexing='ij')

methods = ["True", "FNO", "ANI-2", "ANI-4"]

def plot_uv_grid(u_list, v_list, t_label, methods):
    fig = plt.figure(figsize=(4.5*len(methods), 8))
    gs = fig.add_gridspec(2, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])

    umin, umax = np.min(u_list), np.max(u_list)
    vmin, vmax = np.min(v_list), np.max(v_list)

    # u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, u_list[col], cmap='viridis', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
        # for c in im_u.collections:
            # c.set_rasterized(True)
        ax.set_rasterized(True) 
        ax.set_title(f"{method} - u")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        axes_u.append(ax)
    cax_u = fig.add_subplot(gs[0, -1])
    norm  = mcolors.Normalize(vmin=umin, vmax=umax)
    sm    = cm.ScalarMappable(norm=norm, cmap=im_u.cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax_u, label="u value")
    # fig.colorbar(im_u, cax=cax_u, label="u value")

    # v
    axes_v = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, col])
        im_v = ax.contourf(X, Y, v_list[col], cmap='viridis', origin='lower',
                           levels=100, vmin=vmin, vmax=vmax)
        # for c in im_v.collections:
            # c.set_rasterized(True)
        ax.set_rasterized(True) 
        ax.set_title(f"{method} - v")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        axes_v.append(ax)
    cax_v = fig.add_subplot(gs[1, -1])
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm    = cm.ScalarMappable(norm=norm, cmap=im_v.cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax_v, label="v value")
    # fig.colorbar(im_v, cax=cax_v, label="v value")

    plt.tight_layout()
    plt.savefig(f"../../results/NS/uv_grid_{t_label}_model.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

plot_uv_grid(
    [u_true_t199, u_Baseline_t199, u_2th_t199, u_4th_t199],
    [v_true_t199, v_Baseline_t199, v_2th_t199, v_4th_t199],
    "2s",
    methods
)

plot_uv_grid(
    [u_true_t299, u_Baseline_t299, u_2th_t299, u_4th_t299],
    [v_true_t299, v_Baseline_t299, v_2th_t299, v_4th_t299],
    "3s",
    methods
)

plot_uv_grid(
    [u_true_t349, u_Baseline_t349, u_2th_t349, u_4th_t349],
    [v_true_t349, v_Baseline_t349, v_2th_t349, v_4th_t349],
    "3.5s",
    methods
)

methods = ["FNO", "ANI-2", "ANI-4"]

def plot_error_grid(u_true, v_true, u_methods, v_methods, t_label,
                    signed=False, cmap_abs='viridis', cmap_signed='RdBu'):
    """
    u_true, v_true: 2D numpy arrays
    u_methods, v_methods: lists of numpy arrays [u_2th, u_4th, u_baseline]
    signed: True → signed error; False → absolute error
    Each subplot has its own colorbar.
    """
    # compute error
    if signed:
        err_u = [m - u_true for m in u_methods]
        err_v = [m - v_true for m in v_methods]
        cmap = cmap_signed
    else:
        err_u = [np.abs(m - u_true) for m in u_methods]
        err_v = [np.abs(m - v_true) for m in v_methods]
        cmap = cmap_abs

    for i in err_u:
        print(np.linalg.norm(i))
    for i in err_v:
        print(np.linalg.norm(i))

    fig, axes = plt.subplots(2, len(methods), figsize=(4*len(methods), 8))
    for col, method in enumerate(methods):
        # u
        # im_u = axes[0, col].imshow(err_u[col], origin='lower', cmap=cmap, aspect='auto')
        im_u = axes[0, col].contourf(X, Y, err_u[col], cmap=cmap, origin='lower', levels=100)
        # for c in im_u.collections:
            # c.set_rasterized(True)
        axes[0, col].set_rasterized(True) 
        axes[0, col].set_title(f"{method} — {'u error'}")
        axes[0, col].axis('off')
        axes[0, col].set_aspect('equal', adjustable='box')
        plt.colorbar(im_u, ax=axes[0, col], fraction=0.046, pad=0.04)

        # v
        # im_v = axes[1, col].imshow(err_v[col], origin='lower', cmap=cmap, aspect='auto')
        im_v = axes[1, col].contourf(X, Y, err_v[col], cmap=cmap, origin='lower', levels=100)
        # for c in im_v.collections:
            # c.set_rasterized(True)
        axes[1, col].set_rasterized(True)
        axes[1, col].set_title(f"{method} — {'v error'}")
        axes[1, col].axis('off')
        axes[1, col].set_aspect('equal', adjustable='box')
        plt.colorbar(im_v, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f"../../results/NS/error_grid_{t_label}_model.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


plot_error_grid(
    u_true_t199, v_true_t199,
    [u_Baseline_t199, u_2th_t199, u_4th_t199],
    [v_Baseline_t199, v_2th_t199, v_4th_t199],
    "2s"
)

plot_error_grid(
    u_true_t299, v_true_t299,
    [u_Baseline_t299, u_2th_t299, u_4th_t299],
    [v_Baseline_t299, v_2th_t299, v_4th_t299],
    "3s"
)

plot_error_grid(
    u_true_t349, v_true_t349,
    [u_Baseline_t349, u_2th_t349, u_4th_t349],
    [v_Baseline_t349, v_2th_t349, v_4th_t349],
    "3.5s"
)

model = A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, device=device)
u_true_t199 = torch.tensor(u_true_t199).unsqueeze(0).to(device)
v_true_t199 = torch.tensor(v_true_t199).unsqueeze(0).to(device)
u_2th_t199 = torch.tensor(u_2th_t199).unsqueeze(0).to(device)
v_2th_t199 = torch.tensor(v_2th_t199).unsqueeze(0).to(device)
u_4th_t199 = torch.tensor(u_4th_t199).unsqueeze(0).to(device)
v_4th_t199 = torch.tensor(v_4th_t199).unsqueeze(0).to(device)
u_Baseline_t199 = torch.tensor(u_Baseline_t199).unsqueeze(0).to(device)
v_Baseline_t199 = torch.tensor(v_Baseline_t199).unsqueeze(0).to(device)

u_true_t299 = torch.tensor(u_true_t299).unsqueeze(0).to(device)
v_true_t299 = torch.tensor(v_true_t299).unsqueeze(0).to(device)
u_2th_t299 = torch.tensor(u_2th_t299).unsqueeze(0).to(device)
v_2th_t299 = torch.tensor(v_2th_t299).unsqueeze(0).to(device)
u_4th_t299 = torch.tensor(u_4th_t299).unsqueeze(0).to(device)
v_4th_t299 = torch.tensor(v_4th_t299).unsqueeze(0).to(device)
u_Baseline_t299 = torch.tensor(u_Baseline_t299).unsqueeze(0).to(device)
v_Baseline_t299 = torch.tensor(v_Baseline_t299).unsqueeze(0).to(device)

u_true_t349 = torch.tensor(u_true_t349).unsqueeze(0).to(device)
v_true_t349 = torch.tensor(v_true_t349).unsqueeze(0).to(device)
u_2th_t349 = torch.tensor(u_2th_t349).unsqueeze(0).to(device)
v_2th_t349 = torch.tensor(v_2th_t349).unsqueeze(0).to(device)
u_4th_t349 = torch.tensor(u_4th_t349).unsqueeze(0).to(device)
v_4th_t349 = torch.tensor(v_4th_t349).unsqueeze(0).to(device)
u_Baseline_t349 = torch.tensor(u_Baseline_t349).unsqueeze(0).to(device)
v_Baseline_t349 = torch.tensor(v_Baseline_t349).unsqueeze(0).to(device)

omega_true_t199 = model._velocity_to_vorticity_spectral(u_true_t199, v_true_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_2th_t199 = model._velocity_to_vorticity_spectral(u_2th_t199, v_2th_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_4th_t199 = model._velocity_to_vorticity_spectral(u_4th_t199, v_4th_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_Baseline_t199 = model._velocity_to_vorticity_spectral(u_Baseline_t199, v_Baseline_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

omega_true_t299 = model._velocity_to_vorticity_spectral(u_true_t299, v_true_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_2th_t299 = model._velocity_to_vorticity_spectral(u_2th_t299, v_2th_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_4th_t299 = model._velocity_to_vorticity_spectral(u_4th_t299, v_4th_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_Baseline_t299 = model._velocity_to_vorticity_spectral(u_Baseline_t299, v_Baseline_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

omega_true_t349 = model._velocity_to_vorticity_spectral(u_true_t349, v_true_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_2th_t349 = model._velocity_to_vorticity_spectral(u_2th_t349, v_2th_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_4th_t349 = model._velocity_to_vorticity_spectral(u_4th_t349, v_4th_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
omega_Baseline_t349 = model._velocity_to_vorticity_spectral(u_Baseline_t349, v_Baseline_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

def plot_omega_grid(omega_list, t_label, methods):

    fig = plt.figure(figsize=(4*len(methods), 3.5))
    gs = fig.add_gridspec(1, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])
    umin, umax = np.min(omega_list), np.max(omega_list)

    # u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, omega_list[col], cmap='viridis', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
        # for c in im_u.collections:
            # c.set_rasterized(True)
        ax.set_rasterized(True)
        ax.set_title(fr"{method} - $\omega$")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        axes_u.append(ax)
    cax_u = fig.add_subplot(gs[0, -1])
    norm  = mcolors.Normalize(vmin=umin, vmax=umax)
    sm    = cm.ScalarMappable(norm=norm, cmap=im_u.cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax_u, label=r"$\omega$ value")
    # fig.colorbar(im_u, cax=cax_u, label=r"$\omega$ value")

    plt.tight_layout()
    plt.savefig(f"../../results/NS/omega_grid_{t_label}_model.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


methods = ["True", "FNO", "ANI-2", "ANI-4"]

plot_omega_grid(
    [omega_true_t199, omega_Baseline_t199, omega_2th_t199, omega_4th_t199],
    "2s",
    methods
)

plot_omega_grid(
    [omega_true_t299, omega_Baseline_t299, omega_2th_t299, omega_4th_t299],
    "3s",
    methods
)

plot_omega_grid(
    [omega_true_t349, omega_Baseline_t349, omega_2th_t349, omega_4th_t349],
    "3.5s",
    methods
)

def plot_omega_error_grid(omega_true, omega_methods, t_label, methods,
                          signed=False, cmap_abs='viridis', cmap_signed='RdBu'):
    if signed:
        err = [m - omega_true for m in omega_methods]
        cmap = cmap_signed
    else:
        err = [np.abs(m - omega_true) for m in omega_methods]
        cmap = cmap_abs

    for i, method in zip(err, methods):
        print(f"{method} error norm:", np.linalg.norm(i))

    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for col, method in enumerate(methods):
        im = axes[col].contourf(X, Y, err[col], cmap=cmap, origin='lower', levels=100)
        # for c in im.collections:
            # c.set_rasterized(True)
        axes[col].set_rasterized(True)
        axes[col].set_title(f"{method} — ω error")
        axes[col].axis('off')
        axes[col].set_aspect('equal', adjustable='box')
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"../../results/NS/omega_error_grid_{t_label}_model.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

plot_omega_error_grid(
    omega_true_t199,
    [omega_Baseline_t199, omega_2th_t199, omega_4th_t199],
    "2s",
    ["FNO", "ANI-2", "ANI-4"],
    signed=False,
    cmap_abs='viridis',
    cmap_signed='RdBu'
)

plot_omega_error_grid(
    omega_true_t299,
    [omega_Baseline_t299, omega_2th_t299, omega_4th_t299],
    "3s",
    ["FNO", "ANI-2", "ANI-4"],
    signed=False,
    cmap_abs='viridis',
    cmap_signed='RdBu'
)

plot_omega_error_grid(
    omega_true_t349,
    [omega_Baseline_t349, omega_2th_t349, omega_4th_t349],
    "3.5s",
    ["FNO", "ANI-2", "ANI-4"],
    signed=False,
    cmap_abs='viridis',
    cmap_signed='RdBu'
)

def compute_isotropic_spectrum(u, v, w=None, L=1.0, nbins=None):
    if u.ndim == 3:
        u = u[0]
    if v.ndim == 3:
        v = v[0]

    N = u.shape[0]

    # FFT
    u_hat = np.fft.fft2(u) / N**2
    v_hat = np.fft.fft2(v) / N**2

    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    kx = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2).ravel()
    E_flat = E_hat.ravel()

    k_max = np.pi * N / L * np.sqrt(2)
    if nbins is None:
        nbins = N // 2
    
    k_bins = np.linspace(0.0, k_max, nbins + 1)
    dk = k_bins[1] - k_bins[0]
    E_k_sum, _ = np.histogram(k_mag, bins=k_bins, weights=E_flat)
    k_values = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_k = E_k_sum

    return k_values, E_k

def plot_isotropic_spectrum(u_list, v_list, t_label="t", methods=None, L=1.0):
    if methods is None:
        methods = [f"method_{i}" for i in range(len(u_list))]

    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#733497', '#CC79A7', '#F0E442', '#56B4E9']

    u_first, v_first = u_list[0], v_list[0]
    k, E_k = compute_isotropic_spectrum(u_first.cpu().numpy(), v_first.cpu().numpy(), L=L)
    
    for i, (u, v, method) in enumerate(zip(u_list, v_list, methods)):
        k_i, E_k_i = compute_isotropic_spectrum(u.cpu().numpy(), v.cpu().numpy(), L=L)
        plt.loglog(k_i[1:], E_k_i[1:], color=colors[i % len(colors)], linewidth=2, label=method)

    k_range_3 = np.array([35, 80], dtype=np.float64)
    idx_3 = (np.abs(k - k_range_3[0])).argmin()
    C3 = E_k[idx_3] * (k[idx_3]**3)
    E_ref_3 = C3 * (k_range_3**-3) / 2
    plt.loglog(k_range_3, E_ref_3, '--', color='black', linewidth=2) # 移除 label
    
    plt.text(k_range_3[1], E_ref_3[1], r' $k^{-3}$', fontsize=14, color='black', 
             ha='left', va='bottom', rotation=-30) # rotation使标签跟随线条斜率

    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$E(k)$', fontsize=14)
    plt.tick_params(axis='both', which='major')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    
    plt.xlim(right=plt.xlim()[1] * 1.5)

    plt.tight_layout()
    plt.savefig(f"../../results/NS/isotropic_spectrum_{t_label}_model.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


plot_isotropic_spectrum(
    [u_Baseline_t199, u_2th_t199, u_4th_t199],
    [v_Baseline_t199, v_2th_t199, v_4th_t199],
    "2s",
    ["FNO", "ANI-2", "ANI-4"],
    L=1.0
)

plot_isotropic_spectrum(
    [u_Baseline_t299, u_2th_t299, u_4th_t299],
    [v_Baseline_t299, v_2th_t299, v_4th_t299],
    "3s",
    ["FNO", "ANI-2", "ANI-4"],
    L=1.0
)

plot_isotropic_spectrum(
    [u_Baseline_t349, u_2th_t349, u_4th_t349],
    [v_Baseline_t349, v_2th_t349, v_4th_t349],
    "3.5s",
    ["FNO", "ANI-2", "ANI-4"],
    L=1.0
)