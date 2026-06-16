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

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

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
        omega_k = (
            E.unsqueeze(0) * omega_k_n +
            dt * (A_coeffs.unsqueeze(0) * N0_k +
                  B_coeffs.unsqueeze(0) * (Na_k + Nb_k) +
                  C_coeffs.unsqueeze(0) * Nc_k)
        )
        
        omega_real = torch.fft.ifft2(omega_k).real
        return omega_real.unsqueeze(1)
    
    def single_step(self, u: torch.Tensor, dts: float = None, dt_max: float = 5e-4) -> torch.Tensor:
        remain_time = dts 
        while remain_time > 0:
            dt = min(dt_max, remain_time)
            u = self._etdrk4_step(u, dt)
            remain_time -= dt
        return u
    

u_true_t99 = np.load('2th/u_99_true.npy').squeeze(0)
v_true_t99 = np.load('2th/v_99_true.npy').squeeze(0)
u_2th_t99 = np.load('2th/u_99_2th.npy').squeeze(0)
v_2th_t99 = np.load('2th/v_99_2th.npy').squeeze(0)
u_4th_t99 = np.load('4th/u_99_4th.npy').squeeze(0)
v_4th_t99 = np.load('4th/v_99_4th.npy').squeeze(0)
u_les_t99 = np.load('2th/u_99_les.npy').squeeze(0)
v_les_t99 = np.load('2th/v_99_les.npy').squeeze(0)

u_true_149 = np.load('2th/u_149_true.npy').squeeze(0)
v_true_149 = np.load('2th/v_149_true.npy').squeeze(0)
u_2th_149 = np.load('2th/u_149_2th.npy').squeeze(0)
v_2th_149 = np.load('2th/v_149_2th.npy').squeeze(0)
u_4th_149 = np.load('4th/u_149_4th.npy').squeeze(0)
v_4th_149 = np.load('4th/v_149_4th.npy').squeeze(0)
u_les_149 = np.load('2th/u_149_les.npy').squeeze(0)
v_les_149 = np.load('2th/v_149_les.npy').squeeze(0)

u_true_t199 = np.load('2th/u_199_true.npy').squeeze(0)
v_true_t199 = np.load('2th/v_199_true.npy').squeeze(0)
u_2th_t199 = np.load('2th/u_199_2th.npy').squeeze(0)
v_2th_t199 = np.load('2th/v_199_2th.npy').squeeze(0)
u_4th_t199 = np.load('4th/u_199_4th.npy').squeeze(0)
v_4th_t199 = np.load('4th/v_199_4th.npy').squeeze(0)
u_les_t199 = np.load('2th/u_199_les.npy').squeeze(0)
v_les_t199 = np.load('2th/v_199_les.npy').squeeze(0)

u_true_t249 = np.load('2th/u_249_true.npy').squeeze(0)
v_true_t249 = np.load('2th/v_249_true.npy').squeeze(0)
u_2th_t249 = np.load('2th/u_249_2th.npy').squeeze(0)
v_2th_t249 = np.load('2th/v_249_2th.npy').squeeze(0)
u_4th_t249 = np.load('4th/u_249_4th.npy').squeeze(0)
v_4th_t249 = np.load('4th/v_249_4th.npy').squeeze(0)
u_les_t249 = np.load('2th/u_249_les.npy').squeeze(0)
v_les_t249 = np.load('2th/v_249_les.npy').squeeze(0)

u_true_t299 = np.load('2th/u_299_true.npy').squeeze(0)
v_true_t299 = np.load('2th/v_299_true.npy').squeeze(0)
u_2th_t299 = np.load('2th/u_299_2th.npy').squeeze(0)
v_2th_t299 = np.load('2th/v_299_2th.npy').squeeze(0)
u_4th_t299 = np.load('4th/u_299_4th.npy').squeeze(0)
v_4th_t299 = np.load('4th/v_299_4th.npy').squeeze(0)
u_les_t299 = np.load('2th/u_299_les.npy').squeeze(0)
v_les_t299 = np.load('2th/v_299_les.npy').squeeze(0)

u_true_t349 = np.load('2th/u_349_true.npy').squeeze(0)
v_true_t349 = np.load('2th/v_349_true.npy').squeeze(0)
u_2th_t349 = np.load('2th/u_349_2th.npy').squeeze(0)
v_2th_t349 = np.load('2th/v_349_2th.npy').squeeze(0)
u_4th_t349 = np.load('4th/u_349_4th.npy').squeeze(0)
v_4th_t349 = np.load('4th/v_349_4th.npy').squeeze(0)
u_les_t349 = np.load('2th/u_349_les.npy').squeeze(0)
v_les_t349 = np.load('2th/v_349_les.npy').squeeze(0)

time_steps = [99, 149, 199, 249, 299]
data = {
    'u_true': {}, 'v_true': {},
    'u_2th': {}, 'v_2th': {},
    'u_4th': {}, 'v_4th': {},
    'u_les': {}, 'v_les': {},
    'omega_2th' : {}, 'omega_true' : {},
    'omega_4th' : {}, 'omega_les' : {}
}
for t in time_steps:
    data['u_true'][t] = np.load(f'2th/u_{t}_true.npy').squeeze(0)
    data['v_true'][t] = np.load(f'2th/v_{t}_true.npy').squeeze(0)
    data['u_2th'][t]  = np.load(f'2th/u_{t}_2th.npy').squeeze(0)
    data['v_2th'][t]  = np.load(f'2th/v_{t}_2th.npy').squeeze(0)
    data['u_4th'][t]  = np.load(f'4th/u_{t}_4th.npy').squeeze(0)
    data['v_4th'][t]  = np.load(f'4th/v_{t}_4th.npy').squeeze(0)
    data['u_les'][t]  = np.load(f'2th/u_{t}_les.npy').squeeze(0)
    data['v_les'][t]  = np.load(f'2th/v_{t}_les.npy').squeeze(0)

time_map = {
    99:  "0.5s",
    149: "0.75s",
    199: "1.0s",
    249: "1.25s",
    299: "1.5s",
}

x = np.linspace(0, 1, 128)
y = np.linspace(0, 1, 128)
X, Y = np.meshgrid(x, y, indexing='ij')

methods = ["True", "LES", "ANI-2", "ANI-4"]

def plot_uv_grid(u_list, v_list, t_label, methods):
    fig = plt.figure(figsize=(4.5*len(methods), 8))
    gs = fig.add_gridspec(2, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])

    # 统一颜色范围
    umin, umax = np.min(u_list), np.max(u_list)
    vmin, vmax = np.min(v_list), np.max(v_list)

    # 第一行：u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, u_list[col], cmap='RdBu_r', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
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

    # 第二行：v
    axes_v = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, col])
        im_v = ax.contourf(X, Y, v_list[col], cmap='RdBu_r', origin='lower',
                           levels=100, vmin=vmin, vmax=vmax)
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
    plt.savefig(f"uv_grid_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
# t=0s
# plot_uv_grid(
#     [u_true_t99, u_les_t99, u_2th_t99, u_4th_t99],
#     [v_true_t99, v_les_t99, v_2th_t99, v_4th_t99],
#     "0.5s",
#     methods
# )

# plot_uv_grid(
#     [u_true_149, u_les_149, u_2th_149, u_4th_149],
#     [v_true_149, v_les_149, v_2th_149, v_4th_149],
#     "0.75s",
#     methods
# )

# plot_uv_grid(
#     [u_true_t199, u_les_t199, u_2th_t199, u_4th_t199],
#     [v_true_t199, v_les_t199, v_2th_t199, v_4th_t199],
#     "1.0s",
#     methods
# )

# plot_uv_grid(
#     [u_true_t249, u_les_t249, u_2th_t249, u_4th_t249],
#     [v_true_t249, v_les_t249, v_2th_t249, v_4th_t249],
#     "1.25s",
#     methods
# )

# plot_uv_grid(
#     [u_true_t299, u_les_t299, u_2th_t299, u_4th_t299],
#     [v_true_t299, v_les_t299, v_2th_t299, v_4th_t299],
#     "1.5s",
#     methods
# )

# plot_uv_grid(
#     [u_true_t349, u_les_t349, u_2th_t349, u_4th_t349],
#     [v_true_t349, v_les_t349, v_2th_t349, v_4th_t349],
#     "1.75s",
#     methods
# )
for t, time_label in time_map.items():
    u_list = [
        data['u_true'][t],
        data['u_les'][t],
        data['u_2th'][t],
        data['u_4th'][t]
    ]
    
    v_list = [
        data['v_true'][t],
        data['v_les'][t],
        data['v_2th'][t],
        data['v_4th'][t]
    ]
    plot_uv_grid(u_list, v_list, time_label, methods)


methods = ["LES", "ANI-2", "ANI-4"]

def plot_error_grid(u_true, v_true, u_methods, v_methods, t_label,
                    signed=False, cmap_abs='viridis', cmap_signed='RdBu_r'):
    """
    u_true, v_true: 2D numpy arrays
    u_methods, v_methods: lists of numpy arrays [u_2th, u_4th, u_LES]
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


    fig, axes = plt.subplots(2, len(methods), figsize=(4*len(methods), 8))
    for col, method in enumerate(methods):
        # u
        # im_u = axes[0, col].imshow(err_u[col], origin='lower', cmap=cmap, aspect='auto')
        im_u = axes[0, col].contourf(X, Y, err_u[col], cmap=cmap, origin='lower', levels=100)
        axes[0, col].set_rasterized(True)
        axes[0, col].set_title(f"{method} — {'u error'}")
        axes[0, col].axis('off')
        axes[0, col].set_aspect('equal', adjustable='box')
        plt.colorbar(im_u, ax=axes[0, col], fraction=0.046, pad=0.04)

        # v
        # im_v = axes[1, col].imshow(err_v[col], origin='lower', cmap=cmap, aspect='auto')
        im_v = axes[1, col].contourf(X, Y, err_v[col], cmap=cmap, origin='lower', levels=100)
        axes[1, col].set_rasterized(True)
        axes[1, col].set_title(f"{method} — {'v error'}")
        axes[1, col].axis('off')
        axes[1, col].set_aspect('equal', adjustable='box')
        plt.colorbar(im_v, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0,0,1,0.95])
    # plt.show()
    plt.savefig(f"error_grid_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# plot_error_grid(
#     u_true_t99, v_true_t99,
#     [u_les_t99, u_2th_t99, u_4th_t99],
#     [v_les_t99, v_2th_t99, v_4th_t99],
#     "0.5s"
# )

# plot_error_grid(
#     u_true_149, v_true_149,
#     [u_les_149, u_2th_149, u_4th_149],
#     [v_les_149, v_2th_149, v_4th_149],
#     "0.75s"
# )

# plot_error_grid(
#     u_true_t199, v_true_t199,
#     [u_les_t199, u_2th_t199, u_4th_t199],
#     [v_les_t199, v_2th_t199, v_4th_t199],
#     "1.0s"
# )

# plot_error_grid(
#     u_true_t249, v_true_t249,
#     [u_les_t249, u_2th_t249, u_4th_t249],
#     [v_les_t249, v_2th_t249, v_4th_t249],
#     "1.25s"
# )

# plot_error_grid(
#     u_true_t299, v_true_t299,
#     [u_les_t299, u_2th_t299, u_4th_t299],
#     [v_les_t299, v_2th_t299, v_4th_t299],
#     "1.5s"
# )

# plot_error_grid(
#     u_true_t349, v_true_t349,
#     [u_les_t349, u_2th_t349, u_4th_t349],
#     [v_les_t349, v_2th_t349, v_4th_t349],
#     "1.75s"
# )

# def get_var(prefix, t):
#     name_with_t = f"{prefix}_t{t}"
#     name_without_t = f"{prefix}_{t}"
#     if name_with_t in locals() or name_with_t in globals():
#         return globals().get(name_with_t, locals().get(name_with_t))
#     return globals().get(name_without_t, locals().get(name_without_t))

# for t, time_label in time_map.items():
#     u_true = data['u_true'][t]
#     v_true = data['v_true'][t]
    
#     # u_preds = [get_var("u_les", t), get_var("u_2th", t), get_var("u_4th", t)]
#     # v_preds = [get_var("v_les", t), get_var("v_2th", t), get_var("v_4th", t)]
#     u_preds = [data['u_les'][t], data['u_2th'][t], data['u_4th'][t]]
#     v_preds = [data['v_les'][t], data['v_2th'][t], data['v_4th'][t]]
    
#     if u_true is not None:
#         plot_error_grid(
#             u_true, v_true,
#             u_preds, v_preds,
#             time_label
#         )

model = A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device)
# convert u_true_t99 -> tensor
# u_true_t99 = torch.tensor(u_true_t99).unsqueeze(0).to(device)
# v_true_t99 = torch.tensor(v_true_t99).unsqueeze(0).to(device)
# u_2th_t99 = torch.tensor(u_2th_t99).unsqueeze(0).to(device)
# v_2th_t99 = torch.tensor(v_2th_t99).unsqueeze(0).to(device)
# u_4th_t99 = torch.tensor(u_4th_t99).unsqueeze(0).to(device)
# v_4th_t99 = torch.tensor(v_4th_t99).unsqueeze(0).to(device)
# u_les_t99 = torch.tensor(u_les_t99).unsqueeze(0).to(device)
# v_les_t99 = torch.tensor(v_les_t99).unsqueeze(0).to(device)

# u_true_149 = torch.tensor(u_true_149).unsqueeze(0).to(device)
# v_true_149 = torch.tensor(v_true_149).unsqueeze(0).to(device)
# u_2th_149 = torch.tensor(u_2th_149).unsqueeze(0).to(device)
# v_2th_149 = torch.tensor(v_2th_149).unsqueeze(0).to(device)
# u_4th_149 = torch.tensor(u_4th_149).unsqueeze(0).to(device)
# v_4th_149 = torch.tensor(v_4th_149).unsqueeze(0).to(device)
# u_les_149 = torch.tensor(u_les_149).unsqueeze(0).to(device)
# v_les_149 = torch.tensor(v_les_149).unsqueeze(0).to(device)

# u_true_t199 = torch.tensor(u_true_t199).unsqueeze(0).to(device)
# v_true_t199 = torch.tensor(v_true_t199).unsqueeze(0).to(device)
# u_2th_t199 = torch.tensor(u_2th_t199).unsqueeze(0).to(device)
# v_2th_t199 = torch.tensor(v_2th_t199).unsqueeze(0).to(device)
# u_4th_t199 = torch.tensor(u_4th_t199).unsqueeze(0).to(device)
# v_4th_t199 = torch.tensor(v_4th_t199).unsqueeze(0).to(device)
# u_les_t199 = torch.tensor(u_les_t199).unsqueeze(0).to(device)
# v_les_t199 = torch.tensor(v_les_t199).unsqueeze(0).to(device)

# u_true_t249 = torch.tensor(u_true_t249).unsqueeze(0).to(device)
# v_true_t249 = torch.tensor(v_true_t249).unsqueeze(0).to(device)
# u_2th_t249 = torch.tensor(u_2th_t249).unsqueeze(0).to(device)
# v_2th_t249 = torch.tensor(v_2th_t249).unsqueeze(0).to(device)
# u_4th_t249 = torch.tensor(u_4th_t249).unsqueeze(0).to(device)
# v_4th_t249 = torch.tensor(v_4th_t249).unsqueeze(0).to(device)
# u_les_t249 = torch.tensor(u_les_t249).unsqueeze(0).to(device)
# v_les_t249 = torch.tensor(v_les_t249).unsqueeze(0).to(device)

# u_true_t299 = torch.tensor(u_true_t299).unsqueeze(0).to(device)
# v_true_t299 = torch.tensor(v_true_t299).unsqueeze(0).to(device)
# u_2th_t299 = torch.tensor(u_2th_t299).unsqueeze(0).to(device)
# v_2th_t299 = torch.tensor(v_2th_t299).unsqueeze(0).to(device)
# u_4th_t299 = torch.tensor(u_4th_t299).unsqueeze(0).to(device)
# v_4th_t299 = torch.tensor(v_4th_t299).unsqueeze(0).to(device)
# u_les_t299 = torch.tensor(u_les_t299).unsqueeze(0).to(device)
# v_les_t299 = torch.tensor(v_les_t299).unsqueeze(0).to(device)

# u_true_t349 = torch.tensor(u_true_t349).unsqueeze(0).to(device)
# v_true_t349 = torch.tensor(v_true_t349).unsqueeze(0).to(device)
# u_2th_t349 = torch.tensor(u_2th_t349).unsqueeze(0).to(device)
# v_2th_t349 = torch.tensor(v_2th_t349).unsqueeze(0).to(device)
# u_4th_t349 = torch.tensor(u_4th_t349).unsqueeze(0).to(device)
# v_4th_t349 = torch.tensor(v_4th_t349).unsqueeze(0).to(device)
# u_les_t349 = torch.tensor(u_les_t349).unsqueeze(0).to(device)
# v_les_t349 = torch.tensor(v_les_t349).unsqueeze(0).to(device)


# omega_true_t99 = model._velocity_to_vorticity_spectral(u_true_t99, v_true_t99, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_t99 = model._velocity_to_vorticity_spectral(u_2th_t99, v_2th_t99, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_t99 = model._velocity_to_vorticity_spectral(u_4th_t99, v_4th_t99, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_t99 = model._velocity_to_vorticity_spectral(u_les_t99, v_les_t99, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

# omega_true_149 = model._velocity_to_vorticity_spectral(u_true_149, v_true_149, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_149 = model._velocity_to_vorticity_spectral(u_2th_149, v_2th_149, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_149 = model._velocity_to_vorticity_spectral(u_4th_149, v_4th_149, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_149 = model._velocity_to_vorticity_spectral(u_les_149, v_les_149, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

# omega_true_t199 = model._velocity_to_vorticity_spectral(u_true_t199, v_true_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_t199 = model._velocity_to_vorticity_spectral(u_2th_t199, v_2th_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_t199 = model._velocity_to_vorticity_spectral(u_4th_t199, v_4th_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_t199 = model._velocity_to_vorticity_spectral(u_les_t199, v_les_t199, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

# omega_true_t249 = model._velocity_to_vorticity_spectral(u_true_t249, v_true_t249, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_t249 = model._velocity_to_vorticity_spectral(u_2th_t249, v_2th_t249, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_t249 = model._velocity_to_vorticity_spectral(u_4th_t249, v_4th_t249, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_t249 = model._velocity_to_vorticity_spectral(u_les_t249, v_les_t249, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

# omega_true_t299 = model._velocity_to_vorticity_spectral(u_true_t299, v_true_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_t299 = model._velocity_to_vorticity_spectral(u_2th_t299, v_2th_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_t299 = model._velocity_to_vorticity_spectral(u_4th_t299, v_4th_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_t299 = model._velocity_to_vorticity_spectral(u_les_t299, v_les_t299, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

# omega_true_t349 = model._velocity_to_vorticity_spectral(u_true_t349, v_true_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_2th_t349 = model._velocity_to_vorticity_spectral(u_2th_t349, v_2th_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_4th_t349 = model._velocity_to_vorticity_spectral(u_4th_t349, v_4th_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()
# omega_les_t349 = model._velocity_to_vorticity_spectral(u_les_t349, v_les_t349, model.Kx_fine, model.Ky_fine).squeeze(0).cpu().numpy()

methods = ["true", "2th", "4th", "les"]
steps = [99, 149, 199, 249, 299]
for t in steps:
    for m in methods:
        u_key = f'u_{m}'
        v_key = f'v_{m}'
        omega_key = f'omega_{m}'
        
        if t in data[u_key] and t in data[v_key]:
            u_np = data[u_key][t]
            v_np = data[v_key][t]
            
            u_tensor = torch.tensor(u_np).unsqueeze(0).to(device)
            v_tensor = torch.tensor(v_np).unsqueeze(0).to(device)
            
            with torch.no_grad():
                omega_tensor = model._velocity_to_vorticity_spectral(
                    u_tensor, v_tensor, model.Kx_fine, model.Ky_fine
                )
            
            # 将计算结果存回字典 (以 Numpy 形式，方便绘图)
            data[omega_key][t] = omega_tensor.squeeze(0).cpu().numpy()
            
            # 如果你后续训练还需要 Tensor，也可以选择把 Tensor 存下来：
            # data[f'{u_key}_tensor'][t] = u_tensor 
            

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.contourf(X, Y, omega_true_t249, levels=100, cmap='RdBu_r', origin='lower')
# ax.axis('off')
# ax.set_aspect('equal', adjustable='box')
# plt.tight_layout()
# ax.set_rasterized(True)
# plt.savefig("omega_grid.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
# plt.close()

def plot_omega_grid(omega_list, t_label, methods):

    fig = plt.figure(figsize=(4*len(methods), 3.5))
    gs = fig.add_gridspec(1, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])

    # 统一颜色范围
    umin, umax = np.min(omega_list), np.max(omega_list)

    # 第一行：u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, omega_list[col], cmap='RdBu_r', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
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
    plt.savefig(f"omega_grid_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


methods_legend = ["True", "LES", "ANI-2", "ANI-4"]
# plot_omega_grid(
#     [omega_true_t99, omega_les_t99, omega_2th_t99, omega_4th_t99],
#     "0.5s",
#     methods
# )

# plot_omega_grid(
#     [omega_true_149, omega_les_149, omega_2th_149, omega_4th_149],
#     "0.75s",
#     methods
# )

# plot_omega_grid(
#     [omega_true_t199, omega_les_t199, omega_2th_t199, omega_4th_t199],
#     "1.0s",
#     methods
# )

# plot_omega_grid(
#     [omega_true_t249, omega_les_t249, omega_2th_t249, omega_4th_t249],
#     "1.25s",
#     methods
# )

# plot_omega_grid(
#     [omega_true_t299, omega_les_t299, omega_2th_t299, omega_4th_t299],
#     "1.5s",
#     methods
# )

# plot_omega_grid(
#     [omega_true_t349, omega_les_t349, omega_2th_t349, omega_4th_t349],
#     "1.75s",
#     methods
# )


for t, time_label in time_map.items():
    try:
        omega_list = [
            data['omega_true'][t],
            data['omega_les'][t],
            data['omega_2th'][t],
            data['omega_4th'][t]
        ]
        
        plot_omega_grid(
            omega_list,
            time_label,
            methods_legend
        )
        
    except KeyError:
        print(f"Skipping Step {t}: Data not found in data dictionary.")

def plot_omega_error_grid(omega_true, omega_methods, t_label, methods,
                          signed=False, cmap_abs='viridis', cmap_signed='RdBu_r'):
    if signed:
        err = [m - omega_true for m in omega_methods]
        cmap = cmap_signed
    else:
        err = [np.abs(m - omega_true) for m in omega_methods]
        cmap = cmap_abs



    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for col, method in enumerate(methods):
        im = axes[col].contourf(X, Y, err[col], cmap=cmap, origin='lower', levels=100)
        axes[col].set_rasterized(True)
        axes[col].set_title(f"{method} — ω error")
        axes[col].axis('off')
        axes[col].set_aspect('equal', adjustable='box')
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"omega_error_grid_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# plot_omega_error_grid(
#     omega_true_t99,
#     [omega_les_t99, omega_2th_t99, omega_4th_t99],
#     "0.5s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

# plot_omega_error_grid(
#     omega_true_149,
#     [omega_les_149, omega_2th_149, omega_4th_149],
#     "0.75s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

# plot_omega_error_grid(
#     omega_true_t199,
#     [omega_les_t199, omega_2th_t199, omega_4th_t199],
#     "1.0s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

# plot_omega_error_grid(
#     omega_true_t249,
#     [omega_les_t249, omega_2th_t249, omega_4th_t249],
#     "1.25s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

# plot_omega_error_grid(
#     omega_true_t299,
#     [omega_les_t299, omega_2th_t299, omega_4th_t299],
#     "1.5s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

# plot_omega_error_grid(
#     omega_true_t349,
#     [omega_les_t349, omega_2th_t349, omega_4th_t349],
#     "1.75s",
#     ["LES", "ANI-2", "ANI-4"],
#     signed=False,
#     cmap_abs='viridis',
#     cmap_signed='RdBu_r'
# )

methods_label = ["LES", "ANI-2", "ANI-4"]
plot_config = {
    'signed': False,
    'cmap_abs': 'viridis',
    'cmap_signed': 'RdBu_r'
}

# for t, time_label in time_map.items():
    
#     try:
#         o_true = data['omega_true'][t]
#         o_preds = [
#             data['omega_les'][t],
#             data['omega_2th'][t],
#             data['omega_4th'][t]
#         ]
        
#         print(f"Generating Omega Error Grid for {time_label} (Step {t})...")
#         plot_omega_error_grid(
#             o_true,
#             o_preds,
#             time_label,
#             methods_label,
#             **plot_config  
#         )
        
#     except KeyError:
#         if t > 349:
#             print(f"Notice: Data for Step {t} ({time_label}) not found in 'data' dictionary. Skipping.")

def compute_kinetic_energy(u, v, L=1.0):
    # u [1, 128, 128]
    # v [1, 128, 128]
    # L = 1.0
    dx = L / u.shape[1]
    dy = L / u.shape[2]
    u = u.squeeze(0).cpu().numpy()
    v = v.squeeze(0).cpu().numpy()
    E = 0.5 * (u**2 + v**2) * dx * dy
    return E

def plot_kinetic_energy(u_list, v_list, t_label, methods):
    E_list = [compute_kinetic_energy(u, v) for u, v in zip(u_list, v_list)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for E, method in zip(E_list, methods):
        ax.plot(E, label=method)
    ax.set_title(f"Kinetic Energy — {t_label}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Kinetic Energy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"kinetic_energy_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# plot_kinetic_energy(
#     [u_true_t99, u_les_t99, u_2th_t99, u_4th_t99],
#     [v_true_t99, v_les_t99, v_2th_t99, v_4th_t99],
#     "1s",
#     ["True", "LES", "ANI-2", "ANI-4"]
# )

def compute_isotropic_spectrum(u, v, w=None, L=1.0, nbins=None):
    """
    计算速度场的各向同性能谱 (修正版)
    适用于 2D (u,v) 或 3D (u,v,w) 速度场
    """
    # ... (前面的代码，如 tensor 转 numpy, FFT 等保持不变) ...
    # 假设 N, L 已经定义
    # 假设 u_hat, v_hat 已经通过 FFT 计算得出
    
    # ------------------- 从这里开始修改 -------------------
    u = u.detach().cpu().numpy() if torch.is_tensor(u) else np.asarray(u)
    v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

    if u.ndim == 3:
        u = u[0]
    if v.ndim == 3:
        v = v[0]

    N = u.shape[0]

    # FFT
    u_hat = np.fft.fft2(u) / N**2
    v_hat = np.fft.fft2(v) / N**2

    # 能量谱密度 (傅里叶空间)
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    # 波数网格
    kx = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2).ravel()
    E_flat = E_hat.ravel()

    # 最大波数 (Nyquist)
    k_max = np.pi * N / L * np.sqrt(2)
    if nbins is None:
        nbins = N // 2
    
    # 分箱
    k_bins = np.linspace(0.0, k_max, nbins + 1)
    dk = k_bins[1] - k_bins[0]

    # bin 内能量总和
    E_k_sum, _ = np.histogram(k_mag, bins=k_bins, weights=E_flat)

    k_values = 0.5 * (k_bins[:-1] + k_bins[1:])
    # shell_area = dk
    # shell_area[shell_area == 0] = 1.0  # 避免除零

    # 各向同性能谱 (密度形式)
    E_k = E_k_sum

    return k_values, E_k

def plot_isotropic_spectrum(u_list, v_list, t_label="t", methods=None, L=1.0):
    """
    绘制专业级的各向同性能量谱图 (使用行内标签，无图例)
    """
    if methods is None:
        methods = [f"method_{i}" for i in range(len(u_list))]

    # 1. 设置样式和颜色
    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#733497', '#CC79A7', '#F0E442', '#56B4E9']

    u_first, v_first = u_list[0], v_list[0]
    k, E_k = compute_isotropic_spectrum(u_first, v_first, L=L)
    
    # 绘制所有方法的数据曲线，并在线条末端添加标签
    for i, (u, v, method) in enumerate(zip(u_list, v_list, methods)):
        k_i, E_k_i = compute_isotropic_spectrum(u, v, L=L)
        plt.loglog(k_i[1:], E_k_i[1:], color=colors[i % len(colors)], linewidth=2, label=method)
        # 在数据曲线的末端添加文字标签
        # plt.text(k_i[-1], E_k_i[-1], f'  {method}', color=colors[i % len(colors)], 
                #  fontsize=12, ha='left', va='center')


    # 3. 智能地绘制参考线并添加行内标签
    # k^-3 参考线
    k_range_3 = np.array([35, 80], dtype=np.float64)
    idx_3 = (np.abs(k - k_range_3[0])).argmin()
    C3 = E_k[idx_3] * (k[idx_3]**3)
    E_ref_3 = C3 * (k_range_3**-3) * 2
    plt.loglog(k_range_3, E_ref_3, '--', color='black', linewidth=2) # 移除 label
    
    # 在 k^-3 线的右上角添加文字
    plt.text(k_range_3[1], E_ref_3[1], r' $k^{-3}$', fontsize=14, color='black', 
             ha='left', va='bottom', rotation=-30) # rotation使标签跟随线条斜率

    # k^-5/3 参考线
    k_range_53 = np.array([15, 35], dtype=np.float64)
    idx_53 = (np.abs(k - k_range_53[0])).argmin()
    C53 = E_k[idx_53] * (k[idx_53]**(5/3))
    E_ref_53 = C53 * (k_range_53**(-5/3)) * 2
    plt.loglog(k_range_53, E_ref_53, '--', color='dimgray', linewidth=2) # 移除 label
    
    # 在 k^-5/3 线的右上角添加文字
    plt.text(k_range_53[1], E_ref_53[1], r' $k^{-5/3}$', fontsize=14, color='dimgray',
             ha='left', va='bottom', rotation=-25) # rotation使标签跟随线条斜率


    # 4. 美化图表细节
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$E(k)$', fontsize=14)
    plt.tick_params(axis='both', which='major')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    
    # 调整坐标轴范围，给右侧的文字留出空间
    plt.xlim(right=plt.xlim()[1] * 1.5)

    plt.tight_layout()
    plt.savefig(f"isotropic_spectrum_{t_label}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# plot_isotropic_spectrum(
#     [u_true_t99, u_les_t99, u_2th_t99, u_4th_t99],
#     [v_true_t99, v_les_t99, v_2th_t99, v_4th_t99],
#     "0.5s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

# plot_isotropic_spectrum(
#     [u_true_149, u_les_149, u_2th_149, u_4th_149],
#     [v_true_149, v_les_149, v_2th_149, v_4th_149],
#     "0.75s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

# plot_isotropic_spectrum(
#     [u_true_t199, u_les_t199, u_2th_t199, u_4th_t199],
#     [v_true_t199, v_les_t199, v_2th_t199, v_4th_t199],
#     "1.0s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

# plot_isotropic_spectrum(
#     [u_true_t249, u_les_t249, u_2th_t249, u_4th_t249],
#     [v_true_t249, v_les_t249, v_2th_t249, v_4th_t249],
#     "1.25s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

# plot_isotropic_spectrum(
#     [u_true_t299, u_les_t299, u_2th_t299, u_4th_t299],
#     [v_true_t299, v_les_t299, v_2th_t299, v_4th_t299],
#     "1.5s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

# plot_isotropic_spectrum(
#     [u_true_t349, u_les_t349, u_2th_t349, u_4th_t349],
#     [v_true_t349, v_les_t349, v_2th_t349, v_4th_t349],
#     "1.75s",
#     ["True", "LES", "ANI-2", "ANI-4"],
#     L=1.0
# )

methods_label = ["True", "LES", "ANI-2", "ANI-4"]

for t, time_label in time_map.items():
    try:
        u_list = [
            data['u_true'][t],
            data['u_les'][t],
            data['u_2th'][t],
            data['u_4th'][t]
        ]
        
        v_list = [
            data['v_true'][t],
            data['v_les'][t],
            data['v_2th'][t],
            data['v_4th'][t]
        ]
        
        print(f"Plotting Isotropic Spectrum for {time_label} (Step {t})...")
        
        plot_isotropic_spectrum(
            u_list,
            v_list,
            time_label,
            methods_label,
            L=1.0
        )
        
    except KeyError:
        print(f"Skipping Step {t}: Data not found in 'data' dictionary.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib import cm
import torch


# ============================================================
# Nature-style global settings
# ============================================================
def mm_to_inch(mm):
    return mm / 25.4


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


COLORS = {
    "True":  "#0072B2",
    "LES":   "#009E73",
    "ANI-2": "#7B4FB3",
    "ANI-4": "#CC79A7",
}


# ============================================================
# Helpers
# ============================================================
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ============================================================
# Panel a: trajectory error
# ============================================================
def load_error_series(path, n=400):
    df = pd.read_csv(path, sep="\t")
    y = df["avg_relative_error"].to_numpy()
    return y[:n]


def plot_error_panel(ax, file_les, file_ani2, file_ani4, n=400):
    y_les  = load_error_series(file_les, n=n)
    y_ani2 = load_error_series(file_ani2, n=n)
    y_ani4 = load_error_series(file_ani4, n=n)

    x = np.arange(len(y_les))

    ax.plot(
        x, y_les,
        label="LES",
        color=COLORS["True"],
        linestyle="-.",
        linewidth=1.0,
    )
    ax.plot(
        x, y_ani2,
        label="ANI-2",
        color=COLORS["LES"],
        linestyle="-",
        linewidth=1.0,
    )
    ax.plot(
        x, y_ani4,
        label="ANI-4",
        color="#E69F00",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average relative error")
    ax.grid(True, linewidth=0.35, alpha=0.30)
    ax.legend(frameon=False, loc="best", handlelength=2.5)

    add_panel_label(ax, "a")


# ============================================================
# Panel b: isotropic spectrum
# ============================================================
def compute_isotropic_spectrum(u, v, L=1.0, nbins=None):
    u = to_numpy(u)
    v = to_numpy(v)

    if u.ndim == 3:
        u = u[0]
    if v.ndim == 3:
        v = v[0]

    N = u.shape[0]

    u_hat = np.fft.fft2(u) / N**2
    v_hat = np.fft.fft2(v) / N**2

    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2).ravel()
    E_flat = E_hat.ravel()

    k_max = np.pi * N / L * np.sqrt(2)
    if nbins is None:
        nbins = N // 2

    k_bins = np.linspace(0.0, k_max, nbins + 1)
    E_k_sum, _ = np.histogram(k_mag, bins=k_bins, weights=E_flat)

    k_values = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_k = E_k_sum

    return k_values, E_k


def add_slope_reference(ax, k, E_ref_base, k_range, power, text, color="black",
                        text_xy=None):
    idx0 = (np.abs(k - k_range[0])).argmin()
    C = E_ref_base[idx0] * (k[idx0] ** power)
    E_ref = C * (k_range ** (-power)) * 2.0

    ax.loglog(
        k_range, E_ref,
        linestyle="--",
        color=color,
        linewidth=1.0,
    )

    if text_xy is None:
        text_xy = (k_range[-1], E_ref[-1])

    ax.text(
        text_xy[0], text_xy[1],
        text,
        fontsize=6.5,
        color=color,
        ha="left",
        va="bottom",
        rotation=-28 if power == 3 else -23,
    )


def plot_spectrum_panel(ax, u_list, v_list, methods, L=1.0):
    for u, v, method in zip(u_list, v_list, methods):
        k, E_k = compute_isotropic_spectrum(u, v, L=L)
        ax.loglog(
            k[1:],
            E_k[1:],
            label=method,
            linewidth=1.0,
            color=COLORS[method] if method in COLORS else None,
        )

    # reference lines use the first curve as anchor
    k0, E0 = compute_isotropic_spectrum(u_list[0], v_list[0], L=L)

    # k^-3
    k_range_3 = np.array([70, 120], dtype=float)
    add_slope_reference(
        ax,
        k0,
        E0,
        k_range_3,
        power=3,
        text=r"$k^{-3}$",
        color="black",
        text_xy=(k_range_3[-1] * 1.02, None if False else (E0[(np.abs(k0-k_range_3[0])).argmin()] * (k0[(np.abs(k0-k_range_3[0])).argmin()]**3) * (k_range_3[-1]**-3) * 2.0 * 0.9))
    )

    # k^-5/3
    k_range_53 = np.array([30, 55], dtype=float)
    idx53 = (np.abs(k0 - k_range_53[0])).argmin()
    C53 = E0[idx53] * (k0[idx53] ** (5 / 3))
    E_ref53 = C53 * (k_range_53 ** (-5 / 3)) * 2.0
    ax.loglog(k_range_53, E_ref53, linestyle="--", color="dimgray", linewidth=1.0)
    ax.text(
        k_range_53[-1] * 1.03,
        E_ref53[-1] * 0.95,
        r"$k^{-5/3}$",
        fontsize=6.5,
        color="dimgray",
        ha="left",
        va="bottom",
        rotation=-22,
    )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.30)
    ax.legend(frameon=False, loc="best", handlelength=2.5)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    add_panel_label(ax, "b")


# ============================================================
# Panel c: omega snapshots
# ============================================================
OMEGA_CMAP = "seismic"
OMEGA_BOX_XY = (0.07, 0.25)
OMEGA_BOX_WIDTH = 0.18
OMEGA_BOX_HEIGHT = 0.18


def plot_omega_row(fig, outer_spec, omega_list, method_titles):
    omega_np = [to_numpy(w) for w in omega_list]
    omega_np = [w[0] if w.ndim == 3 else w for w in omega_np]

    # Use a symmetric colour range; better for vorticity fields.
    absmax = max(np.nanmax(np.abs(w)) for w in omega_np)
    vmin, vmax = -absmax, absmax

    gs = outer_spec.subgridspec(
        1, 5,
        width_ratios=[1, 1, 1, 1, 0.075],
        wspace=0.10,
    )

    axes = []
    im = None

    for i, (w, title) in enumerate(zip(omega_np, method_titles)):
        ax = fig.add_subplot(gs[0, i])

        # im = ax.imshow(
        #     w,
        #     cmap=OMEGA_CMAP,
        #     origin="lower",
        #     vmin=vmin,
        #     vmax=vmax,
        #     interpolation="bilinear",   # smoother than nearest
        #     resample=True,
        #     rasterized=True,
        # )
        im = ax.contourf(
            X,
            Y,
            w,
            levels=100,
            cmap=OMEGA_CMAP,
            vmin=vmin,
            vmax=vmax,
        )
        try:
            for coll in im.collections:
                coll.set_rasterized(True)
        except AttributeError:
            # 新版本 Matplotlib (>= 3.8)
            im.set_rasterized(True)
        # for coll in im.collections:
        #     coll.set_rasterized(True)
        # ax.set_rasterized(True)
        ax.set_title(title, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Axes-fraction coordinates: adjust these constants above to move the box.
        box = plt.Rectangle(
            OMEGA_BOX_XY,
            OMEGA_BOX_WIDTH,
            OMEGA_BOX_HEIGHT,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )
        ax.add_patch(box)

        axes.append(ax)

    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\omega$ value", labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.5, width=0.6, length=2.5, pad=1)

    add_panel_label(axes[0], "c", x=-0.18, y=1.10)


# ============================================================
# Main figure
# ============================================================
def main():
    # --------------------------------------------------------
    # Panel a inputs: files
    # --------------------------------------------------------
    file_ani2 = "2th/traj_error.txt"
    file_ani4 = "4th/traj_error.txt"
    file_les  = "2th/traj_error_les.txt"

    # --------------------------------------------------------
    # Panel b inputs: use your already-prepared arrays
    # Example:
    #   u_true_spec = u_true_t299
    #   v_true_spec = v_true_t299
    #   ...
    # --------------------------------------------------------
    u_true_spec = u_true_t299
    v_true_spec = v_true_t299

    u_les_spec = u_les_t299
    v_les_spec = v_les_t299

    u_ani2_spec = u_2th_t299
    v_ani2_spec = v_2th_t299

    u_ani4_spec = u_4th_t299
    v_ani4_spec = v_4th_t299

    # --------------------------------------------------------
    # Panel c inputs: omega snapshots at 1.0 s
    # Replace omega_step_1s with your own index.
    # --------------------------------------------------------
    omega_step_1s = 199  
    omega_list = [
        data["omega_true"][omega_step_1s],
        data["omega_les"][omega_step_1s],
        data["omega_2th"][omega_step_1s],
        data["omega_4th"][omega_step_1s],
    ]

    omega_titles = [
        r"True - $\omega$",
        r"LES - $\omega$",
        r"ANI-2 - $\omega$",
        r"ANI-4 - $\omega$",
    ]

    # --------------------------------------------------------
    # Build the figure
    # --------------------------------------------------------
    width_mm = 180
    height_mm = 118

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    outer = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 0.78],
        width_ratios=[1.0, 1.0],
        hspace=0.40,
        wspace=0.35,
    )

    # Panel a
    ax_a = fig.add_subplot(outer[0, 0])
    plot_error_panel(
        ax_a,
        file_les=file_les,
        file_ani2=file_ani2,
        file_ani4=file_ani4,
        n=400,
    )

    # Panel b
    ax_b = fig.add_subplot(outer[0, 1])
    plot_spectrum_panel(
        ax_b,
        [u_true_spec, u_les_spec, u_ani2_spec, u_ani4_spec],
        [v_true_spec, v_les_spec, v_ani2_spec, v_ani4_spec],
        ["True", "LES", "ANI-2", "ANI-4"],
        L=1.0,
    )

    # Panel c
    plot_omega_row(fig, outer[1, :], omega_list, omega_titles)

    fig.subplots_adjust(
        left=0.065,
        right=0.955, 
        bottom=0.075,
        top=0.965,
        wspace=0.34,
        hspace=0.42,
    )

    fig.savefig("ns_main_figure_nature.pdf", format="pdf", dpi=600)
    fig.savefig("ns_main_figure_nature.png", format="png", dpi=600)
    plt.close(fig)

    print("Saved: ns_main_figure_nature.pdf")
    print("Saved: ns_main_figure_nature.png")


if __name__ == "__main__":
    main()
