# from ANI import N0
import torch
import os
import numpy as np
import torch.fft
import math
import random
import torch.nn.functional as F
from torch.fft import fft2, ifft2
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A():
    def __init__(self, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0, Re=1e4, dt_step=5e-4, device='cuda', factor=1):
        # super().__init__()
        self.Nx = Nx  # Now represents the fine-grid Nx
        self.Ny = Ny  # Now represents the fine-grid Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re  # Reynolds number
        self.device = device
        self.dt_step = dt_step * factor
        self.factor = factor

        # Grid spacing for the original (fine) resolution
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        x       = torch.linspace(0, Lx, Nx+1, device=device)[:-1]
        y       = torch.linspace(0, Ly, Ny+1, device=device)[:-1]
        self.x, self.y = torch.meshgrid(x, y, indexing='ij')
        # external force -(16\pi)^3 sin(16 \pi y) 10^{-4}
        self.F = -8 * math.pi * torch.cos(8 * math.pi * self.y) * 1e-1
        # self.F = 0
        # self.F = 32 * math.pi**3 * torch.sin(2*math.pi*self.x) * torch.sin(2*math.pi*self.y) * 1e-4
        self._cached_dt_coeffs = None

        # Cache coefficients for the original (fine) grid
        self._cache_coefficients()
        self._cache_dealiasing_mask()

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
        # The (0,0) component corresponds to the mean flow and should not be affected by Laplacian,
        # so it's set to zero to prevent division by zero or infinite values if used in denominators.
        laplacian_k[0, 0] = 0.0

        # Denominator for projection operator and stream function calculation: |k|^2 = k_x^2 + k_y^2
        denom = (Kx**2 + Ky**2)
        
        # Create a safe version of the denominator to avoid division by zero at (0,0) wavenumber.
        # The (0,0) mode will be handled separately (e.g., set to zero by projection).
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

        # Set (0,0) wavenumber components of projection operator to 0.
        # This ensures that the mean flow component remains unaffected and that
        # there are no artifacts from the (0,0) division.
        Pkx_x[0,0] = 0.0
        Pky_y[0,0] = 0.0
        Pkx_y[0,0] = 0.0
        Pky_x[0,0] = 0.0

        return Kx, Ky, laplacian_k, Pkx_x, Pky_y, Pkx_y, Pky_x, denom_safe

    def _cache_coefficients(self):
        (self.Kx_fine, self.Ky_fine, self.laplacian_k_fine, 
         self.Pkx_x_fine, self.Pky_y_fine, self.Pkx_y_fine, self.Pky_x_fine, self.denom_safe_fine) = \
            self._get_spectral_operators(self.Nx, self.Ny, self.Lx, self.Ly, self.device)
        (self.Kx_coarse, self.Ky_coarse, self.laplacian_k_coarse,
         self.Pkx_x_coarse, self.Pky_y_coarse, self.Pkx_y_coarse, self.Pky_x_coarse, self.denom_safe_coarse) = \
            self._get_spectral_operators(self.Nx // self.factor, self.Ny // self.factor, self.Lx, self.Ly, self.device)

    def _cache_dealiasing_mask(self):
        """
        2/3 de-aliasing mask (Orszag-style), built from *linear wavenumber indices*,
        similar to your CuPy implementation:
            kc = (2/3) * (n/2)
            mask = (KX^2 + KY^2) < 1.4 * kc^2
        For Nx != Ny we use an elliptical form.
        """
        Nx_f = self.Nx // self.factor
        Ny_f = self.Ny // self.factor

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

    def non_linear_term(self, omega_real: torch.Tensor, Kx: torch.Tensor, Ky: torch.Tensor, denom_safe: torch.Tensor) -> torch.Tensor:
        """
        Computes the nonlinear advection term, -(u · ∇)ω, and applies de-aliasing.
        This is the "Nonlinear" function 'N(ω)' for the ETDRK4 scheme.
        """
        # Convert vorticity to velocity to compute the advection term
        u_x, u_y = self._vorticity_to_velocity_spectral(omega_real, Kx, Ky, denom_safe)
        
        # Compute derivatives of vorticity: ∇ω
        dom_dx, dom_dy = self._compute_derivatives_spectral(omega_real, Kx, Ky)
        
        # Calculate the nonlinear advection term: -(u_x * dω/dx + u_y * dω/dy) + F
        nonlinear_term_real = -(u_x * dom_dx + u_y * dom_dy) + self.F
        
        # Apply 2/3 de-aliasing filter in Fourier space
        nonlinear_term_k = torch.fft.fft2(nonlinear_term_real)
        
        return nonlinear_term_k * self.dealias_mask.unsqueeze(0)

    def _fourier_resample_2d(self, field_real, target_size):
        B, H, W = field_real.shape
        H_new, W_new = target_size

        # Perform 2D FFT
        field_k = torch.fft.fft2(field_real, norm='ortho')
        
        # Initialize the resampled Fourier field with zeros
        field_k_resampled = torch.zeros(B, H_new, W_new, dtype=field_k.dtype, device=field_k.device)

        # Determine the ranges for copying based on new size.
        # This handles both upsampling (where H_new/W_new > H/W) and downsampling (where H_new/W_new < H/W).
        # Positive frequencies (0 to H_half - 1)
        h_pos_copy = min(H_new // 2, H // 2)
        w_pos_copy = min(W_new // 2, W // 2)
        
        # Negative frequencies (from H - H_neg_copy to H - 1)
        h_neg_copy = min(H_new - H_new // 2, H - H // 2)
        w_neg_copy = min(W_new - W_new // 2, W - W // 2)

        # Copy positive frequency components
        field_k_resampled[:, :h_pos_copy, :w_pos_copy] = field_k[:, :h_pos_copy, :w_pos_copy]
        field_k_resampled[:, :h_pos_copy, -w_neg_copy:] = field_k[:, :h_pos_copy, (W - w_neg_copy):]
        
        # Copy negative frequency components
        field_k_resampled[:, -h_neg_copy:, :w_pos_copy] = field_k[:, (H - h_neg_copy):, :w_pos_copy]
        field_k_resampled[:, -h_neg_copy:, -w_neg_copy:] = field_k[:, (H - h_neg_copy):, (W - w_neg_copy):]

        # Perform inverse 2D FFT to get the real-space resampled field
        field_resampled_real = torch.fft.ifft2(field_k_resampled, norm='ortho').real
        return field_resampled_real

    def _cubic_resample_2d(self, field_real, target_size):
        """
        使用 torch 的 bicubic 插值对 2D 实数场进行重采样。

        参数:
            field_real: [B, H, W] 的实数张量
            target_size: (H_new, W_new)，目标尺寸

        返回:
            field_resampled_real: [B, H_new, W_new] 的重采样张量
        """
        B, H, W = field_real.shape
        H_new, W_new = target_size

        # 需要先变成 [B, 1, H, W] 的形状以使用 interpolate
        field_real = field_real.unsqueeze(1)  # [B, 1, H, W]

        # 使用 bicubic 插值
        field_resampled_real = F.interpolate(field_real, size=(H_new, W_new), mode='bicubic', align_corners=True)

        # 去掉 channel 维度
        return field_resampled_real.squeeze(1)  # [B, H_new, W_new]

    def _compute_etdrk4_coefficients(self, dt):
        """
        Computes the ETDRK4 coefficients based on the linear term (Laplacian)
        and the time step dt using a numerical integration method (M=64).
        """
        L = self.laplacian_k_coarse / self.Re

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

        r = torch.exp(1j * np.pi * (torch.arange(1, M+1, dtype=torch.float64) - 0.5) / M).to(device)
        LR = z_nz + r.unsqueeze(0)

        Q  = torch.mean( (torch.exp(LR / 2.0)-1.0) / LR, dim=-1)
        A_coeffs = torch.mean( (-4.0-LR+torch.exp(LR) * (4.0-3.0*LR+LR**2)) / LR**3, dim=-1)
        B_coeffs = torch.mean( (2.0+LR+torch.exp(LR)*(-2.0+LR)) / LR**3, dim=-1)
        C_coeffs = torch.mean( (-4-3*LR - LR**2 + torch.exp(LR)*(4-LR)) / LR**3, dim=-1)

        return E, E2, Q, A_coeffs, B_coeffs, C_coeffs

    def etdrk4_step(self, omega: torch.Tensor, dt: float) -> torch.Tensor:
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
        N0_k = self.non_linear_term(omega_real, self.Kx_coarse, self.Ky_coarse, self.denom_safe_coarse)

        # === Stage 2 ===
        omega_a_k = E2.unsqueeze(0) * omega_k_n + dt * phi1.unsqueeze(0) * N0_k
        omega_a_real = torch.fft.ifft2(omega_a_k).real
        Na_k = self.non_linear_term(omega_a_real, self.Kx_coarse, self.Ky_coarse, self.denom_safe_coarse)

        # === Stage 3 ===
        omega_b_k = E2.unsqueeze(0) * omega_k_n + dt * phi1.unsqueeze(0) * Na_k
        omega_b_real = torch.fft.ifft2(omega_b_k).real
        Nb_k = self.non_linear_term(omega_b_real, self.Kx_coarse, self.Ky_coarse, self.denom_safe_coarse)

        # === Stage 4 ===
        omega_c_k = E2.unsqueeze(0) * omega_a_k + dt * phi1.unsqueeze(0) * (2*Nb_k - N0_k)
        omega_c_real = torch.fft.ifft2(omega_c_k).real
        Nc_k = self.non_linear_term(omega_c_real, self.Kx_coarse, self.Ky_coarse, self.denom_safe_coarse)

        # === Final Update Step ===
        omega_k_new = (
            E.unsqueeze(0) * omega_k_n +
            dt * (A_coeffs.unsqueeze(0) * N0_k +
                  2.0 * B_coeffs.unsqueeze(0) * (Na_k + Nb_k) +
                  C_coeffs.unsqueeze(0) * Nc_k)
        )
        
        omega_new_real = torch.fft.ifft2(omega_k_new).real
        return omega_new_real.unsqueeze(1)

    def single_step(self, u: torch.Tensor, dts: float = None, dt_max: float = 1e-3) -> torch.Tensor:
        """
        Performs a single simulation step using ETDRK4.
        
        The function takes either vorticity (1 channel) or velocity (2 channels)
        as input, performs the time-stepping on the vorticity, and returns
        the result in the original input format.
        """
        dt_max *= self.factor
        B, C, Nx_fine, Ny_fine = u.shape
        if C not in [1, 2]:
            raise ValueError(f"Input channel count must be 1 (omega) or 2 (u,v), got {C}.")

        # Determine the return type and convert input to vorticity
        if C == 2:
            omega_current = self._velocity_to_vorticity_spectral(u[:, 0], u[:, 1], self.Kx_fine, self.Ky_fine)
            return_type = 'velocity'
        else:
            omega_current = u[:, 0]
            return_type = 'vorticity'
        
        omega_current = self._cubic_resample_2d(omega_current, (Nx_fine // self.factor, Ny_fine // self.factor))

        remaining_dts = dts
        dt_min = 1e-9
        
        if dts is None:
            # Use a fixed maximum time step
            omega_new_tensor = self.etdrk4_step(omega_current.unsqueeze(1), dt_max)
            omega_new = omega_new_tensor.squeeze(1)
        else:
            # Use adaptive time stepping based on CFL condition
            while remaining_dts > dt_min:
                current_dt_step = max(dt_min, min(self.dt_step, dt_max, remaining_dts))
                
                # Perform the ETDRK4 step
                omega_current_tensor = self.etdrk4_step(omega_current.unsqueeze(1), current_dt_step)
                omega_current = omega_current_tensor.squeeze(1)
                
                remaining_dts -= current_dt_step
                
                if torch.isnan(omega_current).any():
                    # raise RuntimeError("NaN detected during time stepping.")
                    return omega_current, False
            
            omega_new = omega_current
        
        omega_new = self._cubic_resample_2d(omega_new, (Nx_fine, Ny_fine))

        # Convert output back to original format
        if return_type == 'vorticity':
            return omega_new.unsqueeze(1), True  # [B, 1, Nx, Ny]
        else:
            u_x_new, u_y_new = self._vorticity_to_velocity_spectral(
                omega_new, self.Kx_fine, self.Ky_fine, self.denom_safe_fine
            )
            u_new = torch.stack([u_x_new, u_y_new], dim=1)  # [B, 2, Nx, Ny]
            return u_new, True

# def init_standard_shear(x, z, Lx, n_shear=2, n_blobs=2, width=1.0):
#     shear = np.zeros((x.shape[0], z.shape[1]), dtype=np.float64)
#     velocity = np.zeros((x.shape[0], z.shape[1]), dtype=np.float64)
#     z_shear = np.linspace(0, 1, n_shear, endpoint=False) + 1/n_shear
#     for i, z1 in enumerate(z_shear):
#         sign = 2 * (i%2) - 1
#         zs = n_shear * (z-z1) / 2 / width
#         shear += sign * 1/2 * np.tanh(zs/0.1)
#         velocity += 1/2 * np.sin(sign*n_blobs*np.pi*x/Lx) * np.exp(-zs**2/0.01)
#     shear += 0.5
#     return shear*2, velocity*2

# def generate_initial_velocity(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0, device='cuda'):
#     x = np.linspace(0, Lx, Nx+1)[:-1]
#     z = np.linspace(0, Ly, Ny+1)[:-1]
#     X_np, Z_np = np.meshgrid(x, z, indexing='ij')

#     # Initialize the output tensor on the specified device
#     velocity_fields = torch.zeros((batch_size, 2, Nx, Ny), device=device)
    
#     # Loop through the batch dimension and apply the init function
#     for i in range(batch_size):
#         # Randomly sample parameters for the current sample
#         w = random.uniform(0.5, 2.0)
#         n_shear = 2 * random.randint(1, 2)
#         n_blobs = random.randint(2, 5)
        
#         # Call the original NumPy init function
#         shear_ux_np, velocity_uy_np = init_standard_shear(X_np, Z_np, Lx, n_shear=n_shear, n_blobs=n_blobs, width=w)
        
#         # Convert NumPy arrays to PyTorch tensors and transfer to the device
#         shear_ux_torch = torch.from_numpy(shear_ux_np).to(device)
#         velocity_uy_torch = torch.from_numpy(velocity_uy_np).to(device)
        
#         vor = A()._velocity_to_vorticity_spectral(shear_ux_torch, velocity_uy_torch, A().Kx_fine, A().Ky_fine)
#         u, v = A()._vorticity_to_velocity_spectral(vor, A().Kx_fine, A().Ky_fine, A().denom_safe_fine)
#         velocity_fields[i, 0, :, :] = u
#         velocity_fields[i, 1, :, :] = v
#     return velocity_fields

# def generate_random_vortices(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0,
#                               n_vortices_range=(10, 20), sigma_range=(0.05, 0.20),
#                               device='cuda', seed=None):
#     """
#     随机生成多个带正负旋涡的二维涡量场，适用于不可压NS方程。
#     - 使用周期边界，支持 batch 生成
#     """
#     if seed is not None:
#         torch.manual_seed(seed)

#     x = torch.linspace(0, Lx, Nx, device=device)
#     y = torch.linspace(0, Ly, Ny, device=device)
#     X, Y = torch.meshgrid(x, y, indexing='ij')  # [Nx, Ny]

#     X = X[None, None, :, :]  # [1, 1, Nx, Ny]
#     Y = Y[None, None, :, :]  # [1, 1, Nx, Ny]

#     omega = torch.zeros(batch_size, 1, Nx, Ny, device=device)

#     for b in range(batch_size):
#         n_vortices = torch.randint(n_vortices_range[0], n_vortices_range[1]+1, (1,)).item()

#         for _ in range(n_vortices):
#             # 随机参数
#             x0 = torch.rand(1).item() * Lx
#             y0 = torch.rand(1).item() * Ly
#             sigma = torch.empty(1).uniform_(*sigma_range).item()
#             sign = 1 if torch.rand(1).item() < 0.5 else -1
#             strength = sign * torch.empty(1).uniform_(5, 10).item()  # 强度范围

#             # 构造距离
#             dx = ((X - x0 + Lx/2) % Lx - Lx/2)
#             dy = ((Y - y0 + Ly/2) % Ly - Ly/2)
#             r2 = dx**2 + dy**2

#             # 加入高斯涡旋
#             omega[b:b+1] += strength * torch.exp(-r2 / (2 * sigma**2))

#     omega_fft = torch.fft.fft2(omega.squeeze(1), norm='ortho')  # [B, Nx, Ny]
#     noise = (torch.randn_like(omega_fft.real) + 1j * torch.randn_like(omega_fft.imag)) * 0.1
#     omega_fft += noise
#     omega_noisy = torch.fft.ifft2(omega_fft, norm='ortho').real.unsqueeze(1)

#     return omega_noisy  # [B, 1, Nx, Ny]

# def generate_initial_velocity(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0, device='cuda'):
#     omega = generate_random_vortices(batch_size, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device)
#     vel   = []
#     model = A()
#     for b in range(batch_size):
#         u, v = model._vorticity_to_velocity_spectral(omega[b:b+1].squeeze(1), model.Kx_fine, model.Ky_fine, model.denom_safe_fine)
#         u_max = torch.max(torch.abs(u))
#         v_max = torch.max(torch.abs(v))
#         u_factor = np.random.uniform(0.5, 1.0) / u_max
#         v_factor = np.random.uniform(0.5, 1.0) / v_max
#         factor   = min(u_factor, v_factor)
#         u = u * factor
#         v = v * factor
#         vel.append(torch.cat([u.unsqueeze(0), v.unsqueeze(0)], dim=1))
#         print(torch.cat([u.unsqueeze(0), v.unsqueeze(0)], dim=1).shape)
#     return torch.cat(vel, dim=0)

def generate_kolmogorov_flow_vorticity(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0,
                                    scales=[(5,10,0.2,0.5),   # 大涡: (min_n, max_n, min_r, max_r)
                                            (10,20,0.05,0.2),  # 中涡
                                            (20,40,0.01,0.05)],# 小涡
                                    max_strength=20.0,
                                    k_kolmogorov=[2,6],   # Kolmogorov 低波数模式
                                    noise_level=0.05,
                                    device='cuda:1'):
    """
    生成复杂多尺度涡 + Kolmogorov流 + 小噪声的初始涡量场
    """
    x = torch.linspace(0, Lx, Nx+1, device=device)[:-1]
    y = torch.linspace(0, Ly, Ny+1, device=device)[:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')  # [Nx, Ny]

    omega = torch.zeros(batch_size, Nx, Ny, device=device)

    for b in range(batch_size):
    #     # --- 多尺度随机涡 ---
    #     for scale in scales:
    #         min_n, max_n, min_r, max_r = scale
    #         n_vortices = torch.randint(min_n, max_n+1, (1,)).item()
    #         for _ in range(n_vortices):
    #             x0 = torch.rand(1, device=device) * Lx
    #             y0 = torch.rand(1, device=device) * Ly
    #             radius = torch.rand(1, device=device) * (max_r - min_r) + min_r
    #             strength = (torch.rand(1, device=device) - 0.5) * 2 * max_strength
    #             r2 = (X - x0)**2 + (Y - y0)**2
    #             omega[b] += strength * torch.exp(-r2 / (2*radius**2))

        # --- Kolmogorov流低波数背景 ---
        for k in k_kolmogorov:
            phase = torch.rand(1, device=device) * 2 * math.pi
            strength = torch.rand(1, device=device) * max_strength
            omega[b] += strength * torch.sin(2*math.pi*k*Y + phase)

        # --- 小噪声扰动 ---
        omega[b] += noise_level * max_strength * torch.randn(Nx, Ny, device=device)

        omega[b] = omega[b] - torch.mean(omega[b])

    return omega.unsqueeze(1)  # [B,1,Nx,Ny]
# def generate_kolmogorov_flow_vorticity(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0,
#                                         min_vortices=10, max_vortices=30,
#                                         max_radius=0.10, max_strength=20.0, device='cuda'):
#     x = torch.linspace(0, Lx, Nx+1, device=device)[:-1]
#     y = torch.linspace(0, Ly, Ny+1, device=device)[:-1]
#     X, Y = torch.meshgrid(x, y, indexing='ij')  # [Nx, Ny]

#     omega = torch.zeros(batch_size, Nx, Ny, device=device)
#     for b in range(batch_size):
#         n_vortices = torch.randint(min_vortices, max_vortices + 1, (1,)).item()
#         for _ in range(n_vortices):
#             x0 = torch.rand(1, device=device) * Lx
#             y0 = torch.rand(1, device=device) * Ly
#             radius = torch.rand(1, device=device) * max_radius * 0.5 + max_radius * 0.5
#             strength = (torch.rand(1, device=device) - 0.5) * 2 * max_strength

#             r2 = (X - x0)**2 + (Y - y0)**2
#             blob = strength * torch.exp(-r2 / (2 * radius**2))
#             omega[b] += blob

#     return omega.unsqueeze(1)  # [B,1,Nx,Ny]

def generate_initial_velocity(batch_size, Nx=1024, Ny=1024, Lx=1.0, Ly=1.0, device='cuda'):
    """
    生成Kolmogorov Flow的初始速度场
    """
    # 使用新的函数生成符合Kolmogorov Flow特性的涡量场
    omega = generate_kolmogorov_flow_vorticity(batch_size, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device)
    
    vel = []
    # 假设 A() 和 _vorticity_to_velocity_spectral() 已经定义并能正确工作
    model = A()
    for b in range(batch_size):
        u, v = model._vorticity_to_velocity_spectral(omega[b:b+1].squeeze(1), model.Kx_fine, model.Ky_fine, model.denom_safe_fine)
        u_max = torch.max(torch.abs(u))
        v_max = torch.max(torch.abs(v))
        # 调整缩放因子以匹配外场强度，确保速度合理
        u_factor = np.random.uniform(0.5, 1.0) / u_max
        v_factor = np.random.uniform(0.5, 1.0) / v_max
        factor = min(u_factor, v_factor)
        u = u * factor
        v = v * factor
        vel.append(torch.cat([u.unsqueeze(0), v.unsqueeze(0)], dim=1))
    return torch.cat(vel, dim=0)

def generate_and_save_dataset(model, batch_size, batch, steps, dt, device, save_dir, split_name):
    os.makedirs(save_dir, exist_ok=True)

    input_list = []
    output_list = []

    batch_idx = 0

    while batch_idx < batch:
        print(f"[{split_name}] Batch {batch_idx+1}/{batch}")

        # 1. 生成初始条件
        u0 = generate_initial_velocity(batch_size, device=device)  # [B, 2, Nx, Ny]
        u = u0.clone().to(device)

        trajectory_list = []
        status = True

        # 2. 生成完整轨迹
        for step in range(steps):
            print(f"  Step {step+1}/{steps}", end='\r')
            trajectory_list.append(u.cpu().numpy()[:,:,::8,::8])  # [B, 2, Nx, Ny]
            # trajectory_list.append(F.avg_pool2d(u, kernel_size=8, stride=8, count_include_pad=False).cpu().numpy())
            u, status = model.single_step(u, dts=dt)
            if status == False:
                break

        if status == False:
            continue

        batch_idx += 1

        # 3. 转置维度 → [B, steps, 2, Nx, Ny]
        trajectory = np.stack(trajectory_list, axis=0)
        trajectory = np.transpose(trajectory, (1, 0, 2, 3, 4))

        # 4. 提取所有时间对
        for traj_idx in range(batch_size):
            for t in range(steps - 1):
                u_n = trajectory[traj_idx, t]
                u_np1 = trajectory[traj_idx, t+1]

                input_list.append(u_n)
                output_list.append(u_np1)

    # 5. 打乱并转换为张量
    input_array = np.stack(input_list).astype(np.float64)
    output_array = np.stack(output_list).astype(np.float64)

    perm = np.random.permutation(len(input_array))
    input_tensor = torch.tensor(input_array[perm], dtype=torch.float64)
    output_tensor = torch.tensor(output_array[perm], dtype=torch.float64)

    # 6. 保存
    torch.save(input_tensor, os.path.join(save_dir, f"{split_name}_input_new.pt"))
    torch.save(output_tensor, os.path.join(save_dir, f"{split_name}_output_new.pt"))
    print(f"[{split_name}] saved {input_tensor.shape[0]} samples.")

def _downsample_omega_to_128(omega: torch.Tensor) -> torch.Tensor:
    B, C, H, W = omega.shape
    if H == 128 and W == 128:
        return omega

    if (H % 128 == 0) and (W % 128 == 0):
        sx = H // 128
        sy = W // 128
        return omega[:, :, ::sx, ::sy]

    return F.interpolate(omega, size=(128, 128), mode="bilinear", align_corners=False)


@torch.no_grad()
def generate_and_save_omega_trajectories(
    model,
    batch_size: int,
    batch: int,
    dt: float,
    device,
    save_dir: str,
    split_name: str,
    T: int = 1000,
    burn_in_seconds: float = 0.0,
    dtype=np.float64,
):
    os.makedirs(save_dir, exist_ok=True)

    # 总轨迹数
    N_target = batch * batch_size
    burn_steps = int(round(burn_in_seconds / dt))
    print(f"[{split_name}] N={N_target}, T={T}, dt={dt}, burn_in={burn_in_seconds}s -> {burn_steps} steps")

    out_path = os.path.join(save_dir, f"{split_name}_omega_traj_{N_target}x{T}x128x128.npy")
    traj_mm = open_memmap(out_path, mode="w+", dtype=dtype, shape=(N_target, T, 128, 128))

    filled = 0
    batch_idx = 0

    while filled < N_target:
        batch_idx += 1
        print(f"[{split_name}] Generating batch {batch_idx} (filled {filled}/{N_target})")

        u0 = generate_initial_velocity(batch_size, device=device) 
        u0 = u0.to(device)

        omega = model._velocity_to_vorticity_spectral(
            u0[:, 0], u0[:, 1], model.Kx_fine, model.Ky_fine
        ).unsqueeze(1)

        status = True

        for s in range(burn_steps):
            omega, status = model.single_step(omega, dts=dt)
            # if status is False:
            #     break
        # if status is False:
        #     print(f"[{split_name}] burn-in failed, resampling...")
        #     continue

        omega_batch_frames = []
        for t in range(T):
            if (t + 1) % 50 == 0 or t == 0:
                print(f"  Save step {t+1}/{T}", end="\r")

            omega_128 = _downsample_omega_to_128(omega)      # [B,1,128,128]
            omega_128 = omega_128[:, 0]                      # [B,128,128]
            omega_batch_frames.append(omega_128.detach().cpu())

            omega, status = model.single_step(omega, dts=dt)
            if status is False:
                break

        # if status is False or len(omega_batch_frames) != T:
        # if len(omega_batch_frames) != T:
        #     print(f"\n[{split_name}] rollout failed mid-way, resampling...")
        #     continue

        omega_batch = torch.stack(omega_batch_frames, dim=1).numpy().astype(dtype, copy=False)

        take = min(batch_size, N_target - filled)
        traj_mm[filled:filled + take] = omega_batch[:take]
        filled += take
        traj_mm.flush()

        print(f"\n[{split_name}] wrote {take} trajectories -> filled {filled}/{N_target}")

    if split_name == "test":
        pt_path = os.path.join(save_dir, "test_trajectory_new.pt")
        traj_mm.flush()
        torch.save(torch.from_numpy(traj_mm), pt_path)
        print(f"[{split_name}] ✅ saved PyTorch trajectory to: {pt_path}")

    print(f"[{split_name}] ✅ saved trajectories to: {out_path}")
    return out_path

if __name__ == '__main__':
    model = A(Nx=1024, Ny=1024, Lx=1.0, Ly=1.0, Re=1e4, device=device)
    dt = 1e-2
    generate_and_save_omega_trajectories(model=model, batch_size=1, batch=40, dt=5e-3, device=device, save_dir='./', split_name='train', T=800)
    # generate_and_save_omega_trajectories(model=model, batch_size=1, batch=4, dt=5e-3, device=device, save_dir='./', split_name='test', T=800)
    # generate_and_save_dataset(model, batch_size=1, batch=32, steps=1000, dt=dt, device=device, save_dir='./', split_name='train')

    # # # 验证集 10个样本，500步
    # generate_and_save_dataset(model, batch_size=1, batch=4, steps=1000, dt=dt, device=device, save_dir='./', split_name='val')

    # generate_test_dataset(model, batch_size=1, batch=4, steps=400, dt=dt, device=device, save_path='test_trajectory_new.pt')
    # vor = model._velocity_to_vorticity_spectral(u[:,0], u[:,1], model.Kx_fine, model.Ky_fine)

    # x = torch.linspace(0, 1, 1024+1)[:-1]
    # y = torch.linspace(0, 1, 1024+1)[:-1]
    # # omega = generate_kolmogorov_flow_vorticity(1, Nx=1024, Ny=1024, Lx=1., Ly=1., device=device)
    # u0 = generate_initial_velocity(1, device=device)
    # # print(u0.shape)
    # omega = model._velocity_to_vorticity_spectral(u0[:, 0], u0[:, 1], model.Kx_fine, model.Ky_fine).unsqueeze(1)

    # for l in range(10):
    #     for i in range(1000):
    #         print(f"{i}/1000", end='\r')
    #         omega, _ = model.single_step(omega, dts=1e-3)
    #     X,Y = torch.meshgrid(x,y)
    #     # plot omega of [1, 1, 1024, 1024]

    #     if hasattr(omega, 'detach'): # 如果是 torch tensor
    #         omega_plot = omega.detach().cpu().numpy().squeeze().T
    #     else: # 如果是 cupy array
    #         omega_plot = omega.get().squeeze().T

    #     # 2. 绘图
    #     plt.figure(figsize=(8, 8))
    #     # 使用 pcolormesh 或 imshow。对于 1024x1024，imshow 性能更好
    #     # cmap='RdBu_r' 是流体可视化中最常用的，红色代表正涡度（逆时针），蓝色代表负涡度（顺时针）
    #     im = plt.imshow(omega_plot, 
    #                     extent=[0, 1, 0, 1], 
    #                     origin='lower', 
    #                     cmap='RdBu_r')

    #     # 3. 细节美化
    #     plt.colorbar(im, label=rf'Vorticity $\omega$')
    #     plt.title(f"Kolmogorov Flow at $t = {1}$")
    #     plt.xlabel("$x$")
    #     plt.ylabel("$y$")

    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig(f'omega_{l}.png', dpi=300)
    #     plt.close()


    # # plot u
    # plt.figure(figsize=(10, 10))
    # plt.contourf(X, Y, vor[0].cpu().numpy(), levels=100, cmap='viridis')
    # plt.colorbar()
    # plt.savefig('vorticity.png')


    # ux  = torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y)
    # uy  = -torch.cos(2*np.pi*X) * torch.sin(2*np.pi*Y)
    # u0 = torch.stack([ux, uy], dim=0).unsqueeze(0)
    # # print(u0.shape)
    # t = 1.0

    # u_out, status = model.single_step(u0.to(device), dts=t)
    # print(status)

    # u_real_x = torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y) 
    # u_real_y = -torch.cos(2*np.pi*X) * torch.sin(2*np.pi*Y)

    # error_x = torch.abs(u_out[0, 0].cpu()-u_real_x.cpu())
    # error_y = torch.abs(u_out[0, 1].cpu()-u_real_y.cpu())

    # print(error_x.abs().max(), error_y.abs().max())

    # x = torch.linspace(0, 1, 1024+1)[:-1]
    # y = torch.linspace(0, 1, 1024+1)[:-1]

    # X,Y = torch.meshgrid(x,y)
    # omega = torch.rand(1, 1, 1024, 1024)
    # omega = omega - torch.mean(omega)
