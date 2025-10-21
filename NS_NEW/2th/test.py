# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from ANI import N0, ANIBASE, ResidualBlockWithT, MLP, FNO2d
# from ANI_NS_2th import ODEPairDataset
# from ANI_NS_2th import NavierStokes, A  # 假设这些类已经按原来定义好了
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# np.random.seed(int(time.time()))

# torch.set_default_dtype(torch.float64)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# train_input = torch.load('../dataset/val_input.pt')
# train_output = torch.load('../dataset/val_output.pt')

# # random_idx = 422
# random_idx = np.random.randint(len(train_input), size=1)[0]
# # print(f"random_idx: {random_idx}")

# u_random_input = train_input[random_idx:random_idx+1].clone().to(device).permute(0, 3, 1, 2)
# u_output       = train_output[random_idx:random_idx+1].clone().to(device).permute(0, 3, 1, 2)

# A_1 = A(device=device, downsample_factor=32)
# n=2
# dt_tensor = torch.tensor(1e-1/n, dtype=torch.float64, device=device)

# for _ in range(n):
#     u_random_input = A_1.single_step(u_random_input, dts=dt_tensor)
#     error = torch.abs(u_random_input -  u_output).squeeze(0).permute(1,2,0)
#     print(f"norm of error is {error.norm()}")

# # plot error[:,:,0] 和 error[:,:,1], 二维，两张图


# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(error[:,:,0].cpu().detach().numpy(), cmap='coolwarm')
# plt.colorbar()
# plt.title('Error in u')

# plt.subplot(1, 2, 2)
# plt.imshow(error[:,:,1].cpu().detach().numpy(), cmap='coolwarm')
# plt.colorbar()
# plt.title('Error in v')

# plt.savefig(f'error_ns.png')

# import torch
# import os
# import random
# from ANI_NS_2th import NavierStokes, A, ODEPairDataset  # 假设你有这些类
# import matplotlib.pyplot as plt
# import numpy as np

# @torch.no_grad()
# def test_one_sample(model_path="models/best_model.pth", data_dir="../dataset", device="cuda:2"):
#     # 加载验证集
#     val_dataset = ODEPairDataset(
#         os.path.join(data_dir, "val_input.pt"),
#         os.path.join(data_dir, "val_output.pt"),
#     )

#     # 随机选一个样本
#     idx = random.randint(0, len(val_dataset) - 1)
#     inputs, targets = val_dataset[idx]
#     inputs = inputs.unsqueeze(0).to(device).permute(0, 3, 1, 2)   # [1, N, C]
#     targets = targets.unsqueeze(0).to(device).permute(0, 3, 1, 2) # [1, N, C]

#     # 构建模型
#     model = NavierStokes(N0_SCHEME=A(device=device), modes1=8, modes2=8, width=32, dt=0.1, device=device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # 预测
#     preds = model(inputs)

#     # 计算误差
#     l2_error = torch.norm(preds - targets).item()
#     relative_error = l2_error / torch.norm(targets).item()

#     print(f"Test sample idx: {idx}")
#     print(f"L2 error      : {l2_error:.4e}")
#     print(f"Relative error: {relative_error:.4%}")

#     # [Nx, Ny, 2]
#     preds_np = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     targets_np = targets.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     abs_err = np.abs(preds_np - targets_np)

#     plt.figure(figsize=(10, 6))
#     for i, name in enumerate(["$u_x$", "$u_y$"]):
#         plt.subplot(2, 2, 2 * i + 1)
#         plt.imshow(preds_np[:, :, i], cmap="jet", origin="lower")
#         plt.colorbar()
#         plt.title(f"Pred {name}")

#         plt.subplot(2, 2, 2 * i + 2)
#         plt.imshow(abs_err[:, :, i], cmap="hot", origin="lower")
#         plt.colorbar()
#         plt.title(f"Error {name}")

#     plt.tight_layout()
#     plt.savefig("test_sample_pred_and_error.png")
#     plt.close()

# test_one_sample()

import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from ANI_NS_2th import NavierStokes, A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import torch.fft as fft
import time

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

class LES():
    def __init__(self, Nx=256, Ny=512, Lx=1.0, Ly=2.0, Re=1e4, Cs=0.18, device='cuda:2'):
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
        # self.F = -8 * math.pi * torch.cos(8 * math.pi * self.y) * 1e-1
        self.F = torch.zeros_like(self.y)
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

dt_tensor = 0.01/2

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

@torch.no_grad()
def evaluate_on_trajectory(model, init_states, true_trajs, dt=0.1):
    model.eval()
    T = true_trajs.shape[1]
    batch_size = init_states.shape[0]

    curr_states = init_states.clone()
    pred_trajs = []

    # A1 = A(device=device)
    A1 = LES(Nx=256, Ny=256, Lx=1, Ly=1)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for t in range(T):
        # print(f"t = {t}/{T-1}", end='\r')
        # eps_batch = epsilons.view(-1, 1)  # [4, 1]
        next_states = model.predict(curr_states)  # [4, 128, 128]
        # u, v = curr_states[:, 0], curr_states[:, 1]
        # next_states = A1._velocity_to_vorticity_spectral(u, v, A1.Kx_fine, A1.Ky_fine).unsqueeze(1)
        # for i in range(2):
        #     next_states = A1.single_step(next_states, dts=dt_tensor)
        # next_states = next_states.squeeze(1)
        # u, v = A1._vorticity_to_velocity_spectral(next_states, A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine)
        # next_states = torch.stack([u, v], dim=1)
        # if t == 0:
            # error = (next_states - true_trajs[:, t+1])
            # error = true_trajs[:, 0] 
            # error = true_trajs[:, 99] 
        # pred_trajs.append(next_states)
        curr_states = next_states

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")

    # pred_trajs = torch.stack(pred_trajs, dim=1)  # [4, T, 128, 128]

    # u_99 = pred_trajs[0, 99, 0, :, :]
    # v_99 = pred_trajs[0, 99, 1, :, :]

    # u_149 = pred_trajs[0, 149, 0, :, :]
    # v_149 = pred_trajs[0, 149, 1, :, :]

    # u_199 = pred_trajs[0, 199, 0, :, :]
    # v_199 = pred_trajs[0, 199, 1, :, :]

    # u_249 = pred_trajs[0, 249, 0, :, :]
    # v_249 = pred_trajs[0, 249, 1, :, :]

    # u_299 = pred_trajs[0, 299, 0, :, :]
    # v_299 = pred_trajs[0, 299, 1, :, :]

    # u_349 = pred_trajs[0, 349, 0, :, :]
    # v_349 = pred_trajs[0, 349, 1, :, :]

    # np.save('u_99_2th.npy', u_99.cpu().numpy())
    # np.save('v_99_2th.npy', v_99.cpu().numpy())
    # np.save('u_149_2th.npy', u_149.cpu().numpy())
    # np.save('v_149_2th.npy', v_149.cpu().numpy())
    # np.save('u_199_2th.npy', u_199.cpu().numpy())
    # np.save('v_199_2th.npy', v_199.cpu().numpy())
    # np.save('u_249_2th.npy', u_249.cpu().numpy())
    # np.save('v_249_2th.npy', v_249.cpu().numpy())
    # np.save('u_299_2th.npy', u_299.cpu().numpy())
    # np.save('v_299_2th.npy', v_299.cpu().numpy())
    # np.save('u_349_2th.npy', u_349.cpu().numpy())
    # np.save('v_349_2th.npy', v_349.cpu().numpy())

    # np.save('u_99_true.npy', true_trajs[0, 100, 0, :, :].cpu().numpy())
    # np.save('v_99_true.npy', true_trajs[0, 100, 1, :, :].cpu().numpy())
    # np.save('u_149_true.npy', true_trajs[0, 150, 0, :, :].cpu().numpy())
    # np.save('v_149_true.npy', true_trajs[0, 150, 1, :, :].cpu().numpy())
    # np.save('u_199_true.npy', true_trajs[0, 200, 0, :, :].cpu().numpy())
    # np.save('v_199_true.npy', true_trajs[0, 200, 1, :, :].cpu().numpy())
    # np.save('u_249_true.npy', true_trajs[0, 250, 0, :, :].cpu().numpy())
    # np.save('v_249_true.npy', true_trajs[0, 250, 1, :, :].cpu().numpy())
    # np.save('u_299_true.npy', true_trajs[0, 300, 0, :, :].cpu().numpy())
    # np.save('v_299_true.npy', true_trajs[0, 300, 1, :, :].cpu().numpy())
    # np.save('u_349_true.npy', true_trajs[0, 350, 0, :, :].cpu().numpy())
    # np.save('v_349_true.npy', true_trajs[0, 350, 1, :, :].cpu().numpy())


    # avg_l2_per_t = []
    # avg_rel_per_t = []

    # for t in range(T-1):
    #     l2s = []
    #     rels = []
    #     for i in range(batch_size):
    #         l2 = mse_loss(pred_trajs[i, t], true_trajs[i, t+1])
    #         rel = relative_l2_error(pred_trajs[i, t], true_trajs[i, t+1])
    #         l2s.append(l2.item())
    #         rels.append(rel.item())
    #     avg_l2_per_t.append(sum(l2s) / batch_size)
    #     avg_rel_per_t.append(sum(rels) / batch_size)
    
    # # plot error [B,2,256,512], so plot u, v
    # print(error.norm())
    # u = error[0, 0]
    # v = error[0, 1]
    # x = np.linspace(0, 1, 256, endpoint=False)
    # y = np.linspace(0, 1, 256, endpoint=False)
    # X, Y = np.meshgrid(x, y)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # # plt.imshow(u.cpu().numpy(), cmap="jet", origin="lower")
    # plt.contourf(X, Y, u.cpu().numpy().T, cmap="jet", levels=100)
    # plt.colorbar()
    # plt.title('Error in u')

    # plt.subplot(1, 2, 2)
    # # plt.imshow(v.cpu().numpy(), cmap="jet", origin="lower")
    # plt.contourf(X, Y, v.cpu().numpy().T, cmap="jet", levels=100)
    # plt.colorbar()
    # plt.title('Error in v')

    # plt.tight_layout()
    # plt.savefig("error_A.png")
    # plt.close()

    # return avg_l2_per_t, avg_rel_per_t, pred_trajs

if __name__ == "__main__":
    # 加载模型
    model = NavierStokes(N0_SCHEME=A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, device=device, Re=1e4), modes1=32, modes2=32, width=64, dt=0.01, device=device).to(device)
    model.load_state_dict(torch.load("models/best_model_fno.pth", map_location=device))

    # 加载 trajectory 数据
    traj_data = torch.load("../dataset/test_trajectory.pt").to(device).to(torch.float64)   # [N, T, 128, 128]

    # init_states = traj_data[:, 0].clone().permute(0, 3, 1, 2)     # [4, 128, 128]
    # true_trajs  = traj_data[:].clone().permute(0, 1, 4, 2, 3)       # [4, T, 128, 128]
    init_states = traj_data[:, 0].clone()     # [4, 128, 128]
    true_trajs  = traj_data[:].clone()       # [4, T, 128, 128]

    print(init_states.shape, true_trajs.shape)

    evaluate_on_trajectory(model, init_states, true_trajs)

    # avg_l2_per_t, avg_rel_per_t, pred_trajs = evaluate_on_trajectory(model, init_states, true_trajs)

    # # 写入文件，每行写一个时间点的误差
    # with open("traj_error.txt", "w") as f:
    #     f.write("t_index\tavg_L2_error\tavg_relative_error\n")
    #     for t, (l2, rel) in enumerate(zip(avg_l2_per_t, avg_rel_per_t)):
    #         f.write(f"{t}\t{l2:.6e}\t{rel:.6e}\n")

    # print("✅ Trajectory error evaluation done.")

    # B, T, _, H, W = true_trajs.shape

    # 假设已有 true_trajs [B, T, 2, H, W]
    # u_seq = true_trajs[0, :, 0].cpu().numpy()  # [T, H, W]
    # v_seq = true_trajs[0, :, 1].cpu().numpy()  # [T, H, W]
    # u_seq = pred_trajs[0, :, 0].cpu().numpy()  # [T, H, W]
    # v_seq = pred_trajs[0, :, 1].cpu().numpy()  # [T, H, W]
    # speed_seq = np.sqrt(u_seq**2 + v_seq**2)  # [T, H, W]

    # T, H, W = speed_seq.shape
    # x = np.linspace(0, 1, H, endpoint=False)
    # y = np.linspace(0, 1, W, endpoint=False)
    # X, Y = np.meshgrid(x, y)

    # fig, ax = plt.subplots(figsize=(6, 5))
    # cf = ax.contourf(X, Y, speed_seq[0].T, levels=100, cmap='viridis', vmin=0, vmax=1)
    # cbar = fig.colorbar(cf, ax=ax)
    # cbar.set_label('Speed')


    # def update(frame):
    #     ax.clear()
    #     cf = ax.contourf(X, Y, speed_seq[frame].T, levels=100, cmap='viridis', vmin=0, vmax=1)
    #     ax.set_title(f'Time step: {frame}')
    #     return []

    # ani = animation.FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    # ani.save("pred_traj.gif", writer='pillow')
    # plt.close()

    # u_last = true_trajs[0, 10, 0].cpu().numpy()  # [H, W]
    # v_last = true_trajs[0, 10, 1].cpu().numpy()  # [H, W]

    # # 创建坐标网格
    # x = np.linspace(0, 1, H, endpoint=False)
    # y = np.linspace(0, 1, W, endpoint=False)
    # X, Y = np.meshgrid(x, y)

    # # 创建绘图
    # fig, ax = plt.subplots(figsize=(8, 8))

    # # 绘制流线
    # # 我们使用 (X, Y) 网格和 u, v 速度分量来绘制流线
    # ax.streamplot(X, Y, u_last.T, v_last.T, density=1.5)

    # # 添加颜色条来表示速度大小
    # # 为了颜色条，我们必须先创建一个可映射对象（mappable）
    # # 这里我们用一个临时的 pcolormesh 来创建它
    # speed_last = np.sqrt(u_last.T**2 + v_last.T**2)
    # mappable = ax.pcolormesh(X, Y, speed_last, cmap='viridis', shading='auto')
    # cbar = fig.colorbar(mappable, ax=ax)
    # cbar.set_label('Speed')

    # ax.set_title(f'Streamlines at time step: {T-1}')
    # ax.set_xlabel('X-coordinate')
    # ax.set_ylabel('Y-coordinate')
    # # plt.show()


