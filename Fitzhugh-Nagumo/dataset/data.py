import torch
import torch.fft
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 512
# X, Y = np.meshgrid(x, y, indexing='ij')
x = torch.linspace(0, 1, N+1)[:-1].to(device)
y = torch.linspace(0, 1, N+1)[:-1].to(device)
X, Y = torch.meshgrid(x, y, indexing='ij')

torch.manual_seed(0)
np.random.seed(0)


# ------------------------------
# ✅ 全局缓存类
# ------------------------------
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
            Du=1e-2, Dv=1e-2, a=0.7, b=0.8, eps=0.1,
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
        a   = torch.tensor(a, device=self.device, dtype=u.dtype)
        if a.ndim == 0:
            a = a.expand(B)
        a = a.view(B, 1, 1)
        b = torch.tensor(b, device=self.device, dtype=u.dtype)
        if b.ndim == 0:
            b = b.expand(B)
        b = b.view(B, 1, 1)

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

def sample_init(batch_size, N=64, device='cuda'):
    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')  # [N, N]

    # 扩展成 batch 形式
    X = X.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, N]
    Y = Y.unsqueeze(0).repeat(batch_size, 1, 1)

    # 傅里叶频率限制
    K = 8
    r = torch.rand(batch_size, device=device) * 0.4 + 0.6  # ∈ [0.6, 1.0]
    u0 = torch.zeros(batch_size, N, N, device=device)
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            amp = (torch.randn(batch_size, device=device) * 2).unsqueeze(-1).unsqueeze(-1)
            phase = 2 * torch.pi * torch.rand(batch_size, device=device).unsqueeze(-1).unsqueeze(-1)
            mode = amp * torch.sin(i * torch.pi * X + phase) * torch.sin(j * torch.pi * Y + phase)
            decay = ((i**2 + j**2) ** (-r)).unsqueeze(-1).unsqueeze(-1)
            u0 += decay * mode

    # u0 += 0.1 * torch.randn_like(u0)  # 添加高斯噪声
    return u0


def generate_fitzhugh_nagumo_dataset(
    num_samples=10000,
    num_val=2000,
    batch_size=256,
    N=512,
    Du=1e-2,
    Dv=1e-2,
    fixed_a=(0.7, 0.7),  # Fixed 'a'
    fixed_b=(0.8, 0.8),  # Fixed 'b'
    fixed_dt_solver=5e-4, # Internal solver timestep
    T_save_interval=0.01,  # Time interval between saved snapshots
    eps_range=(4, 6), # Range for 'eps'
    total_time_steps=60, # Total number of snapshots to save per simulation (including initial)
    downsample_factor=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_samples % batch_size != 0:
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Adjusted num_samples to {num_samples} to be a multiple of batch_size.")
    if num_val % batch_size != 0:
        num_val = (num_val // batch_size) * batch_size
        print(f"Adjusted num_val to {num_val} to be a multiple of batch_size.")

    total_dataset_size = num_samples + num_val
    num_batches = total_dataset_size // batch_size

    # Lists to hold all data, broken down into inputs, outputs, and eps
    all_train_inputs = []
    all_train_outputs = []
    all_train_eps = []

    all_val_inputs = []
    all_val_outputs = []
    all_val_eps = []

    solver = FitzHughNagumo2D(N=N, device=device)

    print(f"🚀 Generating FitzHugh-Nagumo time-series data on {device} with batch size {batch_size}, total snapshots {total_time_steps + 1}")

    for i_batch in range(num_batches):
        u_curr = sample_init(batch_size, N=N, device=device)
        v_curr = sample_init(batch_size, N=N, device=device)
        eps_batch = (torch.rand(batch_size, device=device) * (eps_range[1] - eps_range[0]) + eps_range[0])
        a_batch   = (torch.rand(batch_size, device=device) * (fixed_a[1] - fixed_a[0]) + fixed_a[0])
        b_batch   = (torch.rand(batch_size, device=device) * (fixed_b[1] - fixed_b[0]) + fixed_b[0])

        # Initial state before any simulation
        initial_stacked_state = torch.stack([u_curr, v_curr], dim=1).cpu()

        # Simulate for 'total_time_steps' intervals
        for step_idx in range(total_time_steps):
            print(f"Batch {i_batch + 1}/{num_batches}, Simulating step {step_idx + 1}/{total_time_steps}...", end='\r')

            # The current state will be the input for this time step
            # Downsample and store as input
            input_state_crop = initial_stacked_state[:, :, ::downsample_factor, ::downsample_factor]

            # Advance the solution by T_save_interval
            u_next, v_next = solver.solve(
                u0=u_curr,
                v0=v_curr,
                dt=fixed_dt_solver,
                t_end=T_save_interval,
                Du=Du,
                Dv=Dv,
                a=a_batch,
                b=b_batch,
                eps=eps_batch,
                return_all=False
            )

            # Update u_curr, v_curr for the next iteration
            u_curr = u_next.clone()
            v_curr = v_next.clone()

            # The next state will be the output for this time step
            # Stack u_next and v_next, then downsample and store as output
            output_state_stacked = torch.stack([u_next, v_next], dim=1).cpu()
            output_state_crop = output_state_stacked[:, :, ::downsample_factor, ::downsample_factor]

            # Prepare eps tensor for saving (make it [B])
            # eps_to_save = eps_batch.cpu()

            # Append to train or validation lists
            if i_batch * batch_size < num_samples: # Simplified condition based on total_samples
                all_train_inputs.append(input_state_crop)
                all_train_outputs.append(output_state_crop)
                # all_train_eps.append(eps_to_save)
                # stack eps a b into all_train_eps
                all_train_eps.append(torch.stack([eps_batch, a_batch, b_batch], dim=1).cpu())
            else:
                all_val_inputs.append(input_state_crop)
                all_val_outputs.append(output_state_crop)
                # all_val_eps.append(eps_to_save)
                all_val_eps.append(torch.stack([eps_batch, a_batch, b_batch], dim=1).cpu())

            # The output of this step becomes the input for the next step (for sequence generation)
            initial_stacked_state = output_state_stacked


    os.makedirs('data', exist_ok=True)

    def save_split_npy(inputs_list, outputs_list, eps_list, prefix):
        inputs_tensor = torch.cat(inputs_list, dim=0)
        outputs_tensor = torch.cat(outputs_list, dim=0)
        eps_tensor = torch.cat(eps_list, dim=0).cpu().numpy()

        inputs_tensor = inputs_tensor.permute(0, 2, 3, 1).contiguous()
        outputs_tensor = outputs_tensor.permute(0, 2, 3, 1).contiguous()

        inputs_tensor = inputs_tensor.cpu().numpy()
        outputs_tensor = outputs_tensor.cpu().numpy()

        print(f"\nShape of {prefix} inputs: {inputs_tensor.shape}")
        print(f"Shape of {prefix} outputs: {outputs_tensor.shape}")
        print(f"Shape of {prefix} eps: {eps_tensor.shape}")

        # Shuffle the data (optional but good practice)
        perm = np.random.permutation(inputs_tensor.shape[0])
        inputs_tensor = inputs_tensor[perm]
        outputs_tensor = outputs_tensor[perm]
        eps_tensor = eps_tensor[perm]

        np.save(f'data/{prefix}_inputs_all.npy', inputs_tensor)
        np.save(f'data/{prefix}_outputs_all.npy', outputs_tensor)
        np.save(f'data/{prefix}_eps_all.npy', eps_tensor)
        print(f"✅ Saved {prefix} data: {inputs_tensor.shape[0]} samples.")

    # Save the generated data
    save_split_npy(all_train_inputs, all_train_outputs, all_train_eps, 'train_fhn')
    save_split_npy(all_val_inputs, all_val_outputs, all_val_eps, 'val_fhn')

    print("🎉 Dataset generation complete!")

def generate_fitzhugh_nagumo_dataset_dt(
    num_samples=10000,
    num_val=2000,
    batch_size=256,
    N=512,  # Adjusted N to match the solver's default
    fixed_dt=5e-4,
    total_steps=60,
    Du=1e-2,
    Dv=1e-2,
    a_range=0.7,  # Range for 'a'
    b_range=0.8,  # Range for 'b'
    eps_range=5, # Range for 'eps'
    downsample_factor=4 # How much to downsample the output
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_samples % batch_size != 0:
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Adjusted num_samples to {num_samples} to be a multiple of batch_size.")
    if num_val % batch_size != 0:
        num_val = (num_val // batch_size) * batch_size
        print(f"Adjusted num_val to {num_val} to be a multiple of batch_size.")

    total_samples = num_samples + num_val
    num_batches = total_samples // batch_size

    T_choices = [0.01, 0.005, 0.0025] # Time horizons for saving

    all_train_inputs, all_train_outputs = [], []
    all_train_dts = []

    all_val_inputs, all_val_outputs = [], []
    all_val_dts = []

    # Initialize the solver
    solver = FitzHughNagumo2D(N=N, device=device)

    print(f"🚀 Generating FitzHugh-Nagumo data on {device} with batch size {batch_size}, total steps {total_steps}")

    for i_batch in range(num_batches):
        u_curr = sample_init(batch_size, N=N, device=device) # Initial values [B, N, N]
        v_curr = sample_init(batch_size, N=N, device=device) # Initial values [B, N, N]

        for step_idx in range(total_steps):
            print(f"Batch {i_batch + 1}/{num_batches}, Step {step_idx + 1}/{total_steps}", end='\r')

            # Randomly sample current time horizon for this step
            T_save = float(np.random.choice(T_choices))

            input_curr = torch.stack([u_curr, v_curr], dim=1) # Shape: [B, 2, N, N]
            # Advance the solution using the FitzHughNagumo2D solver
            # The solve method handles internal dt steps
            u_next, v_next = solver.solve(
                u0=u_curr,
                v0=v_curr,
                dt=fixed_dt,
                t_end=T_save,
                Du=Du,
                Dv=Dv,
                a=a_range,
                b=b_range,
                eps=eps_range,
                return_all=False # We only need the final state for each T_save
            )

            u_curr = u_next.clone()
            v_curr = v_next.clone()

            # Stack u_next and v_next for the output
            output_next = torch.stack([u_next, v_next], dim=1) # Shape: [B, 2, N, N]

            # Downsample and move to CPU
            input_curr_crop = input_curr[:, :, ::downsample_factor, ::downsample_factor].cpu()
            output_next_crop = output_next[:, :, ::downsample_factor, ::downsample_factor].cpu()

            dt_tensor = torch.full((batch_size,), T_save, dtype=torch.float64)

            if i_batch * batch_size < num_samples:
                all_train_inputs.append(input_curr_crop)
                all_train_outputs.append(output_next_crop)
                all_train_dts.append(dt_tensor)
            else:
                all_val_inputs.append(input_curr_crop)
                all_val_outputs.append(output_next_crop)
                all_val_dts.append(dt_tensor)

    os.makedirs('data_dt', exist_ok=True)

    def save_split(inputs, outputs, dts, prefix):
        inputs_tensor = torch.cat(inputs, dim=0)
        outputs_tensor = torch.cat(outputs, dim=0)
        dts_tensor = torch.cat(dts, dim=0)

        inputs_tensor = inputs_tensor.permute(0, 2, 3, 1).contiguous()
        outputs_tensor = outputs_tensor.permute(0, 2, 3, 1).contiguous()

        print(f"\nShape of {prefix} inputs: {inputs_tensor.shape}")
        print(f"Shape of {prefix} outputs: {outputs_tensor.shape}")
        print(f"Shape of {prefix} dts: {dts_tensor.shape}")

        perm = torch.randperm(inputs_tensor.shape[0])
        inputs_tensor = inputs_tensor[perm].cpu().numpy()
        outputs_tensor = outputs_tensor[perm].cpu().numpy()
        dts_tensor = dts_tensor[perm].cpu().numpy()

        # np.save(inputs_tensor, f'data_dt/{prefix}_inputs.npy')
        # np.save(outputs_tensor, f'data_dt/{prefix}_outputs.npy')
        # np.save(dts_tensor, f'data_dt/{prefix}_dt.npy')
        np.save(f'data_dt/{prefix}_inputs.npy', inputs_tensor)
        np.save(f'data_dt/{prefix}_outputs.npy', outputs_tensor)
        np.save(f'data_dt/{prefix}_dt.npy', dts_tensor)
        print(f"✅ Saved {prefix} data: {inputs_tensor.shape[0]} samples.")

    save_split(all_train_inputs, all_train_outputs,
               all_train_dts, 'train')
    save_split(all_val_inputs, all_val_outputs,
               all_val_dts, 'val')

    print("🎉 Dataset generation complete!")

def generate_fhn_trajectory_dataset(
    num_samples=100,
    batch_size=10,
    N=128,
    Du=1e-2,
    Dv=1e-2,
    fixed_a=(0.7, 0.7),
    fixed_b=(0.8, 0.8),
    fixed_dt_solver=5e-4,
    t_save=0.01,
    total_steps=30,
    eps_range=(4, 6),
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_samples % batch_size != 0:
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Adjusted num_samples to {num_samples} to be a multiple of batch_size.")

    num_batches = num_samples // batch_size

    solver = FitzHughNagumo2D(N=N, device=device)

    all_trajectories = []
    all_eps = []

    for i_batch in range(num_batches):
        # 初始状态采样
        u_curr = sample_init(batch_size, N=N, device=device)
        v_curr = sample_init(batch_size, N=N, device=device)
        eps_batch = (torch.rand(batch_size, device=device) * (eps_range[1] - eps_range[0]) + eps_range[0])
        a_batch   = (torch.rand(batch_size, device=device) * (fixed_a[1] - fixed_a[0]) + fixed_a[0])
        b_batch   = (torch.rand(batch_size, device=device) * (fixed_b[1] - fixed_b[0]) + fixed_b[0])
        # eps_batch = torch.full((batch_size,), 3.0 + i_batch, dtype=torch.float64, device=device)

        # 预先分配数组保存轨迹：shape = (batch_size, total_steps, N, N, 2)
        traj_batch = torch.zeros(batch_size, total_steps, N, N, 2, dtype=torch.float64)

        for step in range(total_steps):
            # 保存当前状态
            traj_batch[:, step, :, :, 0] = u_curr.cpu()
            traj_batch[:, step, :, :, 1] = v_curr.cpu()

            # 向前推进一步
            u_next, v_next = solver.solve(
                u0=u_curr,
                v0=v_curr,
                dt=fixed_dt_solver,
                t_end=t_save,
                Du=Du,
                Dv=Dv,
                a=a_batch,
                b=b_batch,
                eps=eps_batch,
                return_all=False
            )

            u_curr = u_next
            v_curr = v_next

        all_trajectories.append(traj_batch[:, :, ::4, ::4, :])
        # all_eps.append(eps_batch.cpu())
        all_eps.append(torch.stack([eps_batch, a_batch, b_batch], dim=1).cpu())

        print(f"Batch {i_batch+1}/{num_batches} done.", end='\r')

    all_trajectories = torch.cat(all_trajectories, dim=0).numpy()  # (N, 30, 128, 128, 2)
    all_eps = torch.cat(all_eps, dim=0).numpy()                    # (N,)

    os.makedirs('data', exist_ok=True)
    np.save('data/fhn_test_trajectory_plot.npy', all_trajectories)
    np.save('data/fhn_test_eps_plot.npy', all_eps)

    print(f"\nSaved test trajectory dataset with shape {all_trajectories.shape}")
    print(f"Saved eps array with shape {all_eps.shape}")

def generate_fhn_ab_grid_dataset(
    N=128,
    Du=1e-2,
    Dv=1e-2,
    fixed_dt_solver=5e-4,
    t_save=0.01,
    T_target=1.0,
    eps_value=5.5,
    a_values=np.linspace(0.6, 0.8, 20),
    b_values=np.linspace(0.7, 0.9, 20),
    num_inits=4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver = FitzHughNagumo2D(N=N, device=device)

    ab_pairs = list(product(a_values, b_values))   # 25 个 (a,b) 组合
    all_trajectories = []
    all_eps = []

    for (a_val, b_val) in ab_pairs:
        print(f"Processing (a, b) = ({a_val}, {b_val})...")
        for k in range(num_inits):
            # 初值
            u_curr = sample_init(1, N=N, device=device)
            v_curr = sample_init(1, N=N, device=device)

            # 保存 t=0
            traj = torch.zeros(1, 2, N, N, 2, dtype=torch.float64)  
            traj[0, 0, :, :, 0] = u_curr.cpu()
            traj[0, 0, :, :, 1] = v_curr.cpu()

            # 时间推进到 t=1
            u_next, v_next = solver.solve(
                u0=u_curr,
                v0=v_curr,
                dt=fixed_dt_solver,
                t_end=T_target,
                Du=Du,
                Dv=Dv,
                a=torch.tensor([a_val], device=device),
                b=torch.tensor([b_val], device=device),
                eps=torch.tensor([eps_value], device=device),
                return_all=False
            )

            # 保存 t=1
            traj[0, 1, :, :, 0] = u_next.cpu()
            traj[0, 1, :, :, 1] = v_next.cpu()

            all_trajectories.append(traj[:, :, ::4, ::4, :])  # 下采样 4 倍
            all_eps.append(torch.tensor([[eps_value, a_val, b_val]], dtype=torch.float64))

    all_trajectories = torch.cat(all_trajectories, dim=0).numpy()   # (100, 2, N/4, N/4, 2)
    all_eps = torch.cat(all_eps, dim=0).numpy()                     # (100, 3)

    os.makedirs('data', exist_ok=True)
    np.save('data/fhn_ab_grid_traj.npy', all_trajectories)
    np.save('data/fhn_ab_grid_eps.npy', all_eps)

    print(f"Saved trajectories: {all_trajectories.shape}")  # (100, 2, N/4, N/4, 2)
    print(f"Saved eps array: {all_eps.shape}")              # (100, 3)

if __name__ == '__main__':
    # generate_fitzhugh_nagumo_dataset(num_samples=128, num_val=64, batch_size=64, N=512, eps_range=(4, 6), fixed_a=(0.6, 0.8), fixed_b=(0.7, 0.9))
    # generate_fitzhugh_nagumo_dataset_dt(num_samples=128, num_val=64, batch_size=64, N=512)
    # generate_fhn_trajectory_dataset(num_samples=1, batch_size=1, N=512, eps_range=(5., 5.), total_steps=100, fixed_a=(0.75, 0.75), fixed_b=(0.75, 0.75))
    generate_fhn_ab_grid_dataset(N=512)
    # u = sample_init(1, N=512, device=device)
    # v = sample_init(1, N=512, device=device)
    # print(u.shape, v.shape)
    # eps = 5
    # Solver = FitzHughNagumo2D(N=512, device=device)
    # u1, v1 = Solver.solve(u, v, dt=5e-4, t_end=0.5, Du=1e-2, Dv=1e-2, a=0.7, b=0.8, eps=eps, return_all=False)
    # # print(u.shape, v.shape)
    # u2, v2 = Solver.solve(u, v, dt=5e-4, t_end=0.5+1e-2, Du=1e-2, Dv=1e-2, a=0.7, b=0.8, eps=eps+2, return_all=False)
    # error_u = torch.abs(u1-u2)
    # error_v = torch.abs(v1-v2)
    # # plot u, v
    # x = torch.linspace(0, 1, 512+1)[:-1]
    # y = torch.linspace(0, 1, 512+1)[:-1]
    # X, Y = torch.meshgrid(x, y)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # contour = ax.contourf(X, Y, error_u[0, :, :].cpu().numpy(), cmap='coolwarm', levels=50)  # 保存contourf的返回值
    # ax.set_aspect('equal')
    # plt.colorbar(contour)  # 将contourf的返回值传递给colorbar
    # plt.savefig('test2.png')