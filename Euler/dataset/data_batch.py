import numpy as np
import torch
from Euler_Solver import FD, Pri2Con
import os
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === 参数设置 ===
Nx = 256
dt_target = 0.01
batch_size = 16
train_samples = 10000
val_samples = 1000
save_dir = './dataset/burgers_fd'
os.makedirs(save_dir, exist_ok=True)

# def Pri_init(x):  # x 是 [Nx] 的 torch.Tensor（周期网格坐标）
#     Nx = x.shape[0]
#     Ny = x.shape[1] 
#     rho = torch.rand(Nx, Ny, dtype=torch.float64, device=x.device)
#     p   = torch.rand(Nx, Ny, dtype=torch.float64, device=x.device)
#     u   = -2 + 4 * torch.rand(Nx, Ny, dtype=torch.float64, device=x.device)
#     return torch.stack([rho, u, p], dim=2)  # [Nx, 3]

# def Pri_init(x_batch):
#     B, Nx = x_batch.shape
#     device = x_batch.device
#     dtype = x_batch.dtype

#     # 独立对每个样本生成随机 rho, u, p
#     rho = torch.empty(B, Nx, device=device, dtype=dtype).uniform_(0.1, 1.0)
#     u   = torch.empty(B, Nx, device=device, dtype=dtype).uniform_(-1.0, 1.0)
#     p   = torch.empty(B, Nx, device=device, dtype=dtype).uniform_(0.1, 1.0)

#     pri_batch = torch.stack([rho, u, p], dim=2)  # [B, Nx, 3]
#     return pri_batch

def Pri_init(x_batch, p=7):
    """
    CE-RPUI-inspired 1D initial condition generator for batches,
    fully vectorized to avoid explicit loops.

    x_batch: [B, N] tensor of grid points in [0,1) for each batch sample.
             B is batch size, N is number of grid points.
    p: Number of random terms for sigma and groups.

    Returns: [B, N, 3] tensor with (rho, u, p_)
    """
    B, N = x_batch.shape
    device = x_batch.device
    dtype = x_batch.dtype

    # === σ(x) 随机扰动项 ===
    # Generate alpha and beta for each sample in the batch.
    # alpha: [B, p], beta: [B, p]
    alpha = torch.empty(B, p, device=device, dtype=dtype).uniform_(-0.01, 0.01)
    beta = torch.empty(B, p, device=device, dtype=dtype).uniform_(0.0, 1.0)

    # Calculate frequencies.
    # freqs: [p] -> view as [1, 1, p] for broadcasting
    freqs = 2 * torch.pi * (torch.arange(p, device=device, dtype=dtype) + 1 + 2 * p**2)
    freqs_broadcast = freqs.view(1, 1, p) # [1, 1, p]

    # Expand x_batch to [B, N, 1] for broadcasting with freqs and alpha/beta.
    x_batch_expanded = x_batch.unsqueeze(2) # [B, N, 1]

    # Expand alpha and beta to [B, 1, p] for broadcasting.
    alpha_expanded = alpha.unsqueeze(1) # [B, 1, p]
    beta_expanded = beta.unsqueeze(1)   # [B, 1, p]

    # Calculate the argument for sin: (freq * x + 2 * pi * beta)
    # This results in a [B, N, p] tensor, where each (b, n, i) element
    # corresponds to freq_i * x_bn + 2 * pi * beta_bi.
    sin_arg = freqs_broadcast * x_batch_expanded + 2 * torch.pi * beta_expanded # [B, N, p]

    # Multiply by alpha and sum along the 'p' dimension to get sigma.
    # alpha_expanded: [B, 1, p] * sin(sin_arg): [B, N, p] -> [B, N, p]
    # Summing along dim=2 collapses the 'p' dimension, resulting in [B, N].
    sigma = (alpha_expanded * torch.sin(sin_arg)).sum(dim=2) # [B, N]

    # === Fractional part of x + σ(x) ===
    x_perturbed = (x_batch + sigma) % 1.0

    # === 分区编号 ===
    K = p + 1
    # group_ids will be [B, N]
    group_ids = torch.floor(x_perturbed * K).long()

    # === 对每个区域随机初始化物理量 ===
    # For each batch sample, generate K random values for rho, u, p.
    # These are [B, K] tensors.
    rho_vals = torch.empty(B, K, device=device, dtype=dtype).uniform_(0.01, 1.0)
    u_vals   = torch.empty(B, K, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    p_vals   = torch.empty(B, K, device=device, dtype=dtype).uniform_(0.01, 1.0)

    # Use torch.gather to select the correct value for each grid point
    # based on its group_id.
    # For gather, input: [B, K], dim: 1, index: [B, N]
    rho = torch.gather(rho_vals, 1, group_ids) # [B, N]
    print(rho)
    u   = torch.gather(u_vals, 1, group_ids)   # [B, N]
    p_  = torch.gather(p_vals, 1, group_ids)   # [B, N]

    # Stack the results to get the final [B, N, 3] tensor.
    pri_batch = torch.stack([rho, u, p_], dim=2)
    return pri_batch



# === 批量生成一批数据 (inputs, outputs) ===
def generate_batch(batch_size, dt=dt_target):
    solver = FD(Nx, batch_size=batch_size, device=device)
    solver.x_beg = 0
    solver.x_end = 1
    solver.CFD   = 0.8
    solver.t_end = dt

    solver.Init(Pri_init)

    u0 = solver.Get_Cons().cpu().numpy()  # [B, N, 3]

    solver.t_end = dt
    solver.Solve()

    u1 = solver.Get_Cons().cpu().numpy()  # [B, N, 3]

    return u0, u1

# === 生成训练或验证集 ===
def generate_dataset(total_samples, prefix):
    all_inputs = []
    all_outputs = []

    for _ in tqdm(range(0, total_samples, batch_size), desc=f"Generating {prefix}"):
        current_batch = min(batch_size, total_samples - len(all_inputs))
        x, y = generate_batch(current_batch)
        all_inputs.append(x[:,::2,:])
        all_outputs.append(y[:,::2,:])

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    np.save(f"{save_dir}/{prefix}_input.npy", all_inputs)
    np.save(f"{save_dir}/{prefix}_output.npy", all_outputs)
    print(f"{prefix} saved: input {all_inputs.shape}, output {all_outputs.shape}")

# === 生成一个测试轨迹（演化1秒，每0.05s保存一次） ===
def generate_test_trajectory(dt=dt_target, total_time=1.0):
    solver = FD(N=Nx, batch_size=1, device=device)

    # 初始条件
    rho = torch.rand(Nx, dtype=torch.float64, device=device)
    p   = torch.rand(Nx, dtype=torch.float64, device=device)
    u   = -2 + 4 * torch.rand(Nx, dtype=torch.float64, device=device)
    Pri_init = torch.stack([rho, u, p], dim=1)  # [N, 3]

    Ng = solver.Ng
    solver.Pris[:, Ng:Ng+Nx, :] = Pri_init.unsqueeze(0)
    solver.Cons[:, Ng:Ng+Nx, :] = Pri2Con(Pri_init).unsqueeze(0)

    # 保存初始状态
    snapshots = [solver.Get_Cons()[0].cpu().numpy()]  # [128, 3]

    num_steps = int(total_time / dt)
    for _ in tqdm(range(num_steps), desc="Generating test trajectory"):
        solver.t_end = solver.t + dt
        solver.Solve()
        snapshots.append(solver.Get_Cons()[0].cpu().numpy())

    trajectory = np.stack(snapshots[:,::2,:])  # [21, 128, 3]
    np.save(f"{save_dir}/test_trajectory.npy", trajectory)
    print(f"Test trajectory saved: shape = {trajectory.shape}")

# === 主程序入口 ===
if __name__ == "__main__":
    generate_dataset(train_samples, "train")
    generate_dataset(val_samples, "val")
    generate_test_trajectory()