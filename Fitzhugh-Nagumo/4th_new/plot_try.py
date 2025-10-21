import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from ANI_4th import AllenCahn, A
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

model = AllenCahn(N0_SCHEME=A(), modes1=8, modes2=8, width=20, dt=0.01).to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))

# u_data = np.load("../2th_new/u0_true.npy")
# v_data = np.load("../2th_new/v0_true.npy")

# state  = np.stack([u_data, v_data], axis=0)[None, ...]
data = np.load("../dataset/data/fhn_test_trajectory_plot.npy") # (1, 2, 128, 128, 10)
# print(data.shape)
state = data[:, 39]  # (1, 2, 128, 128)
state = torch.tensor(state, device=device)
eps    = torch.tensor([[5.0, 0.75, 0.75]], device=device)
# print(state.shape)
omega_init, omega_save_1, omega_save_2, omega_save_3, omega_save_4, omega_save_5, omega_save_6, omega_final = model(torch.tensor(state, device=device),eps)

x = np.linspace(0, 1, 128)
y = np.linspace(0, 1, 128)
X, Y = np.meshgrid(x, y, indexing='ij')
# plot each omega, 每个单独保存
# 把每个结果 squeeze 并转成 numpy
omegas = [
    omega_init,
    omega_save_1,
    omega_save_2,
    omega_save_3,
    omega_save_4,
    omega_save_5,
    omega_save_6,
    omega_final
]

for i, omega in enumerate(omegas):
    omega_np = omega[0,:,:,1].detach().cpu().numpy()  # [128,128]
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, omega_np, levels=100, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"u_{i}.png", dpi=300)
    plt.close()