import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from ANI_4th_back import NavierStokes, A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

model = NavierStokes(N0_SCHEME=A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device, Re=1e4), modes1=32, modes2=32, width=64, dt=0.01, device=device).to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))

u_data = np.load("../2th_new/u_99_true.npy") # [128, 128]
v_data = np.load("../2th_new/v_99_true.npy") # [128, 128]
# convert to [1, 2, 128, 128]
state  = np.stack([u_data, v_data], axis=0)[None, ...]
state  = torch.tensor(state, device=device)

# print(state.shape)
with torch.no_grad():
    omega_init, omega_save_1, omega_save_2, omega_save_3, omega_save_4, omega_save_5, omega_save_6, omega_final, result = model(state)
    _, omega_save_1, omega_save_2, omega_save_3, omega_save_4, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, omega_save_1, omega_save_2, omega_save_3, omega_save_4, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, omega_save_1, omega_save_2, omega_save_3, omega_save_4, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, _, omega_save_2, omega_save_3, _, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, _, omega_save_2, omega_save_3, _, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, _, omega_save_2, omega_save_3, _, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, _, omega_save_2, omega_save_3, _, omega_save_5, omega_save_6, omega_final, result = model(result)
    _, _, _, omega_save_3, _, _, omega_save_6, omega_final, result = model(result)
    _, _, _, omega_save_3, _, _, omega_save_6, omega_final, result = model(result)
    _, _, _, omega_save_3, _, _, omega_save_6, omega_final, result = model(result)
    _, _, _, omega_save_3, _, _, omega_save_6, omega_final, result = model(result)
    _, _, _, _, _, _, _, omega_final, result = model(result)
    _, _, _, _, _, _, _, omega_final, result = model(result)
    _, _, _, _, _, _, _, omega_final, result = model(result)
    _, _, _, _, _, _, _, omega_final, result = model(result)

x = np.linspace(0, 1, 128+1)[:-1]
y = np.linspace(0, 1, 128+1)[:-1]
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

v_min = min([omega.min().item() for omega in omegas])
v_max = max([omega.max().item() for omega in omegas])

for i, omega in enumerate(omegas):
    omega_np = omega.squeeze(0).squeeze(0).detach().cpu().numpy()  # [128,128]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(X, Y, omega_np, levels=100, cmap="jet", vmin=v_min, vmax=v_max)

    ax.set_aspect("equal")      # 保持正方形比例
    ax.axis("off")              # 去掉坐标轴和刻度
    for spine in ax.spines.values():
        spine.set_visible(False)  # 去掉边框

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 去掉figure的白边
    fig.savefig(f"omega_{i}.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

