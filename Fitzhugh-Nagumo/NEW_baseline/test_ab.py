import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from model import AllenCahn, A
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

import torch
from torch.nn.functional import mse_loss

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

dt_tensor = torch.tensor(0.01, dtype=torch.float64).to(device)


if __name__ == "__main__":
    # 加载模型
    model = AllenCahn(N0_SCHEME=A(), modes1=8, modes2=8, width=20, dt=0.01).to(device)
    model.load_state_dict(torch.load("models/best_model_new_RK4.pth", map_location=device))
    model.eval()

    # 加载 trajectory 数据
    # traj_data = torch.load("../dataset/allen_cahn_traj.pt").to(device)   # [N, T, 128, 128]
    # epsilons  = torch.load("../dataset/allen_cahn_traj_eps.pt").to(device)  # [N]
    traj_data = np.load('../dataset/data/fhn_ab_grid_traj.npy')
    epsilons  = np.load('../dataset/data/fhn_ab_grid_eps.npy')
    a_list    = np.linspace(0.6, 0.8, 20)
    b_list    = np.linspace(0.7, 0.9, 20)
    # group epsilons by a and b , eps [B, 3] [1:2] is a and b
    epsilons = torch.tensor(epsilons, dtype=torch.float64).to(device)
    traj_data = torch.tensor(traj_data, dtype=torch.float64).to(device)
    unique_pairs = torch.unique(epsilons[:, 1:3], dim=0)

    filename = 'pred_error.txt'

    for a, b in unique_pairs.cpu().numpy():
        print(f"[{a}-{b}]")
        mask = np.isclose(epsilons[:, 1].cpu().numpy(), a) & np.isclose(epsilons[:, 2].cpu().numpy(), b)
        epsilons_a_b = epsilons[mask]
        traj_data_a_b = traj_data[mask]
        with torch.no_grad():
            init_state = traj_data_a_b[:, 0]
            for i in range(100):
                print(f"[{a}-{b}] step: {i}/100", end="\r")
                init_state = model(init_state, epsilons_a_b)
                # init_state = model.N0_SCHEME.single_step(init_state, dts=dt_tensor)
            error = relative_l2_error(init_state, traj_data_a_b[:, -1])
            with open(filename, 'a') as f:
                f.write(f"{a},{b},{error.item():.6f}\n")

    
            