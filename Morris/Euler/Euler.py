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
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A(N0):
    def __init__(self, mu, sigma):
        super(A, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def F(self, x: torch.Tensor):
        # 反标准化（矢量化方式）
        x = x * self.sigma + self.mu  # shape: (..., 4)
        V = x[:, 0]
        w = x[:, 1]

        C = 20.0     # Membrane capacitance
        gL = 2.0     # Leak conductance
        VL = -60.0   # Leak reversal potential
        gCa = 4.0    # Calcium conductance
        VCa = 120.0  # Calcium reversal potential
        gK = 8.0     # Potassium conductance
        VK = -84.0   # Potassium reversal potential
        v1 = -1.2    # Half-activation voltage for m_infinity
        v2 = 18.0    # Slope for m_infinity
        v3 = 12.0    # Half-activation voltage for w_infinity
        v4 = 17.4    # Slope for w_infinity
        phi = 0.066  # Activation time constant factor
        I_app = 60.0

        m_infinity = 0.5 * (1 + torch.tanh((V - v1) / v2)) 

        # Equilibrium open fraction for K+ current (w_infinity)
        # w_infinity = 0.5 * [1 + tanh((V - v3) / v4)]
        w_infinity = 0.5 * (1 + torch.tanh((V - v3) / v4))

        # Activation time constant for the delayed rectifier (tau)
        # tau = 1 / cosh((V - v3) / (2 * v4))
        tau = 1 / torch.cosh((V - v3) / (2*v4))
        # Add a small epsilon to tau to prevent division by zero if cosh evaluates to 0
        tau = tau + 1e-9 # Small epsilon to prevent potential division by zero if tau gets too small

        # --- Calculate the derivatives ---

        # dV/dt equation
        # dV/dt = (-gCa * (V - VCa) * m_infinity - gK * (V - VK) * w - gL * (V - VL) + I_app) / C
        dVdt = (-gCa * (V - VCa) * m_infinity - gK* (V - VK) * w - gL * (V - VL) + I_app) / C

        # dw/dt equation
        # dw/dt = phi * (w_infinity - w) / tau
        dwdt = phi * (w_infinity - w) / tau

        # Stack the outputs [N, 2]
        dxdt = torch.stack([dVdt, dwdt], dim=1)

        return dxdt / self.sigma

    def single_step(self, u: torch.Tensor, parameters=None, t=None, dt=None, mu=None, sigma=None) -> torch.Tensor:
        # k1 = self.F(u)
        # k2 = self.F(u + dt/2.0 * k1)
        # k3 = self.F(u + dt/2.0 * k2)
        # k4 = self.F(u + dt * k3)
        return u + dt * self.F(u)
        # return u + dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None):
        raw_inputs  = np.load(input_file)[:20000, :]
        raw_outputs = np.load(output_file)[:20000, :]

        num_vars = raw_outputs.shape[1]

        if mu is None or sigma is None:
            # 只标准化状态变量部分（raw_inputs 的前 num_vars 列）
            x_min = raw_inputs[:, :num_vars].min(axis=0)
            x_max = raw_inputs[:, :num_vars].max(axis=0)

            mu = (x_min + x_max) / 2.0
            sigma = (x_max - x_min) / 2.0

        # 标准化 inputs（仅状态变量部分）
        normalized_inputs = raw_inputs.copy()
        normalized_inputs[:, :num_vars] = (raw_inputs[:, :num_vars] - mu) / sigma
        # dt（最后一列）保持不变

        # 标准化 outputs（全部是状态变量）
        normalized_outputs = raw_outputs.copy()
        normalized_outputs[:, :num_vars] = (raw_outputs[:, :num_vars] - mu) / sigma

        self.inputs = torch.from_numpy(normalized_inputs).to(device)
        self.outputs = torch.from_numpy(normalized_outputs).to(device)

        self.mu = torch.tensor(mu, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, device=device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
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


if __name__ == "__main__":
    # # 首先加载 train dataset，并提取 mu, sigma
    train_dataset = ODEPairDataset("../dataset/train_inputs.npy", "../dataset/train_outputs.npy")
    mu, sigma     = train_dataset.mu, train_dataset.sigma

    A_Euler = A(mu, sigma)

    test_trajectories = np.load("../dataset/test_trajectories.npy")

    test_loss = 0.0
    abs_test_errors = np.zeros(test_trajectories[0].shape, dtype=np.float64)
    rel_test_errors = np.zeros(test_trajectories[0].shape[0], dtype=np.float64)
    with torch.no_grad():
        # for inputs, targets in test_loader:
        #     preds = model(inputs)
        #     test_loss += criterion(preds, targets).item() * inputs.size(0)
        j = 0
        for traj in test_trajectories:
            # print(traj.shape)
            u0 = traj[0, :2]  # 初始条件
            u_lists = [u0]  # 存储 u(t) 的列表，初始值 u(0)
            u0 = torch.tensor(u0, dtype=torch.float64, device=device)
            dt = torch.tensor([5e-2], dtype=torch.float64, device=device)
            for i in range(1, traj.shape[0]):
                # input_data = torch.tensor([u0[0], u0[1], u0[2], dt], dtype=torch.float64).to(device).reshape(1, 4)
                u1 = (u0 - mu) / sigma
                # input_data = torch.cat([u1, dt]).reshape(1,-1)
                u1 = A_Euler.single_step(u1.reshape(1, 2), dt=5e-2, mu=mu, sigma=sigma)
                u0 = sigma * u1 + mu
                # output = model.predict(input_data)
                # u0 = output[0]  # 更新 u0 为下一个时刻
                u_lists.append(u0[0].cpu().numpy())
            u_lists = np.array(u_lists)
            if j == 0:
                np.save("u_euler.npy", u_lists)
                # exit(-1)
            # if j <= 3:
            #     plot_trajectories(u_lists, traj[:, :3], title=f"Trajectory Comparison for Test Trajectory {i}", i=j)
            # error = np.mean(np.square(u_lists - traj[:, :2]), axis=0)  # 计算均方误差
            abs_test_errors += np.abs(u_lists - traj[:, :2])
            rel_test_errors += np.linalg.norm(u_lists - traj[:, :2], axis=1) / np.linalg.norm(traj[:, :2], axis=1)

            # abs_test_errors = np.append(abs_test_errors, abs_error)
            # rel_test_errors = np.append(rel_test_errors, rel_error)
            # test_loss += np.mean(abs_error)
            # test_loss_file.write(f"{abs_error[0]:.5e} {abs_error[1]:.5e}\n")
            # test_loss_rel_file.write(f"{rel_error:.5e}\n")
            j += 1
    # test_loss /= len(test_trajectories)
    abs_test_errors /= len(test_trajectories)
    rel_test_errors /= len(test_trajectories)

    np.savetxt("abs_test_errors_small.txt", abs_test_errors.reshape(-1, 2))
    np.savetxt("rel_test_errors_small.txt", rel_test_errors)
    # test_loss /= len(test_loader.dataset)
    # print(f"\nFinal Test Loss: {test_loss:.5e}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        B, T, D = test_trajectories.shape
        dt = 5e-2
        # 初始化 u
        u = test_trajectories[:, 0, :]       # [B, D]
        # pred_trajs = [u]                     # list 保存所有时间步预测
        u = torch.tensor(u, device=device, dtype=torch.float64)
        for t in range(1, T):
            # 拼接 dt
            # input_data = torch.cat([u, dt*torch.ones(B, 1, device=device, dtype=torch.float64)], dim=1)  # [B, D+1]
            # u = A_Euler.predict(input_data)   # [B, D]
            # pred_trajs.append(u)

            u1 = (u - mu) / sigma
            u1 = A_Euler.single_step(u1.reshape(B, 2), dt=5e-2, mu=mu, sigma=sigma)
            u  = sigma * u1 + mu

        # 最终结果
        # pred_trajs = torch.stack(pred_trajs, dim=1)  # [B, T, D]


    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")