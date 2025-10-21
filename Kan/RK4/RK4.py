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
from torch import exp
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

    def F(self, u: torch.Tensor):
        x_1 = u[:,0]
        x_2 = u[:,1]
        x_3 = u[:,2]

        dudt1 = 0.120709256920938*exp(-0.117332653272611*x_1 - 0.0619250945606703*exp(1.6*x_3) + 0.0270342893799585*exp(-1.2*x_2)) - 0.240559352715717*exp(-0.0922711846875956*x_1 - 0.0369556899334119*(-x_3 - 0.18)**2 - 0.0260433074495818*exp(-2.4*x_2)) - 0.0261227442273856*exp(0.395073171507276*x_1 - 0.512685231016291*x_3 + 0.280203634825687*exp(-2.2*x_2)) + 0.000581093535397713*exp(1.74002911773549*x_1 - 1.38837079632477*x_3 + 1.21105026295561*exp(-2.0*x_2)) + 0.15012282505425

        dudt2 = 0.00344176124290824*exp(-0.823069107306826*x_1 + 1.06809423128394*x_3 - 0.583757572553514*exp(-2.2*x_2)) - 0.0251348445452428*exp(-0.509276814946972*x_1 + 0.406352428192615*x_3 - 0.354453735499203*exp(-2.0*x_2)) - 0.0532871637473543*exp(-0.127999258115575*x_1 - 0.0675546486116403*exp(1.6*x_3) + 0.0294919520508638*exp(-1.2*x_2)) - 0.26749876911961*exp(0.0461355923437977*x_1 + 0.0184778449667059*(-x_3 - 0.18)**2 + 0.0130216537247909*exp(-2.4*x_2)) + 0.338550263766672

        dudt3 = 0.0605833513957377*exp(-0.0959994435866816*x_1 - 0.0506659864587302*exp(1.6*x_3) + 0.0221189640381479*exp(-1.2*x_2)) - 0.159306883613958*exp(-0.0922711846875956*x_1 - 0.0369556899334119*(-x_3 - 0.18)**2 - 0.0260433074495818*exp(-2.4*x_2)) + 0.208342008395171*exp(-0.0848794691578286*x_1 + 0.0677254046987691*x_3 - 0.0590756225832005*exp(-2.0*x_2)) - 0.10350243167659*exp(-0.0823069107306825*x_1 + 0.106809423128394*x_3 - 0.0583757572553513*exp(-2.2*x_2)) - 0.00277809678379121

        return torch.stack([dudt1, dudt2, dudt3], dim=1)

    def single_step(self, u: torch.Tensor, parameters=None, t=None, dt=None, mu=None, sigma=None) -> torch.Tensor:
        u1 = self.sigma*u + self.mu
        k1 = self.F(u1)
        k2 = self.F(u1 + dt/2.0 * k1)
        k3 = self.F(u1 + dt/2.0 * k2)
        k4 = self.F(u1 + dt * k3)
        # return u + dt * self.F(u)
        out = u1 + dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out = (out - self.mu) / self.sigma
        return out

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
            u0 = traj[0, :3]  # 初始条件
            u_lists = [u0]  # 存储 u(t) 的列表，初始值 u(0)
            u0 = torch.tensor(u0, dtype=torch.float64, device=device)
            dt = torch.tensor([1e-1], dtype=torch.float64, device=device)
            for i in range(1, traj.shape[0]):
                # input_data = torch.tensor([u0[0], u0[1], u0[2], dt], dtype=torch.float64).to(device).reshape(1, 4)
                u1 = (u0 - mu) / sigma
                # input_data = torch.cat([u1, dt]).reshape(1,-1)
                u1 = A_Euler.single_step(u1.reshape(1, 3), dt=1e-1, mu=mu, sigma=sigma)
                u0 = sigma * u1 + mu
                # output = model.predict(input_data)
                # u0 = output[0]  # 更新 u0 为下一个时刻
                u_lists.append(u0[0].cpu().numpy())
            u_lists = np.array(u_lists)
            if j == 0:
                np.save("u_base.npy", u_lists)
                # exit(-1)
            # if j <= 3:
            #     plot_trajectories(u_lists, traj[:, :3], title=f"Trajectory Comparison for Test Trajectory {i}", i=j)
            # error = np.mean(np.square(u_lists - traj[:, :2]), axis=0)  # 计算均方误差
            abs_test_errors += np.abs(u_lists - traj[:, :3])
            rel_test_errors += np.linalg.norm(u_lists - traj[:, :3], axis=1) / np.linalg.norm(traj[:, :3], axis=1)

            # abs_test_errors = np.append(abs_test_errors, abs_error)
            # rel_test_errors = np.append(rel_test_errors, rel_error)
            # test_loss += np.mean(abs_error)
            # test_loss_file.write(f"{abs_error[0]:.5e} {abs_error[1]:.5e}\n")
            # test_loss_rel_file.write(f"{rel_error:.5e}\n")
            j += 1
    # test_loss /= len(test_trajectories)
    abs_test_errors /= len(test_trajectories)
    rel_test_errors /= len(test_trajectories)

    np.savetxt("abs_test_errors_small.txt", abs_test_errors.reshape(-1, 3))
    np.savetxt("rel_test_errors_small.txt", rel_test_errors)
    # test_loss /= len(test_loader.dataset)
    # print(f"\nFinal Test Loss: {test_loss:.5e}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        B, T, D = test_trajectories.shape
        dt = 1e-1
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
            u1 = A_Euler.single_step(u1.reshape(B, 3), dt=1e-1, mu=mu, sigma=sigma)
            u  = sigma * u1 + mu

        # 最终结果
        # pred_trajs = torch.stack(pred_trajs, dim=1)  # [B, T, D]


    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")