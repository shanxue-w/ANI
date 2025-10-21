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
        self.A_mul = torch.tensor([
            [-2.61910367,  -0.77, -8.684, 4.051, 1.149, 3.965, 6.286],
            [4.962, -5.75810367, 18.247, -7.686, -7.584, -7.97, -12.49],
            [0., 3.637, -20.43010367, -5.173, -5.431, 1.418, 1.756],
            [0., 1.095, 21.883, -24.44310367, -19.398, -1.49, 11.778],
            [0., 2.567, 1.397, -14.926, -35.25310367, 0., 0.],
            [-4.886, 0.858, 24.967 , 19.343, 2.623, 3.58289633, 9.305],
            [0., 0., 0., 1.3, 0., 0., -3.23710367]], dtype=torch.float64, device=device)

        self.b = torch.tensor([
            -4.195, 14.236, -1.228, 5.407, 5.584, -9.311, 0.
        ], dtype=torch.float64, device=device).reshape(1,7)

    def F(self, x:torch.Tensor):
        # x0 = self.sigma[0] * x[..., 0:1] + self.mu[0]
        # x1 = self.sigma[1] * x[..., 1:2] + self.mu[1]
        # x2 = self.sigma[2] * x[..., 2:3] + self.mu[2]
        # x3 = self.sigma[3] * x[..., 3:4] + self.mu[3]
        # x4 = self.sigma[4] * x[..., 4:5] + self.mu[4]
        # x5 = self.sigma[5] * x[..., 5:6] + self.mu[5]
        # x6 = self.sigma[6] * x[..., 6:7] + self.mu[6]
        # x_full = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=-1)
        x_full = self.sigma * x + self.mu
        # return torch.cat([
        #     -4.195 + -2.61910367 * x0 + -0.77 * x1 + -8.684 * x2 + 4.051 * x3 + 1.149 * x4 + 3.965 * x5 + 6.286 * x6,
        # 14.236 + 4.962 * x0 + -5.75810367 * x1 + 18.247 * x2 + -7.686 * x3 + -7.584 * x4 + -7.97 * x5 + -12.49 * x6,
        # -1.228 + 0. * x0 + 3.637 * x1 + -20.43010367 * x2 + -5.173 * x3 + -5.431 * x4 + 1.418 * x5 + 1.756 * x6,
        # 5.407  + 0. * x0 + 1.095 * x1 + 21.883 * x2 + -24.44310367 * x3 + -19.398 * x4 + -1.49 * x5 + 11.778 * x6,
        # 5.584  + 0. * x0 + 2.567 * x1 + 1.397 * x2 + -14.926 * x3 + -35.25310367 * x4 + 0. * x5 + 0. * x6,
        # -9.311 + -4.886 * x0 + 0.858 * x1 + 24.967 * x2 + 19.343 * x3 + 2.623 * x4 + 3.58289633 * x5 + 9.305 * x6,
        # 1.3 * x3 + -3.23710367 * x6
        # ], dim=-1) / self.sigma
        return (x_full @ self.A_mul.T + self.b) / self.sigma

    def single_step(self, u: torch.Tensor, parameters=None, t=None, dt=None, mu=None, sigma=None) -> torch.Tensor:
        k1 = self.F(u)
        k2 = self.F(u + dt/2.0 * k1)
        k3 = self.F(u + dt/2.0 * k2)
        k4 = self.F(u + dt * k3)
        # return u + dt * self.F(u)
        return u + dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_mlp(input_dim, output_dim, hidden_layers, hidden_dim, activation=nn.Tanh):
    layers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

class Lorenz(ANIBASE):
    def __init__(self, N0_SCHEME: N0, input_dim: int, output_dim: int, hidden_layers: int = 4, hidden_dim: int = 64):
        super(Lorenz, self).__init__(N0_SCHEME, input_dim, output_dim, hidden_layers, hidden_dim)
        # self.mlp = build_mlp(input_dim, output_dim, hidden_layers, hidden_dim, nn.GELU)
        # self.mlp.output_dim = output_dim
        # self.residual_block = ResidualBlockWithT(self.mlp)

    def modify_input(self, x):
        # return (x - self.mu) / self.sigma
        # x[..., 0:1] = (x[..., 0:1] - self.mu[0]) / self.sigma[0]
        # x[..., 1:2] = (x[..., 1:2] - self.mu[1]) / self.sigma[1]
        # x[..., 2:3] = (x[..., 2:3] - self.mu[2]) / self.sigma[2]
        # return x
        return (x[..., :self.mu.shape[0]] - self.mu) / self.sigma
    
    def modify_output(self, x):
        # return self.sigma * x + self.mu
        # x[..., 0:1] = self.sigma[0] * x[..., 0:1] + self.mu[0]
        # x[..., 1:2] = self.sigma[1] * x[..., 1:2] + self.mu[1]
        # x[..., 2:3] = self.sigma[2] * x[..., 2:3] + self.mu[2]
        # return x
        return self.sigma * x[..., :self.mu.shape[0]] + self.mu

    def SplitConditions(self, x: torch.Tensor):
        return x[:, 0:7], x[:, 7:8]
    
    def forward(self, x):
        u, dts = self.SplitConditions(x)
        u      = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u      = self.residual_block(torch.cat([u, dts], dim=-1))
        # u      = self.deeponet(u, dts)
        # u      = self.film_residual_block(torch.cat([u, dts], dim=-1))
        u      = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        return u
    def predict(self, x):
        # x = self.N0_SCHEME.single_step(x, x[..., -1])
        u, dts = self.SplitConditions(x)
        u = self.modify_input(u)
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u = self.residual_block(torch.cat([u, dts], dim=-1))
        # u = self.deeponet(u, dts)
        # u = self.film_residual_block(torch.cat([u, dts], dim=-1))
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u = self.modify_output(u)
        return u


# train_input = np.load('train_inputs.npy')
# train_output = np.load('train_outputs.npy')

# print(f"Train input shape: {train_input.shape}, Train output shape: {train_output.shape}")



class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None):
        raw_inputs  = np.load(input_file)[:16000, :]
        raw_outputs = np.load(output_file)[:16000, :]

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

# save np array as .mat file
# import scipy.io as sio

# def plot_trajectories(NN_traj, true_traj, title="Trajectories Comparison", i = 0):
#     plt.figure()
#     plt.plot(NN_traj[:, 0], NN_traj[:, 1], label='NN Prediction', color='blue')
#     plt.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory', color='orange', linestyle='--')
#     plt.xlabel('Theta (rad)')
#     plt.ylabel('Theta Dot (rad/s)')
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     # plt.show()
#     plt.savefig(f"trajectory_comparison_small_{i}.png")
#     plt.close()

#     # save them as .mat file
#     sio.savemat(f"ANI_Lorenz_2th_{i}.mat", {"NN_traj": NN_traj})
#     sio.savemat(f"ANI_Lorenz_true_{i}.mat", {"true_traj": true_traj})

import numpy as np
from pysindy.feature_library import PolynomialLibrary
from pysindy import SINDy
import pysindy as ps

if __name__ == "__main__":
    # # 首先加载 train dataset，并提取 mu, sigma
    train_dataset = ODEPairDataset("../dataset/train_inputs.npy", "../dataset/train_outputs.npy")
    mu, sigma     = train_dataset.mu, train_dataset.sigma

    # 用相同 mu, sigma 加载 val/test
    val_dataset   = ODEPairDataset("../dataset/val_inputs.npy", "../dataset/val_outputs.npy", 
                                   mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
    # test_dataset  = ODEPairDataset("../dataset/test_inputs_small.npy", "../dataset/test_outputs_small.npy", 
                                #    mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())

    # 后面继续创建 dataloader 和训练模型...

    batch_size   = 90
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader  = DataLoader(test_dataset, batch_size=16)

    test_trajectories = np.load("../dataset/test_trajectories.npy")

    # new_traj = np.load("../dataset/test_trajectories_small.npy")
    # new_traj_1 = [new_traj[i] for i in range(new_traj.shape[0])] 
    # # new_traj = (new_traj - mu.cpu().numpy()) / sigma.cpu().numpy()

    # opt = ps.STLSQ(threshold=.5, alpha=.5)
    # poly = PolynomialLibrary(degree=1)
    # model = SINDy(optimizer=opt, feature_library=poly)
    # model.fit(new_traj_1, t=1e-3, multiple_trajectories=True)
    # model.print()

    # from sympy import symbols
    # import re
    # def export_sindy_as_single_torch_function_keep_vars(model, tol=1e-10):
    #     """
    #     生成单一函数 f(x, device=None)，返回 torch.tensor([...])
    #     变量名保持原样，不替换成 x[1]，但补乘号和幂运算符。
    #     """
    #     coef_matrix = model.coefficients()
    #     feature_names = model.get_feature_names()

    #     exprs = []
    #     for coefs in coef_matrix:
    #         terms = []
    #         for c, name in zip(coefs, feature_names):
    #             if abs(c) > tol:
    #                 if name == '1':
    #                     terms.append(f"{c:.5f}")
    #                 else:
    #                     # 用正则把 ^ 替换为 **
    #                     name_fixed = re.sub(r'\^', r'**', name)
    #                     # 用乘号替换变量间空格，比如 "x1 x5" -> "x1*x5"
    #                     name_fixed = '*'.join(name_fixed.split())
    #                     terms.append(f"{c:.5f} * {name_fixed}")
    #         expr = " + ".join(terms) if terms else "0.0"
    #         exprs.append(expr)

    #     joined_exprs = ",\n        ".join(exprs)
    #     func_str = f"""
    #     return torch.cat([
    #         {joined_exprs}
    #     ], dim=-1)
    # """
    #     return func_str


    # print(export_sindy_as_single_torch_function_keep_vars(model))

    A1 = A(mu, sigma)
    # # 初始化模型
    model = Lorenz(N0_SCHEME=A(mu, sigma), input_dim=8, output_dim=7, hidden_layers=6, hidden_dim=32).to(device)
    model.get_mu_and_sigma(mu, sigma)
    model = torch.compile(model, mode="max-autotune")

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 1000
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-5)

    best_val_loss = float('inf')

    train_loss_lists = []
    val_loss_lists = []

    train_loss_file = open("train_loss.txt", "w")
    val_loss_file = open("val_loss.txt", "w")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_lists.append(train_loss)
        train_loss_file.write(f"{train_loss:.5e}\n")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                val_loss += criterion(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model!")
    

    plot_losses(train_loss_lists, val_loss_lists)

    # # ---------------------
    # # 测试评估
    # # ---------------------
    print("\nTesting best model on test set...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

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
            u0 = traj[0, :7]  # 初始条件
            u_lists = [u0]  # 存储 u(t) 的列表，初始值 u(0)
            u0 = torch.tensor(u0, dtype=torch.float64, device=device)
            dt = torch.tensor([5e-2], dtype=torch.float64, device=device)
            for i in range(1, traj.shape[0]):
                # input_data = torch.tensor([u0[0], u0[1], u0[2], dt], dtype=torch.float64).to(device).reshape(1, 4)
                input_data = torch.cat([u0, dt]).reshape(1, -1)
                output = model.predict(input_data)
                u0 = output[0]  # 更新 u0 为下一个时刻
                u_lists.append(u0.cpu().numpy())
            u_lists = np.array(u_lists)
            # if j <= 3:
            #     plot_trajectories(u_lists, traj[:, :3], title=f"Trajectory Comparison for Test Trajectory {i}", i=j)
            # error = np.mean(np.square(u_lists - traj[:, :2]), axis=0)  # 计算均方误差
            if j == 0:
                np.save("u_2th.npy", u_lists)
                np.save("traj_2th.npy", traj[:, :7])
            #     exit(-1)
            abs_test_errors += np.abs(u_lists - traj[:, :7])
            rel_test_errors += np.linalg.norm(u_lists - traj[:, :7], axis=1) / np.linalg.norm(traj[:, :7], axis=1)

            # abs_test_errors = np.append(abs_test_errors, abs_error)
            # rel_test_errors = np.append(rel_test_errors, rel_error)
            # test_loss += np.mean(abs_error)
            # test_loss_file.write(f"{abs_error[0]:.5e} {abs_error[1]:.5e}\n")
            # test_loss_rel_file.write(f"{rel_error:.5e}\n")
            j += 1
    # test_loss /= len(test_trajectories)
    abs_test_errors /= len(test_trajectories)
    rel_test_errors /= len(test_trajectories)

    # train_loss_file.close()
    # val_loss_file.close()
    np.savetxt("abs_test_errors_small.txt", abs_test_errors.reshape(-1, 7))
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
            input_data = torch.cat([u, dt*torch.ones(B, 1, device=device, dtype=torch.float64)], dim=1)  # [B, D+1]
            u = model.predict(input_data)   # [B, D]
            # pred_trajs.append(u)

        # 最终结果
        # pred_trajs = torch.stack(pred_trajs, dim=1)  # [B, T, D]


    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")