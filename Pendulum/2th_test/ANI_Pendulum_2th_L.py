from ANI import N0, ANIBASE
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
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class A(N0):
    def __init__(self):
        super(A, self).__init__()
        # self.mu = mu
        # self.sigma = sigma
        # self.a = a

    def F(self, u, mu, sigma):
        # 安全地构造新张量，避免 inplace 错误
        # u0 = u[..., 0:1]
        # u1 = u[..., 1:2]
        x = sigma * u + mu
        u0 = x[..., 0:1]
        u1 = x[..., 1:2]

        # new_u0 = u0 + dt * (sigma[1]/sigma[0] * u1 + mu[1]/sigma[0]) 
        # new_u1 = u1 - 9.80665 / sigma[1] * dt * torch.sin(sigma[0] * u0 + mu[0])
        new_u0 = u1 / sigma[0]
        new_u1 = -9.80665 * torch.sin(u0) / sigma[1]

        return torch.cat([new_u0, new_u1], dim=-1)

    def single_step(self, u: torch.Tensor, parameters=None, t=None, dt=None, mu=None, sigma=None) -> torch.Tensor:
        # k1 = self.F(u, mu, sigma)
        # k2 = self.F(u + dt / 2 * k1, mu, sigma)
        # k3 = self.F(u + dt / 2 * k2, mu, sigma)
        # k4 = self.F(u + dt * k3, mu, sigma)
        # return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # https://www.researchgate.net/figure/The-Butcher-Tableau-for-the-RK6-Method_tbl3_318284280
        k1 = dt * self.F(u, mu, sigma)
        k2 = dt * self.F(u + 1/3 * k1, mu, sigma)
        k3 = dt * self.F(u + 2/3 * k2, mu, sigma)
        k4 = dt * self.F(u + 1/12 * k1 + 1/3 * k2 - 1/12 * k3, mu, sigma)
        k5 = dt * self.F(u - 1/16 * k1 + 9/8 * k2 - 3/16 * k3 - 3/8 * k4, mu, sigma)
        k6 = dt * self.F(u + 9/8 * k2 - 3/8 * k3 - 3/4 * k4 + 1/2 * k5, mu, sigma)
        k7 = dt * self.F(u + 9/44 * k1 - 9/11 * k2 + 63/44 * k3 + 18/11 * k4 - 16/11 * k6, mu, sigma)
        return u + 11/120 * k1 + 27/40 * k3 + 27/40 * k4 - 4/15 * k5 - 4/15 * k6 + 11/120 * k7

class FiLM(nn.Module):
    def __init__(self, time_dim=1, hidden_dim=128):
        super().__init__()
        # 三层 MLP, 输入是 t
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)  # 输出 gamma 和 beta
        )

    def forward(self, h, t):
        """
        h: [B, hidden_dim]
        t: [B, time_dim]
        """
        film_params = self.mlp(t)  # [B, 2*hidden_dim]
        gamma, beta = film_params.chunk(2, dim=-1)  # 各 [B, hidden_dim]
        return gamma * h + beta

class FiLM_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=6, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        # self.instancenorm = nn.InstanceNorm1d(hidden_dim, affine=False)

        self.in_layer = nn.Linear(input_dim - 1, hidden_dim)

        self.film_layers = nn.ModuleList()
        self.hidden_layers_list = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 保存中间 hidden_dim 状态以确保最终输出维度正确
        current_dim = hidden_dim

        # 上升阶段
        for _ in range(int((hidden_layers - 1) / 2)):
            next_dim = current_dim * 2
            self.hidden_layers_list.append(nn.Linear(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            self.film_layers.append(FiLM(time_dim=1, hidden_dim=next_dim))
            current_dim = next_dim

        # 下降阶段
        for _ in range(int((hidden_layers - 1) / 2)):
            next_dim = current_dim // 2
            self.hidden_layers_list.append(nn.Linear(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            self.film_layers.append(FiLM(time_dim=1, hidden_dim=next_dim))
            current_dim = next_dim

        # 输出层结构分支
        if hidden_layers % 2 == 0:
            self.output_layer = nn.Linear(current_dim, output_dim)
            self.final_fc1 = None
            self.final_fc2 = None
        else:
            self.final_fc1 = nn.Linear(current_dim, current_dim // 2)
            self.final_fc2 = nn.Linear(current_dim // 2, output_dim)
            self.output_layer = None

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_feat = x[:, :-1]
        t = x[:, -1:].detach()

        h = F.gelu(self.in_layer(x_feat))

        for layer, norm, film in zip(self.hidden_layers_list, self.norms, self.film_layers):
            h = layer(h)
            # h = self.instancenorm(h.unsqueeze(2)).squeeze(2)
            h = film(h, t)
            h = F.gelu(h)

        if self.output_layer is not None:
            return self.output_layer(h)
        else:
            h = F.gelu(self.final_fc1(h))
            return self.final_fc2(h)

class Pendulum(ANIBASE):
    def __init__(self, N0_SCHEME: N0, input_dim: int, output_dim: int, hidden_layers: int = 4, hidden_dim: int = 64):
        super(Pendulum, self).__init__(N0_SCHEME, input_dim, output_dim, hidden_layers, hidden_dim)
        # initialize each block
        self.mlp = nn.Sequential(
            nn.Linear(3, 10),
            nn.GELU(),
            nn.Linear(10, 20),
            nn.GELU(),
            nn.Linear(20, 20),
            nn.GELU(),
            nn.Linear(20, 10),
            nn.GELU(),
            nn.Linear(10, 2)
        )

    def modify_input(self, x):
        # return (x - self.mu) / self.sigma
        x[..., 0:1] = (x[..., 0:1] - self.mu[0]) / self.sigma[0]
        x[..., 1:2] = (x[..., 1:2] - self.mu[1]) / self.sigma[1]
        return x
    
    def modify_output(self, x):
        # return self.sigma * x + self.mu
        x[..., 0:1] = self.sigma[0] * x[..., 0:1] + self.mu[0]
        x[..., 1:2] = self.sigma[1] * x[..., 1:2] + self.mu[1]
        return x

    def SplitConditions(self, x: torch.Tensor):
        return x[:, 0:2], x[:, 2:3]
    
    def forward(self, x):
        u, dts = self.SplitConditions(x)
        u      = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        # u = self.residual_block(torch.cat([u, dts], dim=-1))
        u = u + dts * self.mlp(torch.cat([u, dts], dim=-1))
        u      = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        return u
    def predict(self, x):
        # x = self.N0_SCHEME.single_step(x, x[..., -1])
        u, dts = self.SplitConditions(x)
        u = self.modify_input(u)
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        # input_RK = torch.cat([u, unused], dim=-1)
        # u = self.rk4(input_RK, dts)
        # u = self.residual_block(torch.cat([u, dts], dim=-1))
        u = u + dts * self.mlp(torch.cat([u, dts], dim=-1))
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u = self.modify_output(u)
        return u



# train_input = np.load('train_inputs.npy')
# train_output = np.load('train_outputs.npy')

# print(f"Train input shape: {train_input.shape}, Train output shape: {train_output.shape}")

class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None, limit=1000):
        raw_inputs  = np.load(input_file)[:limit, :]
        raw_outputs = np.load(output_file)[:limit, :]

        if mu is None or sigma is None:
            # 用于 train 集，计算 mu 和 sigma
            x_min = raw_inputs[:, 0].min()
            x_max = raw_inputs[:, 0].max()
            y_min = raw_inputs[:, 1].min()
            y_max = raw_inputs[:, 1].max()

            mu = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
            sigma = np.array([(x_max - x_min) / 2.0, (y_max - y_min) / 2.0])

        # 标准化输入的前两列
        normalized_inputs = raw_inputs.copy()
        normalized_inputs[:, 0] = (raw_inputs[:, 0] - mu[0]) / sigma[0]
        normalized_inputs[:, 1] = (raw_inputs[:, 1] - mu[1]) / sigma[1]
        # dt 保留不变

        normalized_outputs = raw_outputs.copy()
        normalized_outputs[:, 0] = (raw_outputs[:, 0] - mu[0]) / sigma[0]
        normalized_outputs[:, 1] = (raw_outputs[:, 1] - mu[1]) / sigma[1]

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

def plot_trajectories(NN_traj, true_traj, title="Trajectories Comparison", i = 0):
    plt.figure()
    plt.plot(NN_traj[:, 0], NN_traj[:, 1], label='NN Prediction', color='blue')
    plt.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory', color='orange', linestyle='--')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Theta Dot (rad/s)')
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f"trajectory_comparison_small_{i}.png")
    plt.close()

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

if __name__ == "__main__":
    # # 首先加载 train dataset，并提取 mu, sigma
    train_dataset = ODEPairDataset("../dataset/train_inputs_var.npy", "../dataset/train_outputs_var.npy")
    mu, sigma     = train_dataset.mu, train_dataset.sigma

    # 用相同 mu, sigma 加载 val/test
    val_dataset   = ODEPairDataset("../dataset/val_inputs_var.npy", "../dataset/val_outputs_var.npy", 
                                   mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
    # test_dataset  = ODEPairDataset("../dataset/test_inputs_var.npy", "../dataset/test_outputs_var.npy", 
                                #    mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())

    # 后面继续创建 dataloader 和训练模型...


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)
    # test_loader  = DataLoader(test_dataset, batch_size=16)

    test_trajectories = np.load("../dataset/test_trajectories_var.npy")

    # # 初始化模型
    model = Pendulum(N0_SCHEME=A(), input_dim=3, output_dim=2, hidden_layers=4, hidden_dim=10).to(device)
    model.get_mu_and_sigma(mu, sigma)
    model = torch.compile(model, mode="max-autotune")

    # 损失和优化器
    # criterion = torch.nn.MSELoss()
    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 3000
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
        train_loss_file.write(f"{train_loss:.14f}\n")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                val_loss += criterion(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.14f}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.14f} | Val Loss: {val_loss:.14f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_small.pth")
            print("Saved best model!")
    

    plot_losses(train_loss_lists, val_loss_lists)

    # # ---------------------
    # # 测试评估
    # # ---------------------
    # print("\nTesting best model on test set...")
    model.load_state_dict(torch.load("best_model_small.pth"))
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
            u0 = traj[0, :2]  # 初始条件
            dt = 1e-1
            u_lists = [u0]  # 存储 u(t) 的列表，初始值 u(0)
            for i in range(1, traj.shape[0]):
                input_data = torch.tensor([u0[0], u0[1], dt], dtype=torch.float64).to(device).reshape(1, 3)
                output = model.predict(input_data)
                u0 = output[0]  # 更新 u0 为下一个时刻
                u_lists.append(u0.cpu().numpy())
            u_lists = np.array(u_lists)
            if j <= 3:
                plot_trajectories(u_lists, traj[:, :2], title=f"Trajectory Comparison for Test Trajectory {i}", i=j)
            # if j == 0:
            #     # save u and traj for j = 0
            #     np.save("traj_test.npy", traj[:, :2])
            #     exit(-1)
            error = np.mean(np.square(u_lists - traj[:, :2]), axis=0)  # 计算均方误差
            abs_test_errors += np.abs(u_lists - traj[:, :2])
            rel_test_errors += np.linalg.norm(u_lists - traj[:, :2], axis=1) / np.linalg.norm(traj[:, :2], axis=1)

            # abs_test_errors = np.append(abs_test_errors, abs_error)
            # rel_test_errors = np.append(rel_test_errors, rel_error)
            # test_loss += np.mean(abs_error)
            # test_loss_file.write(f"{abs_error[0]:.14f} {abs_error[1]:.14f}\n")
            # test_loss_rel_file.write(f"{rel_error:.14f}\n")
            j += 1
    test_loss /= len(test_trajectories)
    abs_test_errors /= len(test_trajectories)
    rel_test_errors /= len(test_trajectories)

    train_loss_file.close()
    val_loss_file.close()
    np.savetxt("abs_test_errors_small.txt", abs_test_errors.reshape(-1, 2))
    np.savetxt("rel_test_errors_small.txt", rel_test_errors)
    # test_loss /= len(test_loader.dataset)
    # print(f"\nFinal Test Loss: {test_loss:.14f}")



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
    # # test_loss /= len(test_trajectories)
    # abs_test_errors /= len(test_trajectories)
    # rel_test_errors /= len(test_trajectories)

    # train_loss_file.close()
    # val_loss_file.close()
    # np.savetxt("abs_test_errors_small.txt", abs_test_errors.reshape(-1, 2))
    # np.savetxt("rel_test_errors_small.txt", rel_test_errors)
    # test_loss /= len(test_loader.dataset)
    # print(f"\nFinal Test Loss: {test_loss:.14f}")

