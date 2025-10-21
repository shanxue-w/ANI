import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import os
import time

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

# =====================
# 数据集定义
# =====================
class ODEDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None, limit=1000):
        raw_inputs = np.load(input_file)[:limit, :2]
        raw_outputs = np.load(output_file)[:limit, :2]

        if mu is None or sigma is None:
            mu = np.mean(raw_inputs, axis=0)
            sigma = np.std(raw_inputs, axis=0)

        norm_in = (raw_inputs - mu) / sigma
        norm_out = (raw_outputs - mu) / sigma

        self.inputs = torch.from_numpy(norm_in).to(device)
        self.outputs = torch.from_numpy(norm_out).to(device)
        self.mu = torch.tensor(mu, dtype=torch.float64, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, device=device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# =====================
# Neural ODE 模型定义
# =====================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # 使用 nn.GELU 而不是 F.gelu
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SimpleMLP()

    def forward(self, t, y):
        return self.mlp(y)

class NeuralODE(nn.Module):
    def __init__(self, func, dt=1e-1, solver="dopri5"):
        super().__init__()
        self.func = func
        self.dt = dt
        self.solver = solver

    def forward(self, x):
        t = torch.tensor([0.0, self.dt], device=x.device, dtype=torch.float64)
        out = odeint(self.func, x, t, method=self.solver)
        return out[1]  # return x(t + dt)

def plot_losses(losses, losses_val):
    plt.figure()
    plt.plot(losses, label="Train")
    plt.plot(losses_val, label="Validation")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig("loss_plot.png")
    plt.close()

# =====================
# 训练函数
# =====================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=3000):
    best_val = float("inf")
    train_loss_file = open("train_loss_baseline.txt", "w")
    val_loss_file = open("val_loss_baseline.txt", "w")
    train_loss_lists = np.array([])
    val_loss_lists = np.array([])
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_loss_lists = np.append(train_loss_lists, avg_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_lists = np.append(val_loss_lists, val_loss)

        print(f"[{epoch+1:03d}] Train Loss: {avg_loss:.5e} | Val Loss: {val_loss:.5e}")

        train_loss_file.write(f"{epoch+1}\t{avg_loss}\n")
        val_loss_file.write(f"{epoch+1}\t{val_loss}\n")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_neural_ode.pth")

        scheduler.step()

    plot_losses(train_loss_lists, val_loss_lists)

    train_loss_file.close()
    val_loss_file.close()
# =====================
# Trajectory rollout
# =====================
def rollout(model, u0, steps=1000):
    traj = [u0]
    u = torch.tensor(u0, dtype=torch.float64, device=device).unsqueeze(0)
    for _ in range(steps):
        u = model(u)
        traj.append(u.squeeze(0).detach().cpu().numpy())
    return np.stack(traj)

# =====================
# 绘图函数
# =====================
def plot_trajectory(pred, true, i=0):
    plt.figure()
    plt.plot(pred[:, 0], pred[:, 1], label="NeuralODE", color="blue")
    plt.plot(true[:, 0], true[:, 1], "--", label="True", color="orange")
    plt.xlabel("Theta")
    plt.ylabel("Theta_dot")
    plt.title(f"Trajectory {i}")
    plt.legend()
    plt.grid()
    plt.savefig(f"trajectory_baseline_{i}.png")
    plt.close()

# =====================
# 主程序入口
# =====================
def main():
    # 载入数据
    train_data = ODEDataset("../dataset/train_inputs_var.npy", "../dataset/train_outputs_var.npy", limit=1000)
    val_data = ODEDataset("../dataset/val_inputs_var.npy", "../dataset/val_outputs_var.npy",
                          mu=train_data.mu.cpu().numpy(), sigma=train_data.sigma.cpu().numpy(), limit=1000)
    mu, sigma = train_data.mu, train_data.sigma

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # 初始化模型
    func = ODEFunc()
    model = NeuralODE(func).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    criterion = nn.MSELoss()

    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # 加载最佳模型
    model.load_state_dict(torch.load("best_neural_ode.pth"))
    model.eval()

    # =====================
    # 测试轨迹
    # =====================
    test_trajs = np.load("../dataset/test_trajectories_var.npy")
    abs_test_errors = np.zeros(test_trajs[0].shape, dtype=np.float64)
    rel_test_errors = np.zeros(test_trajs[0].shape[0], dtype=np.float64)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for i, traj in enumerate(test_trajs):
            true = traj[:, :2]
            u0 = (true[0] - mu.cpu().numpy()) / sigma.cpu().numpy()
            pred = rollout(model, u0, steps=true.shape[0] - 1)
            pred = pred * sigma.cpu().numpy() + mu.cpu().numpy()
            # plot_trajectory(pred, true, i=i)
            if i <= 3:
                plot_trajectory(pred, true, i=i)
            # if i == 0:
            #     np.save("u_baseline.npy", pred)
            #     exit(-1)

            abs_test_errors += np.abs(pred - true)
            rel_test_errors += np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1)


    abs_test_errors /= len(test_trajs)
    rel_test_errors /= len(test_trajs)

    np.savetxt("abs_test_errors_baseline.txt", abs_test_errors.reshape(-1, 2))
    np.savetxt("rel_test_errors_baseline.txt", rel_test_errors)

    print("Test results saved.")
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")


if __name__ == "__main__":
    main()
