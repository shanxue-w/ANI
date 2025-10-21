import torch
import torch.nn as nn
from ANI import MLP  # 你自己的 MLP 实现
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class RK4Baseline(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=128, hidden_layers=4):
        super(RK4Baseline, self).__init__()
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim,
                                 hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    def get_mu_and_sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def modify_input(self, x):
        return (x[..., :self.mu.shape[0]] - self.mu) / self.sigma
    
    def modify_output(self, x):
        return self.sigma * x[..., :self.mu.shape[0]] + self.mu

    def SplitConditions(self, x: torch.Tensor):
        return x[:, 0:4], x[:, 4:5]
    
    def f(self, x):
        f1 = x[:, 1] - x[:, 0]
        f2 = x[:, 0] * (26-x[:,2]) - x[:,1]
        f3 = x[:,0]*x[:,1] - 0.7*x[:,2]
        f4 = torch.zeros_like(x[:, 0])
        # return [f1, f2, f3, f4] + MLP(x)
        return torch.stack([f1/self.sigma[0], f2/self.sigma[1], f3/self.sigma[2], f4], dim=-1) + self.mlp(x)

    def forward(self, x):
        u0 = x[:, :4]  # 取出 u0
        dt = x[:, 4:5]  # 取出时间步长 dt
        # u = self.single_step(u0, dt=dt)
        # delta = self.residual_net(x)  # 残差预测
        # return u + dt * delta  # 返回 u0 + 残差
        # k1 = self.mlp(u0)
        # k2 = self.mlp(u0 + dt * k1 / 2.0)
        # k3 = self.mlp(u0 + dt * k2 / 2.0)
        # k4 = self.mlp(u0 + dt * k3)
        k1 = self.f(u0)
        k2 = self.f(u0 + dt / 2.0 * k1)
        k3 = self.f(u0 + dt / 2.0 * k2)
        k4 = self.f(u0 + dt * k3)

        return u0 + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    def predict(self, x):
        u0 = x[:, :4]  # 取出 u0
        dt = x[:, 4:5]  # 取出时间步长 dt

        u0 = self.modify_input(u0)
        # u1 = self.residual_net(torch.cat([u0, dt], dim=-1))
        # u  = self.single_step(u0, dt=dt) + dt * u1
        u = self.forward(torch.cat([u0, dt], dim=-1))

        return self.modify_output(u)


class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None):
        raw_inputs  = np.load(input_file)[:10000, :]
        raw_outputs = np.load(output_file)[:10000, :]

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
import scipy.io as sio

def plot_trajectories(NN_traj, true_traj, title="Trajectories Comparison", i = 0):
    # plt.figure()
    # plt.plot(NN_traj[:, 0], NN_traj[:, 1], label='NN Prediction', color='blue')
    # plt.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory', color='orange', linestyle='--')
    # plt.xlabel('Theta (rad)')
    # plt.ylabel('Theta Dot (rad/s)')
    # plt.title(title)
    # plt.legend()
    # plt.grid()
    # # plt.show()
    # plt.savefig(f"trajectory_comparison_small_{i}.png")
    # plt.close()

    # save them as .mat file
    sio.savemat(f"ANI_Lorenz_stenflo_base_{i}.mat", {"NN_traj": NN_traj})
    sio.savemat(f"ANI_Lorenz_stenflo_true_{i}.mat", {"true_traj": true_traj})


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

    batch_size   = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader  = DataLoader(test_dataset, batch_size=16)

    test_trajectories = np.load("../dataset/test_trajectories.npy")

    # # 初始化模型
    model = RK4Baseline(input_dim=4, output_dim=4, hidden_layers=4, hidden_dim=20).to(device)
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

    train_loss_file = open("train_loss_test.txt", "w")
    val_loss_file = open("val_loss_test.txt", "w")

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
            torch.save(model.state_dict(), "best_model_test.pth")
            print("Saved best model!")
    

    plot_losses(train_loss_lists, val_loss_lists)

    # ---------------------
    # 测试评估
    # ---------------------
    print("\nTesting best model on test set...")
    model.load_state_dict(torch.load("best_model_test.pth"))
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
            u0 = traj[0, :4]  # 初始条件
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
            plot_trajectories(u_lists, traj[:, :3], title=f"Trajectory Comparison for Test Trajectory {i}", i=j)
            # error = np.mean(np.square(u_lists - traj[:, :2]), axis=0)  # 计算均方误差
            abs_test_errors += np.abs(u_lists - traj[:, :4])
            rel_test_errors += np.linalg.norm(u_lists - traj[:, :4], axis=1) / np.linalg.norm(traj[:, :4], axis=1)

            # abs_test_errors = np.append(abs_test_errors, abs_error)
            # rel_test_errors = np.append(rel_test_errors, rel_error)
            # test_loss += np.mean(abs_error)
            # test_loss_file.write(f"{abs_error[0]:.5e} {abs_error[1]:.5e}\n")
            # test_loss_rel_file.write(f"{rel_error:.5e}\n")
            j += 1
    # test_loss /= len(test_trajectories)
    abs_test_errors /= len(test_trajectories)
    rel_test_errors /= len(test_trajectories)

    train_loss_file.close()
    val_loss_file.close()
    np.savetxt("abs_test_errors_small_test.txt", abs_test_errors.reshape(-1, 4))
    np.savetxt("rel_test_errors_small_test.txt", rel_test_errors)
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

