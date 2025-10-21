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
from kan import KAN, LBFGS
from torch import exp, sin
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)
class KAN_ODE_Function(nn.Module):
    def __init__(self, kan_model):
        super(KAN_ODE_Function, self).__init__()
        self.kan_model = kan_model

    def forward(self, u):
        # t is provided by odeint, but we only use u
        # KAN expects batch input, so unsqueeze u
        # u will be (batch_size, state_dim) if odeint processes in batches,
        # or (state_dim,) if odeint passes one by one (then unsqueeze needed).
        # To be safe, we'll ensure it's (batch_size, state_dim) for KAN.
        if u.dim() == 1: # If u is (state_dim,), make it (1, state_dim)
            u_input = u.unsqueeze(0)
        else: # If u is already (batch_size, state_dim)
            u_input = u
            
        # Forward pass through KAN
        output = self.kan_model(u_input)
        
        # Ensure output matches expected shape for odeint (batch_size, state_dim) or (state_dim,)
        if u.dim() == 1:
            return output.squeeze(0) # Remove batch dim if it was added
        else:
            return output

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
        out = (out - mu) / sigma
        return out

class Kan(ANIBASE):
    def __init__(self, N0_SCHEME: N0, input_dim: int, output_dim: int, hidden_layers: int = 4, hidden_dim: int = 64):
        super(Kan, self).__init__(N0_SCHEME, input_dim, output_dim, hidden_layers, hidden_dim)
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
        return x[:, 0:3], x[:, 3:4]
    
    def block_tau(self, u: torch.Tensor, dts: torch.Tensor):
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u = self.residual_block(torch.cat([u, dts], dim=-1))
        # u = self.deeponet(u, dts)
        u = self.N0_SCHEME.single_step(u, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        return u

    def forward(self, x):
        u, dts = self.SplitConditions(x)

        k1 = self.block_tau(u, dts)
        k2 = self.block_tau(self.block_tau(u, dts/2.0), dts/2.0)

        # 最终组合
        return -1.0/3.0 * k1 + 4.0/3.0 * k2
        # u1 = self.linear(k1, k2)
        # return u1

    
    def predict(self, x):
        # x = self.N0_SCHEME.single_step(x, x[..., -1])
        u, dts = self.SplitConditions(x)
        u = self.modify_input(u)

        k1 = self.block_tau(u, dts)
        k2 = self.block_tau(self.block_tau(u, dts/2.0), dts/2.0)

        u  = -1.0/3.0 * k1 + 4.0/3.0 * k2
        # u  = self.linear(torch.cat([k1, k2], dim=-1)).squeeze(-1)
        # u = self.linear(k1, k2)

        u = self.modify_output(u)
        return u

# train_input = np.load('train_inputs.npy')
# train_output = np.load('train_outputs.npy')

# print(f"Train input shape: {train_input.shape}, Train output shape: {train_output.shape}")



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

    batch_size   = 50
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader  = DataLoader(test_dataset, batch_size=16)

    test_trajectories = np.load("../dataset/test_trajectories.npy")

    # # 初始化模型
    model = Kan(N0_SCHEME=A(mu, sigma), input_dim=4, output_dim=3, hidden_layers=4, hidden_dim=30).to(device)
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

    # ---------------------
    # 测试评估
    # ---------------------
    print("\nTesting best model on test set...")
    model.load_state_dict(torch.load("best_model.pth"))
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
            u0 = traj[0, :3]  # 初始条件
            u_lists = [u0]  # 存储 u(t) 的列表，初始值 u(0)
            u0 = torch.tensor(u0, dtype=torch.float64, device=device)
            dt = torch.tensor([1e-1], dtype=torch.float64, device=device)
            for i in range(1, traj.shape[0]):
                # input_data = torch.tensor([u0[0], u0[1], u0[2], dt], dtype=torch.float64).to(device).reshape(1, 4)
                input_data = torch.cat([u0, dt]).reshape(1,-1)
                output = model.predict(input_data)
                u0 = output[0]  # 更新 u0 为下一个时刻
                u_lists.append(u0.cpu().numpy())
            u_lists = np.array(u_lists)
            if j == 0:
                np.save("u_4th.npy", u_lists)
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

    train_loss_file.close()
    val_loss_file.close()
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