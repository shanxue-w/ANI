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
import torch.nn.functional as F
import torch.fft as fft
import torch.fft
import math


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

class CNEXTUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_width=64, n_blocks=4):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_width = base_width

        # 编码器 blocks（Conv → InstanceNorm → FiLM → GELU）
        self.encoder = nn.ModuleList()
        for _ in range(n_blocks):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, base_width, 3, padding=1),
                nn.InstanceNorm2d(base_width, affine=False),
            ))
            in_channels = base_width  # only first uses in_channels=4

        # 中间处理
        self.middle = nn.Sequential(
            self.make_res_block(base_width),
            self.make_res_block(base_width),
            self.make_res_block(base_width)
        )

        # 解码器（可选添加更多层）
        self.decoder = nn.Sequential(
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 1)
        )

    def make_res_block(self, width):
        return nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1)
        )


    def forward(self, x):
        B = x.shape[0]

        # 编码阶段，每层都用 FiLM
        for i, block in enumerate(self.encoder):
            x = block(x)  # Conv + InstanceNorm
            x = F.gelu(x)

        # 中间层
        for res_block in self.middle:
            x = x + res_block(x)

        # 解码器
        x = self.decoder(x)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.complex128,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):    
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, film_input=1, num_layers=3):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.num_layers = num_layers

        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for (x, y)

        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(width, affine=False) for _ in range(num_layers)
        ])

        # MLP FiLM 输出 gamma/beta： [num_layers, 2, width]
        self.film_mlp = nn.Sequential(
            nn.Linear(film_input, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 2 * width * num_layers)
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, x):
        B, H, W, _ = x.shape
        gridx = torch.linspace(0, 1, W+1, device=x.device)[:-1]
        gridy = torch.linspace(0, 1, H+1, device=x.device)[:-1]
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing="xy")
        grid = torch.stack((gridx, gridy), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def apply_film(self, x, gamma, beta):
        return gamma * x + beta

    def forward(self, x, dts=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x)
        x = torch.cat([x, grid], dim=-1)  # [B, H, W, C+2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.pad(x, [0, self.padding, 0, self.padding], mode='circular')

        # FiLM 调制参数
        if dts is None:
            gamma_beta = torch.cat([
                torch.ones(B, self.num_layers, self.width, device=x.device),
                torch.zeros(B, self.num_layers, self.width, device=x.device)
            ], dim=-1)  # [B, num_layers, 2*width]
        else:
            if dts.dim() == 0:
                dts = dts.repeat(B, 1)
            elif dts.dim() == 1:
                dts = dts.view(B, -1)
            gamma_beta = self.film_mlp(dts)  # [B, 2*width*num_layers]
            gamma_beta = gamma_beta.view(B, self.num_layers, 2, self.width)  # [B, L, 2, width]

        for i in range(self.num_layers):
            x_res = x
            x = self.norms[i](x)

            gamma = gamma_beta[:, i, 0].unsqueeze(-1).unsqueeze(-1)  # [B, width, 1, 1]
            beta = gamma_beta[:, i, 1].unsqueeze(-1).unsqueeze(-1)
            x = self.apply_film(x, gamma, beta)

            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x)
            x = F.gelu(x1 + x2) + x_res

        x = x[..., :-self.padding, :-self.padding]  # 去掉 padding
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # [B, H, W, out_channels]
        return x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

class ODEPairDataset(Dataset):
    def __init__(self, input_file, output_file, mu=None, sigma=None, limit=8000):
        # Load all three files
        raw_inputs  = torch.load(input_file)
        raw_outputs = torch.load(output_file)

        # Limit data to 2000 samples if they exist, otherwise take all
        num_samples_to_load = min(limit, raw_inputs.shape[0])

        self.inputs = (raw_inputs[:num_samples_to_load]).to(torch.float64)
        self.outputs = (raw_outputs[:num_samples_to_load]).to(torch.float64)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Return inputs, outputs, and epsilons
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

def spectral_loss(pred, target, model, alpha=0.1):
    """
    pred, target: [B, C, H, W], real tensors
    alpha: 高频权重放大因子
    """
    u_pred = pred[:, 0, :, :]
    v_pred = pred[:, 1, :, :]
    u_target = target[:, 0, :, :]
    v_target = target[:, 1, :, :]

    omega_pred = model._velocity_to_vorticity_spectral(u_pred, v_pred, model.Kx_fine, model.Ky_fine)
    omega_target = model._velocity_to_vorticity_spectral(u_target, v_target, model.Kx_fine, model.Ky_fine)

    return F.mse_loss(omega_pred, omega_target)

def total_loss(pred, target, λ_spatial=1.0, λ_spectral=1.0, alpha=0.1):
    # spatial = F.mse_loss(pred, target)
    # spectral = spectral_loss(pred, target, model, alpha)
    # return λ_spatial * spatial + λ_spectral * spectral
    return F.mse_loss(pred, target)


if __name__ == "__main__":
    batch_size = 8
    data_dir = "../dataset"

    train_dataset = ODEPairDataset(
        os.path.join(data_dir, "train_input.pt"),
        os.path.join(data_dir, "train_output.pt"),
        limit=4000,
    )
    val_dataset = ODEPairDataset(
        os.path.join(data_dir, "val_input.pt"),
        os.path.join(data_dir, "val_output.pt"),
        limit=1000,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # model = CNEXTUNet().to(device)
    model = FNO2d(modes1=32, modes2=32, width=64, in_channels=2, out_channels=2).to(device)
    dt    = torch.tensor(0.01).to(device)

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 100
    Tmax   = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-5)

    best_val_loss = float('inf')

    train_loss_lists = []
    val_loss_lists = []

    # Create directories for saving loss files if they don't exist
    os.makedirs('logs', exist_ok=True)
    train_loss_file = open(os.path.join('logs', "train_loss.txt"), "w")
    val_loss_file = open(os.path.join('logs', "val_loss.txt"), "w")

    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Unpack inputs, targets, and epsilons from the DataLoader
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"[train] Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs, dt)
            loss = total_loss(preds, targets)
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
            # Unpack inputs, targets, and epsilons from the DataLoader
            # for inputs, targets in val_loader:
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                print(f"[val] Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(val_loader)}", end='\r')
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs, dt)
                val_loss += total_loss(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create directory for saving models if it doesn't exist
            os.makedirs('.', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', "best_model_fno.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")