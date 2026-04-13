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
from einops import rearrange
from torch.utils.checkpoint import checkpoint


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

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
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, num_layers=2):
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


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x)
        x = torch.cat([x, grid], dim=-1)  # [B, H, W, C+2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.pad(x, [0, self.padding, 0, self.padding], mode='circular')

        for i in range(self.num_layers):
            x_res = x

            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x)
            x = F.gelu(x1 + x2) + x_res

        x = x[..., :-self.padding, :-self.padding]  # 去掉 padding
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # [B, H, W, out_channels]
        return x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]
    
class FNO2d_NS(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, in_channels=1, num_layers=4, dt=5e-3):
        super().__init__()
        self.dt = dt
        self.fno = FNO2d(modes1=modes1, modes2=modes2, width=width, num_layers=num_layers, in_channels=in_channels)
    
    def forward(self, x):
        return x + self.dt * self.fno(x)
    
    def predict(self, x):
        return x + self.dt * self.fno(x)
    
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

def spectral_loss(pred, target):
    return F.mse_loss(pred, target)


def total_loss(pred, target):
    return F.l1_loss(pred, target)

if __name__ == "__main__":
    data_dir = "../dataset" # Or "../dataset" if that's where your data is

    train_dataset = ODEPairDataset(
        os.path.join(data_dir, "train_input_new.pt"),
        os.path.join(data_dir, "train_output_new.pt"),
        limit=8000,
    )
    val_dataset   = ODEPairDataset(
        os.path.join(data_dir, "test_input_new.pt"),
        os.path.join(data_dir, "test_output_new.pt"),
        limit=2000,
    )

    batch_size   = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    model = FNO2d_NS(modes1=16, modes2=16, width=64, dt=5e-3).to(device)

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
            preds = model(inputs)
            # loss = criterion(preds, targets)
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
                preds = model(inputs)
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
            torch.save(model.state_dict(), os.path.join('models', "best_model_cnextunet.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")