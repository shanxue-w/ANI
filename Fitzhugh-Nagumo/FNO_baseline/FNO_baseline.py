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
import time

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(555)
np.random.seed(555)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cdouble))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cdouble))

    #Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = torch.einsum("bixy,ioxy->boxy",a,b)
        return op

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1)))
        return x

class FiLM(nn.Module):
    def __init__(self, width, hidden_dim=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(width, affine=False, dtype=torch.float64)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden_dim, width, dtype=torch.float64)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden_dim, width, dtype=torch.float64)
        )

    def forward(self, x, eps):
        # x: [B, C, H, W], eps: [B]
        x = self.norm(x)
        gamma = self.gamma_mlp(eps)
        beta = self.beta_mlp(eps)
        
        # Reshape to [Batch, Width, 1, 1] for broadcasting
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        
        return gamma * x + beta
    
class FNO2d_FiLM(nn.Module):
    def __init__(self, modes1, modes2, width, N=128):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(4, width, dtype=torch.float64)  # [u,v,x,y]

        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.film1 = FiLM(width)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.film2 = FiLM(width)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.film3 = FiLM(width)

        self.fc1 = nn.Linear(width, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 2, dtype=torch.float64)

        x = torch.linspace(0, 1, N+1, dtype=torch.float64)[:-1]
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1)

    def forward(self, x, eps):
        # x: [B, N, N, 2]
        grid = self.grid.to(x.device).to(dtype=torch.float64)
        grid = grid.repeat(x.size(0), 1, 1, 1)
        x    = torch.cat((x, grid), dim=-1)  # [B, N, N, 4]
        x = self.fc0(x).permute(0, 3, 1, 2)  # [B, C, H, W], float64

        x1 = self.conv1(x)
        x1 = self.film1(x1, eps)
        x1 = F.gelu(x1)

        x2 = self.conv2(x1)
        x2 = self.film2(x2, eps)
        x2 = F.gelu(x1 + x2)

        x3 = self.conv3(x2)
        x3 = self.film3(x3, eps)
        x = F.gelu(x2 + x3)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AllenCahn(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, dt=1e-2):
        super(AllenCahn, self).__init__()
        self.dt = dt
        self.fno = FNO2d_FiLM(modes1=modes1, modes2=modes2, width=width)

    def forward(self, x, eps):
        return x + self.dt * self.fno(x, eps)

    def predict(self, x, eps):
        return x + self.dt * self.fno(x, eps)
    

class FHN_Dataset(Dataset):
    def __init__(self, data_dir, split, limit=2000, device='cpu'):
        self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs_all.npy'))[:limit]
        self.outputs = np.load(os.path.join(data_dir, f'{split}_outputs_all.npy'))[:limit]
        self.dt = np.load(os.path.join(data_dir, f'{split}_eps_all.npy'))[:limit]
        self.device = device

        # Ensure data is float32 and move to device if applicable later
        # It's often better to convert to tensor and move to device in __getitem__ or during DataLoader iteration
        # to manage memory efficiently, especially for large datasets.

        # Check for consistency
        assert len(self.inputs) == len(self.outputs) == len(self.dt), "Mismatch in dataset sizes!"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors and move to device
        input_tensor = torch.from_numpy(self.inputs[idx]).to(self.device)
        output_tensor = torch.from_numpy(self.outputs[idx]).to(self.device)
        dt_tensor = torch.tensor(self.dt[idx], dtype=torch.float64).to(self.device) # dt is called eps in this context

        return input_tensor, output_tensor, dt_tensor

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
    # Ensure the 'data' directory exists and contains the generated files
    # For this example, I'll assume the files are in a 'data' directory
    # relative to where this script is run.
    # Adjust paths if your data is in a different location.
    data_dir = "../dataset/data" # Or "../dataset" if that's where your data is

    train_dataset = FHN_Dataset(
        data_dir, 'train_fhn', device=device, limit=4000
    )
    val_dataset   = FHN_Dataset(
        data_dir, 'val_fhn', device=device, limit=1000
    )

    batch_size   = 24
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    model = AllenCahn(modes1=16, modes2=16, width=40, dt=0.01).to(device)
    # checkpoint = torch.load("models/best_model.pth", map_location=device)
    # model.load_state_dict(checkpoint)
    # model = torch.compile(model, mode="max-autotune") # Uncomment if you want to use torch.compile

    # 损失和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )

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
        for batch_idx, (inputs, targets, epsilons) in enumerate(train_loader):
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            inputs, targets, epsilons = inputs.to(device), targets.to(device), epsilons.to(device)
            optimizer.zero_grad()
            # Pass epsilons to the model's forward method
            preds = model(inputs, epsilons)
            loss = criterion(preds, targets)
            # clip gradient
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for inputs, targets, epsilons in val_loader:
                # Pass epsilons to the model's forward method
                inputs, targets, epsilons = inputs.to(device), targets.to(device), epsilons.to(device)
                preds = model(inputs, epsilons)
                val_loss += criterion(preds, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create directory for saving models if it doesn't exist
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', "best_model.pth"))
            print("Saved best model!")
    
    train_loss_file.close()
    val_loss_file.close()

    plot_losses(train_loss_lists, val_loss_lists)
    print("\nTraining complete!")