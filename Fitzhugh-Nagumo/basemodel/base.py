import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time # For timing the training

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(4, self.width) # input channel is 4: (U, V, x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class FNO_ETDRK4(nn.Module):
    def __init__(self, modes1, modes2, width, N=128, Du=1e-2, Dv=1e-2, device='cuda:1'):
        super(FNO_ETDRK4, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.N = N # Spatial resolution
        self.Du = Du # Diffusion coefficient for u
        self.Dv = Dv # Diffusion coefficient for v
        self.device = device

        self.fno = FNO2d(modes1, modes2, width)

        self._init_fft_grid()
        self.etdrk4_coeff_cache = {}

    def _init_fft_grid(self):
        k_x = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(k_x, k_y, indexing='ij')
        self.K2 = -(KX**2 + KY**2) # [N, N]

        # Prepare grid for FNO input
        grid_x = torch.linspace(0, 1, self.N+1, device=self.device)[:-1]
        grid_y = torch.linspace(0, 1, self.N+1, device=self.device)[:-1]
        # Stack as [N, N, 2]
        self.grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1) # [N, N, 2]

    def _compute_etdrk4_coeffs(self, dt_val):
        # dt_val should be a scalar
        L_u = self.Du * self.K2 # Linear part for u [N, N]
        L_v = self.Dv * self.K2 # Linear part for v [N, N]

        # Calculate coefficients for u
        E_u = torch.exp(dt_val * L_u) # [N, N]
        E2_u = torch.exp(dt_val * L_u / 2) # [N, N]
        Q_u, f1_u, f2_u, f3_u = self._phi_functions(dt_val, L_u) # All [N, N]

        # Calculate coefficients for v
        E_v = torch.exp(dt_val * L_v) # [N, N]
        E2_v = torch.exp(dt_val * L_v / 2) # [N, N]
        Q_v, f1_v, f2_v, f3_v = self._phi_functions(dt_val, L_v) # All [N, N]

        # Stack coefficients along the channel dimension (dim=0)
        # Resulting shapes: [2, N, N]
        E = torch.stack([E_u, E_v], dim=0)     # [2, N, N]
        E2 = torch.stack([E2_u, E2_v], dim=0)  # [2, N, N]
        Q = torch.stack([Q_u, Q_v], dim=0)     # [2, N, N]
        f1 = torch.stack([f1_u, f1_v], dim=0)  # [2, N, N]
        f2 = torch.stack([f2_u, f2_v], dim=0)  # [2, N, N]
        f3 = torch.stack([f3_u, f3_v], dim=0)  # [2, N, N]

        return E, E2, Q, f1, f2, f3

    def _phi_functions(self, dt, L):
        # Helper function to compute the phi functions for ETDRK4
        # L: [N, N]
        # Returns Q, f1, f2, f3 for a single component (u or v)

        M = 64 # Number of points for contour integral
        j = torch.arange(1, M + 1, device=self.device)
        r = torch.exp(2j * torch.pi * (j - 0.5) / M) # [M]
        # Reshape L to [N, N, 1] for broadcasting with r, then to [N*N, 1] for easier complex ops
        LR = dt * L.unsqueeze(-1) + r # [N, N, M]

        # Ensure that division by zero (LR=0) is handled
        epsilon = 1e-10 # Small epsilon to avoid division by zero
        LR_safe = LR + epsilon * (LR == 0).float() # Add epsilon only where LR is zero

        exp_LR = torch.exp(LR)
        exp_LR_half = torch.exp(LR / 2)

        Q_integral = torch.mean((exp_LR_half - 1) / LR_safe, dim=-1).real
        f1_integral = torch.mean((-4 - LR_safe + exp_LR * (4 - 3 * LR_safe + LR_safe**2)) / (LR_safe**3), dim=-1).real
        f2_integral = torch.mean((2 + LR_safe + exp_LR * (-2 + LR_safe)) / (LR_safe**3), dim=-1).real
        f3_integral = torch.mean((-4 - 3 * LR_safe - LR_safe**2 + exp_LR * (4 - LR_safe)) / (LR_safe**3), dim=-1).real

        Q = dt * Q_integral
        f1 = dt * f1_integral
        f2 = dt * f2_integral
        f3 = dt * f3_integral

        return Q, f1, f2, f3


    def forward(self, x, dts):
        # x: [B, N, N, 2] (u and v as channels)
        # dts: [B] (dt value for each sample in the batch)
        if isinstance(dts, torch.Tensor) and len(dts.shape) == 0:
            dts = dts.repeat(x.shape[0])

        u = x[:,:,:,0:2].permute(0,3,1,2)
        # grid = x[:,:,:,2:]
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, grid], dim=-1) # [B, N, N, 4]

        B, C, H, W = u.shape # C will be 2 (for u, v)

        # Get unique dt values and their inverse indices
        unique_dts, inverse_indices = torch.unique(dts, return_inverse=True)

        # Prepare coefficients for the batch
        # Coefficients will be [B, C, H, W]
        batch_E = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_E2 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_Q = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f1 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f2 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)
        batch_f3 = torch.empty((B, C, H, W), device=self.device, dtype=u.dtype)

        for i, dt_val in enumerate(unique_dts):
            dt_val_item = dt_val.item() # Convert tensor scalar to Python float for dict key
            if dt_val_item not in self.etdrk4_coeff_cache:
                # _compute_etdrk4_coeffs returns [C, H, W]
                E, E2, Q, f1, f2, f3 = self._compute_etdrk4_coeffs(dt_val)
                self.etdrk4_coeff_cache[dt_val_item] = (E, E2, Q, f1, f2, f3)
            else:
                E, E2, Q, f1, f2, f3 = self.etdrk4_coeff_cache[dt_val_item]

            # Assign coefficients to the correct batch indices
            indices_in_batch = (inverse_indices == i).nonzero(as_tuple=True)[0]
            # E, E2, ... are [C, H, W]. Need to expand them to [1, C, H, W] before assigning
            # and then broadcast them to the batch indices.
            batch_E[indices_in_batch] = E.unsqueeze(0)
            batch_E2[indices_in_batch] = E2.unsqueeze(0)
            batch_Q[indices_in_batch] = Q.unsqueeze(0)
            batch_f1[indices_in_batch] = f1.unsqueeze(0)
            batch_f2[indices_in_batch] = f2.unsqueeze(0)
            batch_f3[indices_in_batch] = f3.unsqueeze(0)


        
        # stage 1
        u_hat = torch.fft.fft2(u)
        nu = (self.fno(x)).permute(0,3,1,2)
        nu_hat = torch.fft.fft2(nu)
        
        a_hat = batch_E2 * u_hat + batch_Q * nu_hat
        a = torch.fft.ifft2(a_hat).real
        
        # stage 2
        au = torch.cat((a.permute(0,2,3,1), grid), dim=-1)
        na = (self.fno(au)).permute(0,3,1,2)
        na_hat = torch.fft.fft2(na)
        
        b_hat = batch_E2 * u_hat + batch_Q * na_hat
        b = torch.fft.ifft2(b_hat).real
        
        # stage 3
        bu = torch.cat((b.permute(0,2,3,1), grid), dim=-1)
        nb = (self.fno(bu)).permute(0,3,1,2)
        nb_hat = torch.fft.fft2(nb)
        
        c_hat = batch_E2 * a_hat + batch_Q * (2*nb_hat - nu_hat)
        c = torch.fft.ifft2(c_hat).real
        
        # stage 4
        cu = torch.cat((c.permute(0,2,3,1), grid), dim=-1)
        nc = (self.fno(cu)).permute(0,3,1,2)
        nc_hat = torch.fft.fft2(nc)
        
        u_hat = batch_E * u_hat + batch_f1 * nu_hat + 2 * batch_f2 * (na_hat + nb_hat) + batch_f3 * nc_hat
        u = torch.fft.ifft2(u_hat).real
        
        return u.permute(0,2,3,1)

class FHN_Dataset(Dataset):
    def __init__(self, data_dir, split, device='cpu'):
        self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs.npy'))[:3000]
        self.outputs = np.load(os.path.join(data_dir, f'{split}_outputs.npy'))[:3000]
        self.dt = np.load(os.path.join(data_dir, f'{split}_dt.npy'))[:3000]
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

# --- Training Setup ---
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    print(f"\n--- Starting Training on {device} ---")
    model.train()

    best_val_loss = float('inf')  # 初始化为正无穷

    for epoch in range(num_epochs):
        start_time = time.time()
        total_train_loss = 0.0

        for batch_idx, (inputs, targets, dts) in enumerate(train_loader):
            # inputs, targets, dts = inputs.to(device), targets.to(device), dts.to(device)

            optimizer.zero_grad()
            predictions = model(inputs, dts)
            loss = criterion(predictions, targets)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.5e}")

        avg_train_loss = total_train_loss / len(train_loader)
        end_time = time.time()
        epoch_duration = end_time - start_time

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs_val, targets_val, dts_val in val_loader:
                # inputs_val, targets_val, dts_val = inputs_val.to(device), targets_val.to(device), dts_val.to(device)
                predictions_val = model(inputs_val, dts_val)
                val_loss = criterion(predictions_val, targets_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # ✨ Save best model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {avg_val_loss:.5e}")

        model.train()

        print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_duration:.2f}s")
        print(f"Train Loss: {avg_train_loss:.5e}, Val Loss: {avg_val_loss:.5e}")

    print("\n--- Training Complete ---")

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Define Hyperparameters and Device
    DATA_DIR = '../dataset/data_dt' # Directory where your .npy files are stored
    BATCH_SIZE = 32 # Use a smaller batch size for initial testing
    NUM_EPOCHS = 200 # Number of training epochs
    LEARNING_RATE = 1e-3
    # Automatically select GPU if available, otherwise CPU
    
    print(f"Using device: {DEVICE}")

    # Model parameters (adjust these based on your FNO architecture)
    MODES1 = 8
    MODES2 = 8
    WIDTH = 40
    N_SPATIAL = 128 # This should match the N used when generating the data (downsampled N)

    # 2. Load Data
    print("Loading datasets...")
    train_dataset = FHN_Dataset(DATA_DIR, 'train', device=DEVICE)
    val_dataset = FHN_Dataset(DATA_DIR, 'val', device=DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")

    # 3. Instantiate Model, Loss, and Optimizer
    model = FNO_ETDRK4(modes1=MODES1, modes2=MODES2, width=WIDTH, N=N_SPATIAL, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight_decay
    criterion = nn.MSELoss() # Mean Squared Error Loss

    # 4. Start Training
    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)

    # Optional: Save the trained model
    # torch.save(model.state_dict(), 'fno_etdrk4_fhn_model.pth')
    # print("Model saved to fno_etdrk4_fhn_model.pth")