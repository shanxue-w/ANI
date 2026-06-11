import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize 
import torch.nn.utils as utils
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import random # Added for scheduled sampling

# ================= 1. Config =================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_PATH = "../dataset/processed_battery_data_rollout.pt"
BATCH_SIZE = 256
EPOCHS = 400
LR = 2e-3

LAMBDA_SMOOTH = 0.1  

class A(nn.Module):
    def __init__(self, R_val, C_val, w_vals):
        super().__init__()
        self.register_buffer("R", torch.tensor(R_val, dtype=torch.float64))
        self.register_buffer("C", torch.tensor(C_val, dtype=torch.float64))
        self.register_buffer("w0", torch.tensor(w_vals[0], dtype=torch.float64))
        self.register_buffer("w1", torch.tensor(w_vals[1], dtype=torch.float64))
        self.register_buffer("tau", torch.tensor(R_val * C_val, dtype=torch.float64))

    def get_ocv(self, soc):
        return self.w0 + self.w1 * soc
    
    def forward(self, x):
        # x: [V, I_load, SoC, Cycle, dt]
        V_t = x[:, 0]
        I_t = x[:, 1]
        SoC_t = x[:, 2]
        dt = x[:, 4]

        V_ocv = self.get_ocv(SoC_t)
        V_inf = V_ocv - I_t * self.R

        dt = torch.clamp(dt, min=0.0)
        alpha = torch.exp(-dt / self.tau)
        V_next = V_inf + (V_t - V_inf) * alpha
        return V_next.unsqueeze(1)
    
class ANI2(nn.Module):
    def __init__(self, R_val, C_val, w_vals, hidden_dim=64, hidden_layers=4):
        super().__init__()
        self.prior = A(R_val, C_val, w_vals)

        layers = []
        layers.append(nn.Linear(5, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.nn = nn.Sequential(*layers)

    # def forward(self, x):
    #     dt = x[:, 4:5]
    #     dt_half = dt * 0.5

    #     x_half = x.clone()
    #     x_half[:, 4:5] = dt_half
    #     V_half_prior = self.prior(x_half)

    #     x_nn = x.clone()
    #     x_nn[:, 0:1] = V_half_prior
        
    #     delta = self.nn(x_nn)
        
    #     V_mid = V_half_prior + delta

    #     x_next = x.clone()
    #     x_next[:, 0:1] = V_mid
    #     x_next[:, 4:5] = dt_half

    #     V_next = self.prior(x_next)
    #     return V_next

    def forward(self, x):
        dt = x[:, 4:5]
        dt_half = dt * 0.5

        x_half = x.clone()
        x_half[:, 4:5] = dt_half
        V_half_prior = self.prior(x_half)

        x_nn = x.clone()
        x_nn[:, 0:1] = V_half_prior
        
        delta = self.nn(x_nn)
        
        V_mid = V_half_prior + delta

        x_next = x.clone()
        x_next[:, 0:1] = V_mid
        x_next[:, 4:5] = dt_half

        V_next = self.prior(x_next)
        return V_next

    def forward_prior(self, x):
        dt = x[:, 4:5]
        dt_half = dt * 0.5

        x_half = x.clone()
        x_half[:, 4:5] = dt_half
        V_half_prior = self.prior(x_half)

        x_next = x.clone()
        x_next[:, 0:1] = V_half_prior
        x_next[:, 4:5] = dt_half

        V_next = self.prior(x_next)
        return V_next

# ================= 3. Training Utils =================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    
    data = torch.load(path, map_location='cpu')
    print(f"Loaded data from {path}")
    print(f"Prior Params: R={data['prior_params']['R']:.4f}, C={data['prior_params']['C']:.1f}")
    
    X_train = data['X_train'].to(torch.float64)
    Y_train = data['Y_train'].to(torch.float64)
    X_val = data['X_val'].to(torch.float64)
    Y_val = data['Y_val'].to(torch.float64)
    X_test = data['X_test'].to(torch.float64)
    Y_test = data['Y_test'].to(torch.float64)
    
    R_val = data['prior_params']['R']
    C_val = data['prior_params']['C']
    w_vals = data['prior_params']['w']
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (R_val, C_val, w_vals)

def train_epoch(model, loader, optimizer, criterion, device='cuda', teacher_forcing_ratio=0.0):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    V_IDX = 0 
    
    for i, (x, y) in enumerate(loader):
        print(f"{i}", end='\r')
        x, y = x.to(device), y.to(device)
        
        model_dtype = next(model.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
            y = y.to(dtype=model_dtype)
            
        optimizer.zero_grad()
        
        batch_size, seq_len, n_feats = x.shape
        preds = []
        
        current_V = x[:, 0, V_IDX] 
        
        for t in range(seq_len):
            step_input = x[:, t, :].clone() 
            
            if t > 0:
                use_ground_truth = random.random() < teacher_forcing_ratio
                if use_ground_truth:
                    step_input[:, V_IDX] = x[:, t, V_IDX] 
                else:
                    step_input[:, V_IDX] = current_V

            V_next = model(step_input) 
            preds.append(V_next)
            current_V = V_next.squeeze(-1) 
            
        preds = torch.stack(preds, dim=1)
        if preds.shape != y.shape:
            preds = preds.view_as(y)
        
        
        mse_loss = torch.mean((preds - y) ** 2)
        # change to L1 loss
        # mse_loss = torch.mean(torch.abs(preds - y))
        
        diff = preds[:, 1:] - preds[:, :-1]
        smooth_loss = torch.mean(diff ** 2)
        
        loss = mse_loss + LAMBDA_SMOOTH * smooth_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return total_loss / total_samples if total_samples > 0 else 0.0

def validate(model, loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    V_IDX = 0
    
    with torch.no_grad(): 
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            model_dtype = next(model.parameters()).dtype
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype)
                y = y.to(dtype=model_dtype)
            
            batch_size, seq_len, n_feats = x.shape
            preds = []
            
            current_V = x[:, 0, V_IDX]
            
            for t in range(seq_len):
                step_input = x[:, t, :].clone()
                step_input[:, V_IDX] = current_V
                
                V_next = model(step_input)
                preds.append(V_next)
                
                current_V = V_next.squeeze(-1)
            
            preds = torch.stack(preds, dim=1)

            if preds.shape != y.shape:
                preds = preds.view_as(y)
                
            loss = criterion(preds, y)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    return total_loss / total_samples if total_samples > 0 else 0.0

if __name__ == "__main__":
    # 1. Load Data
    train_data, val_data, test_data, prior_params = load_data(DATA_PATH)
    
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    R_val, C_val, w_vals = prior_params
    model = ANI2(R_val, C_val, w_vals).to(device)
    # model = torch.compile(model, mode='max-autotune')
    
    print(f"Model initialized on {device}")
    
    # 3. Setup Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_model_path = "best_ani2_model.pth"
    history = {'train_loss': [], 'val_loss': []}
    
    # 4. Training Loop
    print("Start Training...")
    for epoch in range(EPOCHS):
        decay_steps = EPOCHS * 0.2
        tf_ratio = max(0.0, 1.0 - epoch / decay_steps)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=tf_ratio)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{EPOCHS} | TF: {tf_ratio:.2f} | Train: {train_loss:.2e} | Val: {val_loss:.2e} (*) Saved")
        else:
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | TF: {tf_ratio:.2f} | Train: {train_loss:.2e} | Val: {val_loss:.2e}")
                
    # 5. Final Evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_loss = validate(model, val_loader, criterion, device)
    print(f"\nTraining Finished. Best Val Loss: {best_val_loss:.2e}")
    print(f"Val Set MSE Loss: {test_loss:.2e}")
    
    # 6. Visualization
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = test_data[0].to(device), test_data[1].to(device)
        
        prior_pred = model.prior(x_sample).squeeze(-1).cpu().numpy()
        ani2_pred = model(x_sample).squeeze(-1).cpu().numpy()

        y_true = y_sample.squeeze(-1).cpu().numpy()


        plt.figure(figsize=(12, 5))
        plt.plot(np.abs(prior_pred - y_true), 'r--', label='Prior Only (Linear)', alpha=0.5)
        plt.plot(np.abs(ani2_pred - y_true), 'g--', label='ANI2 (Prior+Res)', alpha=0.8)
        plt.legend()
        plt.title("Test Set Prediction: Prior vs ANI2")
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.savefig("ani2_result.png")
        print("Result plot saved to ani2_result.png")