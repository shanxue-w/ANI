import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

# ================= 1. Config =================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASIN_ID = "01031500"
DATA_FILE = f"../dataset/data_{BASIN_ID}.pt"
MODEL_FILE = f"ANI2_Best_{BASIN_ID}.pth"

SEQ_LEN = 60
PRED_STEPS = 30
BATCH_SIZE = 64
HIDDEN_DIM = 16
LR = 1e-3
EPOCHS = 100
DT = 1.0

NOISE_STD = 1e-3
DRIFT_WEIGHT = 1e-2

# ================= 2. Dataset =================
class CatchmentRolloutDataset(Dataset):
    def __init__(self, data_dict, part='train', seq_len=30, pred_steps=7):
        self.X_raw = data_dict[part]['X_raw']   # [T, d]
        self.X_norm = data_dict[part]['X_norm'] # [T, d]
        self.y_raw = data_dict[part]['y_raw']   # [T] or [T,1]
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.n_samples = len(self.y_raw) - seq_len - pred_steps + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_hist_norm = self.X_norm[idx : idx + self.seq_len]  # [L,d]
        x_future_raw = self.X_raw[idx + self.seq_len - 1 : idx + self.seq_len + self.pred_steps]   # [K+1,d]
        x_future_norm = self.X_norm[idx + self.seq_len - 1 : idx + self.seq_len + self.pred_steps] # [K+1,d]
        q_start = self.y_raw[idx + self.seq_len - 1]
        y_target = self.y_raw[idx + self.seq_len : idx + self.seq_len + self.pred_steps]
        return x_hist_norm, x_future_raw, x_future_norm, q_start, y_target

# ================= 3. Model =================
class SimpleTransformerCorrector(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=1, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model + 1 + input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, history_norm, q_curr_norm, x_norm_t):
        B, L, _ = history_norm.shape
        x = self.input_proj(history_norm)                               # [B,L,D]
        pos = self.pos_emb(torch.arange(L, device=x.device)).unsqueeze(0)
        x = x + pos  # add position

        cls = self.cls.expand(B, -1, -1)                                 # [B,1,D]
        x = torch.cat([cls, x], dim=1)                                   # [B,1+L,D]
        h = self.encoder(x)
        h_pool = h[:, 0]                                                 # CLS pooling

        q_curr_norm = q_curr_norm.view(-1, 1)
        x_norm_t = x_norm_t.view(B, -1)
        feat = torch.cat([h_pool, q_curr_norm, x_norm_t], dim=-1)
        return self.out(feat)


class PhysicalA(nn.Module):
    def __init__(self, k, alpha):
        super().__init__()
        self.register_buffer("k", torch.tensor(float(k)))
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, Q_raw, P_raw, T_raw, dt):
        # return Q_raw + self.k * (P_raw - self.alpha * T_raw - Q_raw) * dt
        gamma = torch.exp(-self.k * dt)
        forcing = P_raw - self.alpha * T_raw
        return gamma * Q_raw + (1 - gamma) * forcing

class ANI_2th_Hydro(nn.Module):
    def __init__(self, ode_prior, stats, input_dim, hidden_dim=64):
        super().__init__()
        self.A = PhysicalA(ode_prior['k'], ode_prior['alpha'])
        self.NN = SimpleTransformerCorrector(input_dim=input_dim, d_model=hidden_dim)

        self.register_buffer('y_mean', stats['y_mean'])
        self.register_buffer('y_std', stats['y_std'])

    def _strang_step(self, q, x_raw_t, x_norm_t, history_norm, dt):
        P_raw, T_raw = x_raw_t[:, 0:1], x_raw_t[:, 1:2]

        # prior half-step
        q_half = self.A(q, P_raw, T_raw, dt/2)

        # NN correction (NN 能看到 forcing)
        q_half_norm = (q_half - self.y_mean) / (self.y_std + 1e-12)
        drift = self.NN(history_norm, q_half_norm, x_norm_t)  # [B,1]

        # corrected
        q_corr = q_half + drift * dt * self.y_std

        # # prior half-step
        q_next = self.A(q_corr, P_raw, T_raw, dt/2)
        # q_next = self.A(q_half, P_raw, T_raw, dt/2)
        
        return F.softplus(q_next), drift

    def forward(self, history_norm, x_future_raw, x_future_norm, q_start, pred_steps, dt=1.0):
        preds, drifts = [], []
        curr_q = q_start
        if curr_q.dim() == 1:
            curr_q = curr_q.unsqueeze(-1)

        curr_history = history_norm.clone()

        for t in range(pred_steps):
            x_raw_t = x_future_raw[:, t+1]
            x_norm_t = x_future_norm[:, t+1]

            curr_q, d = self._strang_step(curr_q, x_raw_t, x_norm_t, curr_history, dt)
            preds.append(curr_q)
            drifts.append(d)

            # 更新 history（把当前 forcing 加进去）
            curr_history = torch.cat([curr_history[:, 1:], x_norm_t.unsqueeze(1)], dim=1)

        return torch.stack(preds, dim=1), torch.stack(drifts, dim=1)

class PeakWeightedMSELoss(nn.Module):
    def __init__(self, drift_weight=1e-2):
        super().__init__()
        self.drift_weight = drift_weight

    def forward(self, y_pred, y_true, drifts=None):
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)
        
        weights = torch.log1p(torch.abs(y_true)) + 1.0
        
        mse = torch.mean(weights * (y_pred - y_true) ** 2)
        
        drift_loss = torch.mean(drifts ** 2) if drifts is not None else 0.0
        return mse + self.drift_weight * drift_loss

# ================= 5. Training Loop with Curriculum =================
if __name__ == "__main__":
    assert os.path.exists(DATA_FILE), f"DATA_FILE not found: {DATA_FILE}"
    full_data = torch.load(DATA_FILE, weights_only=False)

    input_dim = int(full_data['train']['X_norm'].shape[-1])

    train_ds = CatchmentRolloutDataset(full_data, 'train', SEQ_LEN, PRED_STEPS)
    val_ds = CatchmentRolloutDataset(full_data, 'val', SEQ_LEN, PRED_STEPS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ANI_2th_Hydro(full_data['ode_prior'], full_data['stats'], input_dim, HIDDEN_DIM).to(device)
    
    criterion = PeakWeightedMSELoss(drift_weight=DRIFT_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_val_nse = -float('inf')

    def nse(true, pred):
        true = true.reshape(-1)
        pred = pred.reshape(-1)
        num = torch.sum((true - pred) ** 2)
        den = torch.sum((true - torch.mean(true)) ** 2) + 1e-8
        return 1 - num / den

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        
        if epoch < 5:
            curr_steps = 1
        else:
            progress = (epoch - 5) / (EPOCHS - 5)
            curr_steps = int(1 + progress * (PRED_STEPS - 1))
            curr_steps = min(curr_steps, PRED_STEPS) # 确保不超过


        for x_hist, x_f_raw, x_f_norm, q_start, y_target in train_loader:
            x_hist = x_hist.to(device)
            x_f_raw = x_f_raw.to(device)
            x_f_norm = x_f_norm.to(device)
            q_start = q_start.to(device)
            y_target = y_target.to(device)

            if q_start.dim() == 1: q_start = q_start.unsqueeze(-1)
            if y_target.dim() == 2: y_target = y_target.unsqueeze(-1)

            x_hist = x_hist + torch.randn_like(x_hist) * NOISE_STD

            optimizer.zero_grad()
            
            y_pred, drifts = model(x_hist, x_f_raw, x_f_norm, q_start, curr_steps, dt=DT)
            
            loss = criterion(y_pred, y_target[:, :curr_steps], drifts[:, :curr_steps])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss_sum += loss.item() * x_hist.size(0)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_hist, x_f_raw, x_f_norm, q_start, y_target in val_loader:
                x_hist = x_hist.to(device)
                x_f_raw = x_f_raw.to(device)
                x_f_norm = x_f_norm.to(device)
                q_start = q_start.to(device)
                y_target = y_target.to(device)

                if q_start.dim() == 1: q_start = q_start.unsqueeze(-1)
                if y_target.dim() == 2: y_target = y_target.unsqueeze(-1)

                y_pred, _ = model(x_hist, x_f_raw, x_f_norm, q_start, PRED_STEPS, dt=DT)
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())

        full_preds = torch.cat(all_preds, dim=0)
        full_targets = torch.cat(all_targets, dim=0)
        val_nse = nse(full_targets, full_preds)
        
        # val_loss = torch.nn.MSELoss()(full_preds, full_targets)
        
        scheduler.step(val_nse)
        # scheduler.step()

        if val_nse > best_val_nse:
            best_val_nse = val_nse
            torch.save(model.state_dict(), MODEL_FILE)
            

        print(f"Epoch {epoch+1:03d}/{EPOCHS} [Steps={curr_steps}] | Loss: {train_loss_sum/len(train_loader.dataset):.4f} | Val NSE: {val_nse.item():.4f} | Best: {best_val_nse:.4f}")