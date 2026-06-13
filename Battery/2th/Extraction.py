# ============================================================
# Extraction.py
# Pure-NN rollout -> SINDy to extract interpretable ODE for V
#
# State: V
# Controls: I, SOC, CYCLE
#
# Rollout rule:
#   V_{k+1}  = NN([V_k, I_k, SOC_k, CYCLE_k, DT_k])
#   SOC_{k+1}= clip( SOC_k - I_{k+1} * DT_{k+1} / Q )
#   DT_{k+1} ~ Uniform(9, 11), I_{k+1}=2.0
#
# Then SINDy fits:
#   dV/dt = f(V, I, SOC, CYCLE)
# ============================================================

import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import pysindy as ps
from ANI2 import A, load_data, DATA_PATH
import matplotlib.pyplot as plt
# -------------------------
# 0) Config
# -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

    def forward(self, x):
        dt = x[:, 4:5]               
        dt_half = dt * 0.5

        x_half = x.clone()
        x_half[:, 4:5] = dt_half           
        V_half_prior = self.prior(x_half)   

        x_nn = x.clone()
        x_nn[:, 0:1] = V_half_prior         
        
        delta = self.nn(x_nn)          
        
        residual = delta       
        V_mid = V_half_prior + residual

        dVdt = residual / dt
        # [I, SOC, CYCLE]
        output = torch.cat([x_nn[:, 0:1], x[:, 1:4]], dim=1)

        x_next = x.clone()
        x_next[:, 0:1] = V_mid 
        x_next[:, 4:5] = dt_half       

        V_next = self.prior(x_next)        
        return V_next, dVdt, output

# Indices: [V, I, SOC, CYCLE, DT]
V_IDX, I_IDX, SOC_IDX, CYCLE_IDX, DT_IDX = 0, 1, 2, 3, 4

# Battery capacity for SOC integration
Q_total_As = 2.0 * 3600.0  # 2Ah -> As

# Rollout settings
NX = 3
T_STEPS = 30
I_CONST = 2.0
DT_LOW, DT_HIGH = 9.0, 11.0

# SINDy settings
MIN_TRAJ_LEN = 6  # to avoid finite-difference crash
POLY_DEGREE = 2
THRESHOLD = 1e-3
RIDGE_ALPHA = 1e-6

# Model checkpoint
MODEL_CKPT_PATH = "best_ani2_model.pth"
USE_MODEL_ATTR_NN = True                # True: use model.nn(x), False: use model(x)

# -------------------------
# 1) Initial conditions
# -------------------------
x0 = torch.tensor(
    [
        [4.2012, 0.0000, 1.0000, 0.8982,  9.3590],
        [4.1951, 0.0000, 1.0000, 0.9042, 10.1710],
        [4.1957, 0.0022297, 1.0000, 0.91018, 9.3440],
    ],
    dtype=torch.float64,
    device=device
)

def load_model():
    model = ANI2(
        0,
        0,
        (0,0)
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_CKPT_PATH, map_location=device)
    )
    model.eval()

    print(f"Loaded ANI2 model from {MODEL_CKPT_PATH} on {device}")
    return model


if __name__ == "__main__":
    MODEL_PATH = "best_ani2_model.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    print(f"Loading data from {DATA_PATH} ...")
    train_data, val_data, test_data, prior_params = load_data(DATA_PATH)
    test_x_all, test_y_all = test_data
    
    test_x_all = test_x_all.to(device)
    print(f"Test data shape: {test_x_all.shape}")

    model_origin = ANI2(R_val=0, C_val=0, w_vals=(0,0)).to(device)
    model_origin.load_state_dict(torch.load(MODEL_PATH))
    

    print("Extracting (State, Derivative) pairs from Neural Network...")
    
    BATCH_SIZE = 4096
    X_list, Y_list = [], []

    with torch.no_grad():
        for i in range(0, len(test_x_all), BATCH_SIZE):
            batch_x = test_x_all[i : i + BATCH_SIZE]
            
            V, dVdt, states = model_origin(batch_x)
            
            X_list.append(states.cpu().numpy())
            Y_list.append(dVdt.cpu().numpy())

    X_train = np.concatenate(X_list, axis=0) 
    Y_train = np.concatenate(Y_list, axis=0) 

    print(f"SINDy Training Data: X={X_train.shape}, Y={Y_train.shape}")
    
    print("Fitting SINDy model...")
    
    poly_lib = ps.PolynomialLibrary(degree=2, include_bias=True)
    
    optimizer = ps.STLSQ(threshold=0.02, alpha=1e-5)

    model_sindy = ps.SINDy(
        feature_names=["V", "I", "SOC", "Cyc"],
        feature_library=poly_lib,
        optimizer=optimizer
    )

    model_sindy.fit(x=X_train, x_dot=Y_train)

    print("\n" + "="*50)
    print(" >>> Discovered Physics from ANI2 Residual <<<")
    print(" Equation: d(V_res)/dt = ...")
    print("="*50)
    model_sindy.print()
    
    score = model_sindy.score(X_train, x_dot=Y_train)
    print(f"\nR^2 Score: {score:.4f}")

    idx = np.random.choice(len(X_train), 100)
    y_pred_sindy = model_sindy.predict(X_train[idx])
    y_true_nn = Y_train[idx]

    plt.figure(figsize=(8, 4))
    plt.plot(y_true_nn, label='NN Output (True)', marker='o', linestyle='None', alpha=0.6)
    plt.plot(y_pred_sindy, label='SINDy Prediction', marker='x', linestyle='None', alpha=0.8)
    plt.title("SINDy Fit Verification (dV/dt)")
    plt.xlabel("Sample Index")
    plt.ylabel("dV/dt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("Verification plot created.")
