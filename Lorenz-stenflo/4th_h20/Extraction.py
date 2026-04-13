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
import time
import pysindy
from ANI4 import A, LorenzStenflo, ODEPairDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

train_dataset = ODEPairDataset("../dataset/train_inputs.npy", "../dataset/train_outputs.npy")
mu, sigma     = train_dataset.mu, train_dataset.sigma
print(mu, sigma)

model = LorenzStenflo(N0_SCHEME=A(mu, sigma), input_dim=5, output_dim=4, hidden_layers=4, hidden_dim=20).to(device)
model.get_mu_and_sigma(mu, sigma)
model = torch.compile(model, mode="max-autotune")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


nx = 5
x0 = torch.rand((nx, 4)).to(device)
x_lists = [[] for i in range(nx)]
dts = torch.tensor([[2.5e-2]]).to(device)
for i in range(nx):
    x = x0[i:i+1, :].to(device)
    x_lists[i].append(x.cpu().detach().numpy())
    for j in range(1000):
        x = model.residual_block(torch.cat([x, dts], dim=-1))
        x_lists[i].append(x.cpu().detach().numpy())

import numpy as np
import pysindy as ps


train_trajectories = []

for traj in x_lists:
    clean_traj = np.vstack(traj)
    train_trajectories.append(clean_traj)

dt_val = 2.5e-2 

feature_library = ps.PolynomialLibrary(degree=1, include_bias=True)

optimizer = ps.STLSQ(threshold=5e-3)

model = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["x", "y", "z", "w"]  
)

model.fit(train_trajectories, t=dt_val, multiple_trajectories=True)


A = model.coefficients()[:, 1:]
b = model.coefficients()[:, 0:1]
Sigma = np.diag(sigma.cpu().numpy())
mu = mu.cpu().numpy()
# print(Sigma)
A_true = Sigma @ A @ np.linalg.inv(Sigma)
b_true = Sigma @ b - A_true @ mu.reshape(-1,1)
model.coefficients()[:, 1:] = A_true
model.coefficients()[:, 0:1] = b_true

# item less than 1e-2 set to zero
coef = model.coefficients()
coef[np.abs(coef) < 1e-2] = 0.0
model.coefficients()[:] = coef

print("--- Discovered dynamics ---")
model.print()

# score = model.score(train_trajectories, t=dt_val, multiple_trajectories=True)
# model.score(...) for R^2

print("--- Reference missing-physics form ---")
print(f"(x)' = 1.5w")
print(f"(y)' = 0")
print(f"(z)' = 0")
print(f"(w)' = -x-w")