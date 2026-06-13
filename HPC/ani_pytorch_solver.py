#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ani_pytorch_solver.py

Path 2 (ANI-style with differentiable proxy prior in PyTorch):
  - Use a differentiable RK4 proxy for the prior step map S_dt.
  - Train an ANI Strang-composed corrector end-to-end:
        u1 = C_{dt/2}(u0)
        u2 = S_dt(u1)           # proxy prior in PyTorch (RK4)
        u3 = C_{dt/2}(u2)
    where C_dt(u) = u + dt * g_theta(u).
  - Export the trained model as TorchScript for C++ (libtorch) inference.

Notes:
  - This is the "keep ANI structure" option: gradients flow through the proxy prior.
  - Deployment can export:
        (a) the full pipeline (corrector + proxy prior) as one TorchScript module, or
        (b) the corrector alone, then couple with a compiled prior at inference if desired.

Dataset:
  - Uses the same .npz from data.py: X, Y, dt
    X: state at time t
    Y: reference state at time t + dt (from dt_small=1e-3)
    dt should be dt_flow=1e-1
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainCfg:
    batch_size: int = 1024
    epochs: int = 300
    lr: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 0
    width: int = 128
    depth: int = 3
    dtype: torch.dtype = torch.float64


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def ho_rhs(u: torch.Tensor) -> torch.Tensor:
    """
    Prior ODE (harmonic oscillator):
      x' = y
      y' = -x
    u: [B,2]
    """
    x = u[..., 0]
    y = u[..., 1]
    return torch.stack([y, -x], dim=-1)


def rk4_step_torch(u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    One RK4 step for harmonic oscillator prior. Differentiable.
    """
    dt_t = torch.tensor(dt, dtype=u.dtype, device=u.device)
    k1 = ho_rhs(u)
    k2 = ho_rhs(u + 0.5 * dt_t * k1)
    k3 = ho_rhs(u + 0.5 * dt_t * k2)
    k4 = ho_rhs(u + dt_t * k3)
    return u + (dt_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=128, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.GELU())
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualCorrector(nn.Module):
    """
    C_dt(u) = u + dt * g_theta(u)
    """
    def __init__(self, width=128, depth=3):
        super().__init__()
        self.g = MLP(3, 2, width=width, depth=depth)

    def forward(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        dt_t = torch.tensor(dt, dtype=u.dtype, device=u.device)
        batch_size = u.shape[0]
        if dt_t.dim() == 0: 
             dt_feature = dt_t.view(1, 1).expand(batch_size, 1)
        else: 
             dt_feature = dt_t.view(-1, 1)
             if dt_feature.shape[0] == 1:
                 dt_feature = dt_feature.expand(batch_size, 1)
        net_input = torch.cat([u, dt_feature], dim=1)
        return u + dt_t * self.g(net_input)


class ANI_Strang(nn.Module):
    """
    u_{n+1} = C_{dt/2}( S_dt( C_{dt/2}(u_n) ) )
    """
    def __init__(self, dt: float, width=128, depth=3):
        super().__init__()
        self.dt = float(dt)
        self.corrector = ResidualCorrector(width=width, depth=depth)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        u1 = rk4_step_torch(u0, dt/2)
        u2 = self.corrector(u1, dt)
        u3 = rk4_step_torch(u2, dt/2)
        return u3


@torch.no_grad()
def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.norm(pred - target, dim=-1)
    den = torch.norm(target, dim=-1).clamp_min(eps)
    return (num / den).mean().item()


def train(model: nn.Module, train_loader, test_loader, cfg: TrainCfg, device: torch.device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_mse = float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]

        # if ep == 1 or ep % 20 == 0:
        model.eval()
        with torch.no_grad():
            mse_sum = 0.0
            r_sum = 0.0
            m = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                mse_sum += F.mse_loss(pred, y, reduction="sum").item()
                r_sum += rel_l2(pred, y) * x.shape[0]
                m += x.shape[0]
        print(f"[ANI/Strang] epoch {ep:4d} | train_mse={loss_sum/max(1,n):.3e} | test_mse={mse_sum/max(1,m):.3e} | test_relL2={r_sum/max(1,m):.3e}")
        if mse_sum < best_mse:
            best_mse = mse_sum
            torch.save(model.state_dict(), "best_ani_strang.pth")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Path to .npz produced by data.py")
    p.add_argument("--out_ts_full", type=str, default="ani_strang_full.ts", help="TorchScript output for full pipeline")
    p.add_argument("--out_ts_corrector", type=str, default="ani_corrector_only.ts", help="TorchScript output for corrector module only")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--test_split", type=float, default=0.1)
    args = p.parse_args()

    cfg = TrainCfg(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        width=args.width,
        depth=args.depth,
    )

    set_seed(cfg.seed)
    torch.set_default_dtype(cfg.dtype)
    device = torch.device(args.device)

    npz = np.load(args.dataset)
    X = torch.from_numpy(npz["X"].astype(np.float64))
    Y = torch.from_numpy(npz["Y"].astype(np.float64))
    dt = float(npz["dt"])

    # train/test split
    N = X.shape[0]
    n_test = max(1, int(round(args.test_split * N)))
    idx = torch.randperm(N)
    te = idx[:n_test]
    tr = idx[n_test:]

    train_ds = TensorDataset(X[tr], Y[tr])
    test_ds = TensorDataset(X[te], Y[te])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ANI_Strang(dt=dt, width=cfg.width, depth=cfg.depth).to(device=device, dtype=cfg.dtype)
    train(model, train_loader, test_loader, cfg, device)

    model.load_state_dict(torch.load("best_ani_strang.pth"))

    # Export full pipeline
    model_cpu = model.cpu().eval()
    scripted_full = torch.jit.script(model_cpu)
    out_full = Path(args.out_ts_full)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    scripted_full.save(str(out_full))
    print(f"[ANI/Strang] Saved full TorchScript: {out_full}")

    corr = model_cpu.corrector.eval()
    scripted_corr = torch.jit.script(corr)
    out_corr = Path(args.out_ts_corrector)
    out_corr.parent.mkdir(parents=True, exist_ok=True)
    scripted_corr.save(str(out_corr))
    print(f"[ANI/Strang] Saved corrector TorchScript: {out_corr}")


if __name__ == "__main__":
    main()
