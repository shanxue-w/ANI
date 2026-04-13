"""
Rollout evaluation on the battery test split: one trajectory per held-out discharge cycle.

Use this to report mean ± std over **all** test cycles (and optionally over multiple cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Matches dataset column order in process_and_split_dynamic
V_IDX = 0
I_IDX = 1
SOC_IDX = 2
CYCLE_IDX = 3
DT_IDX = 4

# Rollout coulomb capacity when not using bundle meta (matches ``Battery/2th/test.py``).
DEFAULT_ROLLOUT_Q0_AH = 2.0


def _squeeze_voltage(out: torch.Tensor) -> torch.Tensor:
    if out.ndim == 2 and out.shape[-1] == 1:
        return out.squeeze(-1)
    return out


def predict_voltage(
    model: torch.nn.Module,
    x: torch.Tensor,
    mode: str = "full",
) -> torch.Tensor:
    """
    mode:
      - ``full``: ``model(x)`` (ANI / baseline).
      - ``prior``: ``model.prior(x)`` if that submodule exists.
    """
    if mode == "prior" and hasattr(model, "prior"):
        out = model.prior(x)
    else:
        out = model(x)
    return _squeeze_voltage(out)


def recursive_prediction_single_cycle(
    model: torch.nn.Module,
    cycle_x: torch.Tensor,
    device: torch.device,
    q0_ah: float,
    predict_mode: str = "full",
) -> np.ndarray:
    """Autoregressive voltage rollout; I, dt, cycle id from ground-truth next step; SOC coulomb-counted."""
    model.eval()
    predictions: List[float] = []

    current_input = cycle_x[0].unsqueeze(0).to(device)
    q_total_as = float(q0_ah) * 3600.0
    current_soc = float(current_input[0, SOC_IDX].item())

    with torch.no_grad():
        for i in range(len(cycle_x)):
            pred_v = predict_voltage(model, current_input, mode=predict_mode)
            predictions.append(float(pred_v.item()))

            if i == len(cycle_x) - 1:
                break

            nxt = cycle_x[i + 1].unsqueeze(0).to(device).clone()
            next_i = float(nxt[0, I_IDX].item())
            next_dt = float(nxt[0, DT_IDX].item())

            nxt[0, V_IDX] = pred_v
            next_soc = current_soc - (next_i * next_dt) / q_total_as
            next_soc = max(0.0, min(1.0, next_soc))
            nxt[0, SOC_IDX] = next_soc

            current_input = nxt
            current_soc = next_soc

    return np.asarray(predictions, dtype=np.float64)


@dataclass
class CycleMetrics:
    cycle_feature_id: float
    n_steps: int
    mse: float
    mae: float
    rmse: float
    max_error: float


def evaluate_all_test_cycles(
    model: torch.nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    q0_ah: float,
    min_len: int = 10,
    predict_mode: str = "full",
) -> Tuple[List[CycleMetrics], Dict[str, float]]:
    """
    Returns per-cycle metrics and summary dict with mean/std across cycles.
    """
    all_ids = test_x[:, CYCLE_IDX].detach().cpu().numpy()
    unique = np.unique(all_ids)

    rows: List[CycleMetrics] = []
    for cid in unique:
        mask = all_ids == cid
        cx = test_x[mask]
        cy = test_y[mask]
        if cx.shape[0] < min_len:
            continue

        pred = recursive_prediction_single_cycle(
            model, cx, device, q0_ah=q0_ah, predict_mode=predict_mode
        )
        true = cy.detach().cpu().numpy().reshape(-1)

        err = pred - true
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        rows.append(
            CycleMetrics(
                cycle_feature_id=float(cid),
                n_steps=int(len(true)),
                mse=mse,
                mae=mae,
                rmse=float(np.sqrt(mse)),
                max_error=float(np.max(np.abs(err))),
            )
        )

    if not rows:
        return [], {"n_cycles": 0}

    mses = np.array([r.mse for r in rows])
    maes = np.array([r.mae for r in rows])
    rmses = np.array([r.rmse for r in rows])

    def stat(a: np.ndarray) -> Tuple[float, float]:
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        return m, s

    m_mse, s_mse = stat(mses)
    m_mae, s_mae = stat(maes)
    m_rmse, s_rmse = stat(rmses)

    summary = {
        "n_cycles": float(len(rows)),
        "mse_mean": m_mse,
        "mse_std": s_mse,
        "mae_mean": m_mae,
        "mae_std": s_mae,
        "rmse_mean": m_rmse,
        "rmse_std": s_rmse,
    }
    return rows, summary


def print_cycle_table(rows: List[CycleMetrics], max_rows: int = 20) -> None:
    print(f"{'cycle_id':>12} {'n':>6} {'MSE':>14} {'MAE':>14} {'RMSE':>14}")
    for r in rows[:max_rows]:
        print(
            f"{r.cycle_feature_id:12.6f} {r.n_steps:6d} {r.mse:14.6e} {r.mae:14.6e} {r.rmse:14.6e}"
        )
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more cycles omitted)")
