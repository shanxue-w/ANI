#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No retraining: load an existing ``processed_battery_data_rollout.pt`` and checkpoint,
run **autoregressive rollout for every held-out discharge cycle** in the test split,
report per-cycle MSE / MAE / RMSE, and aggregate **mean ± std across cycles**.
Pass multiple ``.pt`` files for optional cross-cell evaluation with a single checkpoint (e.g. zero-shot).

Run from the ``Battery`` directory in the repo::

    cd Battery
    python eval_multicycle_standalone.py --model ani4 \\
        --checkpoint 4th/best_ani4_model.pt \\
        --data dataset/processed_battery_data_rollout.pt

    New_baseline checkpoints (Thevenin prior + MLP delta) use ``--model new_baseline``;
    ``baseline`` matches ``baseline/base.py`` (MLP on V+delta only, no ``prior`` submodule).

Multiple bundles (one per cell)::

    python eval_multicycle_standalone.py --model ani4 \\
        --checkpoint 4th/best_ani4_model.pt \\
        --data dataset/processed_B0005_rollout.pt dataset/processed_B0006_rollout.pt \\
        --cuda --out_csv summary.csv

``--predict_mode prior``: if the model exposes ``.prior``, use only the prior at each step (matches some ``test.py`` setups).

Default rollout uses **2.0 Ah** for coulomb SOC (same as ``Battery/2th/test.py``). Use ``--use_meta_q0`` for ``meta["Q0_Ah"]`` (aligned with ``dataset/data.py`` / ``4th/test.py``).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from eval_rollout_metrics import DEFAULT_ROLLOUT_Q0_AH

# ---------------------------------------------------------------------------
# Feature order must match ``dataset/data.py`` (process_and_split_dynamic): V, I, SoC, Norm_Cycle, dt
# ---------------------------------------------------------------------------
V_IDX = 0
I_IDX = 1
SOC_IDX = 2
CYCLE_IDX = 3
DT_IDX = 4

ROOT = os.path.dirname(os.path.abspath(__file__))


def _squeeze_voltage(out: torch.Tensor) -> torch.Tensor:
    if out.ndim == 2 and out.shape[-1] == 1:
        return out.squeeze(-1)
    return out


def predict_voltage(
    model: torch.nn.Module, x: torch.Tensor, mode: str = "full"
) -> torch.Tensor:
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
        return [], {"n_cycles": 0.0}

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

    summary: Dict[str, float] = {
        "n_cycles": float(len(rows)),
        "mse_mean": m_mse,
        "mse_std": s_mse,
        "mae_mean": m_mae,
        "mae_std": s_mae,
        "rmse_mean": m_rmse,
        "rmse_std": s_rmse,
    }
    return rows, summary


def print_cycle_table(rows: List[CycleMetrics], max_rows: int = 50) -> None:
    print(f"{'norm_cycle':>14} {'n':>6} {'MSE':>14} {'MAE':>14} {'RMSE':>14} {'max|e|':>14}")
    for r in rows[:max_rows]:
        print(
            f"{r.cycle_feature_id:14.8f} {r.n_steps:6d} "
            f"{r.mse:14.6e} {r.mae:14.6e} {r.rmse:14.6e} {r.max_error:14.6e}"
        )
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more cycles omitted)")


def load_bundle(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def q0_from_bundle(d: Dict[str, Any]) -> float:
    meta = d.get("meta") or {}
    return float(meta.get("Q0_Ah", 2.0))


def build_model(kind: str, prior: Tuple[Any, Any, Any], device: torch.device):
    R_val, C_val, w_vals = prior[0], prior[1], prior[2]

    if kind == "ani2":
        sys.path.insert(0, os.path.join(ROOT, "2th"))
        from ANI2 import ANI2  # type: ignore

        return ANI2(R_val, C_val, w_vals).to(device)

    if kind == "ani4":
        sys.path.insert(0, os.path.join(ROOT, "4th"))
        from ANI4 import ANI4  # type: ignore

        return ANI4(R_val, C_val, w_vals).to(device)

    if kind == "baseline":
        sys.path.insert(0, os.path.join(ROOT, "baseline"))
        from base import Baseline  # type: ignore

        return Baseline().to(device)

    if kind == "new_baseline":
        sys.path.insert(0, os.path.join(ROOT, "New_baseline"))
        from base import Baseline  # type: ignore

        return Baseline(R_val, C_val, w_vals).to(device)

    raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-cycle battery rollout eval (no retrain)")
    parser.add_argument(
        "--model",
        choices=["ani2", "ani4", "baseline", "new_baseline"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="State dict path")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="processed_*_rollout.pt (one or more)",
    )
    parser.add_argument(
        "--predict_mode",
        choices=["full", "prior"],
        default="full",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--use_meta_q0",
        action="store_true",
        help="Use meta['Q0_Ah'] for coulomb SOC (default: 2.0 Ah, same as 2th/test.py)",
    )
    parser.add_argument("--min_len", type=int, default=10, help="Min steps per cycle to evaluate")
    parser.add_argument("--out_csv", type=str, default="", help="Optional per-cycle + summary CSV")
    parser.add_argument("--print_cycles", type=int, default=200, help="Max per-battery cycles to print")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    csv_cycle_rows: List[Dict[str, Any]] = []
    csv_summary_rows: List[Dict[str, Any]] = []
    per_cell_means: List[Dict[str, float]] = []

    for data_path in args.data:
        bundle = load_bundle(data_path)
        prior = (
            bundle["prior_params"]["R"],
            bundle["prior_params"]["C"],
            bundle["prior_params"]["w"],
        )
        meta = bundle.get("meta") or {}
        cell_name = str(meta.get("battery_key", os.path.basename(data_path)))

        model = build_model(args.model, prior, device)
        try:
            sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            sd = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            msg = str(e)
            if args.model == "baseline" and "prior." in msg and "Unexpected key" in msg:
                raise RuntimeError(
                    f"{msg}\n\nHint: checkpoints from Battery/New_baseline/ include a Thevenin "
                    "`prior` submodule. Re-run with `--model new_baseline` (not `baseline`)."
                ) from e
            raise
        model.eval()

        test_x = bundle["X_test"].to(dtype=torch.float64)
        test_y = bundle["Y_test"].to(dtype=torch.float64)
        q0 = q0_from_bundle(bundle) if args.use_meta_q0 else DEFAULT_ROLLOUT_Q0_AH

        rows, summ = evaluate_all_test_cycles(
            model,
            test_x,
            test_y,
            device,
            q0_ah=q0,
            min_len=args.min_len,
            predict_mode=args.predict_mode,
        )

        print(f"\n{'='*60}")
        print(f"Cell / file: {cell_name}  |  {data_path}")
        print(
            f"Rollout Q0_Ah: {q0:.6f} "
            f"({'meta' if args.use_meta_q0 else 'default 2.0, same as 2th/test.py'})"
        )
        if meta.get("test_cycle_ids") is not None:
            print(f"meta.test_cycle_ids (discharge indices): {meta['test_cycle_ids']}")
        print(
            f"Evaluated cycles: {int(summ.get('n_cycles', 0))} "
            f"(grouped by X_test[:, {CYCLE_IDX}] Norm_Cycle)"
        )


        for r in rows:
            csv_cycle_rows.append(
                {
                    "battery_key": cell_name,
                    "data_path": data_path,
                    "norm_cycle_feature": r.cycle_feature_id,
                    "n_steps": r.n_steps,
                    "mse": r.mse,
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "max_abs_error": r.max_error,
                }
            )

        csv_summary_rows.append(
            {
                "battery_key": cell_name,
                "data_path": data_path,
                "n_cycles_evaluated": int(summ.get("n_cycles", 0)),
                **{k: summ[k] for k in ("mse_mean", "mse_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std") if k in summ},
            }
        )

        if summ.get("n_cycles", 0) > 0:
            per_cell_means.append(
                {
                    "mse_mean": summ["mse_mean"],
                    "mae_mean": summ["mae_mean"],
                    "rmse_mean": summ["rmse_mean"],
                }
            )




    if args.out_csv:
        keys_c = list(csv_cycle_rows[0].keys()) if csv_cycle_rows else []
        keys_s = list(csv_summary_rows[0].keys()) if csv_summary_rows else []
        base, _ext = os.path.splitext(args.out_csv)
        path_c = base + "_per_cycle.csv"
        path_s = base + "_per_battery.csv"
        if csv_cycle_rows and keys_c:
            with open(path_c, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys_c)
                w.writeheader()
                w.writerows(csv_cycle_rows)
            print(f"\nWrote {path_c}")
        if csv_summary_rows and keys_s:
            with open(path_s, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys_s)
                w.writeheader()
                w.writerows(csv_summary_rows)
            print(f"Wrote {path_s}")


if __name__ == "__main__":
    main()
