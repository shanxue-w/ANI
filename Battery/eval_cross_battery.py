#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on one or more ``processed_*_rollout.pt`` bundles.

- **Within each cell**: mean ± std over **all** held-out test discharge cycles.
- **Across cells**: mean ± std of the per-cell mean MSE (and MAE / RMSE).

Examples (from ``Battery/`` directory, after setting PYTHONPATH for ``2th`` imports)::

    python eval_cross_battery.py --model ani4 --checkpoint 4th/best_ani4_model.pt \\
        --data dataset/processed_battery_data_rollout.pt

    python eval_cross_battery.py --model baseline --checkpoint baseline/best_baseline_model.pth \\
        --data dataset/processed_B0005_rollout.pt dataset/processed_B0006_rollout.pt

``--predict_mode prior`` uses only ``model.prior(x)`` when available (e.g. ablation on ANI2).

Default rollout uses **2.0 Ah** for coulomb SOC (same as ``Battery/2th/test.py``). Use ``--use_meta_q0`` for ``meta["Q0_Ah"]``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from eval_rollout_metrics import (
    DEFAULT_ROLLOUT_Q0_AH,
    evaluate_all_test_cycles,
    print_cycle_table,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_pt(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def q0_from_bundle(d: Dict[str, Any]) -> float:
    meta = d.get("meta") or {}
    return float(meta.get("Q0_Ah", 2.0))


def build_model(
    kind: str,
    prior: Tuple[float, float, list],
    device: torch.device,
) -> torch.nn.Module:
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
        # Checkpoint from Battery/baseline/ (MLP delta on V only; no prior submodule).
        sys.path.insert(0, os.path.join(ROOT, "baseline"))
        from base import Baseline  # type: ignore

        return Baseline().to(device)

    if kind == "new_baseline":
        sys.path.insert(0, os.path.join(ROOT, "New_baseline"))
        from base import Baseline  # type: ignore

        return Baseline(R_val, C_val, w_vals).to(device)

    raise ValueError(f"Unknown model kind {kind}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ani2", "ani4", "baseline", "new_baseline"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="One or more processed .pt files",
    )
    p.add_argument(
        "--predict_mode",
        choices=["full", "prior"],
        default="full",
        help="Use full model or only .prior submodule when present",
    )
    p.add_argument("--cuda", action="store_true")
    p.add_argument(
        "--use_meta_q0",
        action="store_true",
        help="Use meta['Q0_Ah'] for coulomb SOC (default: 2.0 Ah, same as 2th/test.py)",
    )
    p.add_argument("--out_csv", type=str, default="", help="Optional summary CSV path")
    p.add_argument("--print_cycles", type=int, default=0, help="Print first N per-cell cycle rows")
    args = p.parse_args()

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    per_cell_means: List[Dict[str, float]] = []
    csv_rows: List[Dict[str, Any]] = []

    for data_path in args.data:
        bundle = load_pt(data_path)
        prior_params = (
            bundle["prior_params"]["R"],
            bundle["prior_params"]["C"],
            bundle["prior_params"]["w"],
        )
        meta = bundle.get("meta") or {}
        key = meta.get("battery_key", os.path.basename(data_path))

        model = build_model(args.model, prior_params, device)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        test_x = bundle["X_test"].to(torch.float64)
        test_y = bundle["Y_test"].to(torch.float64)
        q0 = q0_from_bundle(bundle) if args.use_meta_q0 else DEFAULT_ROLLOUT_Q0_AH

        rows, summ = evaluate_all_test_cycles(
            model,
            test_x,
            test_y,
            device,
            q0_ah=q0,
            predict_mode=args.predict_mode,
        )

        print(f"\n=== {key} ({data_path}) ===")
        print(
            f"Rollout Q0_Ah: {q0:.6f} "
            f"({'meta' if args.use_meta_q0 else 'default 2.0, same as 2th/test.py'})"
        )
        print(f"Held-out test cycle ids (discharge index): {meta.get('test_cycle_ids', 'n/a')}")
        print(f"n_eval_cycles={int(summ['n_cycles'])}")
        if summ["n_cycles"] > 0:
            print(
                f"MSE  mean±std: {summ['mse_mean']:.6e} ± {summ['mse_std']:.6e} | "
                f"MAE: {summ['mae_mean']:.6e} ± {summ['mae_std']:.6e} | "
                f"RMSE: {summ['rmse_mean']:.6e} ± {summ['rmse_std']:.6e}"
            )
        if args.print_cycles > 0 and rows:
            print_cycle_table(rows, max_rows=args.print_cycles)

        if summ["n_cycles"] > 0:
            per_cell_means.append(
                {
                    "mse_mean": summ["mse_mean"],
                    "mae_mean": summ["mae_mean"],
                    "rmse_mean": summ["rmse_mean"],
                }
            )

        csv_rows.append(
            {
                "battery_key": key,
                "data_path": data_path,
                "n_cycles": int(summ["n_cycles"]),
                **{k: summ[k] for k in summ if k != "n_cycles"},
            }
        )

    if len(per_cell_means) >= 1:
        print("\n=== Across cells (stats of per-cell **mean** metrics) ===")
        for name in ["mse_mean", "mae_mean", "rmse_mean"]:
            vals = np.array([c[name] for c in per_cell_means], dtype=np.float64)
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            print(f"{name}: {m:.6e} ± {s:.6e}  (n_cells={len(vals)})")

    if args.out_csv and csv_rows:
        keys = list(csv_rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
