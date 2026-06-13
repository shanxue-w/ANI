#!/usr/bin/env python3
"""
Plot MSE vs normalized discharge cycle for ANI-2, ANI-4, and baseline.

Reads:
  - ani2_per_cycle.csv
  - ani4_per_cycle.csv
  - baseline_per_cycle.csv

Each row must share the same norm_cycle_feature order across files (typical eval output).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float_col(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def main() -> None:
    root = Path(__file__).resolve().parent
    p2 = root / "ani2_per_cycle.csv"
    p4 = root / "ani4_per_cycle.csv"
    pb = root / "baseline_per_cycle.csv"

    for p in (p2, p4, pb):
        if not p.is_file():
            raise SystemExit(f"Missing: {p}")

    r2, r4, rb = _read_csv(p2), _read_csv(p4), _read_csv(pb)
    if not (r2 and r4 and rb):
        raise SystemExit("One or more CSV files are empty.")

    c2 = _float_col(r2, "norm_cycle_feature")
    c4 = _float_col(r4, "norm_cycle_feature")
    cb = _float_col(rb, "norm_cycle_feature")
    if not (np.allclose(c2, c4) and np.allclose(c2, cb)):
        raise SystemExit(
            "norm_cycle_feature columns differ across CSVs; merge or sort before plotting."
        )

    mse2 = _float_col(r2, "mse")
    mse4 = _float_col(r4, "mse")
    mseb = _float_col(rb, "mse")

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(c2, mse2, marker="o", ms=4, lw=1.5, label="ANI-2")
    ax.plot(c4, mse4, marker="s", ms=4, lw=1.5, label="ANI-4")
    ax.plot(cb, mseb, marker="^", ms=4, lw=1.5, label="Baseline")
    ax.set_xlabel("Normalized cycle")
    ax.set_ylabel("MSE")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.35)

    out_pdf = root / "per_cycle_mse_compare.pdf"
    out_png = root / "per_cycle_mse_compare.png"
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
