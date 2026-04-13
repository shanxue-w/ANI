#!/usr/bin/env python3
"""
Single discharge-cycle ground-truth voltage from the same processed .pt as Battery/*/test.py.
Matches 2th/test.py: X/Y pointwise tensors, split by CYCLE_IDX, true = Y_test flattened per cycle.
Plot is only the black curve (no axes/title), PDF @ 300 dpi by default.

Note: processed_battery_data_rollout.pt contains discharge cycles only (NASA pipeline in dataset/data.py).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Same feature layout as Battery/2th/test.py (process_and_split_dynamic stack)
CYCLE_IDX = 3


def load_split_xy(data_path: str, split: str):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    key_x = f"X_{split}"
    key_y = f"Y_{split}"
    if key_x not in data or key_y not in data:
        raise KeyError(f"{data_path} missing {key_x!r} or {key_y!r}")
    X = data[key_x].to(torch.float64)
    Y = data[key_y].to(torch.float64)
    return X, Y


def true_voltage_one_cycle(X: torch.Tensor, Y: torch.Tensor, which: int, min_len: int) -> np.ndarray:
    """
    which: index into np.unique(cycle_id), ascending (same iteration order as test.py for loop).
    """
    all_cycles_id = X[:, CYCLE_IDX].numpy()
    unique_cycles = np.unique(all_cycles_id)
    if which < 0 or which >= len(unique_cycles):
        raise IndexError(
            f"which={which} out of range; test split has {len(unique_cycles)} distinct cycles."
        )
    c_id = unique_cycles[which]
    mask = all_cycles_id == c_id
    cycle_y = Y[mask]
    if len(cycle_y) < min_len:
        raise RuntimeError(
            f"Cycle id={c_id} has only {len(cycle_y)} points (< min_len={min_len})."
        )
    return cycle_y.numpy().flatten()


def main():
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--data_file",
        default=os.path.join(here, "dataset", "processed_battery_data_rollout.pt"),
        help="Torch bundle from dataset/data.py export_processed_rollout_pt",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which split to take cycles from (test matches test.py).",
    )
    p.add_argument(
        "--which",
        type=int,
        default=0,
        help="Index among sorted unique cycle ids (0 = same first cycle as test.py results[0]).",
    )
    p.add_argument("--out", default="one_discharge_cycle_truth_pure.pdf")
    p.add_argument("--min_len", type=int, default=10, help="Skip short cycles; same spirit as test.py")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--size_in",
        type=float,
        default=4.0,
        help="Square figure side length in inches (output PDF page is size_in × size_in).",
    )
    p.add_argument("--lw", type=float, default=2.0, help="Match test.py linewidth=2 for 'true'")
    p.add_argument("--color", default="black")
    p.add_argument(
        "--pad_x",
        type=float,
        default=2.5,
        help="Extra horizontal space on each side in step-index units (steep start needs a few px).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.data_file):
        print(f"Data not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    X, Y = load_split_xy(args.data_file, args.split)
    true_v = true_voltage_one_cycle(X, Y, args.which, args.min_len)

    # Identical to plt.plot(res['true'], 'k-') in test.py: y vs implicit step index
    x = np.arange(len(true_v), dtype=np.float64)

    fig = plt.figure(figsize=(args.size_in, args.size_in), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(x, true_v, color=args.color, linewidth=args.lw, linestyle="-", solid_capstyle="round")
    ax.set_axis_off()
    for s in ax.spines.values():
        s.set_visible(False)
    n = len(true_v)
    if n <= 1:
        ax.set_xlim(-args.pad_x, max(1.0, n - 1) + args.pad_x)
    else:
        ax.set_xlim(-args.pad_x, (n - 1) + args.pad_x)
    # Full square page (no tight crop — avoids non-square bbox when x/y data spans differ).
    fig.savefig(args.out, transparent=True, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
