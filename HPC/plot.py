# plot_traj_pair.py
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV from combined.cc")
    ap.add_argument("--out", type=str, required=True, help="Output PDF path")
    ap.add_argument("--model", type=str, default="Model", help="Model label, e.g., Lie or ANI2")
    ap.add_argument("--ref", type=str, default="Reference", help="Reference label")
    ap.add_argument("--skip0", action="store_true", help="Skip k=0 row")
    ap.add_argument("--ms", type=float, default=2.0, help="Marker size for model points")
    ap.add_argument("--every", type=int, default=1, help="Plot every N-th model point (downsample markers)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = ["t", "pred_x", "pred_y", "ref_x", "ref_y"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {args.csv}")

    if args.skip0 and "k" in df.columns:
        df = df[df["k"] > 0].copy()

    # optional downsample for markers
    if args.every > 1:
        df_model = df.iloc[::args.every].copy()
    else:
        df_model = df

    t = df["t"].to_numpy()
    rx = df["ref_x"].to_numpy()
    ry = df["ref_y"].to_numpy()

    tm = df_model["t"].to_numpy()
    pxm = df_model["pred_x"].to_numpy()
    pym = df_model["pred_y"].to_numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        # Page 1: time series
        fig = plt.figure(figsize=(6, 6))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(t, rx, label=f"{args.ref} x(t)")
        ax1.plot(tm, pxm, linestyle="None", marker="o", markersize=args.ms, label=f"{args.model} x(t)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("x")
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(t, ry, label=f"{args.ref} y(t)")
        ax2.plot(tm, pym, linestyle="None", marker="o", markersize=args.ms, label=f"{args.model} y(t)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("y")
        ax2.legend()

        fig.suptitle(f"Trajectory vs time: {args.ref} and {args.model}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: phase portrait
        # fig = plt.figure(figsize=(7.5, 5.2))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(rx, ry, label=args.ref)
        # ax.plot(pxm, pym, linestyle="None", marker="o", markersize=args.ms, label=args.model)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_title(f"Phase portrait: {args.ref} and {args.model}")
        # ax.legend()
        # fig.tight_layout()
        # pdf.savefig(fig)
        # plt.close(fig)

    print(f"Saved PDF: {out.resolve()}")


if __name__ == "__main__":
    main()
