"""
test_trajectories.npy 已为物理轨迹 (x,y,z,w)；先验 f 同 ANI2.py A.F (30–33)；
估 ẏ 后学 g = ẏ - f(y)（SINDy）。

  cd Lorenz-stenflo-small && python learn_missing_physics_compare.py

依赖: numpy, scipy, pysindy
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def f_prior(y: np.ndarray) -> np.ndarray:
    """ANI2.py A.F 在物理坐标下的前四项 (30-33)。"""
    x1, x2, x3 = y[..., 0], y[..., 1], y[..., 2]
    dx1 = x2 - x1
    dx2 = 26.0 * x1 - x1 * x3 - x2
    dx3 = x1 * x2 - 0.7 * x3
    dx4 = np.zeros_like(x1)
    return np.stack([dx1, dx2, dx3, dx4], axis=-1)


def true_missing_g(y: np.ndarray) -> np.ndarray:
    x1, x4 = y[..., 0], y[..., 3]
    z = np.zeros_like(x1)
    return np.stack([1.5 * x4, z, z, -x1 - x4], axis=-1)


def ydot_centered(Y: np.ndarray, dt: float) -> np.ndarray:
    d = np.zeros_like(Y, dtype=np.float64)
    d[1:-1] = (Y[2:] - Y[:-2]) / (2.0 * dt)
    d[0] = (Y[1] - Y[0]) / dt
    d[-1] = (Y[-1] - Y[-2]) / dt
    return d


def ydot_savgol(Y: np.ndarray, dt: float, window: int, polyorder: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter

    out = np.zeros_like(Y, dtype=np.float64)
    for j in range(Y.shape[1]):
        out[:, j] = savgol_filter(Y[:, j], window_length=window, polyorder=polyorder, deriv=1, delta=dt)
    return out


def stack_residuals(trajs_phys: np.ndarray, dt: float, savgol_window: int) -> tuple[np.ndarray, np.ndarray]:
    xs, gs = [], []
    for traj in trajs_phys:
        Y = np.asarray(traj[:, :4], dtype=np.float64)
        if savgol_window >= 5 and savgol_window % 2 == 1 and Y.shape[0] > savgol_window:
            yd = ydot_savgol(Y, dt, savgol_window)
            m = savgol_window // 2
            Ym = Y[m:-m]
            xs.append(Ym)
            gs.append(yd[m:-m] - f_prior(Ym))
        else:
            yd = ydot_centered(Y, dt)
            xs.append(Y[1:-1])
            gs.append(yd[1:-1] - f_prior(Y[1:-1]))
    return np.vstack(xs), np.vstack(gs)


def main() -> None:
    try:
        import pysindy as ps
    except ImportError:
        print("pip install pysindy", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=5e-2)
    p.add_argument("--poly_degree", type=int, default=1)
    p.add_argument("--threshold", type=float, default=5e-3)
    p.add_argument("--savgol_window", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=ROOT / "sindy_missing_physics_out")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    ds = ROOT / "dataset"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y_traj = np.load(ds / "test_trajectories.npy")[..., :4].astype(np.float64)

    X, G = stack_residuals(y_traj, args.dt, args.savgol_window)

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=args.poly_degree, include_bias=True),
        optimizer=ps.STLSQ(threshold=args.threshold),
    )
    model.fit(X, t=args.dt, x_dot=G, feature_names=["x", "y", "z", "w"])

    buf = io.StringIO()
    _o, sys.stdout = sys.stdout, buf
    model.print()
    sys.stdout = _o
    eq = buf.getvalue()
    print(eq)
    (args.out_dir / "sindy_equations.txt").write_text(eq, encoding="utf-8")
    np.save(args.out_dir / "sindy_coefficients.npy", model.coefficients())

    Gp = model.predict(X)
    mse = np.mean((Gp - G) ** 2, axis=0)
    mt = np.mean((Gp - true_missing_g(X)) ** 2, axis=0)
    summary = "MSE vs residual: " + ", ".join(f"{v:.4e}" for v in mse)
    summary += "\nMSE vs theory g: " + ", ".join(f"{v:.4e}" for v in mt)
    print(summary)
    (args.out_dir / "fit_summary.txt").write_text(summary + "\n")

    if args.plot:
        import matplotlib.pyplot as plt

        gt = true_missing_g(X)
        for i, name, fn in [(0, "g1", "scatter_g1.pdf"), (3, "g4", "scatter_g4.pdf")]:
            plt.figure(figsize=(5, 5))
            plt.scatter(gt[:, i], Gp[:, i], s=6, alpha=0.3)
            a = max(np.abs(gt[:, i]).max(), np.abs(Gp[:, i]).max()) * 1.05
            plt.plot([-a, a], [-a, a], "k--", lw=1)
            plt.xlabel(f"true {name}")
            plt.ylabel(f"SINDy {name}")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_dir / fn)
            plt.close()

    print(f"Saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
