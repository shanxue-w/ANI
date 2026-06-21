from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "test_trajectories.npy"


def f_lorenz_stenflo_full(u: np.ndarray) -> np.ndarray:
    x, y, z, w = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    dx = (y - x) + 1.5 * w
    dy = 26.0 * x - x * z - y
    dz = x * y - 0.7 * z
    dw = -x - w
    return np.stack([dx, dy, dz, dw], axis=-1)


def f_prior(u: np.ndarray) -> np.ndarray:
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    dx = y - x
    dy = 26.0 * x - x * z - y
    dz = x * y - 0.7 * z
    dw = np.zeros_like(x)
    return np.stack([dx, dy, dz, dw], axis=-1)


def b_additive_distilled(u: np.ndarray) -> np.ndarray:
    x, y, _, w = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    return np.stack(
        [
            0.012 * x - 0.007 * y + 1.502 * w,
            0.036 * x - 0.011 * y + 0.017 * w,
            np.zeros_like(x),
            -0.998 * x - 0.998 * w,
        ],
        axis=-1,
    )


def b_ani4_distilled(u: np.ndarray) -> np.ndarray:
    x, w = u[..., 0], u[..., 3]
    return np.stack(
        [1.506 * w, np.zeros_like(x), np.zeros_like(x), -0.035 - 1.005 * x - 0.998 * w],
        axis=-1,
    )


def rhs_additive(u: np.ndarray) -> np.ndarray:
    return f_prior(u) + b_additive_distilled(u)


def rhs_ani4(u: np.ndarray) -> np.ndarray:
    return f_prior(u) + b_ani4_distilled(u)


def rk4_step(u: np.ndarray, dt: float, rhs) -> np.ndarray:
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate_batch(u0: np.ndarray, dt_macro: float, h_micro: float, rhs) -> np.ndarray:
    u = np.asarray(u0, dtype=np.float64).copy()
    t = 0.0
    while t < dt_macro - 1e-18:
        h = min(h_micro, dt_macro - t)
        u = rk4_step(u, h, rhs)
        t += h
    return u


def rollout_batch(u0: np.ndarray, n_steps: int, dt_macro: float, h_micro: float, rhs) -> np.ndarray:
    u = np.asarray(u0, dtype=np.float64).copy()
    out = np.empty((u.shape[0], n_steps + 1, u.shape[1]), dtype=np.float64)
    out[:, 0, :] = u
    for k in range(n_steps):
        u = rk4_integrate_batch(u, dt_macro, h_micro, rhs)
        out[:, k + 1, :] = u
    return out


def rollout_single(u0: np.ndarray, n_steps: int, dt_macro: float, h_micro: float, rhs) -> np.ndarray:
    return rollout_batch(u0[None, :], n_steps, dt_macro, h_micro, rhs)[0]


def block_tau(u: np.ndarray, dt_block: float, b_fn, h_micro: float) -> np.ndarray:
    u = rk4_integrate_batch(u, 0.5 * dt_block, h_micro, f_prior)
    u = rk4_integrate_batch(u, dt_block, h_micro, b_fn)
    u = rk4_integrate_batch(u, 0.5 * dt_block, h_micro, f_prior)
    return u


def ani4_strang_step(u: np.ndarray, dt_strang: float, b_fn, h_micro: float) -> np.ndarray:
    k1 = block_tau(u, dt_strang, b_fn, h_micro)
    k2 = block_tau(block_tau(u, 0.5 * dt_strang, b_fn, h_micro), 0.5 * dt_strang, b_fn, h_micro)
    return -1.0 / 3.0 * k1 + 4.0 / 3.0 * k2


def rollout_ani4_split_batch(
    u0: np.ndarray,
    n_steps: int,
    dt_macro: float,
    h_micro: float,
    n_sub_strang: int,
) -> np.ndarray:
    u = np.asarray(u0, dtype=np.float64).copy()
    out = np.empty((u.shape[0], n_steps + 1, u.shape[1]), dtype=np.float64)
    out[:, 0, :] = u
    h = dt_macro / float(n_sub_strang)
    for k in range(n_steps):
        for _ in range(n_sub_strang):
            u = ani4_strang_step(u, h, b_ani4_distilled, h_micro)
        out[:, k + 1, :] = u
    return out


def rollout_ani4_split_single(
    u0: np.ndarray,
    n_steps: int,
    dt_macro: float,
    h_micro: float,
    n_sub_strang: int,
) -> np.ndarray:
    return rollout_ani4_split_batch(u0[None, :], n_steps, dt_macro, h_micro, n_sub_strang)[0]


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - truth, axis=-1)
    den = np.linalg.norm(truth, axis=-1) + 1e-12
    return num / den


def support_size_additive() -> int:
    coeffs = np.array(
        [
            [0.0, 0.012, -0.007, 0.0, 1.502],
            [0.0, 0.036, -0.011, 0.0, 0.017],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.998, 0.0, 0.0, -0.998],
        ]
    )
    return int(np.count_nonzero(np.abs(coeffs) > 0.0))


def support_size_ani4() -> int:
    coeffs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.506],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.035, -1.005, 0.0, 0.0, -0.998],
        ]
    )
    return int(np.count_nonzero(np.abs(coeffs) > 0.0))


def build_window_index(
    n_traj: int,
    n_time: int,
    horizon: int,
    burn_in_steps: int,
    stride: int,
    max_windows: int | None,
) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    last_start = n_time - horizon - 1
    for traj_idx in range(n_traj):
        for start in range(burn_in_steps, last_start + 1, stride):
            starts.append((traj_idx, start))
    if max_windows is not None and max_windows > 0 and len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
        starts = [starts[i] for i in pick]
    return starts


def extract_windows(
    trajectories: np.ndarray, starts: list[tuple[int, int]], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    u0 = np.stack([trajectories[i, s] for i, s in starts], axis=0)
    truth = np.stack([trajectories[i, s : s + horizon + 1] for i, s in starts], axis=0)
    return u0, truth


def choose_representative_window(window_scores: np.ndarray) -> int:
    order = np.argsort(window_scores)
    return int(order[len(order) // 2])


@dataclass
class ClosedLoopResult:
    label: str
    support_size: str
    rel: np.ndarray

    @property
    def mean_curve(self) -> np.ndarray:
        return self.rel.mean(axis=0)

    @property
    def mean_200_step_error(self) -> float:
        return float(self.rel[:, 1:].mean())


def run_closed_loop(
    trajectories: np.ndarray,
    starts: list[tuple[int, int]],
    horizon: int,
    dt_macro: float,
    model_h_micro: float,
    truth_h_micro: float,
    n_sub_strang: int,
) -> tuple[list[ClosedLoopResult], np.ndarray, int]:
    u0, _ = extract_windows(trajectories, starts, horizon)

    # Match symbolic_refinement_rollout.py:
    # every model, including the reference truth, is rerolled from the sampled ICs.
    truth = rollout_batch(u0, horizon, dt_macro, truth_h_micro, f_lorenz_stenflo_full)
    pred_prior = rollout_batch(u0, horizon, dt_macro, model_h_micro, f_prior)
    pred_add = rollout_batch(u0, horizon, dt_macro, model_h_micro, rhs_additive)
    pred_ani = rollout_ani4_split_batch(u0, horizon, dt_macro, model_h_micro, n_sub_strang)

    results = [
        ClosedLoopResult("Frozen prior", "--", rel_l2(pred_prior, truth)),
        ClosedLoopResult("Additive distilled refinement", str(support_size_additive()), rel_l2(pred_add, truth)),
        ClosedLoopResult("ANI-4 distilled refinement", str(support_size_ani4()), rel_l2(pred_ani, truth)),
    ]

    rep_idx = choose_representative_window(results[2].rel[:, 1:].mean(axis=1))
    rep_truth = truth[rep_idx]
    rep_preds = np.stack([pred_prior[rep_idx], pred_add[rep_idx], pred_ani[rep_idx]], axis=0)
    return results, np.concatenate([rep_truth[None, ...], rep_preds], axis=0), rep_idx


def _shared_xyz_lims(*arrays: np.ndarray, pad_frac: float = 0.04) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([a[:, 0] for a in arrays])
    ys = np.concatenate([a[:, 1] for a in arrays])
    zs = np.concatenate([a[:, 2] for a in arrays])

    def span(v: np.ndarray) -> tuple[float, float]:
        lo, hi = float(np.min(v)), float(np.max(v))
        d = hi - lo
        pad = pad_frac * d if d > 1e-12 else 1.0
        return lo - pad, hi + pad

    return span(xs), span(ys), span(zs)


def _style_3d(ax, elev: float = 22.0, azim: float = -58.0) -> None:
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle="-", linewidth=0.5, color="0.78", alpha=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis="both", labelsize=8, pad=2)


def plot_closed_loop_figure(results: list[ClosedLoopResult], representative: np.ndarray, dt_macro: float, out_path: Path) -> None:
    ref, prior, add, ani = representative
    time = np.arange(ref.shape[0]) * dt_macro

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.5))

    ax = axes[0]
    ax.plot(ref[:, 0], ref[:, 3], color="#1f77b4", lw=2.0, label="Reference")
    ax.plot(prior[:, 0], prior[:, 3], color="#1f77b4", ls="--", lw=1.6, label="Prior")
    ax.plot(add[:, 0], add[:, 3], color="#2ca02c", ls="-.", lw=1.6, label="Additive distilled")
    ax.plot(ani[:, 0], ani[:, 3], color="#d62728", ls=":", lw=2.0, label="ANI-4 distilled")
    ax.set_xlabel("x")
    ax.set_ylabel("w")
    ax.set_title("Symbolic closed-loop rerollout", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = [r.label for r in results]
    for color, label, result in zip(colors, labels, results):
        ax.plot(time, result.mean_curve, color=color, lw=1.7, label=f"{label} (mean={result.mean_200_step_error:.3e})")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Mean relative $L^2$ error")
    ax.set_title("Attractor-sampled 200-step mean error", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_attractor_triptych(truth: np.ndarray, add: np.ndarray, ani: np.ndarray, out_path: Path) -> None:
    xlim, ylim, zlim = _shared_xyz_lims(truth, add, ani)
    fig = plt.figure(figsize=(15.4, 5.0), facecolor="white")
    specs = [
        (truth, "Reference"),
        (add, "Additive SINDy"),
        (ani, "ANI-4 SINDy"),
    ]

    for idx, (traj, title) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d", facecolor="white")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#1f77b4", lw=0.95, alpha=0.95)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("x", fontsize=8, labelpad=4)
        ax.set_ylabel("y", fontsize=8, labelpad=4)
        ax.set_zlabel("z", fontsize=8, labelpad=4)
        ax.set_title(title, fontsize=9)
        _style_3d(ax)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_closed_loop_tables(results: list[ClosedLoopResult], out_dir: Path) -> None:
    csv_path = out_dir / "lorenz_symbolic_closed_loop_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Support size", "Mean 200-step relative L2 error"])
        for result in results:
            writer.writerow([result.label, result.support_size, f"{result.mean_200_step_error:.6e}"])

    tex_path = out_dir / "lorenz_symbolic_closed_loop_table.tex"
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Support size} & \textbf{Mean 200-step relative $L^2$ error} \\",
        r"\midrule",
    ]
    for result in results:
        lines.append(
            f"{result.label} & {result.support_size} & "
            + r"$"
            + f"{result.mean_200_step_error:.1e}"
            + r"$ \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_closed_loop_summary(
    results: list[ClosedLoopResult],
    out_dir: Path,
    starts: list[tuple[int, int]],
    representative_idx: int,
    dt_macro: float,
    horizon: int,
    burn_in_steps: int,
    stride: int,
    model_h_micro: float,
    truth_h_micro: float,
    n_sub_strang: int,
) -> None:
    lines = [
        "Closed-loop symbolic write-back rerollout",
        f"dataset={DATASET}",
        f"dt_macro={dt_macro}",
        f"horizon={horizon}",
        f"num_windows={len(starts)}",
        f"burn_in_steps={burn_in_steps}",
        f"stride={stride}",
        f"model_h_micro={model_h_micro}",
        f"truth_h_micro={truth_h_micro}",
        f"n_sub_strang={n_sub_strang}",
        f"representative_window_index={representative_idx}",
        "",
    ]
    for result in results:
        lines.append(f"{result.label}: mean_200_step_relative_l2={result.mean_200_step_error:.6e}")
    (out_dir / "closed_loop_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_long_rollout(
    ic: np.ndarray,
    n_steps: int,
    dt_macro: float,
    h_micro: float,
    n_sub_strang: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = rollout_single(ic, n_steps, dt_macro, h_micro, f_lorenz_stenflo_full)
    add = rollout_single(ic, n_steps, dt_macro, h_micro, rhs_additive)
    ani = rollout_ani4_split_single(ic, n_steps, dt_macro, h_micro, n_sub_strang)
    return truth, add, ani


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Lorenz symbolic write-back tables and figures.")
    parser.add_argument("--closed_dt", type=float, default=1e-2, help="Macro step for the 200-step closed-loop rerollout.")
    parser.add_argument("--closed_h_micro", type=float, default=1e-2, help="Model micro RK4 step used inside each closed-loop macro step.")
    parser.add_argument("--truth_h_micro", type=float, default=1e-3, help="Reference truth micro RK4 step for the closed-loop rerollout.")
    parser.add_argument("--closed_horizon", type=int, default=200, help="Closed-loop horizon in macro steps.")
    parser.add_argument("--n_sub_strang", type=int, default=1, help="ANI-4 macro-step subdivisions for split rollout.")
    parser.add_argument("--burn_in_steps", type=int, default=200, help="Discard the first burn-in steps before sampling attractor windows.")
    parser.add_argument("--stride", type=int, default=5, help="Stride used when sampling attractor windows.")
    parser.add_argument("--max_windows", type=int, default=0, help="If >0, subsample to at most this many windows.")
    parser.add_argument("--long_dt", type=float, default=1e-2, help="Macro step for the long direct symbolic rollout.")
    parser.add_argument("--long_h_micro", type=float, default=1e-2, help="Model micro RK4 step used in the long direct rollout.")
    parser.add_argument("--long_truth_h_micro", type=float, default=1e-3, help="Reference truth micro RK4 step used in the long direct rollout.")
    parser.add_argument("--long_steps", type=int, default=2000, help="Long direct-rollout horizon.")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "symbolic_writeback_repro_out")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = np.load(DATASET).astype(np.float64)
    n_traj, n_time, _ = trajectories.shape

    starts = build_window_index(
        n_traj=n_traj,
        n_time=n_time,
        horizon=args.closed_horizon,
        burn_in_steps=args.burn_in_steps,
        stride=args.stride,
        max_windows=args.max_windows if args.max_windows > 0 else None,
    )
    if not starts:
        raise ValueError("No attractor windows were sampled. Reduce burn_in_steps or stride.")

    closed_results, representative, rep_idx = run_closed_loop(
        trajectories=trajectories,
        starts=starts,
        horizon=args.closed_horizon,
        dt_macro=args.closed_dt,
        model_h_micro=args.closed_h_micro,
        truth_h_micro=args.truth_h_micro,
        n_sub_strang=args.n_sub_strang,
    )
    plot_closed_loop_figure(
        results=closed_results,
        representative=representative,
        dt_macro=args.closed_dt,
        out_path=args.out_dir / "lorenz_symbolic_closed_loop.pdf",
    )
    write_closed_loop_tables(closed_results, args.out_dir)
    write_closed_loop_summary(
        results=closed_results,
        out_dir=args.out_dir,
        starts=starts,
        representative_idx=rep_idx,
        dt_macro=args.closed_dt,
        horizon=args.closed_horizon,
        burn_in_steps=args.burn_in_steps,
        stride=args.stride,
        model_h_micro=args.closed_h_micro,
        truth_h_micro=args.truth_h_micro,
        n_sub_strang=args.n_sub_strang,
    )

    long_ic = representative[0, 0]
    truth_long = rollout_single(long_ic, args.long_steps, args.long_dt, args.long_truth_h_micro, f_lorenz_stenflo_full)
    add_long = rollout_single(long_ic, args.long_steps, args.long_dt, args.long_h_micro, rhs_additive)
    ani_long = rollout_ani4_split_single(long_ic, args.long_steps, args.long_dt, args.long_h_micro, args.n_sub_strang)
    # plot_attractor_triptych(
    #     truth=truth_long,
    #     add=add_long,
    #     ani=ani_long,
    #     out_path=args.out_dir / "lorenz_symbolic_attractor.pdf",
    # )

    np.savez(
        args.out_dir / "symbolic_writeback_rollouts.npz",
        representative_closed_loop=representative,
        mean_curve_prior=closed_results[0].mean_curve,
        mean_curve_additive=closed_results[1].mean_curve,
        mean_curve_ani4=closed_results[2].mean_curve,
        truth_long=truth_long,
        add_long=add_long,
        ani_long=ani_long,
        closed_dt=np.array([args.closed_dt]),
        long_dt=np.array([args.long_dt]),
        closed_h_micro=np.array([args.closed_h_micro]),
        truth_h_micro=np.array([args.truth_h_micro]),
        long_h_micro=np.array([args.long_h_micro]),
        long_truth_h_micro=np.array([args.long_truth_h_micro]),
        n_sub_strang=np.array([args.n_sub_strang]),
    )

    print(f"Saved outputs to: {args.out_dir}")
    for result in closed_results:
        print(f"{result.label:32s} support={result.support_size:>2s}  mean200={result.mean_200_step_error:.6e}")


if __name__ == "__main__":
    main()
