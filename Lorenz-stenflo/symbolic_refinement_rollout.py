"""
Rollout comparison: 4D Lorenz prior (dw/dt=0) plus symbolic corrections vs **reference
truth** = full Lorenz–Stenflo ODE integrated with RK4 (same macro dt and h_micro as models).

Full ODE (MATLAB lorenz_stenflo_ode):
  dx/dt = (y-x) + 1.5*w,  dy/dt = 26*x - x*z - y,  dz/dt = x*y - 0.7*z,  dw/dt = -x - w

Prior (incomplete): sigma=1, rho=26, beta=0.7, dw/dt=0.

ICs: --ic_from test (default) uses only t=0 from test_trajectories.npy per trajectory;
     --ic_from random samples (x,y,z,w) uniformly from the paper box.

Run:
  cd Lorenz-stenflo-small && python symbolic_refinement_rollout.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent
DT_DEFAULT = 5e-2
H_MICRO_DEFAULT = 5e-3
N_SUB_STRANG_DEFAULT = 5
N_STEPS_DEFAULT = 2000
N_TRAJ_DEFAULT = 4


def f_lorenz_stenflo_full(u: np.ndarray) -> np.ndarray:
    """Full Lorenz–Stenflo vector field (reference dynamics)."""
    x, y, z, w = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    dx = (y - x) + 1.5 * w
    dy = 26.0 * x - x * z - y
    dz = x * y - 0.7 * z
    dw = -x - w
    return np.stack([dx, dy, dz, dw], axis=-1)


def f_prior(u: np.ndarray) -> np.ndarray:
    """4D prior: Lorenz(sigma,rho,beta) with dw/dt=0."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    w = u[..., 3]
    dx = y - x
    dy = 26.0 * x - x * z - y
    dz = x * y - 0.7 * z
    dw = np.zeros_like(x)
    return np.stack([dx, dy, dz, dw], axis=-1)


def B_ani4_distilled(u: np.ndarray) -> np.ndarray:
    """SINDy distilled correction from ANI-4 (physical coordinates)."""
    x, w = u[..., 0], u[..., 3]
    z = np.zeros_like(x)
    return np.stack(
        [1.506 * w, z, z, -0.035 - 1.005 * x - 0.998 * w],
        axis=-1,
    )


def B_baseline_sindy(u: np.ndarray) -> np.ndarray:
    """Additive baseline distilled correction."""
    x, y, z, w = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    return np.stack(
        [
            0.012 * x - 0.007 * y + 1.502 * w,
            0.036 * x - 0.011 * y + 0.017 * w,
            np.zeros_like(x),
            -0.998 * x - 0.998 * w,
        ],
        axis=-1,
    )


def rk4_step(u: np.ndarray, dt: float, rhs) -> np.ndarray:
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_integrate(u: np.ndarray, span: float, h_micro: float, rhs) -> np.ndarray:
    """Integrate u' = rhs(u) from current u over time span > 0 using RK4 steps <= h_micro."""
    if span <= 0:
        return u
    t = 0.0
    while t < span - 1e-18:
        h = min(h_micro, span - t)
        u = rk4_step(u, h, rhs)
        t += h
    return u


def block_tau(u: np.ndarray, dt_block: float, B_fn, h_micro: float) -> np.ndarray:
    """Strang-like split: Phi_prior(dt/2) then Phi_B(dt) then Phi_prior(dt/2)."""
    u = rk4_integrate(u, 0.5 * dt_block, h_micro, f_prior)
    u = rk4_integrate(u, dt_block, h_micro, B_fn)
    u = rk4_integrate(u, 0.5 * dt_block, h_micro, f_prior)
    return u


def ani4_strang_step(u: np.ndarray, dt_strang: float, B_fn, h_micro: float) -> np.ndarray:
    k1 = block_tau(u, dt_strang, B_fn, h_micro)
    k2 = block_tau(block_tau(u, 0.5 * dt_strang, B_fn, h_micro), 0.5 * dt_strang, B_fn, h_micro)
    return -1.0 / 3.0 * k1 + 4.0 / 3.0 * k2


def advance_macro_additive(u: np.ndarray, dt_macro: float, B_fn, h_micro: float) -> np.ndarray:

    def rhs(uu):
        return f_prior(uu) + B_fn(uu)

    return rk4_integrate(u, dt_macro, h_micro, rhs)


def advance_macro_ani4(u: np.ndarray, dt_macro: float, B_fn, h_micro: float, n_sub_strang: int) -> np.ndarray:
    h = dt_macro / float(n_sub_strang)
    for _ in range(n_sub_strang):
        u = ani4_strang_step(u, h, B_fn, h_micro)
    return u


def advance_macro_truth(u: np.ndarray, dt_macro: float, h_micro: float) -> np.ndarray:
    """One macro step: full Lorenz–Stenflo, RK4 sub-steps."""
    return rk4_integrate(u, dt_macro, h_micro, f_lorenz_stenflo_full)


def rollout_truth_rk4(
    u0: np.ndarray,
    dt_macro: float,
    n_steps: int,
    h_micro: float,
    log_every: int = 0,
) -> np.ndarray:
    traj = [u0.copy()]
    u = u0.astype(np.float64).copy()
    for i in range(n_steps):
        u = advance_macro_truth(u, dt_macro, h_micro)
        traj.append(u.copy())
        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"      truth macro {i + 1}/{n_steps}", flush=True)
    return np.stack(traj, axis=0)


def rollout_rk4_additive(
    u0: np.ndarray,
    dt_macro: float,
    n_steps: int,
    B_fn,
    h_micro: float,
    log_every: int = 0,
) -> np.ndarray:
    traj = [u0.copy()]
    u = u0.astype(np.float64).copy()
    for i in range(n_steps):
        u = advance_macro_additive(u, dt_macro, B_fn, h_micro)
        traj.append(u.copy())
        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"      macro step {i + 1}/{n_steps}", flush=True)
    return np.stack(traj, axis=0)


def rollout_ani4_symbolic(
    u0: np.ndarray,
    dt_macro: float,
    n_steps: int,
    B_fn,
    h_micro: float,
    n_sub_strang: int,
    log_every: int = 0,
) -> np.ndarray:
    traj = [u0.copy()]
    u = u0.astype(np.float64).copy()
    for i in range(n_steps):
        u = advance_macro_ani4(u, dt_macro, B_fn, h_micro, n_sub_strang)
        traj.append(u.copy())
        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"      macro step {i + 1}/{n_steps}", flush=True)
    return np.stack(traj, axis=0)


def rel_err_per_time(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - true, axis=1)
    den = np.linalg.norm(true, axis=1) + 1e-12
    return num / den


def _style_3d_attractor_axis(ax, *, elev: float, azim: float) -> None:
    """Clean paper-style 3D axes: white panes, light grid, readable ticks."""
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("0.82")
        axis.pane.set_linewidth(0.6)
    ax.grid(True, linestyle="-", linewidth=0.5, color="0.78", alpha=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis="both", labelsize=8, pad=2)


def _lims_from_xyz(
    *arrays: np.ndarray, pad_frac: float = 0.04
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Shared x,y,z limits from one or more (N,3) trajectory arrays."""
    xs = np.concatenate([a[:, 0] for a in arrays])
    ys = np.concatenate([a[:, 1] for a in arrays])
    zs = np.concatenate([a[:, 2] for a in arrays])

    def span1d(v: np.ndarray) -> tuple[float, float]:
        lo, hi = float(np.min(v)), float(np.max(v))
        d = hi - lo
        p = pad_frac * d if d > 1e-12 else 1.0
        return lo - p, hi + p

    return span1d(xs), span1d(ys), span1d(zs)


def sample_ic_random(rng: np.random.Generator) -> np.ndarray:
    """Uniform IC in [-20,20] x [-30,30] x [0,40] x [-20,20]."""
    x = rng.uniform(-20.0, 20.0)
    y = rng.uniform(-30.0, 30.0)
    z = rng.uniform(0.0, 40.0)
    w = rng.uniform(-20.0, 20.0)
    return np.array([x, y, z, w], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=DT_DEFAULT, help="macro step h (e.g. 0.05 s)")
    p.add_argument("--h_micro", type=float, default=H_MICRO_DEFAULT, help="RK4 sub-step size")
    p.add_argument("--n_sub_strang", type=int, default=N_SUB_STRANG_DEFAULT)
    p.add_argument("--out_dir", type=Path, default=ROOT / "symbolic_rollout_out")
    p.add_argument("--log_every", type=int, default=250, help="0 disables intra-rollout prints")
    p.add_argument(
        "--ic_from",
        choices=("test", "random"),
        default="test",
        help="ICs: first state of each test trajectory, or uniform random in paper box",
    )
    p.add_argument("--n_traj", type=int, default=N_TRAJ_DEFAULT, help="used if ic_from=random")
    p.add_argument("--n_steps", type=int, default=N_STEPS_DEFAULT, help="macro steps (T = n_steps+1)")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for ic_from=random")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dt = args.dt
    h_micro = args.h_micro
    n_sub = args.n_sub_strang
    log_every = args.log_every

    if args.ic_from == "test":
        print("symbolic_refinement_rollout: loading test_trajectories for ICs only ...", flush=True)
        test_file = np.load(ROOT / "dataset" / "test_trajectories.npy").astype(np.float64)
        n_traj, T, dim = test_file.shape
        assert dim == 4
        n_steps = T - 1
        initial_states = [test_file[k, 0].copy() for k in range(n_traj)]
        print(f"  ICs from test file: n_traj={n_traj}, T={T}, macro steps={n_steps}", flush=True)
    else:
        n_traj = args.n_traj
        n_steps = args.n_steps
        T = n_steps + 1
        rng = np.random.default_rng(args.seed)
        initial_states = [sample_ic_random(rng) for _ in range(n_traj)]
        print(
            f"  random ICs: n_traj={n_traj}, n_steps={n_steps}, T={T}, seed={args.seed}",
            flush=True,
        )

    print(
        f"  macro dt={dt}, h_micro={h_micro}, n_sub_strang={n_sub}",
        flush=True,
    )
    print(f"  output dir: {args.out_dir.resolve()}", flush=True)
    if log_every > 0:
        print(f"  intra-rollout progress every {log_every} macro steps (--log_every 0 to disable)", flush=True)

    print("[1/3] truth: full Lorenz–Stenflo, RK4 (same dt, h_micro) ...", flush=True)
    t_truth0 = time.perf_counter()
    truth_list = []
    for k in range(n_traj):
        print(f"    truth trajectory {k + 1}/{n_traj} ...", flush=True)
        truth_list.append(
            rollout_truth_rk4(initial_states[k], dt, n_steps, h_micro, log_every)
        )
    truth = np.stack(truth_list, axis=0)
    print(f"  truth rollouts done in {time.perf_counter() - t_truth0:.1f}s", flush=True)

    methods = {
        "rk4_prior_plus_B_baseline": lambda u0, n: rollout_rk4_additive(
            u0, dt, n, B_baseline_sindy, h_micro, log_every
        ),
        "ani4_strang_B_ani4": lambda u0, n: rollout_ani4_symbolic(
            u0, dt, n, B_ani4_distilled, h_micro, n_sub, log_every
        ),
    }

    stored = {
        "truth": truth,
        "dt": dt,
        "t": np.arange(T) * dt,
        "truth_is": "rk4_lorenz_stenflo_full",
    }
    all_rel = {name: np.zeros((n_traj, T), dtype=np.float64) for name in methods}

    for mi, (name, roll) in enumerate(methods.items(), start=2):
        print(f"[{mi}/3] {name}: rolling out ...", flush=True)
        t0 = time.perf_counter()
        preds = []
        for k in range(n_traj):
            print(f"    trajectory {k + 1}/{n_traj} ({n_steps} macro steps) ...", flush=True)
            u0 = initial_states[k]
            pred = roll(u0, n_steps)
            if pred.shape[0] != T:
                pred = pred[:T]
            preds.append(pred)
            all_rel[name][k] = rel_err_per_time(pred, truth[k])
        stored[name] = np.stack(preds, axis=0)
        elapsed = time.perf_counter() - t0
        m_rel = float(np.mean(all_rel[name]))
        print(f"    done in {elapsed:.1f}s, mean rel L2 vs ref: {m_rel:.4e}", flush=True)

    mean_rel_time = {name: all_rel[name].mean(axis=0) for name in methods}
    summary_lines = [
        f"truth = RK4 full Lorenz–Stenflo ODE (same dt={dt}, h_micro={h_micro})",
        f"ic_from={args.ic_from}, n_traj={n_traj}, T={T}, n_sub_strang={n_sub}",
        "mean relative L2 ||pred-ref||/||ref|| (time average over trajs):",
    ]
    for name in methods:
        m = float(np.mean(mean_rel_time[name]))
        summary_lines.append(f"  {name}: {m:.6e}")
    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text, flush=True)
    print("Writing trajectory_error_summary.txt and rel_error_per_traj_time.npz ...", flush=True)
    (args.out_dir / "trajectory_error_summary.txt").write_text(summary_text, encoding="utf-8")
    np.savez(
        args.out_dir / "rel_error_per_traj_time.npz",
        **{f"{k}": all_rel[k] for k in methods},
        **{f"mean_rel_{k}": mean_rel_time[k] for k in methods},
        t=np.arange(T) * dt,
        step=np.arange(T, dtype=np.float64),
    )

    k0 = 0
    truth_k = truth[k0]
    X_base = stored["rk4_prior_plus_B_baseline"][k0]
    X_ani = stored["ani4_strang_B_ani4"][k0]
    (xlim, ylim, zlim) = _lims_from_xyz(truth_k, X_base, X_ani)

    _elev, _azim = 22.0, -58.0
    _attractor_specs: list[tuple[np.ndarray, str]] = [
        (truth_k, "attractor_xyz_traj0_ref.pdf"),
        (X_base, "attractor_xyz_traj0_sindy.pdf"),
        (X_ani, "attractor_xyz_traj0_ani4.pdf"),
    ]
    for X, fname in _attractor_specs:
        fig = plt.figure(figsize=(5.2, 4.6), facecolor="white")
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:blue", linewidth=0.95, alpha=0.95)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("x", fontsize=10, labelpad=5)
        ax.set_ylabel("y", fontsize=10, labelpad=5)
        ax.set_zlabel("z", fontsize=10, labelpad=5)
        _style_3d_attractor_axis(ax, elev=_elev, azim=_azim)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.08)
        out_path = args.out_dir / fname
        print(f"Saving {fname} ...", flush=True)
        plt.savefig(
            out_path,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

    t_axis = np.arange(T) * dt
    step_axis = np.arange(T, dtype=np.float64)

    plt.figure(figsize=(8, 4))
    for name, lab, c in [
        ("rk4_prior_plus_B_baseline", "SINDy", "#009E73"),
        ("ani4_strang_B_ani4", "ANI4", "#E69F00"),
    ]:
        plt.semilogy(t_axis, mean_rel_time[name], label=lab, color=c, lw=1.5)
    plt.xlabel("time (s)")
    plt.ylabel("mean rel. L2 vs ref")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("Saving mean_rel_error_vs_time.pdf ...", flush=True)
    plt.savefig(args.out_dir / "mean_rel_error_vs_time.pdf")
    plt.close()

    plt.figure(figsize=(8, 4))
    for name, lab, c in [
        ("rk4_prior_plus_B_baseline", "SINDy", "#009E73"),
        ("ani4_strang_B_ani4", "ANI4", "#E69F00"),
    ]:
        plt.semilogy(step_axis, mean_rel_time[name], label=lab, color=c, lw=1.5)
    plt.xlabel("macro step index $k$ (state after $k$ steps)")
    plt.ylabel("mean rel. L2 vs ref")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("Saving mean_rel_error_vs_step.pdf ...", flush=True)
    plt.savefig(args.out_dir / "mean_rel_error_vs_step.pdf")
    plt.close()

    method_lines = [
        ("rk4_prior_plus_B_baseline", "SINDy", "#009E73"),
        ("ani4_strang_B_ani4", "ANI4", "#E69F00"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for ax, (name, lab, c) in zip(axes, method_lines):
        for k in range(n_traj):
            ax.semilogy(step_axis, all_rel[name][k], color="0.72", alpha=0.75, lw=0.85)
        ax.semilogy(step_axis, mean_rel_time[name], color=c, lw=2.0, label="mean over trajs")
        ax.set_ylabel("rel. L2 vs ref")
        ax.set_title(lab, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("macro step index $k$")
    plt.tight_layout()
    print("Saving rel_error_vs_step_by_traj.pdf ...", flush=True)
    plt.savefig(args.out_dir / "rel_error_vs_step_by_traj.pdf")
    plt.close()

    print("Saving trajectories_for_chaos.mat ...", flush=True)
    sio.savemat(
        str(args.out_dir / "trajectories_for_chaos.mat"),
        {
            "truth": truth,
            "rk4_prior_plus_B_baseline": stored["rk4_prior_plus_B_baseline"],
            "ani4_strang_B_ani4": stored["ani4_strang_B_ani4"],
            "dt": np.array([dt]),
            "h_micro": np.array([h_micro]),
            "n_sub_strang": np.array([n_sub], dtype=np.int32),
            "t": t_axis.reshape(-1, 1),
            "truth_is": np.array(["rk4_lorenz_stenflo_full"]),
            "ic_from": np.array([args.ic_from]),
        },
        do_compression=True,
    )
    print(f"All done. Outputs -> {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
