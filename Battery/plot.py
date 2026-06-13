import sys
import importlib.util
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec


ROOT = Path(__file__).resolve().parent

ANI2_DIR = ROOT / "2th"
ANI4_DIR = ROOT / "4th"
BASELINE_DIR = ROOT / "baseline"

DATA_PATH = ROOT / "dataset" / "processed_battery_data_rollout.pt"

ANI2_MODEL_PATH = ANI2_DIR / "best_ani2_model.pth"
ANI4_MODEL_PATH = ANI4_DIR / "best_ani4_model.pth"
BASELINE_MODEL_PATH = BASELINE_DIR / "best_baseline_model.pth"

EVAL_METRICS_PATH = ROOT / "eval_rollout_metrics.py"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOT_CYCLE_INDEX = 0

BASELINE_PREDICT_MODE = "full"

Q0_AH_OVERRIDE = None


# ============================================================
# Feature indices
# [V_t, I_t, SoC_t, norm_cycle, dt_t]
# ============================================================
V_IDX = 0
I_IDX = 1
SOC_IDX = 2
CYCLE_IDX = 3
DT_IDX = 4


def mm_to_inch(mm):
    return mm / 25.4


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================
# Import helpers
# ============================================================
@contextmanager
def prepend_sys_path(path: Path):
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def import_from_file(module_name: str, file_path: Path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find module file: {file_path}")

    with prepend_sys_path(file_path.parent):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def q0_ah_from_pt(path: Path) -> float:
    if Q0_AH_OVERRIDE is not None:
        return float(Q0_AH_OVERRIDE)

    try:
        d = torch_load(path, map_location="cpu")
        return float((d.get("meta") or {}).get("Q0_Ah", 2.0))
    except Exception:
        return 2.0


# ============================================================
# Rollout functions
# ============================================================
def scalar_voltage(output):
    """
    Convert model output to scalar voltage.
    """
    if isinstance(output, torch.Tensor):
        return output.reshape(-1)[0]
    return torch.tensor(float(output), device=DEVICE)


def recursive_prediction_ani(model, cycle_x, device, q0_ah, mode="forward"):
    """
    Recursive rollout for ANI2 / ANI4 / prior.

    mode:
        "forward": use model(current_input)
        "prior":   use model.prior(current_input)
    """
    model.eval()
    predictions = []

    current_input = cycle_x[0].unsqueeze(0).to(device)
    current_soc = float(current_input[0, SOC_IDX].item())
    q_total_as = 2.0 * 3600.0

    with torch.no_grad():
        for i in range(len(cycle_x)):
            if mode == "prior":
                pred_v = scalar_voltage(model.forward_prior(current_input))
            elif mode == "forward":
                pred_v = scalar_voltage(model(current_input))
            else:
                raise ValueError(f"Unknown rollout mode: {mode}")

            predictions.append(float(pred_v.item()))

            if i == len(cycle_x) - 1:
                break

            next_input = cycle_x[i + 1].unsqueeze(0).to(device)

            next_i = float(next_input[0, I_IDX].item())
            next_dt = float(next_input[0, DT_IDX].item())

            next_input[0, V_IDX] = pred_v

            next_soc = current_soc - (next_i * next_dt) / q_total_as
            next_soc = max(0.0, min(1.0, next_soc))
            next_input[0, SOC_IDX] = next_soc

            current_input = next_input
            current_soc = next_soc

    return np.asarray(predictions)


def mse(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = min(len(pred), len(true))
    return float(np.mean((pred[:n] - true[:n]) ** 2))


# ============================================================
# Data/model loading
# ============================================================
def load_all():
    ani2_mod = import_from_file("battery_ani2_mod", ANI2_DIR / "ANI2.py")
    ani4_mod = import_from_file("battery_ani4_mod", ANI4_DIR / "ANI4.py")
    base_mod = import_from_file("battery_base_mod", BASELINE_DIR / "base.py")
    metrics_mod = import_from_file("battery_eval_rollout_metrics", EVAL_METRICS_PATH)

    ANI2 = ani2_mod.ANI2
    ANI4 = ani4_mod.ANI4
    Baseline = base_mod.Baseline

    load_data_ani2 = ani2_mod.load_data
    load_data_ani4 = ani4_mod.load_data
    load_data_base = base_mod.load_data

    # Load ANI2 data and prior params
    _, _, test_data_ani2, prior_params2 = load_data_ani2(str(DATA_PATH))
    test_x_ani2, test_y_ani2 = test_data_ani2

    # Load ANI4 prior params
    _, _, _test_data_ani4, prior_params4 = load_data_ani4(str(DATA_PATH))

    # Load baseline test data
    _, _, test_data_base, _ = load_data_base(str(DATA_PATH))
    test_x_base, test_y_base = test_data_base

    q0_ah = q0_ah_from_pt(DATA_PATH)

    # Models
    R2, C2, w2 = prior_params2
    ani2_model = ANI2(R2, C2, w2).to(DEVICE)
    ani2_model.load_state_dict(torch_load(ANI2_MODEL_PATH, map_location=DEVICE))
    ani2_model.eval()

    R4, C4, w4 = prior_params4
    ani4_model = ANI4(R4, C4, w4).to(DEVICE)
    ani4_model.load_state_dict(torch_load(ANI4_MODEL_PATH, map_location=DEVICE))
    ani4_model.eval()

    baseline_model = Baseline().to(DEVICE)
    baseline_model.load_state_dict(torch_load(BASELINE_MODEL_PATH, map_location=DEVICE))
    baseline_model.eval()

    return {
        "ani2_model": ani2_model,
        "ani4_model": ani4_model,
        "baseline_model": baseline_model,
        "metrics_mod": metrics_mod,
        "test_x_ani": test_x_ani2,
        "test_y_ani": test_y_ani2,
        "test_x_base": test_x_base,
        "test_y_base": test_y_base,
        "q0_ah": q0_ah,
    }


def select_cycle(test_x, test_y, cycle_index=0, min_len=10):
    all_cycle_ids = test_x[:, CYCLE_IDX].detach().cpu().numpy()
    unique_cycles = np.unique(all_cycle_ids)

    valid = []
    for cid in unique_cycles:
        mask = np.isclose(all_cycle_ids, cid)
        if int(mask.sum()) >= min_len:
            valid.append(cid)

    if len(valid) == 0:
        raise RuntimeError("No valid test cycle found.")

    if cycle_index >= len(valid):
        raise IndexError(
            f"PLOT_CYCLE_INDEX={cycle_index} but only {len(valid)} valid cycles found."
        )

    cid = valid[cycle_index]
    mask = np.isclose(all_cycle_ids, cid)

    cycle_x = test_x[mask]
    cycle_y = test_y[mask]

    return cid, cycle_x, cycle_y


# ============================================================
# Plotting
# ============================================================
def add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(
        x, y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_one(ax, true, pred, title, pred_label, panel_label):
    err = mse(pred, true)

    ax.plot(
        true,
        color="black",
        linestyle="-",
        linewidth=1.1,
        label="Ground truth",
    )
    ax.plot(
        pred,
        color="#D55E00",
        linestyle="--",
        linewidth=1.0,
        label=fr"{pred_label} (MSE={err:.1e})",
    )

    ax.set_title(title, pad=3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, loc="best", handlelength=2.5)

    add_panel_label(ax, panel_label)


def main():
    pack = load_all()

    q0_ah = pack["q0_ah"]

    cid, cycle_x_ani, cycle_y_ani = select_cycle(
        pack["test_x_ani"],
        pack["test_y_ani"],
        cycle_index=PLOT_CYCLE_INDEX,
        min_len=10,
    )

    # Pick the same cycle for baseline data
    all_cycle_ids_base = pack["test_x_base"][:, CYCLE_IDX].detach().cpu().numpy()
    base_mask = np.isclose(all_cycle_ids_base, cid)
    cycle_x_base = pack["test_x_base"][base_mask]
    cycle_y_base = pack["test_y_base"][base_mask]

    true_vals = cycle_y_ani.detach().cpu().numpy().reshape(-1)

    # 1. Prior from ANI2.prior
    pred_prior = recursive_prediction_ani(
        pack["ani2_model"],
        cycle_x_ani,
        DEVICE,
        q0_ah=q0_ah,
        mode="prior",
    )

    # 2. Baseline, use your existing evaluation protocol
    pred_base = pack["metrics_mod"].recursive_prediction_single_cycle(
        pack["baseline_model"],
        cycle_x_base,
        DEVICE,
        q0_ah=q0_ah,
        predict_mode=BASELINE_PREDICT_MODE,
    )
    pred_base = np.asarray(pred_base).reshape(-1)

    # 3. ANI-2
    pred_ani2 = recursive_prediction_ani(
        pack["ani2_model"],
        cycle_x_ani,
        DEVICE,
        q0_ah=q0_ah,
        mode="forward",
    )

    # 4. ANI-4
    pred_ani4 = recursive_prediction_ani(
        pack["ani4_model"],
        cycle_x_ani,
        DEVICE,
        q0_ah=q0_ah,
        mode="forward",
    )

    print(f"Selected cycle_feature_id={cid:.6f}")
    print(f"Prior    MSE = {mse(pred_prior, true_vals):.6e}")
    print(f"Baseline MSE = {mse(pred_base, true_vals):.6e}")
    print(f"ANI-2    MSE = {mse(pred_ani2, true_vals):.6e}")
    print(f"ANI-4    MSE = {mse(pred_ani4, true_vals):.6e}")

    # Shared y-limits
    all_y = np.concatenate([
        true_vals,
        pred_prior[:len(true_vals)],
        pred_base[:len(true_vals)],
        pred_ani2[:len(true_vals)],
        pred_ani4[:len(true_vals)],
    ])
    ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
    pad = 0.04 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    # Nature double-column figure
    width_mm = 180
    height_mm = 125

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.22,
        hspace=0.32,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    plot_one(
        axes[0],
        true_vals,
        pred_prior,
        "Mechanistic prior",
        "Prior",
        "a",
    )
    plot_one(
        axes[1],
        true_vals,
        pred_base,
        "Data-driven baseline",
        "Baseline",
        "b",
    )
    plot_one(
        axes[2],
        true_vals,
        pred_ani2,
        "ANI-2",
        "ANI-2",
        "c",
    )
    plot_one(
        axes[3],
        true_vals,
        pred_ani4,
        "ANI-4",
        "ANI-4",
        "d",
    )

    for ax in axes:
        ax.set_ylim(*ylim)

    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.085,
        top=0.950,
        wspace=0.24,
        hspace=0.34,
    )

    fig.savefig(
        "battery_rollout_comparison_nature.pdf",
        format="pdf",
        dpi=600,
    )
    fig.savefig(
        "battery_rollout_comparison_nature.png",
        format="png",
        dpi=600,
    )

    plt.close(fig)

    print("Saved: battery_rollout_comparison_nature.pdf")
    print("Saved: battery_rollout_comparison_nature.png")


if __name__ == "__main__":
    main()