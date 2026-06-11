import sys
import importlib.util
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec


# ============================================================
# Directory layout
# ============================================================
ROOT = Path(__file__).resolve().parent

ANI2_DIR = ROOT / "2th"
ANI4_DIR = ROOT / "4th"
BASE_DIR = ROOT / "baseline"


# ============================================================
# Model/path configuration
# ============================================================
# 如果你的 4th 权重文件实际仍然叫 ANI2_Best_{BASIN_ID}.pth，
# 把 ANI4 的 model_path_template 改掉即可。
PANEL_SPECS = [
    {
        "panel": "a",
        "title": "Calibrated prior",
        "folder": ANI2_DIR,
        "module_file": "ANI2.py",
        "class_name": "ANI_2th_Hydro",
        "model_path_template": "ANI2_Best_{BASIN_ID}.pth",
        "mode": "prior",
        "legend_label": "Prior",
    },
    {
        "panel": "b",
        "title": "Transformer-only baseline",
        "folder": BASE_DIR,
        "module_file": "base.py",
        "class_name": "Baseline_Hydro",
        "model_path_template": "Baseline_Best_{BASIN_ID}.pth",
        "mode": "forward",
        "legend_label": "Baseline",
    },
    {
        "panel": "c",
        "title": "ANI-2",
        "folder": ANI2_DIR,
        "module_file": "ANI2.py",
        "class_name": "ANI_2th_Hydro",
        "model_path_template": "ANI2_Best_{BASIN_ID}.pth",
        "mode": "forward",
        "legend_label": "ANI-2",
    },
    {
        "panel": "d",
        "title": "ANI-4",
        "folder": ANI4_DIR,
        "module_file": "ANI4.py",
        "class_name": "ANI_4th_Hydro",
        "model_path_template": "ANI4_Best_{BASIN_ID}.pth",
        "mode": "forward",
        "legend_label": "ANI-4",
    },
]


# ============================================================
# Nature-style plotting
# ============================================================
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
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def import_module_from_file(module_name: str, file_path: Path):
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


def resolve_relative_path(base_dir: Path, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(obs, sim):
    obs = np.asarray(obs).reshape(-1)
    sim = np.asarray(sim).reshape(-1)

    n = min(len(obs), len(sim))
    obs = obs[:n]
    sim = sim[:n]

    denom = np.sum((obs - np.mean(obs)) ** 2) + 1e-12
    nse = 1.0 - np.sum((obs - sim) ** 2) / denom

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-8)
    beta = np.mean(sim) / (np.mean(obs) + 1e-8)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    pbias = 100.0 * np.sum(sim - obs) / (np.sum(obs) + 1e-12)

    return {
        "NSE": float(nse),
        "KGE": float(kge),
        "PBIAS": float(pbias),
    }


def metric_text(m):
    return (
        f"NSE: {m['NSE']:.3f}\n"
        f"KGE: {m['KGE']:.3f}\n"
        f"PBIAS: {m['PBIAS']:+.1f}%"
    )


# ============================================================
# Prediction helpers
# ============================================================
def instantiate_model(module, class_name, full_data):
    ModelClass = getattr(module, class_name)

    input_dim = int(full_data["train"]["X_norm"].shape[-1])

    # Most of your Hydro models use this signature.
    model = ModelClass(
        ode_prior=full_data["ode_prior"],
        stats=full_data["stats"],
        input_dim=input_dim,
        hidden_dim=int(module.HIDDEN_DIM),
    ).to(module.device)

    return model


def rollout_forward(model, test_X_raw, test_X_norm, test_y_raw, seq_len, device):
    history_norm = test_X_norm[:seq_len].unsqueeze(0)
    q_start = test_y_raw[seq_len - 1: seq_len].to(device)

    x_future_raw = test_X_raw[seq_len - 1:].unsqueeze(0)
    x_future_norm = test_X_norm[seq_len - 1:].unsqueeze(0)

    pred_steps = len(test_X_raw) - seq_len

    with torch.no_grad():
        y_pred, _ = model(
            history_norm,
            x_future_raw,
            x_future_norm,
            q_start,
            pred_steps,
        )

    preds = y_pred.squeeze().detach().cpu().numpy()
    obs = test_y_raw[seq_len:].detach().cpu().squeeze().numpy()

    return obs, preds


def rollout_prior_from_ani2(model, test_X_raw, test_X_norm, test_y_raw, seq_len, device):
    """
    Prior-only rollout using ANI2.forward_prior.
    """
    history_norm = test_X_norm[:seq_len].unsqueeze(0)
    q_start = test_y_raw[seq_len - 1: seq_len].to(device)

    x_future_raw = test_X_raw[seq_len - 1:].unsqueeze(0)
    x_future_norm = test_X_norm[seq_len - 1:].unsqueeze(0)

    pred_steps = len(test_X_raw) - seq_len

    if not hasattr(model, "forward_prior"):
        raise AttributeError(
            "ANI2 prior panel requires model.forward_prior(...), but this method was not found."
        )

    with torch.no_grad():
        y_pred, _ = model.forward_prior(
            history_norm,
            x_future_raw,
            x_future_norm,
            q_start,
            pred_steps,
        )

    preds = y_pred.squeeze().detach().cpu().numpy()
    obs = test_y_raw[seq_len:].detach().cpu().squeeze().numpy()

    return obs, preds


def run_one_spec(spec):
    folder = spec["folder"]
    module_file = folder / spec["module_file"]

    module_name = f"basin_{spec['panel']}_{folder.name}_{Path(spec['module_file']).stem}"
    module = import_module_from_file(module_name, module_file)

    data_path = resolve_relative_path(folder, module.DATA_FILE)
    basin_id = module.BASIN_ID
    seq_len = int(module.SEQ_LEN)
    device = module.device

    full_data = torch_load(data_path, map_location=device)

    test_X_raw = full_data["test"]["X_raw"].to(device)
    test_X_norm = full_data["test"]["X_norm"].to(device)
    test_y_raw = full_data["test"]["y_raw"].to(device)

    model = instantiate_model(module, spec["class_name"], full_data)

    model_path = folder / spec["model_path_template"].format(BASIN_ID=basin_id)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find weight file for panel {spec['panel']}: {model_path}\n"
            f"Check model_path_template in PANEL_SPECS."
        )

    model.load_state_dict(torch_load(model_path, map_location=device))
    model.eval()

    if spec["mode"] == "prior":
        obs, preds = rollout_prior_from_ani2(
            model,
            test_X_raw,
            test_X_norm,
            test_y_raw,
            seq_len,
            device,
        )
    elif spec["mode"] == "forward":
        obs, preds = rollout_forward(
            model,
            test_X_raw,
            test_X_norm,
            test_y_raw,
            seq_len,
            device,
        )
    else:
        raise ValueError(f"Unknown mode: {spec['mode']}")

    m = calculate_metrics(obs, preds)

    print(
        f"[{spec['panel']}] {spec['legend_label']} | "
        f"Basin={basin_id} | "
        f"NSE={m['NSE']:.4f}, KGE={m['KGE']:.4f}, PBIAS={m['PBIAS']:+.2f}%"
    )

    return {
        "panel": spec["panel"],
        "title": spec["title"],
        "legend_label": spec["legend_label"],
        "obs": np.asarray(obs).reshape(-1),
        "preds": np.asarray(preds).reshape(-1),
        "metrics": m,
        "basin_id": basin_id,
    }


# ============================================================
# Plotting
# ============================================================
def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_one_panel(ax, result, show_xlabel=True, show_ylabel=True):
    obs = result["obs"]
    preds = result["preds"]
    n = min(len(obs), len(preds))
    obs = obs[:n]
    preds = preds[:n]

    ax.plot(
        obs,
        color="black",
        alpha=0.75,
        linewidth=0.75,
        label="Observed",
    )

    ax.plot(
        preds,
        color="red",
        alpha=0.85,
        linewidth=0.75,
        linestyle="--",
        label=result["legend_label"],
    )

    ax.set_title(result["title"], pad=3)

    if show_xlabel:
        ax.set_xlabel("Time Step")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Streamflow (mm/day)")
    else:
        ax.set_ylabel("")

    ax.grid(True, linewidth=0.35, alpha=0.30)



    ax.text(
        0.02,
        0.95,
        metric_text(result["metrics"]),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        # bbox=dict(
        #     boxstyle="round,pad=0.18",
        #     facecolor="white",
        #     edgecolor="0.4",
        #     linewidth=0.4,
        #     alpha=0.85,
        # ),
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.5,
    )

    add_panel_label(ax, result["panel"])


def main():
    results = [run_one_spec(spec) for spec in PANEL_SPECS]

    # Shared y-limit, so all panels are visually comparable.
    all_vals = []
    for r in results:
        n = min(len(r["obs"]), len(r["preds"]))
        all_vals.append(r["obs"][:n])
        all_vals.append(r["preds"][:n])

    all_vals = np.concatenate(all_vals)
    ymin = min(0.0, float(np.nanmin(all_vals)))
    ymax = float(np.nanmax(all_vals))
    pad = 0.04 * (ymax - ymin + 1e-12)
    ylim = (ymin - pad, ymax + pad)

    # Nature double-column figure
    width_mm = 180
    height_mm = 125

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.20,
        hspace=0.32,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    # top row no x-label; bottom row with x-label
    plot_one_panel(axes[0], results[0], show_xlabel=True, show_ylabel=True)
    plot_one_panel(axes[1], results[1], show_xlabel=True, show_ylabel=True)
    plot_one_panel(axes[2], results[2], show_xlabel=True, show_ylabel=True)
    plot_one_panel(axes[3], results[3], show_xlabel=True, show_ylabel=True)

    for ax in axes:
        ax.set_ylim(*ylim)

    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.085,
        top=0.950,
        wspace=0.22,
        hspace=0.34,
    )

    fig.savefig(
        "basin_rollout_comparison_nature.pdf",
        format="pdf",
        dpi=600,
    )
    fig.savefig(
        "basin_rollout_comparison_nature.png",
        format="png",
        dpi=600,
    )

    plt.close(fig)

    print("Saved: basin_rollout_comparison_nature.pdf")
    print("Saved: basin_rollout_comparison_nature.png")


if __name__ == "__main__":
    main()