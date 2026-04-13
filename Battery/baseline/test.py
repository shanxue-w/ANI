import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from base import Baseline, load_data

_BATTERY_ROOT = Path(__file__).resolve().parents[1]
if str(_BATTERY_ROOT) not in sys.path:
    sys.path.insert(0, str(_BATTERY_ROOT))
from eval_rollout_metrics import (  # noqa: E402
    CYCLE_IDX,
    evaluate_all_test_cycles,
    recursive_prediction_single_cycle,
)

# ================= 配置区域 =================
DATA_PATH = "../dataset/processed_battery_data_rollout.pt"
MODEL_PATH = "best_baseline_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICT_MODE = "full"
NUM_CYCLES_TO_PLOT = 18
PDF_PREFIX = "battery_Baseline"

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def _q0_ah_from_pt(path: str) -> float:
    d = torch.load(path, map_location="cpu")
    return float((d.get("meta") or {}).get("Q0_Ah", 2.0))


if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH} ...")
    train_data, val_data, test_data, _prior_params = load_data(DATA_PATH)
    test_x_all, test_y_all = test_data
    q0_ah = _q0_ah_from_pt(DATA_PATH)
    print(f"Rollout Q0_Ah (meta, default 2.0): {q0_ah}")

    print("Initializing model...")
    model = Baseline().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    rows, summ = evaluate_all_test_cycles(
        model,
        test_x_all,
        test_y_all,
        DEVICE,
        q0_ah=q0_ah,
        min_len=10,
        predict_mode=PREDICT_MODE,
    )

    print(f"\n===== Test (eval_rollout_metrics protocol) =====")
    print(f"predict_mode={PREDICT_MODE!r}")
    n = int(summ.get("n_cycles", 0))
    print(f"Cycles evaluated: {n}")
    if n > 0:
        print(
            f"MSE  mean ± std: {summ['mse_mean']:.6e} ± {summ['mse_std']:.6e}\n"
            f"MAE  mean ± std: {summ['mae_mean']:.6e} ± {summ['mae_std']:.6e}\n"
            f"RMSE mean ± std: {summ['rmse_mean']:.6e} ± {summ['rmse_std']:.6e}"
        )
        mses = [r.mse for r in rows]
        print(f"Worst cycle MSE: {max(mses):.2e}")
        print(f"Best cycle MSE: {min(mses):.2e}")

    all_cycles_id = test_x_all[:, CYCLE_IDX].numpy()
    n_save = min(NUM_CYCLES_TO_PLOT, len(rows))
    for plot_i in range(n_save):
        idx = plot_i
        cid = rows[idx].cycle_feature_id
        mask = all_cycles_id == cid
        cycle_x = test_x_all[mask]
        cycle_y = test_y_all[mask]
        pred_vals = recursive_prediction_single_cycle(
            model, cycle_x, DEVICE, q0_ah=q0_ah, predict_mode=PREDICT_MODE
        )
        true_vals = cycle_y.numpy().flatten()
        mse = float(np.mean((pred_vals - true_vals) ** 2))

        plt.figure(figsize=(8, 6))
        plt.plot(true_vals, "k-", label="Ground Truth", linewidth=2)
        plt.plot(pred_vals, "r--", label=f"Baseline (MSE={mse:.1e})", linewidth=1.5)
        plt.ylabel("Voltage (V)")
        plt.xlabel("Time Step (within cycle)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"{PDF_PREFIX}_{plot_i}.pdf"
        plt.savefig(out_path, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved {out_path} (norm_cycle={cid:.6f}, MSE={mse:.6e})")
