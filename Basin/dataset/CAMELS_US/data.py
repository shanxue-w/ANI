import os
import pandas as pd
import numpy as np
import torch

from scipy.signal import lfilter
from scipy.optimize import minimize

# =====================
# Config
# =====================
BASIN_ID = "01031500"
DATA_ROOT = "."
OUTPUT_FILE = f"data_{BASIN_ID}.pt"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
DT = 1.0


# =====================
# Utilities
# =====================
def get_basin_area(basin_id, root_dir):
    meta_path = os.path.join(root_dir, "basin_metadata", "basin_physical_characteristics.txt")
    default_area = 100.0
    try:
        if not os.path.exists(meta_path):
            return default_area
        df_meta = pd.read_csv(meta_path, sep=r'\s+', dtype={'BASIN_ID': str})
        row = df_meta[df_meta['BASIN_ID'] == str(basin_id)]
        if not row.empty:
            return float(row.iloc[0]['Size(km2)'])
    except Exception:
        pass
    return default_area


def fit_generator_ode(P, T, Q_obs, dt=1.0):
    def loss_fn(params):
        k, alpha = params
        
        if k <= 1e-5: return 1e9
        
        gamma = np.exp(-k * dt)
        
        forcing = P - alpha * T
        
        
        b = [(1 - gamma)]
        a = [1, -gamma]
        
        # 模拟 Q
        Q_sim = lfilter(b, a, forcing)
        
        warmup = 30 
        mse = np.mean((Q_sim[warmup:] - Q_obs[warmup:]) ** 2)
        return mse

    x0 = [0.1, 0.5] 

    bounds = [(1e-4, 5.0), (0.0, 10.0)]

    res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds)
    
    k_best, alpha_best = res.x
    
    return k_best, alpha_best


def process_basin(basin_id, root):
    print(f"Processing Basin: {basin_id}")

    # ---- forcing ----
    huc_code = basin_id[:2]
    forcing_dir = os.path.join(root, "basin_mean_forcing", "daymet", huc_code)

    f_files = [f for f in os.listdir(forcing_dir) if basin_id in f and f.endswith(".txt")]
    if not f_files:
        raise FileNotFoundError("Forcing file not found")

    forcing_path = os.path.join(forcing_dir, f_files[0])
    df_f = pd.read_csv(forcing_path, sep=r"\s+", header=3)

    df_f["Date"] = pd.to_datetime(
        df_f[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "Month"})
    )
    df_f["T_mean"] = 0.5 * (df_f["tmax(C)"] + df_f["tmin(C)"])
    df_f["T_range"] = df_f["tmax(C)"] - df_f["tmin(C)"]

    # ---- streamflow ----
    flow_path = None
    for r, _, files in os.walk(os.path.join(root, "usgs_streamflow")):
        for file in files:
            if basin_id in file and "_streamflow" in file:
                flow_path = os.path.join(r, file)
                break
        if flow_path:
            break

    if not flow_path:
        raise FileNotFoundError("Streamflow file not found")

    df_q = pd.read_csv(
        flow_path,
        sep=r"\s+",
        header=None,
        names=["basin", "Y", "M", "D", "Q_cfs", "QC"],
    )
    df_q["Date"] = pd.to_datetime(
        df_q[["Y", "M", "D"]].rename(columns={"Y": "Year", "M": "Month", "D": "Day"})
    )

    # ---- merge ----
    df = pd.merge(df_f, df_q[["Date", "Q_cfs"]], on="Date", how="inner")

    area = get_basin_area(basin_id, root)
    factor = (0.028316847 * 86400 * 1000) / (area * 1e6)
    df["Q_obs"] = df["Q_cfs"] * factor
    df = df[df["Q_obs"] >= 0].reset_index(drop=True)

    # ---- split ----
    N = len(df)
    train_end = int(N * TRAIN_RATIO)
    val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # ---- ODE prior fit (THIS IS THE KEY) ----
    k_prior, alpha_prior = fit_generator_ode(
        df_train["prcp(mm/day)"].values,
        df_train["T_mean"].values,
        df_train["Q_obs"].values,
        dt=DT,
    )

    print(f"[ODE PRIOR] k = {k_prior:.6f}, alpha = {alpha_prior:.6f}")

    # ---- ML features ----
    feature_cols = [
        "prcp(mm/day)",
        "T_mean",
        "T_range",
        "srad(W/m2)",
        "vp(Pa)",
        "dayl(s)",
    ]
    target_col = "Q_obs"

    X_train = df_train[feature_cols].values
    y_train = df_train[[target_col]].values

    X_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float64)
    X_std = torch.tensor(X_train.std(axis=0), dtype=torch.float64) + 1e-6
    y_mean = torch.tensor(y_train.mean(axis=0), dtype=torch.float64)
    y_std = torch.tensor(y_train.std(axis=0), dtype=torch.float64) + 1e-6

    def pack(df_part):
        X_raw = torch.tensor(df_part[feature_cols].values, dtype=torch.float64)
        y_raw = torch.tensor(df_part[[target_col]].values, dtype=torch.float64)
        return {
            "X_raw": X_raw,
            "y_raw": y_raw,
            "X_norm": (X_raw - X_mean) / X_std,
            "y_norm": (y_raw - y_mean) / y_std,
            "Date": df_part["Date"].dt.strftime("%Y-%m-%d").values,
        }

    data = {
        "basin_id": basin_id,
        "train": pack(df_train),
        "val": pack(df_val),
        "test": pack(df_test),
        "stats": {
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "feature_names": feature_cols,
        },
        "ode_prior": {
            "k": k_prior,
            "alpha": alpha_prior,
            "dt": DT,
            "form": "dQ/dt = k(P - alpha T - Q)",
        },
    }

    torch.save(data, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_basin(BASIN_ID, DATA_ROOT)
