import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


DT = 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EPS = 1e-6


def _simulate_two_reservoir(
    P: np.ndarray,
    T: np.ndarray,
    dt: float,
    k_fast: float,
    k_slow: float,
    split_fast: float,
    alpha: float,
    q0: float,
) -> np.ndarray:
    n = len(P)
    qf = split_fast * q0
    qs = (1.0 - split_fast) * q0
    out = np.zeros(n, dtype=np.float64)
    g_f = np.exp(-k_fast * dt)
    g_s = np.exp(-k_slow * dt)
    for i in range(n):
        forcing = P[i] - alpha * T[i]
        qf = g_f * qf + (1.0 - g_f) * (split_fast * forcing)
        qs = g_s * qs + (1.0 - g_s) * ((1.0 - split_fast) * forcing)
        out[i] = qf + qs
    return out


def fit_two_reservoir_prior(
    P: np.ndarray, T: np.ndarray, Q_obs: np.ndarray, dt: float = 1.0
) -> Tuple[float, float, float, float]:
    """Fit fast/slow linear reservoirs with temperature-modulated forcing."""

    q0 = float(max(Q_obs[0], 0.0))

    def loss_fn(params):
        k_fast, k_slow, split_fast, alpha = params
        # Physical ordering + positivity constraints
        if k_fast <= k_slow or k_slow <= 1e-5:
            return 1e12
        if not (0.02 <= split_fast <= 0.98):
            return 1e12
        q_sim = _simulate_two_reservoir(P, T, dt, k_fast, k_slow, split_fast, alpha, q0)
        warmup = 30
        return float(np.mean((q_sim[warmup:] - Q_obs[warmup:]) ** 2))

    x0 = [0.35, 0.03, 0.65, 0.5]
    bounds = [(1e-3, 8.0), (1e-4, 2.0), (0.02, 0.98), (0.0, 10.0)]
    res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds)
    k_fast, k_slow, split_fast, alpha = [float(v) for v in res.x]
    if k_fast < k_slow:
        k_fast, k_slow = k_slow, k_fast
        split_fast = 1.0 - split_fast
    return k_fast, k_slow, split_fast, alpha


def _find_flow_path(root: str, basin_id: str) -> str:
    for r, _dirs, files in os.walk(os.path.join(root, "usgs_streamflow")):
        for file in files:
            if basin_id in file and "_streamflow" in file:
                return os.path.join(r, file)
    raise FileNotFoundError(f"Streamflow file not found for {basin_id}")


def _get_basin_area_km2(root: str, basin_id: str) -> float:
    """Match single-basin data.py:get_basin_area — cfs→mm/d scaling needs real drainage area."""
    meta_path = os.path.join(root, "basin_metadata", "basin_physical_characteristics.txt")
    default_area = 100.0
    try:
        if not os.path.exists(meta_path):
            return default_area
        df_meta = pd.read_csv(meta_path, sep=r"\s+", dtype={"BASIN_ID": str})
        row = df_meta[df_meta["BASIN_ID"] == str(basin_id)]
        if not row.empty:
            return float(row.iloc[0]["Size(km2)"])
    except Exception:
        pass
    return default_area


def _load_gauge_table(gauge_path: str) -> pd.DataFrame:
    """Parse CAMELS gauge_information.txt; GAGE_NAME contains spaces so plain whitespace CSV fails."""
    rows = []
    with open(gauge_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line.strip():
            return pd.DataFrame()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                lat = float(parts[-3])
                lon = float(parts[-2])
                area = float(parts[-1])
            except ValueError:
                continue
            huc = parts[0]
            gage_id = parts[1]
            name = " ".join(parts[2:-3])
            rows.append(
                {
                    "HUC_02": huc,
                    "GAGE_ID": gage_id,
                    "GAGE_NAME": name,
                    "LAT": lat,
                    "LONG": lon,
                    "DRAINAGE_AREA_KM2": area,
                }
            )
    return pd.DataFrame(rows)


def _load_static_tables(root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    physical_path = os.path.join(root, "basin_metadata", "basin_physical_characteristics.txt")
    gauge_path = os.path.join(root, "basin_metadata", "gauge_information.txt")
    df_phy = pd.read_csv(physical_path, sep=r"\s+", dtype={"BASIN_ID": str})
    df_gauge = _load_gauge_table(gauge_path)
    return df_phy, df_gauge


def _build_static_vector(df_phy: pd.DataFrame, df_gauge: pd.DataFrame, basin_id: str) -> Tuple[np.ndarray, List[str]]:
    row_phy = df_phy[df_phy["BASIN_ID"] == basin_id]
    row_g = df_gauge[df_gauge["GAGE_ID"] == basin_id]
    if row_phy.empty or row_g.empty:
        raise ValueError(f"Missing static metadata for basin {basin_id}")

    # Use all numeric static fields we can reliably extract from CAMELS metadata.
    phy_cols = [c for c in row_phy.columns if c != "BASIN_ID"]
    gauge_cols = ["LAT", "LONG", "DRAINAGE_AREA_KM2"]
    available_gauge_cols = [c for c in gauge_cols if c in row_g.columns]

    phy_vec = row_phy[phy_cols].to_numpy(dtype=np.float64).reshape(-1)
    if available_gauge_cols:
        gauge_vec = row_g[available_gauge_cols].to_numpy(dtype=np.float64).reshape(-1)
        gauge_names = [f"gauge_{c}" for c in available_gauge_cols]
    else:
        gauge_vec = np.zeros(0, dtype=np.float64)
        gauge_names = []

    vec = np.concatenate([phy_vec, gauge_vec], axis=0)
    names = [f"phy_{c}" for c in phy_cols] + gauge_names
    return vec, names


def process_single_basin(root: str, basin_id: str, static_vec: np.ndarray) -> Dict:
    huc_code = basin_id[:2]
    forcing_dir = os.path.join(root, "basin_mean_forcing", "daymet", huc_code)
    if not os.path.isdir(forcing_dir):
        raise FileNotFoundError(f"Forcing directory not found for HUC {huc_code}")

    f_files = [f for f in os.listdir(forcing_dir) if basin_id in f and f.endswith(".txt")]
    if not f_files:
        raise FileNotFoundError(f"Forcing file not found for {basin_id}")
    forcing_path = os.path.join(forcing_dir, f_files[0])

    df_f = pd.read_csv(forcing_path, sep=r"\s+", header=3)
    df_f["Date"] = pd.to_datetime(df_f[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "Month"}))
    df_f["T_mean"] = 0.5 * (df_f["tmax(C)"] + df_f["tmin(C)"])
    df_f["T_range"] = df_f["tmax(C)"] - df_f["tmin(C)"]

    flow_path = _find_flow_path(root, basin_id)
    df_q = pd.read_csv(
        flow_path,
        sep=r"\s+",
        header=None,
        names=["basin", "Y", "M", "D", "Q_cfs", "QC"],
    )
    df_q["Date"] = pd.to_datetime(df_q[["Y", "M", "D"]].rename(columns={"Y": "Year", "M": "Month", "D": "Day"}))

    df = pd.merge(df_f, df_q[["Date", "Q_cfs"]], on="Date", how="inner")
    area = max(_get_basin_area_km2(root, basin_id), 1.0)
    factor = (0.028316847 * 86400 * 1000) / (area * 1e6)
    df["Q_obs"] = df["Q_cfs"] * factor
    df = df[df["Q_obs"] >= 0].reset_index(drop=True)

    n = len(df)
    if n < 200:
        raise ValueError(f"Basin {basin_id} has too few samples ({n})")
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    feature_cols = ["prcp(mm/day)", "T_mean", "T_range", "srad(W/m2)", "vp(Pa)", "dayl(s)"]
    target_col = "Q_obs"
    x_train = df_train[feature_cols].to_numpy(np.float64)
    y_train = df_train[[target_col]].to_numpy(np.float64)

    k_fast, k_slow, split_fast, alpha_prior = fit_two_reservoir_prior(
        df_train["prcp(mm/day)"].to_numpy(np.float64),
        df_train["T_mean"].to_numpy(np.float64),
        df_train["Q_obs"].to_numpy(np.float64),
        dt=DT,
    )

    return {
        "basin_id": basin_id,
        "feature_cols": feature_cols,
        "train_df": df_train,
        "val_df": df_val,
        "test_df": df_test,
        "x_train": x_train,
        "y_train": y_train,
        "static_raw": static_vec.astype(np.float64),
        "ode_prior": {
            "k_fast": k_fast,
            "k_slow": k_slow,
            "split_fast": split_fast,
            "alpha": alpha_prior,
            "k": 0.5 * (k_fast + k_slow),  # legacy field for backward compatibility
            "dt": DT,
            "form": "two_reservoir: dQf/dt=kf(sf*F-Qf), dQs/dt=ks((1-sf)*F-Qs), F=P-alpha*T",
        },
    }


def build_dataset(root: str, basin_ids: List[str]) -> Dict:
    df_phy, df_gauge = _load_static_tables(root)

    records = []
    static_names_ref = None
    for bid in basin_ids:
        try:
            static_vec, static_names = _build_static_vector(df_phy, df_gauge, bid)
            rec = process_single_basin(root, bid, static_vec)
            records.append(rec)
            if static_names_ref is None:
                static_names_ref = static_names
            print(f"[ok] {bid}: train={len(rec['train_df'])} val={len(rec['val_df'])} test={len(rec['test_df'])}")
        except Exception as e:
            print(f"[skip] {bid}: {e}")

    if not records:
        raise RuntimeError("No valid basins were processed.")

    # Global normalization over all train basins.
    x_train_all = np.concatenate([r["x_train"] for r in records], axis=0)
    y_train_all = np.concatenate([r["y_train"] for r in records], axis=0)
    static_all = np.stack([r["static_raw"] for r in records], axis=0)

    x_mean = x_train_all.mean(axis=0)
    x_std = x_train_all.std(axis=0) + EPS
    y_mean = y_train_all.mean(axis=0)
    y_std = y_train_all.std(axis=0) + EPS
    s_mean = static_all.mean(axis=0)
    s_std = static_all.std(axis=0) + EPS

    basins = {}
    basin_id_to_index = {}
    for i, r in enumerate(records):
        bid = r["basin_id"]
        basin_id_to_index[bid] = i

        def pack(df_part: pd.DataFrame):
            x_raw = torch.tensor(df_part[r["feature_cols"]].to_numpy(np.float64), dtype=torch.float64)
            y_raw = torch.tensor(df_part[["Q_obs"]].to_numpy(np.float64), dtype=torch.float64)
            x_norm = (x_raw - torch.tensor(x_mean, dtype=torch.float64)) / torch.tensor(x_std, dtype=torch.float64)
            y_norm = (y_raw - torch.tensor(y_mean, dtype=torch.float64)) / torch.tensor(y_std, dtype=torch.float64)
            return {
                "X_raw": x_raw,
                "X_norm": x_norm,
                "y_raw": y_raw,
                "y_norm": y_norm,
                "Date": df_part["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
            }

        static_raw = torch.tensor(r["static_raw"], dtype=torch.float64)
        static_norm = (static_raw - torch.tensor(s_mean, dtype=torch.float64)) / torch.tensor(s_std, dtype=torch.float64)

        basins[bid] = {
            "train": pack(r["train_df"]),
            "val": pack(r["val_df"]),
            "test": pack(r["test_df"]),
            "static_raw": static_raw,
            "static_norm": static_norm,
            "ode_prior": r["ode_prior"],
        }

    return {
        "basins": basins,
        "basin_ids": list(basins.keys()),
        "basin_id_to_index": basin_id_to_index,
        "stats": {
            "X_mean": torch.tensor(x_mean, dtype=torch.float64),
            "X_std": torch.tensor(x_std, dtype=torch.float64),
            "y_mean": torch.tensor(y_mean, dtype=torch.float64),
            "y_std": torch.tensor(y_std, dtype=torch.float64),
            "static_mean": torch.tensor(s_mean, dtype=torch.float64),
            "static_std": torch.tensor(s_std, dtype=torch.float64),
            "feature_names": records[0]["feature_cols"],
            "static_feature_names": static_names_ref or [],
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--basin_ids", type=str, default="", help="Comma-separated basin IDs. Empty => use top_n by metadata.")
    p.add_argument("--top_n", type=int, default=30, help="Used when --basin_ids is empty.")
    p.add_argument("--output", type=str, default="data_multi_basin.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.basin_ids.strip():
        basin_ids = [x.strip() for x in args.basin_ids.split(",") if x.strip()]
    else:
        phy_path = os.path.join(args.data_root, "basin_metadata", "basin_physical_characteristics.txt")
        df_phy = pd.read_csv(phy_path, sep=r"\s+", dtype={"BASIN_ID": str})
        basin_ids = df_phy["BASIN_ID"].astype(str).tolist()[: args.top_n]

    out = build_dataset(args.data_root, basin_ids)
    torch.save(out, args.output)
    print(f"Saved multi-basin dataset to: {args.output}")
    print(f"Valid basins: {len(out['basin_ids'])}")
