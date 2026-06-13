import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# --------------------------
# Utilities: robust access
# --------------------------
def _as_1d_float(x):
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(np.float64, copy=False)

def _as_str(x):
    # Robust conversion for MATLAB char arrays / bytes / numpy arrays
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="ignore").strip()
    if isinstance(x, str):
        return x.strip()
    arr = np.asarray(x).squeeze()
    if arr.dtype.kind in ("U", "S"):
        # may be array(['discharge'], dtype='<U9') or similar
        try:
            return str(arr.item()).strip()
        except Exception:
            return str(arr).strip()
    return str(arr).strip()

def _get_mat_battery_key(mat, preferred="B0005"):
    if preferred in mat:
        return preferred
    # choose the first non-private key
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise KeyError("MAT file has no non-private variables.")
    return keys[0]

def _get_cycle_iter(batt_obj):
    # batt_obj.cycle may be list/ndarray of mat_struct
    cycles = getattr(batt_obj, "cycle", None)
    if cycles is None:
        raise KeyError("Battery object has no 'cycle' field.")
    # Ensure iterable
    if isinstance(cycles, (list, tuple)):
        return cycles
    cycles = np.asarray(cycles).squeeze()
    if cycles.ndim == 0:
        return [cycles.item()]
    return list(cycles)

# ==========================================
# 1. Data processing (dt + cycle + robust)
# ==========================================
def process_and_split_dynamic(
    mat_file,
    train_ratio=0.8,
    val_ratio=0.1,
    battery_key="B0005",
    dt_min=1e-4,          # seconds, for derivative stability
    min_points=20
):
    print(f"[1/4] Loading {mat_file} ...")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Cannot find {mat_file}")

    # SciPy recommended options for MATLAB structs
    mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)  # :contentReference[oaicite:3]{index=3}
    key = _get_mat_battery_key(mat, preferred=battery_key)
    batt = mat[key]

    cycles = _get_cycle_iter(batt)

    raw_cycles = []
    discharge_count = 0

    # First pass: collect discharge cycles
    for cyc in cycles:
        ctype = _as_str(getattr(cyc, "type", None))
        if ctype.lower() != "discharge":
            continue

        data = getattr(cyc, "data", None)
        if data is None:
            continue

        V = _as_1d_float(getattr(data, "Voltage_measured", []))
        I = _as_1d_float(getattr(data, "Current_measured", []))
        t = _as_1d_float(getattr(data, "Time", []))


        if len(V) < min_points or len(I) < min_points or len(t) < min_points:
            continue

        # Remove NaN/Inf
        m = np.isfinite(V) & np.isfinite(I) & np.isfinite(t)
        V, I, t = V[m], I[m], t[m]
        # t /= 10
        # t /= 60
        
        if np.nanmedian(I) < 0:
            I_load = -I
        else:
            I_load = I

        I_load = np.maximum(I_load, 0.0)

        # is_discharging = I_load > 1e-2
        # V = V[is_discharging]
        # I_load = I_load[is_discharging]
        # t = t[is_discharging]

        if len(V) < min_points:
            continue

        # Ensure time is non-decreasing; drop non-increasing steps later
        # Decide current sign convention: make discharge current positive (I_load)
        # If median is negative, assume discharge current is negative -> flip

          # keep discharge as positive, clip any sign noise

        # I_load = I

        # Coulomb counting SoC using correct dt (first dt = 0)
        dt_full = np.diff(t, prepend=t[0])
        dt_full[0] = 0.0
        # For SoC integration, ignore negative dt contributions
        dt_full = np.maximum(dt_full, 0.0)

        # Use initial capacity estimate:
        # Prefer first cycle's Capacity field if available; else fallback to 2.0Ah.
        cap = getattr(data, "Capacity", None)
        cap_val = None
        try:
            if cap is not None:
                cap_val = float(np.asarray(cap).squeeze())
                if not np.isfinite(cap_val) or cap_val <= 0:
                    cap_val = None
        except Exception:
            cap_val = None

        raw_cycles.append({
            "V": V,
            "I_load": I_load,
            "t": t,
            "cap": cap_val,     # may be None
            "cycle_id": discharge_count
        })
        discharge_count += 1

    print(f"      Valid discharge cycles: {discharge_count}")
    if discharge_count < 3:
        raise RuntimeError("Too few valid discharge cycles after filtering.")

    # Determine Q0 reference (Ah): use first non-null cap, else 2.0
    Q0 = None
    for item in raw_cycles:
        if item["cap"] is not None:
            Q0 = item["cap"]
            break
    if Q0 is None:
        Q0 = 2.0  # dataset docs commonly treat B0005 ~2Ah initial  :contentReference[oaicite:4]{index=4}

    # Split by cycle index (contiguous)
    n_train = max(1, int(discharge_count * train_ratio))
    n_val = max(1, int(discharge_count * val_ratio))
    n_test = discharge_count - n_train - n_val
    if n_test < 1:
        # ensure we have a test set
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_end = n_train - 1
    val_end = train_end + n_val
    train_cycle_ids = np.arange(0, train_end + 1, dtype=int).tolist()
    val_cycle_ids = np.arange(train_end + 1, val_end + 1, dtype=int).tolist()
    test_cycle_ids = np.arange(val_end + 1, discharge_count, dtype=int).tolist()

    split_meta = {
        "n_discharge_cycles": int(discharge_count),
        "n_train_cycles": int(n_train),
        "n_val_cycles": int(n_val),
        "n_test_cycles": int(n_test),
        "train_cycle_ids": train_cycle_ids,
        "val_cycle_ids": val_cycle_ids,
        "test_cycle_ids": test_cycle_ids,
    }

    # Second pass: build samples
    X_list, Y_list, labels = [], [], []

    denom = (discharge_count - 1) if discharge_count > 1 else 1

    for item in raw_cycles:
        V = item["V"]
        I_load = item["I_load"]
        t = item["t"]
        cid = item["cycle_id"]

        # dt between successive points
        dt_steps = np.diff(t)
        # Keep only strictly positive dt and above dt_min
        ok = (dt_steps > 0) & (dt_steps >= dt_min)

        if ok.sum() < (min_points - 1):
            continue
            
        V_t = V[:-1][ok]
        V_next = V[1:][ok]
        I_t = I_load[:-1][ok]
        dt_t = dt_steps[ok]

        # SoC for this cycle (from I_load, dt)
        # integrate charge from start to each time (consistent with filtered indexing)
        dt_full = np.diff(t, prepend=t[0])
        dt_full[0] = 0.0
        dt_full = np.maximum(dt_full, 0.0)
        q = np.cumsum(I_load * dt_full) / 3600.0  # Ah consumed
        SoC_full = 1.0 - (q / Q0)
        SoC_full = np.clip(SoC_full, 0.0, 1.0)
        SoC_t = SoC_full[:-1][ok]

        norm_cycle = cid / denom

        X_cycle = np.stack([
            V_t,             # Voltage at t
            I_t,             # Load current (discharge positive)
            SoC_t,           # SoC
            np.full_like(V_t, norm_cycle, dtype=np.float64),
            dt_t             # dt (s)
        ], axis=1)

        X_list.append(X_cycle)
        Y_list.append(V_next)

        if cid <= train_end:
            lab = 0
        elif cid <= val_end:
            lab = 1
        else:
            lab = 2
        labels.extend([lab] * len(X_cycle))

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    base_meta = {"battery_key": key, "Q0_Ah": Q0, "dt_min": dt_min, "min_points": min_points}
    base_meta.update(split_meta)

    return (X[labels == 0], Y[labels == 0],
            X[labels == 1], Y[labels == 1],
            X[labels == 2], Y[labels == 2],
            base_meta)

import numpy as np
from scipy.optimize import minimize_scalar

def fit_physical_parameters_linear(X_train, Y_train, dt_min=1e-3,
                                     tau_bounds=(1e-2, 1e6),
                                     denom_eps=1e-6):
    """
    Fit Thevenin prior using exact discretization:
        V_{k+1} = V_inf + (V_k - V_inf) * exp(-dt/tau)
        V_inf   = (w0 + w1*SoC) - R*I
    Solve by:
        - 1D search over tau
        - closed-form LS for (w0, w1, R) given tau

    Returns:
        R_est, C_est, [w0, w1], tau_est
    """
    V = X_train[:, 0].astype(np.float64)
    I = X_train[:, 1].astype(np.float64)  # assume I_load (discharge positive)
    S = X_train[:, 2].astype(np.float64)
    dt = X_train[:, 4].astype(np.float64)
    V_next = Y_train.astype(np.float64)

    # Basic filtering
    ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(S) & np.isfinite(dt) & np.isfinite(V_next) & (dt >= dt_min)
    V, I, S, dt, V_next = V[ok], I[ok], S[ok], dt[ok], V_next[ok]
    if V.size < 1000:
        print("Warning: too few samples after filtering; fit may be unstable.")

    # For a given tau, solve linear LS for theta = [w0, w1, R]
    def solve_theta_given_tau(tau):
        a = np.exp(-dt / tau)
        one_minus = 1.0 - a

        # Avoid division/near-singularity when dt is tiny (a~1)
        m = one_minus > denom_eps
        if m.sum() < 100:
            return np.inf, None

        b = V_next[m] - a[m] * V[m]  # left-hand
        A = np.stack([
            one_minus[m],              # (1-a)*w0
            one_minus[m] * S[m],       # (1-a)*w1*SoC
            -one_minus[m] * I[m],      # -(1-a)*R*I
        ], axis=1)

        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = A @ theta - b
        cost = float(np.mean(resid**2))
        return cost, theta

    # 1D search in log-space for numerical stability
    lo, hi = tau_bounds
    if lo <= 0 or hi <= lo:
        raise ValueError("tau_bounds must satisfy 0 < lo < hi")

    def objective(log_tau):
        tau = np.exp(log_tau)
        cost, _ = solve_theta_given_tau(tau)
        return cost

    res = minimize_scalar(
        objective,
        bounds=(np.log(lo), np.log(hi)),
        method="bounded",
        options={"xatol": 1e-4}
    )

    tau_est = float(np.exp(res.x))
    cost, theta = solve_theta_given_tau(tau_est)

    if theta is None or (not np.isfinite(cost)):
        print("Warning: analytic fit failed; falling back to defaults.")
        R_est, C_est, w_est = 0.2, 2000.0, [3.5, 0.5]
        tau_est = R_est * C_est
        return R_est, C_est, w_est, tau_est

    w0, w1, R_est = map(float, theta)

    # Physical sanity: R>0, tau>0 => C = tau/R >0
    if (R_est <= 0) or (tau_est <= 0):
        print("Warning: non-physical (R<=0 or tau<=0). Falling back to defaults.")
        R_est, C_est, w_est = 0.2, 2000.0, [3.5, 0.5]
        tau_est = R_est * C_est
        return R_est, C_est, w_est, tau_est

    C_est = float(tau_est / R_est)
    w_est = [w0, w1]

    print(f"Fit (analytic): tau={tau_est:.6g}s, R={R_est:.6g}Ω, C={C_est:.6g}F, OCV={w0:.6g}+{w1:.6g}*SoC")
    return R_est, C_est, w_est


# ==========================================
# 3. Frozen prior model (for checking)
# ==========================================
class TheveninPriorFixed(nn.Module):
    def __init__(self, R_val, C_val, w_vals, dtype=torch.float64):
        super().__init__()
        self.R = nn.Parameter(torch.tensor(float(R_val), dtype=dtype), requires_grad=False)
        self.C = nn.Parameter(torch.tensor(float(C_val), dtype=dtype), requires_grad=False)
        self.w = nn.Parameter(torch.tensor(w_vals, dtype=dtype), requires_grad=False)

    def get_ocv(self, soc):
        return self.w[0] + self.w[1] * soc

    def forward(self, x):
        # x: [V, I_load, SoC, Cycle, dt]
        V_t = x[:, 0]
        I_t = x[:, 1]
        SoC_t = x[:, 2]
        dt = x[:, 4]

        tau = self.R * self.C
        V_ocv = self.get_ocv(SoC_t)
        V_inf = V_ocv - I_t * self.R

        # Ensure dt is non-negative
        dt = torch.clamp(dt, min=0.0)
        alpha = torch.exp(-dt / tau)
        V_next = V_inf + (V_t - V_inf) * alpha
        return V_next.unsqueeze(1)

# ==========================================
# 4. Run & save
# ==========================================
# ==========================================
# Helper: Create Sliding Window Sequences
# ==========================================
def create_rollout_sequences(X, Y, rollout_len=5):
    """
    X: (N, Features)
    Y: (N,) or (N, 1)
    Returns:
        X_seq: (N - L + 1, L, Features)
        Y_seq: (N - L + 1, L, 1)
    Note: This assumes X and Y are continuous time series. 
    
    Since we don't want to modify the upstream function, we will use this
    but be aware of the minor boundary artifacts (which are usually negligible 
    if N >> number of cycles).
    """
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    if num_samples < rollout_len:
        # Fallback if data is too short
        return (
            X.reshape(1, num_samples, num_features),
            Y.reshape(1, num_samples, 1)
        )

    # Efficient stride tricks could be used, but loop is clear and safe
    X_seq_list = []
    Y_seq_list = []
    
    # Simple sliding window
    # We want input [t, t+1, ..., t+L-1]
    for i in range(num_samples - rollout_len + 1):
        X_seq_list.append(X[i : i + rollout_len])
        Y_seq_list.append(Y[i : i + rollout_len])
        
    X_seq = np.stack(X_seq_list, axis=0) # (N_seq, L, Feat)
    Y_seq = np.stack(Y_seq_list, axis=0) # (N_seq, L)
    
    if Y_seq.ndim == 2:
        Y_seq = Y_seq[..., np.newaxis]   # (N_seq, L, 1)
        
    return X_seq, Y_seq

# ==========================================
# 4. Export .pt bundle (train/val rollout seq + test pointwise)
# ==========================================
def export_processed_rollout_pt(
    mat_path,
    save_path,
    battery_key="B0005",
    train_ratio=0.8,
    val_ratio=0.1,
    dt_min=1e-3,
    min_points=20,
    rollout_len=5,
    do_reload_check=True,
):
    """
    Build the torch dict used for training: rollout windows for train/val,
    raw steps for test. ``meta`` lists held-out train/val/test discharge-cycle ids.
    """
    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw, meta = process_and_split_dynamic(
        mat_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        battery_key=battery_key,
        dt_min=dt_min,
        min_points=min_points,
    )

    print(f"[2/4] Fitting physics parameters on raw 2D data (N={len(X_train_raw)})...")
    R_fit, C_fit, w_fit = fit_physical_parameters_linear(X_train_raw, Y_train_raw, dt_min=meta["dt_min"])

    print(f"[3/4] Converting to rollout sequences (Len={rollout_len})...")

    X_train_seq, Y_train_seq = create_rollout_sequences(X_train_raw, Y_train_raw, rollout_len)
    X_val_seq, Y_val_seq = create_rollout_sequences(X_val_raw, Y_val_raw, rollout_len)

    data_dict = {
        "X_train": torch.tensor(X_train_seq, dtype=torch.float64),
        "Y_train": torch.tensor(Y_train_seq, dtype=torch.float64),
        "X_val": torch.tensor(X_val_seq, dtype=torch.float64),
        "Y_val": torch.tensor(Y_val_seq, dtype=torch.float64),
        "X_test": torch.tensor(X_test_raw, dtype=torch.float64),
        "Y_test": torch.tensor(Y_test_raw, dtype=torch.float64),

        "prior_params": {"R": R_fit, "C": C_fit, "w": w_fit},
        "feature_names": ["Voltage", "I_load(discharge+)", "SoC", "Norm_Cycle", "dt"],
        "description": (
            f"NASA battery rollout sequences (L={rollout_len}); Thevenin prior; "
            f"key={meta.get('battery_key')}"
        ),
        "meta": meta,
    }

    print(f"[4/4] Saving to {save_path} ...")
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(data_dict, save_path)
    print("      Saved.")

    if do_reload_check:
        print("      Reload check ...")
        loaded = torch.load(save_path, map_location="cpu")

        print(f"      Train X shape: {loaded['X_train'].shape}")
        print(f"      Train Y shape: {loaded['Y_train'].shape}")

        prior = TheveninPriorFixed(
            loaded["prior_params"]["R"],
            loaded["prior_params"]["C"],
            loaded["prior_params"]["w"],
            dtype=torch.float64
        )

        with torch.no_grad():
            sample_first_step = loaded["X_train"][:5, 0, :]
            pred = prior(sample_first_step)
            print(f"      Prior forward (on 1st step of seq) OK, output shape: {pred.shape}")

    return data_dict


# ==========================================
# 5. CLI: default B0005 (backward compatible)
# ==========================================
if __name__ == "__main__":
    export_processed_rollout_pt(
        mat_path="B0005.mat",
        save_path="processed_battery_data_rollout.pt",
        battery_key="B0005",
        train_ratio=0.8,
        val_ratio=0.1,
        dt_min=1e-3,
        min_points=20,
        rollout_len=5,
        do_reload_check=True,
    )
    print("Done.")
