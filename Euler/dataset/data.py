import numpy as np
from FD1D_Euler import FD, Con2Pri
import os

def fun_Pri_1(x, k=3):
    """
    CE-RPUI-inspired 1D initial condition generator with jump positions
    aligned to indices that are multiples of 4.
    
    x: [N,] array of grid points in [0,1)
    Returns: [N, 3] array with (rho, u, p)
    """
    import numpy as np

    N = len(x)
    assert np.all((x >= 0) & (x < 1)), "x ∈ [0, 1) assumed"

    # Choose number of jumps (max k)
    k = np.random.randint(1, k + 1)

    # Choose k unique jump indices that are multiples of 4
    valid_idxs = np.arange(4, N, 4)
    assert len(valid_idxs) >= k, "Not enough valid jump points. Increase N or reduce k."
    chosen_idxs = np.sort(np.random.choice(valid_idxs, size=k, replace=False))

    # Get the actual split points in x
    split_points = x[chosen_idxs]
    split_points = np.concatenate([split_points, [1.0]])

    # Sample piecewise constant values
    rho_vals = np.random.uniform(0.1, 2.0, k + 1)
    u_vals   = np.random.uniform(-2.0, 2.0, k + 1)
    p_vals   = np.random.uniform(0.1, 2.0, k + 1)

    # Assign values based on segment
    indices = np.searchsorted(split_points, x, side='right')
    indices = np.clip(indices, 0, k)  # Ensure indices within bounds

    rho = rho_vals[indices]
    u   = u_vals[indices]
    p_  = p_vals[indices]

    return np.stack([rho, u, p_], axis=-1)



def downsample(u, factor=4):
    """
    Downsample a 2D array u [N_high, C] → [N_high // factor, C]
    Using reshape + mean for piecewise average.
    """
    N_high, C = u.shape
    assert N_high % factor == 0
    # return u.reshape(N_high // factor, factor, C).mean(axis=1)
    return u[::factor, :]

def generate_dataset(N_high=512, factor=4, num_samples=10000, val_ratio=0.1, T=150, out_dir="./"):
    os.makedirs(out_dir, exist_ok=True)
    N_low = N_high // factor

    X, Y = [], []

    i = 0
    while i < num_samples:
        solver = FD(N_high)
        solver.CFL = 0.4
        solver.bc = 1
        Ng = solver.Ng

        solver.Init(fun_Pri_1)
        u_t = solver.Cons[Ng:N_high+Ng, :].copy()  # 初始 [N_high, 3]

        traj_X = []
        traj_Y = []

        success = True
        step = 0
        saved = 0
        t_saved = 0
        while saved < T:
            dt = solver.Compute_dt()
            t_saved += dt
            solver.t_end = dt
            success = solver.Solve()
            if not success:
                print(f"[Retry] Sample {i+1} failed at step {step}. Retrying...")
                break

            step += 1
            if step % factor == 0:  # 每隔 4 步保存一次
                u_next = solver.Cons[Ng:N_high+Ng, :].copy()  # [N_high, 3]
                u_input = np.concatenate([u_t, np.full((N_high, 1), t_saved)], axis=-1)  # [N_high, 4]

                # 下采样
                u_input_low = downsample(u_input, factor)  # [N_low, 4]
                u_next_low  = downsample(u_next,  factor)  # [N_low, 3]

                traj_X.append(u_input_low)
                traj_Y.append(u_next_low)
                u_t = u_next  # 更新为下一步

                saved += 1

                t_saved = 0

        if not success:
            continue

        X.extend(traj_X)
        Y.extend(traj_Y)

        i += T
        print(f"\r✅ Generated {i}/{num_samples}", end="")

    X = np.array(X)
    Y = np.array(Y)

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    num_val = int(len(X) * val_ratio)
    np.save(os.path.join(out_dir, "train_input_best.npy"), X[:-num_val])
    np.save(os.path.join(out_dir, "train_output_best.npy"), Y[:-num_val])
    np.save(os.path.join(out_dir, "val_input_best.npy"), X[-num_val:])
    np.save(os.path.join(out_dir, "val_output_best.npy"), Y[-num_val:])

    print(f"\n✅ Dataset saved to {out_dir}")
    print(f"Train size: {len(X)-num_val}, Val size: {num_val}")

if __name__ == "__main__":
    generate_dataset(N_high=128, factor=1, num_samples=15000, val_ratio=0.2, out_dir="./")


# def generate_trajectories(N_high=512, N_low=128, dt=0.05, total_time=1.0, num_traj=10, out_path="./trajectories.npy"):
#     num_steps = int(total_time / dt)
#     all_trajs = []

#     l = int(round(N_high / N_low))  # 下采样倍数

#     traj_idx = 0
#     while traj_idx < num_traj:
#         retry_count = 0
#         while retry_count < 20:
#             solver = FD(N_high)
#             solver.t_end = dt
#             solver.CFL = 0.8
#             solver.bc = 1
#             Ng = solver.Ng

#             solver.Init(fun_Pri)
#             u_init = solver.Cons[Ng:N_high+Ng, :].copy()
#             buffer_frames = [u_init[::l, :]]  # 下采样初始状态

#             success = True
#             for step in range(num_steps):
#                 print(f"step: {step}, traj: {traj_idx+1}/{num_traj}")
#                 if not solver.Solve():
#                     print(f"\n[❌] Traj {traj_idx+1}, step {step+1} failed. Retrying trajectory...")
#                     success = False
#                     break

#                 u_next = solver.Cons[Ng:N_high+Ng, :].copy()
#                 buffer_frames.append(u_next[::l, :])

#                 # 设置下一步
#                 def new_init(_): return Con2Pri(u_next)
#                 solver = FD(N_high)
#                 solver.t_end = dt
#                 solver.CFL = 0.8
#                 solver.bc = 1
#                 solver.Init(new_init)

#             if success:
#                 traj = np.stack(buffer_frames)  # [num_steps+1, N_low, 3]
#                 all_trajs.append(traj)
#                 print(f"\r✅ Trajectory {traj_idx+1}/{num_traj} generated", end="")
#                 traj_idx += 1
#                 break
#             else:
#                 retry_count += 1

#     all_trajs = np.stack(all_trajs)  # [num_traj, num_steps+1, N_low, 3]
#     np.save(out_path, all_trajs)
#     print(f"\n✅ Saved {len(all_trajs)} trajectories to {out_path}")

# if __name__ == "__main__":
#     generate_dataset(N_high=256, dt=0.01, num_samples=10000, val_ratio=0.1)
#     generate_trajectories(num_traj=5, N_high=256, dt=0.01, total_time=1.0)