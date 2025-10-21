import numpy as np
from FD1D_Euler import FD, Con2Pri
import os



def fun_Pri(x):
    N = len(x)
    res = np.zeros((N, 3))
    res[:, 0] = np.random.uniform(0.1, 1.0, N)    # Density
    res[:, 1] = np.random.uniform(-1.0, 1.0, N)    # Velocity
    res[:, 2] = np.random.uniform(0.1, 1.0, N)     # Pressure
    return res

def generate_dataset(N=128, dt=0.05, num_samples=1000, val_ratio=0.1, out_dir="./"):
    os.makedirs(out_dir, exist_ok=True)
    
    X, Y = [], []

    solver = FD(N)
    solver.t_end = dt
    solver.CFL = 0.8
    solver.bc = 1
    Ng = solver.Ng

    i = 0
    while i < num_samples:
        solver.Init(fun_Pri)
        u0_high = solver.Cons[Ng:N+Ng, :].copy()

        success = solver.Solve()
        if not success:
            print(f"[Retry] dt too small at sample {i+1}, retrying...")
            continue

        u1_high = solver.Cons[Ng:N+Ng, :].copy()

        u0 = u0_high
        u1 = u1_high

        X.append(u0)
        Y.append(u1)
        i += 1
        print(f"\rGenerating dataset: {i}/{num_samples}", end="")

    X = np.array(X)  # [num_samples, N_low, 3]
    Y = np.array(Y)

    # Split
    num_val = int(num_samples * val_ratio)
    np.save(os.path.join(out_dir, "train_input.npy"), X[:-num_val])
    np.save(os.path.join(out_dir, "train_output.npy"), Y[:-num_val])
    np.save(os.path.join(out_dir, "val_input.npy"), X[-num_val:])
    np.save(os.path.join(out_dir, "val_output.npy"), Y[-num_val:])

    print(f"\n✅ Dataset saved to {out_dir}")
    print(f"Train size: {len(X)-num_val}, Val size: {num_val}")

def generate_trajectories(N=128, dt=0.05, total_time=1.0, num_traj=10, out_path="./trajectories.npy", Pri_func=None):
    num_steps = int(total_time / dt)
    Ng = 3
    max_retry = 20  # 每个 trajectory 最多尝试多少次

    all_trajs = []  # 存储所有轨迹

    traj_idx = 0
    while traj_idx < num_traj:
        retry_count = 0

        while retry_count < max_retry:
            solver = FD(N)
            solver.t_end = dt
            solver.CFL = 0.4
            solver.bc = 1

            solver.Init(fun_Pri)
            frames = [solver.Cons[Ng:N+Ng, :].copy()]

            success = True
            for step in range(num_steps):
                if not solver.Solve():
                    print(f"\n[❌] Traj {traj_idx+1}, step {step+1} failed (dt too small). Retrying trajectory...")
                    success = False
                    break

                u_next = solver.Cons[Ng:N+Ng, :].copy()
                frames.append(u_next)

                # 用当前帧作为下一步的初始条件
                def new_init(x): return Con2Pri(u_next)
                # solver = FD(N)
                # solver.t_end = dt
                # solver.CFL = 0.4
                # solver.bc = 1
                solver.Init(new_init)

            if success and len(frames) == num_steps + 1:
                traj = np.stack(frames)  # [num_steps+1, N, 3]
                all_trajs.append(traj)
                print(f"\r✅ Trajectory {traj_idx+1}/{num_traj} generated", end="")
                traj_idx += 1
                break  # 成功，跳出 retry 循环
            else:
                retry_count += 1

        if retry_count == max_retry:
            print(f"\n[❌] Failed to generate trajectory {traj_idx+1} after {max_retry} retries. Skipping.")

    all_trajs = np.stack(all_trajs)  # [num_traj, num_steps+1, N, 3]
    np.save(out_path, all_trajs)
    print(f"\n✅ Saved all {len(all_trajs)} trajectories to {out_path}")



if __name__ == "__main__":
    # generate_dataset(N=128, dt=0.05, num_samples=1000, val_ratio=0.1)
    generate_trajectories(num_traj=5, N=128, dt=0.05, total_time=1.0)