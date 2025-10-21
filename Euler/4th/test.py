import torch
import numpy as np
import matplotlib.pyplot as plt
from ANI_Euler_4th import Euler, A  # 假设你的模型类叫 Euler 并保存在 model.py 中
import os
from FD import FD
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fun_Pri(x, k=3):
    """
    Riemann 问题初始条件（物理量）
    """
    N = len(x)
    res = np.zeros((N, 3))  # [rho, u, p]
    for i in range(N):
        # if x[i] < 0.25:
        #     res[i, 0] = 1.0
        #     res[i, 1] = 0.75
        #     res[i, 2] = 1.0
        # else:
        #     res[i, 0] = 0.125
        #     res[i, 1] = 0.0
        #     res[i, 2] = 0.1
        if x[i] < 0.5:
            res[i, 0] = 1.0
            res[i, 1] = 0.0
            res[i, 2] = 1.0
        else:
            res[i, 0] = 0.125
            res[i, 1] = 0.0
            res[i, 2] = 0.1
        # if x[i] < 0.5:
        #     res[i, 0] = 1.4
        #     res[i, 1] = 0.0
        #     res[i, 2] = 1.0
        # else:
        #     res[i, 0] = 1.0
        #     res[i, 1] = 0.0
        #     res[i, 2] = 1.0
    return res


def Pri2Con(Pri, gamma=1.4):
    """
    物理量 → 守恒量
    输入: [N, 3]  (rho, u, p)
    输出: [N, 3]  (rho, rho*u, E)
    """
    rho, u, p = Pri[:, 0], Pri[:, 1], Pri[:, 2]
    m = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, m, E], axis=-1)

def Con2Pri(Con, gamma=1.4):
    """
    守恒量 → 物理量
    输入: [N, 3]  (rho, rho*u, E)
    输出: [N, 3]  (rho, u, p)
    """
    rho, m, E = Con[:, 0], Con[:, 1], Con[:, 2]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return np.stack([rho, u, p], axis=-1)

def compute_dt(Con, dx, CFL=0.5, gamma=1.4):
    """
    Compute time step using CFL condition.
    Con: [1, N, 3]
    """
    with torch.no_grad():
        rho, m, E = Con[0, :, 0], Con[0, :, 1], Con[0, :, 2]
        u = m / rho
        p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
        a = torch.sqrt(gamma * p / rho)  # sound speed
        max_speed = torch.max(torch.abs(u) + a)
        dt = CFL * dx / max_speed
        return dt.item()

@torch.no_grad()
def main():
    # 加载模型
    model = Euler(N0_SCHEME=A(), modes=32, width=128, layers=8).to(device)
    # model = torch.compile(model, mode='max-autotune')
    model.load_state_dict(torch.load("best_model_new.pth", map_location=device))
    model.eval()

    # 初始条件
    N = 128
    x = np.linspace(0, 1, N, endpoint=False)
    dx = 1.0 / N
    pri = fun_Pri(x)              # [N, 3]
    con = Pri2Con(pri)            # [N, 3]
    # con = torch.tensor(con, dtype=torch.float64).unsqueeze(0).to(device)  # [1, N, 3]
    u = torch.tensor(con, dtype=torch.float64).unsqueeze(0).to(device)  # [1, N, 3]

    t = 0.0
    t_end = 0.2
    steps = 0

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    while t < t_end:
        dt_value = compute_dt(u, dx, CFL=0.4)      # float
        if t + dt_value > t_end:
            dt_value = t_end - t
        dt = torch.tensor([dt_value], dtype=torch.float64).to(device)

        dt_broadcast = dt.expand(1, u.shape[1], 1)
        u_input = torch.cat([u, dt_broadcast], dim=-1)  # [B, N, 4]
        u = model(u_input)
        # u = model.N0_SCHEME.single_step(u, dt)
        t += dt_value
        steps += 1
        # print(f"Step {steps:03d} | t = {t:.4f} | dt = {dt_value:.4e}", end='\r')

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")

    # # 转为物理量
    # u_np = u.squeeze(0).cpu()  # [N, 3]
    # pri = Con2Pri(u_np)        # [N, 3]
    # rho, u, p = pri[:, 0], pri[:, 1], pri[:, 2]
    # cons1, cons2, cons3 = u_np[:, 0], u_np[:, 1], u_np[:, 2]
    # # rho, u, p = u_np[:, 0], u_np[:, 1], u_np[:, 2]


    # solver = FD(N)
    # solver.Init(fun_Pri)
    # solver.bc = 1
    # solver.CFL = 0.4
    # solver.t_end = t_end
    # solver.Solve()
    # u_out = solver.Cons[3:128+3, :].copy()
    # pri = Con2Pri(u_out)        # [N, 3]
    # rho_out, u_out1, p_out = pri[:, 0], pri[:, 1], pri[:, 2]
    # # cons_out_1, cons_out_2, cons_out_3 = u_out[:, 0], u_out[:, 1], u_out[:, 2]

    # error_rho = np.abs(rho - rho_out)
    # error_u = np.abs(u - u_out1)
    # error_p = np.abs(p - p_out)
    # np.save("error_rho_sod_4th.npy", error_rho)
    # np.save("error_u_sod_4th.npy", error_u)
    # np.save("error_p_sod_4th.npy", error_p)
    # # error_cons1 = np.abs(cons1 - cons_out_1)
    # # error_cons2 = np.abs(cons2 - cons_out_2)
    # # error_cons3 = np.abs(cons3 - cons_out_3)


    # #  绘图 真解 rho rho_out 等三个物理量对比，相同的在一张图，两个子图
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.plot(x, rho, label='rho', color='red', linestyle='--')  # 虚线
    # plt.plot(x, rho_out, label='rho_weno')
    # plt.xlabel("x")
    # plt.ylabel("rho")
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.plot(x, u, label='u', color='red', linestyle='--')  # 虚线
    # plt.plot(x, u_out1, label='u_weno')
    # plt.xlabel("x")
    # plt.ylabel("u")
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.plot(x, p, label='p', color='red', linestyle='--')  # 虚线
    # plt.plot(x, p_out, label='p_weno')
    # plt.xlabel("x")
    # plt.ylabel("p")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("riemann_trajectory_model_sod.png")

    # np.save("rho_sod_4th.npy", rho)
    # np.save("u_sod_4th.npy", u)
    # np.save("p_sod_4th.npy", p)


    # # 可视化
    # plt.figure(figsize=(12, 4))
    # for i, (var, label) in enumerate(zip([error_rho, error_u, error_p], ['Density', 'Velocity', 'Pressure'])):
    #     plt.subplot(1, 3, i + 1)
    #     plt.plot(x, var, label=label)
    #     plt.xlabel("x")
    #     plt.ylabel(label)
    #     plt.grid(True)
    #     plt.legend()
    # plt.tight_layout()
    # plt.savefig("riemann_trajectory_error_model_2.png")


def data():
    # 加载模型
    model = Euler(N0_SCHEME=A(), modes=32, width=128, layers=8).to(device)
    # model = torch.compile(model, mode='max-autotune')
    model.load_state_dict(torch.load("best_model_new.pth", map_location=device))
    model.eval()

    # 初始条件
    N = 128
    x = np.linspace(0, 1, N, endpoint=False)
    dx = 1.0 / N
    pri = fun_Pri(x)              # [N, 3]
    con = Pri2Con(pri)            # [N, 3]
    # con = torch.tensor(con, dtype=torch.float64).unsqueeze(0).to(device)  # [1, N, 3]
    u = torch.tensor(con, dtype=torch.float64).unsqueeze(0).to(device)  # [1, N, 3]

    t = 0.0
    t_end = 2e-3
    steps = 0

    data_list = [pri]

    with torch.no_grad():
        for i in range(100):
            while t < t_end:
                dt_value = compute_dt(u, dx, CFL=0.4)      # float
                if t + dt_value > t_end:
                    dt_value = t_end - t
                dt = torch.tensor([dt_value], dtype=torch.float64).to(device)

                dt_broadcast = dt.expand(1, u.shape[1], 1)
                u_input = torch.cat([u, dt_broadcast], dim=-1)  # [B, N, 4]
                u = model(u_input)
                # for _ in range(2):
                #     u = model.N0_SCHEME.single_step(u, dt/2)
                t += dt_value
                steps += 1
                print(f"Step {steps:03d} | t = {t:.4f} | dt = {dt_value:.4e}", end='\r')
            t = 0
            u_np = u.squeeze(0).detach().cpu()  # [N, 3]
            pri = Con2Pri(u_np)        # [N, 3]
            # phy_rho, phy_u, phy_p = pri[:, 0], pri[:, 1], pri[:, 2]
            data_list.append(pri)
        data_array = np.array(data_list)
        print(data_array.shape)
        np.save("sod_4th.npy", data_array)

if __name__ == "__main__":
    main()
    # data()