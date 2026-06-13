import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from ANI4 import NavierStokes, A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# dt_tensor = torch.tensor(0.1, dtype=torch.float64, device=device)/2
dt_tensor = 5e-3 / 2

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

@torch.no_grad()
def evaluate_on_trajectory(model, init_states, true_trajs, dt=5e-3):
    model.eval()
    T = true_trajs.shape[1]
    batch_size = init_states.shape[0]

    next_states = init_states.clone()
    print(next_states.shape)
    pred_trajs = []
    n=128

    A1 = A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device, Re=1e4, Cs=0.18)
    warmup_states = init_states.clone()
    _ = model.predict(warmup_states)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for t in range(T):
            print(f"t = {t}/{T-1}", end='\r')
            # eps_batch = epsilons.view(-1, 1)  # [4, 1]
            next_states = model.predict(next_states)  # [4, 128, 128]
            # u, v = curr_states[:, 0], curr_states[:, 1]
            # next_states = A1._velocity_to_vorticity_spectral(u, v, A1.Kx_fine, A1.Ky_fine).unsqueeze(1)
            # for i in range(2):
            #     next_states = A1.single_step(next_states, dts=dt_tensor)
            # next_states = next_states.squeeze(1)
            # u, v = A1._vorticity_to_velocity_spectral(next_states, A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine)
            # next_states = torch.stack([u, v], dim=1)
            # if t == 0:
    # torch.cuda.synchronize()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # with open("time.txt", "w") as f:
    #     f.write(f"Total inference time: {elapsed_time:.6f} s\n")

            # error = (next_states - true_trajs[:, t+1])
    #         # error = true_trajs[:, 0] 
    #         # error = true_trajs[:, 99] 
            pred_trajs.append(next_states)
        # curr_states = next_states

    pred_trajs = torch.stack(pred_trajs, dim=1)  # [4, T, 128, 128]
    # print(pred_trajs.shape)

    # rng = np.random.default_rng(seed=123)
    # n = rng.integers(0, 20)

    n = 1
    time_steps = [99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699, 749]
    file_suffixes = [99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699, 749]
    for t, suffix in zip(time_steps, file_suffixes):
        u_pred, v_pred = A1._vorticity_to_velocity_spectral(
            pred_trajs[n, t], A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine
        )
    
        np.save(f'u_{suffix}_4th.npy', u_pred.cpu().numpy())
        np.save(f'v_{suffix}_4th.npy', v_pred.cpu().numpy())


    time_steps = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
    file_suffixes = [99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699, 749]
    for t, suffix in zip(time_steps, file_suffixes):
        u_true, v_true = A1._vorticity_to_velocity_spectral(
            true_trajs[n, t], A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine
        )
    
        np.save(f'u_{suffix}_true.npy', u_true.cpu().numpy())
        np.save(f'v_{suffix}_true.npy', v_true.cpu().numpy())


    # u_99 = pred_trajs[0, 99, 0, :, :]
    # v_99 = pred_trajs[0, 99, 1, :, :]

    # u_149 = pred_trajs[0, 149, 0, :, :]
    # v_149 = pred_trajs[0, 149, 1, :, :]

    # u_199 = pred_trajs[0, 199, 0, :, :]
    # v_199 = pred_trajs[0, 199, 1, :, :]

    # u_249 = pred_trajs[0, 249, 0, :, :]
    # v_249 = pred_trajs[0, 249, 1, :, :]

    # u_299 = pred_trajs[0, 299, 0, :, :]
    # v_299 = pred_trajs[0, 299, 1, :, :]

    # u_349 = pred_trajs[0, 349, 0, :, :]
    # v_349 = pred_trajs[0, 349, 1, :, :]

    # np.save('u_99_4th.npy', u_99.cpu().numpy())
    # np.save('v_99_4th.npy', v_99.cpu().numpy())
    # np.save('u_149_4th.npy', u_149.cpu().numpy())
    # np.save('v_149_4th.npy', v_149.cpu().numpy())
    # np.save('u_199_4th.npy', u_199.cpu().numpy())
    # np.save('v_199_4th.npy', v_199.cpu().numpy())
    # np.save('u_249_4th.npy', u_249.cpu().numpy())
    # np.save('v_249_4th.npy', v_249.cpu().numpy())
    # np.save('u_299_4th.npy', u_299.cpu().numpy())
    # np.save('v_299_4th.npy', v_299.cpu().numpy())
    # np.save('u_349_4th.npy', u_349.cpu().numpy())
    # np.save('v_349_4th.npy', v_349.cpu().numpy())

    # # np.save('u_99_true.npy', true_trajs[0, 99, 0, :, :].cpu().numpy())
    # # np.save('v_99_true.npy', true_trajs[0, 99, 1, :, :].cpu().numpy())
    # # np.save('u_149_true.npy', true_trajs[0, 149, 0, :, :].cpu().numpy())
    # # np.save('v_149_true.npy', true_trajs[0, 149, 1, :, :].cpu().numpy())
    # # np.save('u_199_true.npy', true_trajs[0, 199, 0, :, :].cpu().numpy())
    # # np.save('v_199_true.npy', true_trajs[0, 199, 1, :, :].cpu().numpy())
    # # np.save('u_249_true.npy', true_trajs[0, 249, 0, :, :].cpu().numpy())
    # # np.save('v_249_true.npy', true_trajs[0, 249, 1, :, :].cpu().numpy())
    # # np.save('u_299_true.npy', true_trajs[0, 299, 0, :, :].cpu().numpy())
    # # np.save('v_299_true.npy', true_trajs[0, 299, 1, :, :].cpu().numpy())
    # # np.save('u_349_true.npy', true_trajs[0, 349, 0, :, :].cpu().numpy())
    # # np.save('v_349_true.npy', true_trajs[0, 349, 1, :, :].cpu().numpy())

    # # np.save('u0_2th.npy', u0.cpu().numpy())
    # # np.save('v0_2th.npy', v0.cpu().numpy())
    # # np.save('u99_2th.npy', u_99.cpu().numpy())
    # # np.save('v99_2th.npy', v_99.cpu().numpy())

    # # np.save('u0_true.npy', true_trajs[1,10,0,:,:].cpu().numpy())
    # # np.save('v0_true.npy', true_trajs[1,10,1,:,:].cpu().numpy())
    # # np.save('u99_true.npy', true_trajs[1,200,0,:,:].cpu().numpy())
    # # np.save('v99_true.npy', true_trajs[1,200,1,:,:].cpu().numpy())


    # avg_l2_per_t = []
    # avg_rel_per_t = []

    # for t in range(T-1):
    #     l2s = []
    #     rels = []
    #     for i in range(batch_size):
    #         omega_pred = pred_trajs[i, t]
    #         omega_true = true_trajs[i, t+1]
    #         l2 = mse_loss(omega_pred, omega_true)
    #         rel = relative_l2_error(omega_pred, omega_true)
    # #         # l2 = mse_loss(pred_trajs[i, t], true_trajs[i, t+1])
    # #         # rel = relative_l2_error(pred_trajs[i, t], true_trajs[i, t+1])
    #         l2s.append(l2.item())
    #         rels.append(rel.item())
    #     avg_l2_per_t.append(sum(l2s) / batch_size)
    #     avg_rel_per_t.append(sum(rels) / batch_size)
    
    # # plot error [B,2,128,512], so plot u, v
    # print(error.norm())
    # u = error[0, 0]
    # v = error[0, 1]
    # x = np.linspace(0, 1, n, endpoint=False)
    # y = np.linspace(0, 1, n, endpoint=False)
    # X, Y = np.meshgrid(x, y)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # # plt.imshow(u.cpu().numpy(), cmap="jet", origin="lower")
    # plt.contourf(X, Y, u.cpu().numpy().T, cmap="jet", levels=100)
    # plt.colorbar()
    # plt.title('Error in u')

    # plt.subplot(1, 2, 2)
    # # plt.imshow(v.cpu().numpy(), cmap="jet", origin="lower")
    # plt.contourf(X, Y, v.cpu().numpy().T, cmap="jet", levels=100)
    # plt.colorbar()
    # plt.title('Error in v')

    # plt.tight_layout()
    # plt.savefig("error.png")
    # plt.close()

    # return avg_l2_per_t, avg_rel_per_t, pred_trajs

if __name__ == "__main__":
    # 加载模型
    model = NavierStokes(N0_SCHEME=A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device, Re=1e4), modes1=32, modes2=32, width=32, dt=5e-3, device=device).to(device)
    model.load_state_dict(torch.load("models/best_model_fno_32.pth", map_location=device))

    # 加载 trajectory 数据
    traj_data = torch.load("../dataset/test_trajectory_new.pt").to(device).to(torch.float64).unsqueeze(2)   # [N, T, 128, 128]

    # init_states = traj_data[:, 0].clone().permute(0, 3, 1, 2)     # [4, 128, 128]
    # true_trajs  = traj_data[:].clone().permute(0, 1, 4, 2, 3)       # [4, T, 128, 128]
    init_states = traj_data[:, 0].clone()     # [4, 128, 128]
    true_trajs  = traj_data[:].clone()       # [4, T, 128, 128]

    print(init_states.shape, true_trajs.shape)

    evaluate_on_trajectory(model, init_states, true_trajs)
    # avg_l2_per_t, avg_rel_per_t, pred_trajs = evaluate_on_trajectory(model, init_states, true_trajs)

    # # 写入文件，每行写一个时间点的误差
    # with open("traj_error.txt", "w") as f:
    #     f.write("t_index\tavg_L2_error\tavg_relative_error\n")
    #     for t, (l2, rel) in enumerate(zip(avg_l2_per_t, avg_rel_per_t)):
    #         f.write(f"{t}\t{l2:.6e}\t{rel:.6e}\n")

    # print("✅ Trajectory error evaluation done.")

    # B, T, _, H, W = true_trajs.shape

    # # 假设已有 true_trajs [B, T, 2, H, W]
    # # u_seq = true_trajs[0, :, 0] # [T, H, W]
    # # v_seq = true_trajs[0, :, 1]  # [T, H, W]
    # u_seq = pred_trajs[0, :, 0] # [T, H, W]
    # v_seq = pred_trajs[0, :, 1]  # [T, H, W]
    # # speed_seq = torch.sqrt(u_seq**2 + v_seq**2)  # [T, H, W]
    # # print(speed_seq[-1].max())
    # model = A(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device=device, Re=1e4)
    # speed_seq = model._velocity_to_vorticity_spectral(u_seq, v_seq, model.Kx_fine, model.Ky_fine).cpu().numpy()

    # T, H, W = speed_seq.shape
    # x = np.linspace(0, 1, H, endpoint=False)
    # y = np.linspace(0, 1, W, endpoint=False)
    # X, Y = np.meshgrid(x, y)

    # # fig, ax = plt.subplots(figsize=(6, 5))
    # # cf = ax.contourf(X, Y, speed_seq[0].T, levels=100, cmap='viridis')
    # # cbar = fig.colorbar(cf, ax=ax)
    # # cbar.set_label('Vorticity')


    # # def update(frame):
    # #     ax.clear()
    # #     cf = ax.contourf(X, Y, speed_seq[frame].T, levels=100, cmap='viridis')
    # #     ax.set_title(f'Time step: {frame}')
    # #     return []

    # # ani = animation.FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    # # ani.save("test_traj_ns.gif", writer='pillow')
    # # plt.close()

    # # u_last = true_trajs[0, -1, 0].cpu().numpy()  # [H, W]
    # # v_last = true_trajs[0, 1, 1].cpu().numpy()  # [H, W]
    # omega = speed_seq[249]
    # # fig, ax = plt.subplots(figsize=(8, 8))
    # # ax.contourf(X, Y, omega.T, cmap='viridis', levels=100)
    # plt.figure(figsize=(8, 8))
    # plt.contourf(X, Y, omega.T, cmap='viridis', levels=100)
    # plt.colorbar()
    # plt.savefig('vor_pred.png')

    # # # 创建坐标网格
    # # x = np.linspace(0, 1, H, endpoint=False)
    # # y = np.linspace(0, 1, W, endpoint=False)
    # # X, Y = np.meshgrid(x, y)

    # # # 创建绘图
    # # fig, ax = plt.subplots(figsize=(8, 8))

    # # # 绘制流线
    # # # 我们使用 (X, Y) 网格和 u, v 速度分量来绘制流线
    # # ax.streamplot(X, Y, u_last.T, v_last.T, density=1.5)

    # # # 添加颜色条来表示速度大小
    # # # 为了颜色条，我们必须先创建一个可映射对象（mappable）
    # # # 这里我们用一个临时的 pcolormesh 来创建它
    # # speed_last = np.sqrt(u_last.T**2 + v_last.T**2)
    # # mappable = ax.pcolormesh(X, Y, speed_last, cmap='viridis', shading='auto')
    # # cbar = fig.colorbar(mappable, ax=ax)
    # # cbar.set_label('Speed')

    # # ax.set_title(f'Streamlines at time step: {T-1}')
    # # ax.set_xlabel('X-coordinate')
    # # ax.set_ylabel('Y-coordinate')
    # # # plt.show()

    # # # 如果想保存图片，可以取消下面这行的注释
    # # fig.savefig("true_streamlines.png")
