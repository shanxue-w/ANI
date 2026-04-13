# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from ANI import N0, ANIBASE, ResidualBlockWithT, MLP, FNO2d
# from ANI_NS_2th import ODEPairDataset
# from ANI_NS_2th import NavierStokes, A  # 假设这些类已经按原来定义好了
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# np.random.seed(int(time.time()))

# torch.set_default_dtype(torch.float64)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# train_input = torch.load('../dataset/val_input.pt')
# train_output = torch.load('../dataset/val_output.pt')

# # random_idx = 422
# random_idx = np.random.randint(len(train_input), size=1)[0]
# # print(f"random_idx: {random_idx}")

# u_random_input = train_input[random_idx:random_idx+1].clone().to(device).permute(0, 3, 1, 2)
# u_output       = train_output[random_idx:random_idx+1].clone().to(device).permute(0, 3, 1, 2)

# A_1 = A(device=device, downsample_factor=32)
# n=2
# dt_tensor = torch.tensor(1e-1/n, dtype=torch.float64, device=device)

# for _ in range(n):
#     u_random_input = A_1.single_step(u_random_input, dts=dt_tensor)
#     error = torch.abs(u_random_input -  u_output).squeeze(0).permute(1,2,0)
#     print(f"norm of error is {error.norm()}")

# # plot error[:,:,0] 和 error[:,:,1], 二维，两张图


# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(error[:,:,0].cpu().detach().numpy(), cmap='coolwarm')
# plt.colorbar()
# plt.title('Error in u')

# plt.subplot(1, 2, 2)
# plt.imshow(error[:,:,1].cpu().detach().numpy(), cmap='coolwarm')
# plt.colorbar()
# plt.title('Error in v')

# plt.savefig(f'error_ns.png')

# import torch
# import os
# import random
# from ANI_NS_2th import NavierStokes, A, ODEPairDataset  # 假设你有这些类
# import matplotlib.pyplot as plt
# import numpy as np

# @torch.no_grad()
# def test_one_sample(model_path="models/best_model.pth", data_dir="../dataset", device="cuda:2"):
#     # 加载验证集
#     val_dataset = ODEPairDataset(
#         os.path.join(data_dir, "val_input.pt"),
#         os.path.join(data_dir, "val_output.pt"),
#     )

#     # 随机选一个样本
#     idx = random.randint(0, len(val_dataset) - 1)
#     inputs, targets = val_dataset[idx]
#     inputs = inputs.unsqueeze(0).to(device).permute(0, 3, 1, 2)   # [1, N, C]
#     targets = targets.unsqueeze(0).to(device).permute(0, 3, 1, 2) # [1, N, C]

#     # 构建模型
#     model = NavierStokes(N0_SCHEME=A(device=device), modes1=8, modes2=8, width=32, dt=0.1, device=device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # 预测
#     preds = model(inputs)

#     # 计算误差
#     l2_error = torch.norm(preds - targets).item()
#     relative_error = l2_error / torch.norm(targets).item()

#     print(f"Test sample idx: {idx}")
#     print(f"L2 error      : {l2_error:.4e}")
#     print(f"Relative error: {relative_error:.4%}")

#     # [Nx, Ny, 2]
#     preds_np = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     targets_np = targets.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     abs_err = np.abs(preds_np - targets_np)

#     plt.figure(figsize=(10, 6))
#     for i, name in enumerate(["$u_x$", "$u_y$"]):
#         plt.subplot(2, 2, 2 * i + 1)
#         plt.imshow(preds_np[:, :, i], cmap="jet", origin="lower")
#         plt.colorbar()
#         plt.title(f"Pred {name}")

#         plt.subplot(2, 2, 2 * i + 2)
#         plt.imshow(abs_err[:, :, i], cmap="hot", origin="lower")
#         plt.colorbar()
#         plt.title(f"Error {name}")

#     plt.tight_layout()
#     plt.savefig("test_sample_pred_and_error.png")
#     plt.close()

# test_one_sample()

import torch
import os
from tqdm import tqdm
import argparse
from torch.nn.functional import mse_loss
from model import FNO2d
from fine import A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

dt    = torch.tensor(0.01).to(device)

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

@torch.no_grad()
def evaluate_on_trajectory(model, init_states, true_trajs):
    model.eval()
    T = true_trajs.shape[1]
    batch_size = init_states.shape[0]

    curr_states = init_states.clone()
    B, _, Nx, Ny = curr_states.shape
    nu_layer = torch.full((B, 1, Nx, Ny), 1e-4, device=device, dtype=torch.float64)
    pred_trajs = []

    A1 = A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, Re=1e4).to(device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for t in range(T):
        # print(f"t = {t}/{T-1}", end='\r')
        # eps_batch = epsilons.view(-1, 1)  # [4, 1]
        omega_input = A1._velocity_to_vorticity_spectral(
            curr_states[:, 0], curr_states[:, 1], A1.Kx_fine, A1.Ky_fine
        )
        omega_input = omega_input.unsqueeze(1)
        combined_input = torch.cat([
            omega_input,      
            curr_states,         
            nu_layer      
        ], dim=1)
        next_states = model(combined_input, dt)  # [4, 128, 128]
        u, v = A1._vorticity_to_velocity_spectral(next_states.squeeze(1), A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine)
        next_states = torch.cat([u.unsqueeze(1), v.unsqueeze(1)], dim=1)
        # next_states = curr_states
        # for i in range(2):
        #     next_states = A1.single_step(next_states, dts=dt_tensor)
        # if t == 0:
            # error = (next_states - true_trajs[:, t+1])
            # error = true_trajs[:, 0] 
            # error = true_trajs[:, 99] 
        pred_trajs.append(next_states)
        curr_states = next_states
    # torch.cuda.synchronize()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # with open("time.txt", "w") as f:
    #     f.write(f"Total inference time: {elapsed_time:.6f} s\n")
    pred_trajs = torch.stack(pred_trajs, dim=1)  # [4, T, 128, 128]

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


    avg_l2_per_t = []
    avg_rel_per_t = []

    for t in range(T-1):
        l2s = []
        rels = []
        for i in range(batch_size):
            l2 = mse_loss(pred_trajs[i, t], true_trajs[i, t+1])
            rel = relative_l2_error(pred_trajs[i, t], true_trajs[i, t+1])
            l2s.append(l2.item())
            rels.append(rel.item())
        avg_l2_per_t.append(sum(l2s) / batch_size)
        avg_rel_per_t.append(sum(rels) / batch_size)
    
    # # plot error [B,2,256,512], so plot u, v
    # print(error.norm())
    # u = error[0, 0]
    # v = error[0, 1]
    # x = np.linspace(0, 1, 256, endpoint=False)
    # y = np.linspace(0, 1, 256, endpoint=False)
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

    return avg_l2_per_t, avg_rel_per_t, pred_trajs

@torch.no_grad()
def evaluate_k_step_rollout(model, traj_data, k=5):
    model.eval()
    B, T, C, Nx, Ny = traj_data.shape
    
    A1 = A(Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0, Re=1e4).to(device)
    nu_layer = torch.full((B, 1, Nx, Ny), 1e-4, device=device, dtype=torch.float64)
    
    total_mse = 0.0
    total_rel = 0.0
    count = 0

    for t_start in tqdm(range(T - k), desc=f"Rolling out {k} steps"):
        curr_states = traj_data[:, t_start].clone() # [B, 2, Nx, Ny]
        
        for step in range(1, k + 1):
            omega_input = A1._velocity_to_vorticity_spectral(
                curr_states[:, 0], curr_states[:, 1], A1.Kx_fine, A1.Ky_fine
            ).unsqueeze(1)
            
            combined_input = torch.cat([omega_input, curr_states, nu_layer], dim=1)
            
            next_vorticity = model(combined_input, dt) # [B, 1, Nx, Ny]
            
            u, v = A1._vorticity_to_velocity_spectral(
                next_vorticity.squeeze(1), A1.Kx_fine, A1.Ky_fine, A1.denom_safe_fine
            )
            curr_states = torch.cat([u.unsqueeze(1), v.unsqueeze(1)], dim=1)
            
            target = traj_data[:, t_start + step]
            
            batch_mse = mse_loss(curr_states, target).item()
            batch_rel = relative_l2_error(curr_states, target).item()
            
            total_mse += batch_mse
            total_rel += batch_rel
            count += 1

    avg_mse = total_mse / count
    avg_rel = total_rel / count
    
    return avg_mse, avg_rel

if __name__ == "__main__":
    # 加载模型
    parser = argparse.ArgumentParser(description="Training script for datasets.")
    parser.add_argument(
        "--train_limit", 
        type=int, 
        default=4000, 
        help="Number of samples to limit in the training dataset (default: 4000)"
    )
    args = parser.parse_args()
    model = FNO2d(modes1=32, modes2=32, width=64, in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(f"models/best_model_fno_{args.train_limit}_final.pth", map_location=device))

    # 加载 trajectory 数据
    traj_data = torch.load("../dataset/test_trajectory.pt").to(device).to(torch.float64)   # [N, T, 128, 128]

    # init_states = traj_data[:, 0].clone().permute(0, 3, 1, 2)     # [4, 128, 128]
    # true_trajs  = traj_data[:].clone().permute(0, 1, 4, 2, 3)       # [4, T, 128, 128]
    init_states = traj_data[:, 0].clone()     # [4, 128, 128]
    true_trajs  = traj_data[:].clone()       # [4, T, 128, 128]

    print(init_states.shape, true_trajs.shape)

    # evaluate_on_trajectory(model, init_states, true_trajs)

    # avg_l2_per_t, avg_rel_per_t, pred_trajs = evaluate_on_trajectory(model, init_states, true_trajs)

    # # 写入文件，每行写一个时间点的误差
    # with open(f"traj_error_{args.train_limit}.txt", "w") as f:
    #     f.write("t_index\tavg_L2_error\tavg_relative_error\n")
    #     for t, (l2, rel) in enumerate(zip(avg_l2_per_t, avg_rel_per_t)):
    #         f.write(f"{t}\t{l2:.6e}\t{rel:.6e}\n")

    # print("✅ Trajectory error evaluation done.")

    K_STEPS = 1
    avg_mse, avg_rel = evaluate_k_step_rollout(model, traj_data, k=K_STEPS)

    output_path = f"k_{K_STEPS}_error_{args.train_limit}.txt"
    with open(output_path, "w") as f:
        f.write(f"Rollout_Steps_K\tAvg_MSE\tAvg_Relative_L2\n")
        f.write(f"{K_STEPS}\t{avg_mse:.6e}\t{avg_rel:.6e}\n")

    print(f"✅ {K_STEPS}-step rollout evaluation done. Results saved to {output_path}")

    # # 假设已有 true_trajs [B, T, 2, H, W]
    # u_seq = true_trajs[2, :, 0].cpu().numpy()  # [T, H, W]
    # v_seq = true_trajs[2, :, 1].cpu().numpy()  # [T, H, W]
    # speed_seq = np.sqrt(u_seq**2 + v_seq**2)  # [T, H, W]

    # T, H, W = speed_seq.shape
    # x = np.linspace(0, 1, H, endpoint=False)
    # y = np.linspace(0, 1, W, endpoint=False)
    # X, Y = np.meshgrid(x, y)

    # fig, ax = plt.subplots(figsize=(6, 5))
    # cf = ax.contourf(X, Y, speed_seq[0].T, levels=100, cmap='viridis')
    # cbar = fig.colorbar(cf, ax=ax)
    # cbar.set_label('Speed')

    # def update(frame):
    #     ax.clear()
    #     cf = ax.contourf(X, Y, speed_seq[frame].T, levels=100, cmap='viridis')
    #     ax.set_title(f'Time step: {frame}')
    #     return []

    # ani = animation.FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    # ani.save("true_traj.gif", writer='pillow')
    # plt.close()
