import torch
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from ANI_2th import AllenCahn, A
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

import torch
from torch.nn.functional import mse_loss

def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

dt_tensor = torch.tensor(0.01, dtype=torch.float64).to(device)

@torch.no_grad()
def evaluate_on_trajectory(model, init_states, epsilons, true_trajs, dt=0.01):
    """
    输入：
        init_states: [4, 128, 128, 2]
        epsilons:    [4]
        true_trajs:  [4, T, 128, 128, 2]
    输出：
        avg_l2_per_t: [T], 每个时间点所有4条轨迹平均L2误差
        avg_rel_per_t: [T], 每个时间点所有4条轨迹平均相对误差
    """
    model.eval()
    T = true_trajs.shape[1]
    batch_size = init_states.shape[0]

    curr_states = init_states.clone()
    pred_trajs = []
    u0 = init_states[0, :, :, 0]

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for t in range(T):
        # print(f"t = {t}/{T-1}", end='\r')
        # eps_batch = epsilons.view(-1, 1)  # [4, 1]
        # next_states = model.predict(curr_states, epsilons)  # [4, 128, 128]
        next_states = curr_states
        next_states = model.N0_SCHEME.single_step(next_states, dts=dt_tensor)
        # pred_trajs.append(next_states)
        curr_states = next_states

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("time_base.txt", "w") as f:
        f.write(f"Total inference time: {elapsed_time:.6f} s\n")

    # pred_trajs = torch.stack(pred_trajs, dim=1)  # [4, T, 128, 128]

    # u0 = pred_trajs[1, 49, :, :, 0]
    # v0 = pred_trajs[1, 49, :, :, 1]

    # u_99 = pred_trajs[1, 99, :, :, 0]
    # v_99 = pred_trajs[1, 99, :, :, 1]

    # np.save('u0_2th.npy', u0.cpu().numpy())
    # np.save('v0_2th.npy', v0.cpu().numpy())
    # np.save('u99_2th.npy', u_99.cpu().numpy())
    # np.save('v99_2th.npy', v_99.cpu().numpy())

    # np.save('u0_true.npy', true_trajs[1,50,:,:,0].cpu().numpy())
    # np.save('v0_true.npy', true_trajs[1,50,:,:,1].cpu().numpy())
    # np.save('u99_true.npy', true_trajs[1,100,:,:,0].cpu().numpy())
    # np.save('v99_true.npy', true_trajs[1,100,:,:,1].cpu().numpy())

    # avg_l2_per_t = []
    # std_l2_per_t = []
    # avg_rel_per_t = []
    # std_rel_per_t = []

    # for t in range(T - 1):
    #     l2s = []
    #     rels = []
    #     for i in range(batch_size):
    #         l2 = mse_loss(pred_trajs[i, t], true_trajs[i, t+1])
    #         rel = relative_l2_error(pred_trajs[i, t], true_trajs[i, t+1])
    #         l2s.append(l2.item())
    #         rels.append(rel.item())

    #     l2_tensor = torch.tensor(l2s)
    #     rel_tensor = torch.tensor(rels)

    #     avg_l2_per_t.append(l2_tensor.mean().item())
    #     std_l2_per_t.append(l2_tensor.std(unbiased=False).item())

    #     avg_rel_per_t.append(rel_tensor.mean().item())
    #     std_rel_per_t.append(rel_tensor.std(unbiased=False).item())

    # return avg_l2_per_t, std_l2_per_t, avg_rel_per_t, std_rel_per_t

if __name__ == "__main__":
    # 加载模型
    model = AllenCahn(N0_SCHEME=A(), modes1=8, modes2=8, width=20, dt=0.01).to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))

    # 加载 trajectory 数据
    # traj_data = torch.load("../dataset/allen_cahn_traj.pt").to(device)   # [N, T, 128, 128]
    # epsilons  = torch.load("../dataset/allen_cahn_traj_eps.pt").to(device)  # [N]
    traj_data = np.load('../dataset/data/fhn_test_trajectory_all.npy')
    epsilons  = np.load('../dataset/data/fhn_test_eps_all.npy')
    # [4] -> [4, 3] (eps, a, b) a = 0.7, b = 0.8
    # epsilons = np.column_stack([epsilons, np.full_like(epsilons, 0.7), np.full_like(epsilons, 0.8)])
    traj_data = torch.from_numpy(traj_data).to(device)
    epsilons  = torch.from_numpy(epsilons).to(device)

    init_states = traj_data[:, 0]     # [4, 128, 128]
    true_trajs  = traj_data[:]        # [4, T, 128, 128]
    epsilons    = epsilons[:]         # [4]

    evaluate_on_trajectory(model, init_states, epsilons, true_trajs)
    # avg_l2_per_t, std_l2_per_t, avg_rel_per_t, std_rel_per_t = evaluate_on_trajectory(model, init_states, epsilons, true_trajs)

    # # 写入文件，每行写一个时间点的误差
    # with open("traj_error_A_5.5.txt", "w") as f:
    #     f.write("t_index\tavg_L2_error\tstd_L2_error\tavg_relative_error\tstd_relative_error\n")
    #     for t, (l2_mean, l2_std, rel_mean, rel_std) in enumerate(
    #         zip(avg_l2_per_t, std_l2_per_t, avg_rel_per_t, std_rel_per_t)
    #     ):
    #         f.write(f"{t}\t{l2_mean:.6e}\t{l2_std:.6e}\t{rel_mean:.6e}\t{rel_std:.6e}\n")


    # print("✅ Trajectory error evaluation done.")

    # u = true_trajs[0, 10, :, :, 0].cpu().numpy()
    # v = true_trajs[0, 10, :, :, 1].cpu().numpy()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(u, cmap='jet')
    # plt.title('u')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(v, cmap='jet')
    # plt.title('v')
    # plt.colorbar()
    # plt.savefig('fhn_0.1.png')
