# 读取 2th 3th 4th resnet下的文件并画图，写个画图函数
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

def plot_error(ANI_2th, ANI_3th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(ANI_2th[:10000], label='ANI_2th')
    plt.plot(ANI_3th[:10000], label='ANI_3th')
    plt.plot(ANI_4th[:10000], label='ANI_4th')
    plt.plot(resnet[:10000], label='Forward Euler')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('error')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

abs_ANI_2th = np.loadtxt('2th/rel_test_errors_small.txt')
abs_ANI_3th = np.loadtxt('3th/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('Euler/rel_test_errors_small.txt')

plot_error(abs_ANI_2th, abs_ANI_3th, abs_ANI_4th, abs_resnet, 'rel_error_1000.png')

data_2th_abs = np.loadtxt("2th/abs_test_errors_small.txt")
data_3th_abs = np.loadtxt("3th/abs_test_errors_small.txt")
data_4th_abs = np.loadtxt("4th/abs_test_errors_small.txt")
abs_resnet   = np.loadtxt("Euler/abs_test_errors_small.txt")

plt.figure(figsize=(8, 6))
plt.plot(abs_resnet[1:200, 0:1], label='Forward Euler', linewidth=2, color="#1f77b4")
plt.plot(data_2th_abs[1:200, 0:1], label="ANI-2", linewidth=2, color="#2ca02c")
# plt.plot(data_3th_abs[1:200, 0:1], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:200, 0:1], label="ANI-4", linewidth=2, color="#733497")

plt.yscale("log")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$V$ Error", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("abs_V_errors_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(abs_resnet[1:200, 1:2], label='Forward Euler', linewidth=2, color="#1f77b4")
plt.plot(data_2th_abs[1:200, 1:2], label="ANI-2", linewidth=2, color="#2ca02c")
# plt.plot(data_3th_abs[1:200, 1:2], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:200, 1:2], label="ANI-4", linewidth=2, color="#733497")

plt.yscale("log")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$w$ Error", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("abs_w_errors_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

u_traj = np.load("2th/u_traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_euler = np.load("Euler/u_euler.npy")
time   = 0.05 * np.arange(0, 200)
# subfigures, plot u and v
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(time, u_traj[:200, 0], label="True")
ax[0].plot(time, u_2th[:200, 0], 'ro', label="ANI-2", markersize=4)
ax[0].plot(time, u_4th[:200, 0], 'gs', label="ANI-4", markersize=4)
ax[0].plot(time, u_euler[:200, 0], 'b^', label="Baseline", markersize=4)
ax[0].legend()
ax[0].set_xlabel("Time")
ax[0].set_ylabel(r"$V$")

ax[1].plot(time, u_traj[:200, 1], label="True")
ax[1].plot(time, u_2th[:200, 1], label="ANI-2")
ax[1].plot(time, u_4th[:200, 1], label="ANI-4")
ax[1].plot(time, u_euler[:200, 1], label="Baseline")
ax[1].legend()
ax[1].set_xlabel("Time")
ax[1].set_ylabel(r"$w$")
# plt.figure()
# plt.plot(u_traj[:200, 0], u_traj[:200, 1], label="True")
# plt.plot(u_2th[:200, 0], u_2th[:200, 1], 'ro', label="ANI-2", markersize=4)
# plt.plot(u_4th[:200, 0], u_4th[:200, 1], 'gs', label="ANI-4", markersize=4)
# plt.plot(u_euler[:200, 0], u_euler[:200, 1], 'b^', label="Baseline", markersize=4)
# plt.tight_layout()
plt.savefig("u_traj.png")
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:, 0], color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("u_traj_1.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:, 1], color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("u_traj_2.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()
