# 读取 2th 3th 4th resnet下的文件并画图，写个画图函数
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 13.5,   # 坐标轴标签字体大小
    "xtick.labelsize": 12,  # x 轴刻度字体大小
    "ytick.labelsize": 12,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

def plot_error(ANI_2th, ANI_3th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[1:200], label='KAN-RK4', color="#1f77b4", linewidth=2)
    plt.plot(ANI_2th[1:200], label='ANI_2th', color="#2ca02c", linewidth=2)
    # plt.plot(ANI_3th[1:200], label='ANI_3th')
    plt.plot(ANI_4th[1:200], label='ANI_4th', color="#733497", linewidth=2)
    plt.legend()
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.yscale('log')
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

abs_ANI_2th = np.loadtxt('2th/rel_test_errors_small.txt')
abs_ANI_3th = np.loadtxt('3th/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('RK4/rel_test_errors_small.txt')

plot_error(abs_ANI_2th, abs_ANI_3th, abs_ANI_4th, abs_resnet, 'rel_error_200.pdf')

u_traj = np.load("2th/u_traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_base = np.load("RK4/u_base.npy")


# 画四个子图，relative error， (u_traj, u_method) 的三维对比图
plt.figure(figsize=(8, 6))
plt.plot(abs_resnet[1:400], label='KAN-RK4', color="#1f77b4", linewidth=2)
plt.plot(abs_ANI_2th[1:400], label='ANI-2', color="#2ca02c", linewidth=2)
plt.plot(abs_ANI_4th[1:400], label='ANI-4', color="#733497", linewidth=2)
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.xlabel('Step', fontsize=14)
plt.ylabel('Relative error', fontsize=14)
plt.savefig("rel_error_200.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()
# fig, ax = plt.subplots(2, 2, figsize=(12, 10))
# ax[0, 0].plot(abs_resnet[1:400], label='KAN-RK4')
# ax[0, 0].plot(abs_ANI_2th[1:400], label='ANI-2')
# ax[0, 0].plot(abs_ANI_4th[1:400], label='ANI-4')
# ax[0, 0].set_yscale('log')
# ax[0, 0].legend()
# ax[0, 0].set_xlabel('step')
# ax[0, 0].set_ylabel('Relative error')

# # 剩下三个画3d (u[:200, 0], u[:200, 1], u[:200, 2])
# # 统一坐标范围
x_min, x_max = u_traj[:,0].min()-0.02, u_traj[:,0].max()+0.02
y_min, y_max = u_traj[:,1].min()-0.02, u_traj[:,1].max()+0.02
z_min, z_max = u_traj[:,2].min()-0.02, u_traj[:,2].max()+0.02


# # 2. baseline
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
ax.plot3D(u_base[:400,0], u_base[:400,1], u_base[:400,2], '--', color="tab:red", alpha=0.8, label='KAN-RK4', linewidth=2)
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
ax.set_xlabel('u1'); ax.set_ylabel('u2'); ax.set_zlabel('u3')
ax.legend()
plt.tight_layout()
plt.savefig("baseline.pdf", format='pdf', dpi=300)
plt.close()
# ax2 = fig.add_subplot(2, 2, 2, projection='3d')
# ax2.plot(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
# ax2.plot(u_base[:400,0], u_base[:400,1], u_base[:400,2], color='tab:red', alpha=0.8, label='KAN-RK4')
# ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
# ax2.set_xlabel('u1'); ax2.set_ylabel('u2'); ax2.set_zlabel('u3')
# ax2.legend()

# # 3. ANI-2
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
ax.plot3D(u_2th[:400,0], u_2th[:400,1], u_2th[:400,2], '--', color="tab:red", alpha=0.8, label='ANI-2', linewidth=2)
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
ax.set_xlabel('u1'); ax.set_ylabel('u2'); ax.set_zlabel('u3')
ax.legend()
plt.tight_layout()
plt.savefig("ANI_2th.pdf", format='pdf', dpi=300)
plt.close()
# ax3 = fig.add_subplot(2, 2, 3, projection='3d')
# ax3.plot(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
# ax3.plot(u_2th[:400,0], u_2th[:400,1], u_2th[:400,2], color='tab:red', alpha=0.8, label='ANI-2')
# ax3.set_xlim(x_min, x_max); ax3.set_ylim(y_min, y_max); ax3.set_zlim(z_min, z_max)
# ax3.set_xlabel('u1'); ax3.set_ylabel('u2'); ax3.set_zlabel('u3')
# ax3.legend()

# # 4. ANI-4
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
ax.plot3D(u_4th[:400,0], u_4th[:400,1], u_4th[:400,2], '--', color="tab:red", alpha=0.8, label='ANI-4', linewidth=2)
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
ax.set_xlabel('u1'); ax.set_ylabel('u2'); ax.set_zlabel('u3')
ax.legend()
plt.tight_layout()
plt.savefig("ANI_4th.pdf", format='pdf', dpi=300)
plt.close()
# ax4 = fig.add_subplot(2, 2, 4, projection='3d')
# ax4.plot(u_traj[:400,0], u_traj[:400,1], u_traj[:400,2], 'k--', linewidth=2, label='True')
# ax4.plot(u_4th[:400,0], u_4th[:400,1], u_4th[:400,2], color='tab:red', alpha=0.8, label='ANI-4')
# ax4.set_xlim(x_min, x_max); ax4.set_ylim(y_min, y_max); ax4.set_zlim(z_min, z_max)
# ax4.set_xlabel('u1'); ax4.set_ylabel('u2'); ax4.set_zlabel('u3')
# ax4.legend()
# plt.tight_layout()
# # plt.show()
# plt.savefig("u_traj.png", dpi=300)
# plt.close()

import scipy.io as sio
data = sio.loadmat("./dataset/kan_test_trajectories_1.mat")
u_traj = data["test_trajectories"][0]
print(u_traj.shape)

fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
plt.plot(u_traj[:,0])
# no label and axis
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_1.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:,1])
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_2.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:,2])
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_3.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.plot(u_traj[:2000, 0], u_traj[:2000, 1], u_traj[:2000, 2], color="#1f77b4")
ax.view_init(elev=18, azim=16)
ax.set_axis_off()
ax.grid(False)
plt.tight_layout(pad=0)
# plt.show()

plt.savefig("data_3d.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()



# data_2th_abs = np.loadtxt("2th/abs_test_errors_small.txt")
# data_3th_abs = np.loadtxt("3th/abs_test_errors_small.txt")
# data_4th_abs = np.loadtxt("4th/abs_test_errors_small.txt")
# abs_resnet   = np.loadtxt("Euler/abs_test_errors_small.txt")

# plt.figure()
# plt.plot(data_2th_abs[1:200, 0:1], label="2th Split", linewidth=2)
# plt.plot(data_3th_abs[1:200, 0:1], label="3th Split", linewidth=2)
# plt.plot(data_4th_abs[1:200, 0:1], label="4th Split", linewidth=2)
# plt.plot(abs_resnet[1:200, 0:1], label="Baseline", linewidth=2)

# plt.yscale("log")
# plt.title(r"Absolute $V$ Errors")
# plt.xlabel("Time Step")
# plt.ylabel(r"$V$ Error")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("abs_V_errors_baseline.png")
# plt.close()

# plt.figure()
# plt.plot(data_2th_abs[1:200, 1:2], label="2th Split", linewidth=2)
# plt.plot(data_3th_abs[1:200, 1:2], label="3th Split", linewidth=2)
# plt.plot(data_4th_abs[1:200, 1:2], label="4th Split", linewidth=2)
# plt.plot(abs_resnet[1:200, 1:2], label="Baseline", linewidth=2)

# plt.yscale("log")
# plt.title(r"Absolute $w$ Errors")
# plt.xlabel("Time Step")
# plt.ylabel(r"$w$ Error")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("abs_w_errors_baseline.png")
# plt.close()