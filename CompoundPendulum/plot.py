# 读取 2th 3th 4th resnet下的文件并画图，写个画图函数
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

def plot_error(ANI_2th, ANI_3th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[:200], label='Baseline', color="#0072B2", linewidth=2)
    plt.plot(ANI_2th[:200], label='ANI-2', color="#009E73", linewidth=2)
    plt.plot(ANI_4th[:200], label='ANI-4', color="#E69F00", linewidth=2)
    plt.legend()
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

abs_ANI_2th = np.loadtxt('2th/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('baseline/rel_test_errors_small.txt')

plot_error(abs_ANI_2th, None, abs_ANI_4th, abs_resnet, 'rel_error_200.pdf')

def plot_error(ANI_2th, ANI_3th, ANI_4th, resnet, filename):
    plt.figure()
    plt.plot(ANI_2th[:1000], label='ANI_2th', linewidth=2)
    # plt.plot(ANI_3th[:1000], label='ANI_3th')
    # plt.plot(ANI_4th[:1000], label='ANI_4th')
    # plt.plot(resnet[:1000], label='Baseline')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('error')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# abs_ANI_2th = np.loadtxt('2th_real/rel_test_errors_small.txt')
# abs_ANI_3th = np.loadtxt('3th_real/rel_test_errors_small.txt')
# abs_ANI_4th = np.loadtxt('4th_real/rel_test_errors_small.txt')
# abs_resnet    = np.loadtxt('baseline_real/rel_test_errors_small.txt')

# plot_error(abs_ANI_2th, None, None, None, 'rel_error_real.png')

u_true = np.load("2th/traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_base = np.load("baseline/u_base.npy")

# figures (2, 2) plot, relative error(1000 steps), (u[:,0], u[:, 1]) (true compare with 2th, 4th, baseline)
# fig, ax = plt.subplots(2, 2, figsize=(12, 10))
# ax[0, 0].plot(abs_resnet[:1000], label='Baseline')
# ax[0, 0].plot(abs_ANI_2th[:1000], label='ANI-2')
# ax[0, 0].plot(abs_ANI_4th[:1000], label='ANI-4')
# ax[0, 0].legend()
# ax[0, 0].set_xlabel('step')
# ax[0, 0].set_ylabel('Relative error')

# ax[0, 1].plot(u_true[600:1000, 0], u_true[600:1000, 1], label='True')
# ax[0, 1].plot(u_base[600:1000, 0], u_base[600:1000, 1], 'ro', label='Baseline', markersize=4)
# ax[0, 1].legend()
# ax[0, 1].set_xlabel(r'$\theta_1$')
# ax[0, 1].set_ylabel(r'$\omega_1$')

# ax[1, 0].plot(u_true[600:1000, 0], u_true[600:1000, 1], label='True')
# ax[1, 0].plot(u_2th[600:1000, 0], u_2th[600:1000, 1], 'ro', label='ANI-2', markersize=4)
# ax[1, 0].legend()
# ax[1, 0].set_xlabel(r'$\theta_1$')
# ax[1, 0].set_ylabel(r'$\omega_1$')

# ax[1, 1].plot(u_true[600:1000, 0], u_true[600:1000, 1], label='True')
# ax[1, 1].plot(u_4th[600:1000, 0], u_4th[600:1000, 1], 'ro', label='ANI-4', markersize=4)
# ax[1, 1].legend()
# ax[1, 1].set_xlabel(r'$\theta_1$')
# ax[1, 1].set_ylabel(r'$\omega_1$')

# plt.savefig('2th_4th_baseline.png')

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#0072B2", label='True', linewidth=2)
plt.plot(u_2th[600:1000, 0], u_2th[600:1000, 1], color="#009E73", marker='o', label='ANI-2', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('2th.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#0072B2", label='True', linewidth=2)
plt.plot(u_4th[600:1000, 0], u_4th[600:1000, 1], color='#E69F00', marker='o', label='ANI-4', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('4th.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#0072B2", label='True', linewidth=2)
plt.plot(u_base[600:1000, 0], u_base[600:1000, 1], color='#D55E00', marker='o', label='Baseline', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('baseline.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

cm = 1 / 2.54
plt.figure(figsize=(2*cm, 2*cm))
# plt.plot(u_true[300:1000, 2], u_true[300:1000, 3], color="#0072B2", label='True', linewidth=2)
plt.plot(u_true[0:1000, 0], u_true[0:1000, 1], color="#0072B2", label='True', linewidth=0.2)
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig('data_traj.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
plt.close()

# def get_coords(u, L1=1.0, L2=1.0):
#     """给定 [T, 4] 的状态 u，返回双摆两个质点的坐标"""
#     theta1 = u[:, 0]
#     theta2 = u[:, 2]
#     x1 = L1 * np.sin(theta1)
#     y1 = -L1 * np.cos(theta1)
#     x2 = x1 + L2 * np.sin(theta2)
#     y2 = y1 - L2 * np.cos(theta2)
#     return x1, y1, x2, y2

# # 计算四组坐标
# x1_a, y1_a, x2_a, y2_a = get_coords(u_true)
# x1_b, y1_b, x2_b, y2_b = get_coords(u_base)
# x1_c, y1_c, x2_c, y2_c = get_coords(u_2th)
# x1_d, y1_d, x2_d, y2_d = get_coords(u_4th)

# err_b = np.sqrt((x2_b - x2_a)**2 + (y2_b - y2_a)**2)
# err_c = np.sqrt((x2_c - x2_a)**2 + (y2_c - y2_a)**2)
# err_d = np.sqrt((x2_d - x2_a)**2 + (y2_d - y2_a)**2)

# plt.figure()
# plt.plot(err_b, label='Base error')
# plt.plot(err_c, label='ANI-2 error')
# plt.plot(err_d, label='ANI-4 error')
# plt.xlabel('Time step')
# plt.ylabel('End position error')
# plt.legend()
# plt.show()


# # 创建 2x2 子图
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# titles = ["Reference (a)", "baseline (b)", "ANI-2 (c)", "ANI-4 (d)"]
# data = [(x1_a, y1_a, x2_a, y2_a),
#         (x1_b, y1_b, x2_b, y2_b),
#         (x1_c, y1_c, x2_c, y2_c),
#         (x1_d, y1_d, x2_d, y2_d)]

# lines, traces = [], []
# trace_x_list, trace_y_list = [], []

# for ax, title in zip(axes.flatten(), titles):
#     ax.set_aspect('equal')
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_title(title)
#     line, = ax.plot([], [], 'o-', lw=2, markersize=8)
#     trace, = ax.plot([], [], 'r-', lw=1, alpha=0.5)
#     lines.append(line)
#     traces.append(trace)
#     trace_x_list.append([])
#     trace_y_list.append([])

# def init():
#     for line, trace in zip(lines, traces):
#         line.set_data([], [])
#         trace.set_data([], [])
#     return lines + traces

# def update(frame):
#     for i, (line, trace, trace_x, trace_y, (x1, y1, x2, y2)) in enumerate(
#         zip(lines, traces, trace_x_list, trace_y_list, data)):
#         line.set_data([0, x1[frame], x2[frame]],
#                       [0, y1[frame], y2[frame]])
#         trace_x.append(x2[frame])
#         trace_y.append(y2[frame])
#         trace.set_data(trace_x, trace_y)
#     return lines + traces

# ani = animation.FuncAnimation(
#     fig, update, frames=min(len(u_true), len(u_base), len(u_2th), len(u_4th)),
#     init_func=init, blit=True, interval=20
# )

# plt.tight_layout()
# # plt.show()
# ani.save('double_pendulum.mp4', writer='ffmpeg', fps=20, dpi=300)