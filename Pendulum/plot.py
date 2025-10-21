import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

data_2th_rel = np.loadtxt("2th_test/rel_test_errors_small.txt")
data_4th_rel = np.loadtxt("4th_test/rel_test_errors_small.txt")
data_6th_rel = np.loadtxt("6th_test/rel_test_errors_small.txt")
data_neural_ode_rel = np.loadtxt("neural_RK4/rel_test_errors_baseline.txt")
data_neural_RK_rel = np.loadtxt("neural_RK4/rel_test_errors_baseline.txt")

# 只有errors，shape 都是 (1001, )
plt.figure(figsize=(8, 6))
plt.plot(data_neural_ode_rel[1:], label="Neural RK6", color="#1f77b4", linewidth=2)
plt.plot(data_2th_rel[1:], label="ANI-2", color="#2ca02c", linewidth=2)
# plt.plot(data_3th_rel[1:], label="3th Split", linewidth=2)
plt.plot(data_4th_rel[1:], label="ANI-4", color="#733497", linewidth=2)
plt.plot(data_6th_rel[1:], label="ANI-6", color="#ff720e", linewidth=2)

plt.yscale("log")
# plt.title("Relative Test Errors")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Relative Error", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("rel_test_errors_test_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()


data_2th_abs = np.loadtxt("2th_test/abs_test_errors_small.txt")
data_4th_abs = np.loadtxt("4th_test/abs_test_errors_small.txt")
data_6th_abs = np.loadtxt("6th_test/abs_test_errors_small.txt")
data_neural_ode_abs = np.loadtxt("neural_RK4/abs_test_errors_baseline.txt")
data_neural_RK_abs = np.loadtxt("neural_RK4/abs_test_errors_baseline.txt")

plt.figure(figsize=(8, 6))
plt.plot(data_2th_abs[1:, 0:1], label="ANI-2", linewidth=2)
# plt.plot(data_3th_abs[1:, 0:1], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:, 0:1], label="ANI-4", linewidth=2)
plt.plot(data_6th_abs[1:, 0:1], label="ANI-6", linewidth=2)
plt.plot(data_neural_ode_abs[1:, 0:1], label="Neural RK6", linewidth=2)

plt.yscale("log")
# plt.title(r"Absolute $\theta$ Errors")
plt.xlabel("Time Step")
plt.ylabel(r"$\theta$ Error")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("abs_theta_errors_test_baseline.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(data_2th_abs[1:, 1:2], label="ANI-2", linewidth=2)
# plt.plot(data_3th_abs[1:, 1:2], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:, 1:2], label="ANI-4", linewidth=2)
plt.plot(data_6th_abs[1:, 1:2], label="ANI-6", linewidth=2)
plt.plot(data_neural_ode_abs[1:, 1:2], label="Neural RK6", linewidth=2)

plt.yscale("log")
# plt.title(r"Absolute $\theta$ Errors")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$\omega$ Error", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("abs_omega_errors_test_baseline.pdf", format='pdf', bbox_inches='tight')
plt.close()

u_omega = np.load("2th_test/traj_test.npy")
# u_omega_2th = np.load("2th_test/u_2th_test.npy")
# u_omega_4th = np.load("4th_test/u_4th_test.npy")
u_omega_6th = np.load("6th_test/u_6th_L.npy")
# u_omega_neural_ode = np.load("neural_RK4/u_baseline.npy")
# print(u_omega.shape, u_omega_2th.shape, u_omega_4th.shape, u_omega_6th.shape, u_omega_neural_ode.shape)
# [1001, 2] plot (\theta, \omega)
plt.figure(figsize=(8, 6))

# True and ANI-6 用实线
plt.plot(u_omega[:, 0], u_omega[:, 1], label="True", color="#1f77b4", linewidth=2)
plt.plot(u_omega_6th[:, 0], u_omega_6th[:, 1], 'o', label="ANI-6", color="#ff720e",
         markersize=3, alpha=0.8)


# 其它方法只画点
# plt.plot(u_omega_2th[:, 0], u_omega_2th[:, 1], 'o', label="ANI-2", markersize=3, alpha=0.7)
# plt.plot(u_omega_4th[:, 0], u_omega_4th[:, 1], 's', label="ANI-4", markersize=3, alpha=0.7)
# plt.plot(u_omega_neural_ode[:, 0], u_omega_neural_ode[:, 1], 'x', label="Neural RK6", markersize=3, alpha=0.7)

plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r"$\omega$", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("traj_test_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
# no axix, only u_true[：, 0]
plt.plot(u_omega[:, 0], u_omega[:, 1], color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_traj.pdf", format='pdf', bbox_inches='tight', dpi=300)