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
    plt.plot(resnet[1:201], label='Baseline', color="#1f77b4", linewidth=2)
    plt.plot(ANI_2th[1:201], label='ANI-2', color="#2ca02c", linewidth=2)
    # plt.plot(ANI_3th[:1000], label='ANI_3th')
    plt.plot(ANI_4th[1:201], label='ANI-4', color="#733497", linewidth=2)
    plt.legend()
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

abs_ANI_2th = np.loadtxt('2th/rel_test_errors_small.txt')
abs_ANI_3th = np.loadtxt('3th/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('NeuralRK4/rel_test_errors_small_test.txt')

plot_error(abs_ANI_2th, abs_ANI_3th, abs_ANI_4th, abs_resnet, 'rel_error_200.pdf')

def plot_trajectory(traj, filename, title='Trajectory'):
    # 三维轨迹图
    x = traj[:10001, 0]
    y = traj[:10001, 1]
    z = traj[:10001, 2]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x, y, z, color="#1f77b4", linewidth=2)
    ax.set_xlabel("x", fontsize=14, labelpad=5)
    ax.set_ylabel("y", fontsize=14, labelpad=5)
    ax.set_zlabel("z", fontsize=14, labelpad=5)
    # ax.tick_params(axis='both', which='major', direction='in', labelsize=8)
    # ax.tick_params(axis='z', which='major', direction='in', labelsize=10)


    ax.grid(True, color='0.8', linestyle='--', linewidth=0.5)  # 浅灰虚线

    # ax.view_init(elev=30, azim=45)  # 可以调整，找最佳观察角度

    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()

lorenz_basedata = np.load('dataset/lorenz_true_plot.npy')
plot_trajectory(lorenz_basedata, 'lorenz_true_plot.pdf')

lorenz_truedata = np.load('dataset/lorenz_stenflo_plot.npy')
plot_trajectory(lorenz_truedata, 'lorenz_stenflo_plot.pdf', title='Lorenz Stenflo True Trajectory')

lorenz_4thdata = np.load('dataset/lorenz_4th_plot.npy')
plot_trajectory(lorenz_4thdata, 'lorenz_4th_plot.pdf', title='Lorenz Stenflo 4th Order Neural ODE')

fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.plot(lorenz_truedata[:,0], lorenz_truedata[:,1], lorenz_truedata[:,2], color="#1f77b4")
ax.set_axis_off()
ax.grid(False)
xyz_min = lorenz_truedata.min(axis=0)
xyz_max = lorenz_truedata.max(axis=0)
ax.set_xlim(xyz_min[0], xyz_max[0])
ax.set_ylim(xyz_min[1], xyz_max[1])
ax.set_zlim(xyz_min[2], xyz_max[2])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
plt.savefig("data_3d_new.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
plt.close()