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
    plt.plot(resnet[:200], label='Neural-RK4', color="#0072B2", linewidth=2)
    plt.plot(ANI_2th[:200], label='ANI-2', color="#009E73", linewidth=2)
    # plt.plot(ANI_3th[:1000], label='ANI_3th')
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
abs_resnet    = np.loadtxt('NeuralRK4/rel_test_errors_small.txt')


plot_error(abs_ANI_2th, None, abs_ANI_4th, abs_resnet, 'rel_error_200.pdf')

def plot_error_compare(ANI_first, ANI_second, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(ANI_first[1:200], label='First', linewidth=2, color='#0072B2')
    plt.plot(ANI_second[1:200], label='Second', linewidth=2, color='#009E73')
    plt.legend()
    plt.xlabel('step', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


phase_true = np.load('2th/traj_2th.npy')
phase_2th  = np.load('2th/u_2th.npy')
phase_4th  = np.load('4th/u_4th.npy')
phase_base = np.load('NeuralRK4/u_baseline.npy')
time       = 0.05 * np.arange(200)
# for i in range(7) 画七个子图 [:, i]， 只画true 和 4th，true用实线，4th用mark，savefig(f"phase_{i}.png")
for i in range(7):
    cm = 1 / 2.54
    plt.figure(figsize=(2*cm, 2*cm))
    plt.plot(time, phase_true[:200, i], color="#0072B2", linewidth=0.4)
    # plt.plot(time, phase_true[:200, i], color="gray", linewidth=0.2)
    # plt.plot(time, phase_4th[:200, i], 'ro', label='ANI-4', markersize=4)
    # plt.plot(time, phase_base[:200, i], 'gx', label='SINDy-RK4', markersize=5)
    # Let Matplotlib decide the best location
    # plt.xlabel('Time (t)')
    # plt.ylabel(f'$u_{i+1}$')
    
    # Only add a legend for the first plot (when i is 0)
    # if i == 0:
        # plt.legend()
    plt.axis("off")
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.savefig(f"phase_{i}.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close()

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

for i in range(7):
    ax = axes[i] # Select the appropriate subplot
    
    # Plot data on the selected subplot axis
    ax.plot(time, phase_true[:200, i], color="#0072B2", label='True', linewidth=2)
    ax.plot(time, phase_base[:200, i], color="#009E73", marker='x', label='Neural-RK4', markersize=4, linestyle='None')
    ax.plot(time, phase_2th[:200, i], color="#E69F00", marker='s', label='ANI-2', markersize=4, linestyle='None')
    ax.plot(time, phase_4th[:200, i], color='#D55E00', marker='o', label='ANI-4', markersize=4, linestyle='None')

    # Set the labels for each subplot
    ax.set_xlabel('Time (t)', fontsize=14)
    ax.set_ylabel(f'$u_{i+1}$', fontsize=14)

# --- Handle the legend and the empty plot ---

# 1. Get the handles and labels from the last plot we made
handles, labels = ax.get_legend_handles_labels()

# 2. Turn off the axis for the unused 8th subplot
axes[7].axis('off')

# 3. Place a single, shared legend in the space of the empty subplot
axes[7].legend(handles, labels, loc='center')

# Adjust layout to prevent titles and labels from overlapping
fig.tight_layout()

plt.savefig("glycolytic_oscillator_all_phases.pdf", format='pdf', bbox_inches='tight', dpi=300)
# plt.show()
plt.close(fig)