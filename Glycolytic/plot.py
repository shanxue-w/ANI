import matplotlib.pyplot as plt
import numpy as np
import os 
os.makedirs("../../results/Glycolytic", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,        
    "axes.labelsize": 14,   
    "xtick.labelsize": 14, 
    "ytick.labelsize": 14, 
    "legend.fontsize": 14,
})

def plot_error(ANI_2th, ANI_4th, resnet, filename, n = 200):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[:n], label='SINDy-RK4', color="#1f77b4", linewidth=2)
    plt.plot(ANI_2th[:n], label='ANI-2', color="#2ca02c", linewidth=2)
    plt.plot(ANI_4th[:n], label='ANI-4', color="#733497", linewidth=2)
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

plot_error(abs_ANI_2th, abs_ANI_4th, abs_resnet, '../../results/Glycolytic/rel_error_200.pdf')
plot_error(abs_ANI_2th, abs_ANI_4th, abs_resnet, '../../results/Glycolytic/rel_error_1000.pdf', n=1000)


phase_true = np.load('2th/traj_2th.npy')
phase_2th  = np.load('2th/u_2th.npy')
phase_4th  = np.load('4th/u_4th.npy')
phase_base = np.load('NeuralRK4/u_baseline.npy')
time       = 0.05 * np.arange(200)
for i in range(7):
    plt.figure(figsize=(8, 8))
    plt.plot(time, phase_true[:200, i], color="#1f77b4", linewidth=2)
    plt.axis("off")
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.savefig(f"../../results/Glycolytic/phase_{i}.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

for i in range(7):
    ax = axes[i] 
    
    ax.plot(time, phase_true[:200, i], color="#1f77b4", label='True', linewidth=2)
    ax.plot(time, phase_base[:200, i], color="#ff720e", marker='x', label='SINDy-RK4', markersize=4, linestyle='None')
    ax.plot(time, phase_2th[:200, i], color="#2ca02c", marker='s', label='ANI-2', markersize=4, linestyle='None')
    ax.plot(time, phase_4th[:200, i], color='#733497', marker='o', label='ANI-4', markersize=4, linestyle='None')

    ax.set_xlabel('Time (t)', fontsize=14)
    ax.set_ylabel(f'$u_{i+1}$', fontsize=14)

handles, labels = ax.get_legend_handles_labels()
axes[7].axis('off')
axes[7].legend(handles, labels, loc='center')
fig.tight_layout()
plt.savefig("../../results/Glycolytic/glycolytic_oscillator_all_phases.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig)