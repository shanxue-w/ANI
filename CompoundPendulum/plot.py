# 读取 2th 3th 4th resnet下的文件并画图，写个画图函数
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os 
os.makedirs("../../results/CompoundPendulum", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,        
    "axes.labelsize": 14,   
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14, 
    "legend.fontsize": 14, 
})

def plot_error(ANI_2th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[:1000], label='Baseline', color="#1f77b4", linewidth=2)
    plt.plot(ANI_2th[:1000], label='ANI-2', color="#2ca02c", linewidth=2)
    plt.plot(ANI_4th[:1000], label='ANI-4', color="#733497", linewidth=2)
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

plot_error(abs_ANI_2th, abs_ANI_4th, abs_resnet, '../../results/CompoundPendulum/rel_error_1000.pdf')

u_true = np.load("2th/traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_base = np.load("baseline/u_base.npy")

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#1f77b4", label='True', linewidth=2)
plt.plot(u_2th[600:1000, 0], u_2th[600:1000, 1], color="#2ca02c", marker='o', label='ANI-2', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('../../results/CompoundPendulum/2th.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#1f77b4", label='True', linewidth=2)
plt.plot(u_4th[600:1000, 0], u_4th[600:1000, 1], color='#733497', marker='o', label='ANI-4', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('../../results/CompoundPendulum/4th.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(u_true[600:1000, 0], u_true[600:1000, 1], color="#1f77b4", label='True', linewidth=2)
plt.plot(u_base[600:1000, 0], u_base[600:1000, 1], color='#ff720e', marker='o', label='Baseline', markersize=4, linestyle='None')
plt.legend(loc="lower left")
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\omega_1$', fontsize=14)
plt.tight_layout()
plt.savefig('../../results/CompoundPendulum/baseline.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_true[300:1000, 2], u_true[300:1000, 3], color="#1f77b4", label='True', linewidth=2)
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig('../../results/CompoundPendulum/data_traj.pdf', format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()