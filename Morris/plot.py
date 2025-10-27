import matplotlib.pyplot as plt
import numpy as np
import os 
os.makedirs("../../results/Morris", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,        
    "axes.labelsize": 14,   
    "xtick.labelsize": 14, 
    "ytick.labelsize": 14,  
    "legend.fontsize": 14,  
})

data_2th_abs = np.loadtxt("2th/abs_test_errors_small.txt")
data_4th_abs = np.loadtxt("4th/abs_test_errors_small.txt")
abs_resnet   = np.loadtxt("Euler/abs_test_errors_small.txt")

plt.figure(figsize=(8, 6))
plt.plot(abs_resnet[1:200, 0:1], label='Forward Euler', linewidth=2, color="#1f77b4")
plt.plot(data_2th_abs[1:200, 0:1], label="ANI-2", linewidth=2, color="#2ca02c")
plt.plot(data_4th_abs[1:200, 0:1], label="ANI-4", linewidth=2, color="#733497")

plt.yscale("log")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$V$ Error", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../results/Morris/abs_V_errors_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(abs_resnet[1:200, 1:2], label='Forward Euler', linewidth=2, color="#1f77b4")
plt.plot(data_2th_abs[1:200, 1:2], label="ANI-2", linewidth=2, color="#2ca02c")
plt.plot(data_4th_abs[1:200, 1:2], label="ANI-4", linewidth=2, color="#733497")

plt.yscale("log")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$w$ Error", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../results/Morris/abs_w_errors_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

u_traj = np.load("2th/u_traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_euler = np.load("Euler/u_euler.npy")
time   = 0.05 * np.arange(0, 200)

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:, 0], color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("../../results/Morris/u_traj_1.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(u_traj[:, 1], color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("../../results/Morris/u_traj_2.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()
