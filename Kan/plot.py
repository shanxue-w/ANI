import matplotlib.pyplot as plt
import numpy as np
import os 
os.makedirs("../../results/Kan", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,       
    "axes.labelsize": 13.5,   
    "xtick.labelsize": 12, 
    "ytick.labelsize": 12,  
    "legend.fontsize": 14, 
})

def plot_error(ANI_2th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[1:200], label='KAN-RK4', color="#1f77b4", linewidth=2)
    plt.plot(ANI_2th[1:200], label='ANI_2th', color="#2ca02c", linewidth=2)
    plt.plot(ANI_4th[1:200], label='ANI_4th', color="#733497", linewidth=2)
    plt.legend()
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.yscale('log')
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

abs_ANI_2th = np.loadtxt('2th/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('RK4/rel_test_errors_small.txt')

plot_error(abs_ANI_2th, abs_ANI_4th, abs_resnet, '../../results/Kan/rel_error_200.pdf')

u_traj = np.load("2th/u_traj.npy")
u_2th  = np.load("2th/u_2th.npy")
u_4th  = np.load("4th/u_4th.npy")
u_base = np.load("RK4/u_base.npy")


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
x_min, x_max = u_traj[:,0].min()-0.02, u_traj[:,0].max()+0.02
y_min, y_max = u_traj[:,1].min()-0.02, u_traj[:,1].max()+0.02
z_min, z_max = u_traj[:,2].min()-0.02, u_traj[:,2].max()+0.02

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

import scipy.io as sio
data = sio.loadmat("./dataset/kan_test_trajectories_1.mat")
u_traj = data["test_trajectories"][0]

fig = plt.figure(figsize=(8, 8))
plt.plot(u_traj[:,0])
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
plt.savefig("data_3d.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()
