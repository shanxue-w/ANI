import numpy as np
import matplotlib.pyplot as plt

import os
os.makedirs("../../results/Euler", exist_ok=True)

plt.rcParams.update({
    "font.size": 18,        
    "axes.labelsize": 22,   
    "xtick.labelsize": 18,  
    "ytick.labelsize": 18,  
    "legend.fontsize": 18,  
})

# Load error data
error_rho_4th = np.load("4th/error_rho_sod_4th.npy")
error_u_4th = np.load("4th/error_u_sod_4th.npy")
error_p_4th = np.load("4th/error_p_sod_4th.npy")


error_rho_2th = np.load("2th/error_rho_sod_2th.npy")
error_u_2th = np.load("2th/error_u_sod_2th.npy")
error_p_2th = np.load("2th/error_p_sod_2th.npy")

error_rho_tvd = np.load("2th/error_rho_sod_tvd.npy")
error_u_tvd = np.load("2th/error_u_sod_tvd.npy")
error_p_tvd = np.load("2th/error_p_sod_tvd.npy")

x = np.linspace(0, 1, len(error_rho_4th))

fig, axs = plt.subplots(1, 3, figsize=(22, 7))

# Density
axs[0].plot(x, error_rho_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[0].plot(x, error_rho_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[0].plot(x, error_rho_4th, label="ANI-4", color="#733497", linewidth=2)
axs[0].set_ylabel(r"$|\rho - \rho_{\mathrm{ref}}|$", fontsize=22)
axs[0].set_xlabel('x', fontsize=22)
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x, error_u_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[1].plot(x, error_u_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[1].plot(x, error_u_4th, label="ANI-4", color="#733497", linewidth=2)
axs[1].set_ylabel(r"$|u - u_{\mathrm{ref}}|$", fontsize=22)
axs[1].set_xlabel('x', fontsize=22)
axs[1].grid(True)
axs[1].legend()

axs[2].plot(x, error_p_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[2].plot(x, error_p_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[2].plot(x, error_p_4th, label="ANI-4", color="#733497", linewidth=2)
axs[2].set_ylabel(r"$|p - p_{\mathrm{ref}}|$", fontsize=22)
axs[2].set_xlabel('x', fontsize=22)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig("../../results/Euler/error_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)

rho_sod = np.load("2th/rho_sod.npy")
u_sod = np.load("2th/u_sod.npy")
p_sod = np.load("2th/p_sod.npy")

rho_2th = np.load("2th/rho_sod_2th.npy")
u_2th = np.load("2th/u_sod_2th.npy")
p_2th = np.load("2th/p_sod_2th.npy")

rho_tvd = np.load("2th/rho_sod_tvd.npy")
u_tvd = np.load("2th/u_sod_tvd.npy")
p_tvd = np.load("2th/p_sod_tvd.npy")

rho_4th = np.load("4th/rho_sod_4th.npy")
u_4th = np.load("4th/u_sod_4th.npy")
p_4th = np.load("4th/p_sod_4th.npy")


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

colors = ["#1f77b4", "#2ca02c", "#733497", "#ff7f0e"]  # True + three models

def plot_models_side_by_side(rho_ref, u_ref, p_ref,
                             rho_models, u_models, p_models,
                             labels, axs):
    axs[0].plot(x, rho_ref, color=colors[0], label='True')
    for i, rho_model in enumerate(rho_models):
        axs[0].plot(x, rho_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[0].set_ylabel(r'$\rho$', fontsize=22)
    axs[0].set_xlabel('x', fontsize=22)
    axs[0].grid(True)
    axs[0].legend()
    
    axins = inset_axes(axs[0], width="35%", height="35%", loc="lower left")  
    axins.plot(x, rho_ref, color=colors[0])
    for i, rho_model in enumerate(rho_models):
        axins.plot(x, rho_model, color=colors[i+1], linestyle='--', linewidth=2)
    x1, x2 = 0.65, 0.75  
    y1, y2 = 0.2, 0.5  
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axs[1].plot(x, u_ref, color=colors[0], label='True')
    for i, u_model in enumerate(u_models):
        axs[1].plot(x, u_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[1].set_ylabel(r'$u$', fontsize=22)
    axs[1].set_xlabel('x', fontsize=22)
    axs[1].grid(True)
    axs[1].legend()

    axins = inset_axes(axs[1], width="35%", height="35%", loc="lower left")
    axins.plot(x, u_ref, color=colors[0])
    for i, u_model in enumerate(u_models):
        axins.plot(x, u_model, color=colors[i+1], linestyle='--', linewidth=2)
    x1, x2 = 0.47, 0.57
    y1, y2 = 0.85, 0.95
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=2)
    
    axs[2].plot(x, p_ref, color=colors[0], label='True')
    for i, p_model in enumerate(p_models):
        axs[2].plot(x, p_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[2].set_ylabel(r'$p$', fontsize=22)
    axs[2].set_xlabel('x', fontsize=22)
    axs[2].grid(True)
    axs[2].legend()

fig, axs = plt.subplots(1, 3, figsize=(22, 7))

plot_models_side_by_side(
    rho_sod, u_sod, p_sod,
    rho_models=[rho_tvd, rho_2th, rho_4th],
    u_models=[u_tvd, u_2th, u_4th],
    p_models=[p_tvd, p_2th, p_4th],
    labels=['TVD', 'ANI-2', 'ANI-4'],
    axs=axs
)

plt.tight_layout()
plt.savefig("../../results/Euler/solution_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)
