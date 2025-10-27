import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os 
os.makedirs("../../results/Fitzhugh-Nagumo", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,       
    "axes.labelsize": 14,   
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14,  
    "legend.fontsize": 14, 
})

files = {
    "Pretrained": "2th_new/pred_error_2th_A.txt",
    "ANI-2": "2th_new/pred_error_2th.txt",
    "ANI-4": "4th_new/pred_error_4th.txt"
}

fig, axes = plt.subplots(1, len(files), figsize=(15, 4), constrained_layout=True)

grid_a = np.linspace(0.6, 0.8, 20)
grid_b = np.linspace(0.7, 0.9, 20)
A_grid, B_grid = np.meshgrid(grid_a, grid_b)

for ax, (method, fname) in zip(axes, files.items()):
    data = np.loadtxt(fname, delimiter=",")
    a, b, err = data[:,0], data[:,1], data[:,2]
    err_clip = np.clip(err, 0, 1)
    ERR_grid = griddata((a, b), err_clip, (A_grid, B_grid), method='cubic')
    im = ax.imshow(ERR_grid, origin='lower', 
                   extent=[grid_a[0], grid_a[-1], grid_b[0], grid_b[-1]],
                   aspect='auto', cmap='viridis')
    ax.set_title(method, fontsize=14)
    ax.set_xlabel("a", fontsize=14)
    ax.set_ylabel("b", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="error")

plt.savefig("../../results/Fitzhugh-Nagumo/ab_error.pdf", format='pdf', bbox_inches='tight', dpi=300)
