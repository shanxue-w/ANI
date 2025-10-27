import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os 
os.makedirs("../../results/KolmogorovFlow", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,       
    "axes.labelsize": 14,   
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14, 
    "legend.fontsize": 14, 
})

file_path_2th = '2th_new/traj_error.txt'
file_path_4th = '4th_new/traj_error.txt'
file_path_les = '2th_new/traj_error_les.txt'

data_2th = pd.read_csv(file_path_2th, sep='\t')
data_2th = data_2th['avg_relative_error']
data_2th = data_2th[0:400]
data_4th = pd.read_csv(file_path_4th, sep='\t')
data_4th = data_4th['avg_relative_error']
data_4th = data_4th[0:400]
data_les = pd.read_csv(file_path_les, sep='\t')
data_les = data_les['avg_relative_error']
data_les = data_les[0:400]

plt.figure(figsize=(8, 6))
plt.plot(data_les, label='LES', linestyle='-.', color='#1f77b4', linewidth=2, marker='^', markersize=2)
plt.plot(data_2th, label='ANI-2', linestyle='-', color='#2ca02c', linewidth=2, marker='o', markersize=2)
plt.plot(data_4th, label='ANI-4', linestyle='--', color='#733497', linewidth=2, marker='s', markersize=2)
plt.xlabel('Time Index', fontsize=14)
plt.ylabel('Average Relative Error', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('../../results/KolmogorovFlow/ns_traj_error_new.pdf', format='pdf', bbox_inches='tight', dpi=300)
