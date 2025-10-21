# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:20:48 2024

@author: qinghe
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm


def runge_kutta_5th_order(f, y0, t0, t_end, h):
    num_steps = int((t_end - t0) / h) + 1
    t = np.linspace(t0, t_end, num_steps)
    y = np.zeros((num_steps, len(y0)))
    y[0, :] = y0

    for i in range(1, num_steps):
        k1 = h * f(t[i - 1], y[i - 1, :])
        k2 = h * f(t[i - 1] + 0.25 * h, y[i - 1, :] + 0.25 * k1)
        k3 = h * f(t[i - 1] + 3 / 8 * h, y[i - 1, :] + 3 / 32 * k1 + 9 / 32 * k2)
        k4 = h * f(t[i - 1] + 12 / 13 * h, y[i - 1, :] + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
        k5 = h * f(t[i - 1] + h, y[i - 1, :] + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
        k6 = h * f(t[i - 1] + 0.5 * h, y[i - 1, :] - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
        
        y[i, :] = y[i - 1, :] + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
    
    return t, y


#%%
# ode function damped pendulum
def ode_function(t, y):
    J0 = 2.5
    k1 = 100
    k2 = 6
    k3 = 16
    k4 = 100
    k5 = 1.28
    k6 = 12
    k  = 1.8
    kappa = 13
    q = 4
    K1 = 0.52
    psi = 0.1
    N = 1
    A = 4
    
    z = np.zeros_like(y)
    t1 = k1 * y[0] * y[5] / (1 + (y[5] / K1)**q)  
    z[0] = J0 - t1    
    
    t2 = k2 * y[1] * (N - y[4])
    z[1] = 2 * t1 - t2 - k6 * y[1] * y[4]
    
    t3 = k3 * y[2] * (A - y[5])
    z[2] = t2 - t3
    
    t4 = k4 * y[3] * y[4]
    z[3] = t3 - t4 - kappa * (y[3] - y[6])
    
    z[4] = t2 - t4 - k6 * y[1] * y[4]
    
    z[5] = -2 * t1 + 2 * t3 - k5 * y[5]
    
    z[6] = psi * kappa * (y[3] - y[6]) - k * y[6]
    
    return z

#%%
# Set common parameters
time_step = 0.05
num_trajectories = 20
num_times = 51
num_parameters = 7

# Generate 1000 initial conditions
initial_conditions = np.zeros((num_trajectories,num_parameters))
initial_conditions[:,0] = np.random.uniform(0.15, 1.6, num_trajectories)
initial_conditions[:,1] = np.random.uniform(0.19, 2.16, num_trajectories)
initial_conditions[:,2] = np.random.uniform(0.04, 0.2, num_trajectories)
initial_conditions[:,3] = np.random.uniform(0.1, 0.35, num_trajectories)
initial_conditions[:,4] = np.random.uniform(0.08, 0.3, num_trajectories)
initial_conditions[:,5] = np.random.uniform(0.14, 2.67, num_trajectories)
initial_conditions[:,6] = np.random.uniform(0.05, 0.1, num_trajectories)

# initial_conditions[0,:] = np.array([1.,1.,0.1,0.1,0.1,1.,0.1])
# Generate and save 1000 trajectories
data = np.zeros((num_trajectories, num_times, num_parameters))
for i, initial_condition in enumerate(initial_conditions):
    data[i,0,:] = initial_condition
    for j in range(num_times-1):
        ttt, y = runge_kutta_5th_order(ode_function, data[i,j,:], 0, time_step, time_step/50)
        data[i,j+1,:] = y[-1,:]

# Save the Data to a file
# np.save('glycolytic_train.npy', data)  # 80*51*7
# np.save('glycolytic_val.npy', data)    # 20*51*7
# np.save('glycolytic_test.npy', data)   # 100*201*7
# np.save('glycolytic_evo.npy', data)

# 构造 (u_in, dt) -> u_out 格式
dt = time_step
num_samples = num_trajectories * (num_times - 1)

inputs = np.zeros((num_samples, num_parameters + 1))
outputs = np.zeros((num_samples, num_parameters))

idx = 0
for i in range(num_trajectories):
    for j in range(num_times - 1):
        inputs[idx, :-1] = data[i, j, :]     # u_in
        inputs[idx, -1] = dt                 # append dt
        outputs[idx, :] = data[i, j+1, :]    # u_out
        idx += 1

# 保存为 numpy 文件
np.save('gly_val_inputs.npy', inputs)
np.save('gly_val_outputs.npy', outputs)


