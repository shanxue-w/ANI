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
    # t = np.linspace(t0, t_end, num_steps)
    # y = np.zeros((num_steps, len(y0)))
    # y[0, :] = y0

    dt = h
    # i = 1
    while np.abs(t0 - t_end) > 1e-5:
        if t_end - t0 < h:
            dt = t_end - t0
        k1 = dt * f(t0, y0)
        k2 = dt * f(t0 + 0.25*dt, y0 + 0.25*k1)
        k3 = dt * f(t0 + 3/8*dt, y0 + 3 / 32 * k1 + 9 / 32 * k2)
        k4 = dt * f(t0 + 12 / 13 * dt, y0 + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
        k5 = dt * f(t0 + dt, y0 + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
        k6 = dt * f(t0 + 0.5 * dt, y0 - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
        y0 =y0 + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
        t0 += dt
        # i  += 1
    # for i in range(1, num_steps):
    #     k1 = h * f(t[i - 1], y[i - 1, :])
    #     k2 = h * f(t[i - 1] + 0.25 * h, y[i - 1, :] + 0.25 * k1)
    #     k3 = h * f(t[i - 1] + 3 / 8 * h, y[i - 1, :] + 3 / 32 * k1 + 9 / 32 * k2)
    #     k4 = h * f(t[i - 1] + 12 / 13 * h, y[i - 1, :] + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
    #     k5 = h * f(t[i - 1] + h, y[i - 1, :] + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
    #     k6 = h * f(t[i - 1] + 0.5 * h, y[i - 1, :] - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
        
    #     y[i, :] = y[i - 1, :] + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
    
    return t0, y0


# ode function damped pendulum
def ode_function(t, y):
    alpha = 0.3
    beta = 9.80665
    z = np.zeros_like(y)
    z[0] = y[1]
    z[1] = - alpha * y[1] - beta * np.sin(y[0])
    return z

# 生成数据
def generate_data(num_samples=50, dt_min=1e-3, dt_max=5e-1):
    inputs = []
    outputs = []

    for i in range(num_samples):
        print(f"{i}/{num_samples}",end='\r')
        # 随机初值
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        theta_dot = np.random.uniform(-np.pi, np.pi)
        y0 = np.array([theta, theta_dot])

        # 随机时间步
        delta_t = np.random.uniform(dt_min, dt_max)
        # delta_t = 1e-1

        # 计算一小步 Runge-Kutta 5
        t_vals, y_vals = runge_kutta_5th_order(ode_function, y0, 0.0, delta_t, h=1e-4)

        u_t = y0
        u_next = y_vals

        # 输入向量多加一维 delta_t
        input_vec = np.concatenate([u_t, [delta_t]])
        output_vec = u_next

        inputs.append(input_vec)
        outputs.append(output_vec)

    return np.array(inputs), np.array(outputs)

def generate_trajectories(num_trajectories=30, delta_t=5e-2, total_time=20.0, h=1e-4):
    trajectories = []

    steps_per_traj = int(total_time / delta_t) + 1

    for _ in range(num_trajectories):
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        theta_dot = np.random.uniform(-np.pi, np.pi)
        y0 = np.array([theta, theta_dot])

        traj = [y0.copy()]

        for _ in range(1, steps_per_traj):
            _, y_vals = runge_kutta_5th_order(ode_function, y0, 0.0, delta_t, h=h)
            y0 = y_vals
            traj.append(y0.copy())

        trajectories.append(np.stack(traj))  # shape: (T, 2)

    return np.stack(trajectories)  # shape: (N, T, 2)


# 生成总数据
total_samples = 10000
inputs, outputs = generate_data(num_samples=total_samples)

# 划分比例
n_train = 9000
n_val = 1000
# n_var = total_samples - n_train - n_val  # = 700

# 划分数据
train_inputs = inputs[:n_train]
train_outputs = outputs[:n_train]

val_inputs = inputs[n_train:n_train+n_val]
val_outputs = outputs[n_train:n_train+n_val]

# test_inputs = inputs[n_train+n_val:]
# test_outputs = outputs[n_train+n_val:]

# 保存为 npy 文件
np.save('train_inputs_test.npy', train_inputs)
np.save('train_outputs_test.npy', train_outputs)

np.save('val_inputs_test.npy', val_inputs)
np.save('val_outputs_test.npy', val_outputs)

# np.save('test_inputs_fixed.npy', test_inputs)
# np.save('test_outputs_fixed.npy', test_outputs)

test_data = generate_trajectories(num_trajectories=50, delta_t=1e-1, total_time=20.0, h=1e-4)
np.save('test_trajectories_test.npy', test_data)