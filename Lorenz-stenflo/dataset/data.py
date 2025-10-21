# 读取 .mat 文件并保存为 npy 文件

import scipy.io as sio
import numpy as np

data_train = sio.loadmat('lorenz_stenflo_train_data.mat')
data_val = sio.loadmat('lorenz_stenflo_val_data.mat')
data_traj = sio.loadmat('lorenz_stenflo_test_trajectories.mat')
data_true_plot = sio.loadmat('lorenz_tol1e10.mat')
data_4th_plot  = sio.loadmat('ANI_Lorenz_stenflo_4th_0.mat')
data_lorenz_stenflo_plot = sio.loadmat('ANI_Lorenz_stenflo_true_0.mat')
# print(data_train)

np.save('train_inputs.npy', data_train['train_inputs'])
np.save('train_outputs.npy', data_train['train_outputs'])
np.save('val_inputs.npy', data_val['val_inputs'])
np.save('val_outputs.npy', data_val['val_outputs'])
# 把 data_traj['test_trajectories'] 从 1*10001*3 转为 10001*3
np.save('test_trajectories.npy', data_traj['test_trajectories'])
# print(data_true_plot)
np.save('lorenz_true_plot.npy', data_true_plot['u1'])
# print(data_4th_plot)
np.save('lorenz_4th_plot.npy', data_4th_plot['NN_traj'])
# print(data_lorenz_stenflo_plot)
np.save('lorenz_stenflo_plot.npy', data_lorenz_stenflo_plot['true_traj'])
