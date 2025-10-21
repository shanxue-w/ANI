# 读取 .mat 文件并保存为 npy 文件

import scipy.io as sio
import numpy as np

data_train = sio.loadmat('gly_train_data.mat')
data_val = sio.loadmat('gly_val_data.mat')
data_traj = sio.loadmat('gly_test_trajectories.mat')
# print(data_train)

np.save('train_inputs.npy', data_train['train_inputs'])
np.save('train_outputs.npy', data_train['train_outputs'])
np.save('val_inputs.npy', data_val['val_inputs'])
np.save('val_outputs.npy', data_val['val_outputs'])
# 把 data_traj['test_trajectories'] 从 1*10001*3 转为 10001*3
np.save('test_trajectories.npy', data_traj['test_trajectories'])

data_small_traj = sio.loadmat('gly_test_trajectories_small.mat')
np.save('test_trajectories_small.npy', data_small_traj['test_trajectories'])