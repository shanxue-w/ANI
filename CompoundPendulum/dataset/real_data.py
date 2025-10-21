import numpy as np

# 读取数据
data = np.loadtxt('real_double_pend_h_1.txt')  # shape: (1520, N)
data = data[:, 2:6]
# 交换第2列和第3列
data[:, [1, 2]] = data[:, [2, 1]]
print(data)

# 拆分数据
train_data = data[:700]
val_data   = data[700:1000]
test_data  = data[1000:]

# 构造输入输出对
def make_io_pairs(seq):
    inputs = seq[:-1]
    outputs = seq[1:]
    return inputs, outputs

train_inputs, train_outputs = make_io_pairs(train_data)
val_inputs, val_outputs     = make_io_pairs(val_data)

# 在输入最后一维拼接常数列 1e-1
dt = 1e-2
train_inputs = np.concatenate([train_inputs, np.full((train_inputs.shape[0], 1), dt)], axis=1)
val_inputs   = np.concatenate([val_inputs,   np.full((val_inputs.shape[0], 1), dt)], axis=1)

# 保存为 .npy 文件
np.save('real_train_inputs.npy', train_inputs)
np.save('real_train_outputs.npy', train_outputs)
np.save('real_val_inputs.npy', val_inputs)
np.save('real_val_outputs.npy', val_outputs)
np.save('real_test_trajectories.npy', test_data)


pinn_input, pinn_output = make_io_pairs(data)

dt = 1e-2
pinn_input = np.concatenate([pinn_input, np.full((pinn_input.shape[0], 1), dt)], axis=1)

# 保存为 npy 文件
np.save('pinn_input.npy', pinn_input)
np.save('pinn_output.npy', pinn_output)