data_true = load('OSNET_Lorenz_stenflo_true_0.mat');
data_true = data_true.true_traj;

data_2th  = load('OSNET_Lorenz_stenflo_2th_0.mat');
data_2th  = data_2th.NN_traj;

data_4th  = load('OSNET_Lorenz_stenflo_4th_0.mat');
data_4th  = data_4th.NN_traj;

data_base = load('OSNET_Lorenz_stenflo_base_0.mat');
data_base = data_base.NN_traj;

a_true   = approximateEntropy(data_true)
a_2th    = approximateEntropy(data_2th)
a_4th    = approximateEntropy(data_4th)
a_base   = approximateEntropy(data_base)

b_true   = correlationDimension(data_true)
b_2th    = correlationDimension(data_2th)
b_4th    = correlationDimension(data_4th)
b_base   = correlationDimension(data_base)

c_true   = lyapunovExponent(data_true, 100)
c_2th    = lyapunovExponent(data_2th, 100)
c_4th    = lyapunovExponent(data_4th, 100)
c_base   = lyapunovExponent(data_base, 100)


result_cdim = abs([b_2th, b_4th, b_base] - b_true) / b_true
result_app = abs([a_2th, a_4th,  a_base] - a_true) / a_true
result_lya = abs([c_2th, c_4th,  c_base] - c_true) / c_true