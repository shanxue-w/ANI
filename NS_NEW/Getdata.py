from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *

auth_token = 'cn.edu.zju.3220104819-8ce9a7a0'
dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'
output_filename = 'test.tsv'

# instantiate the dataset.
dataset = turb_dataset(dataset_title = dataset_title, output_path = output_path, auth_token = auth_token)
variable = 'velocity'
temporal_method = 'none'
spatial_method = 'none'
spatial_operator = 'field'

nx = 64
ny = 64
nz = 64
n_points = nx * ny * nz
x_points = np.linspace(0, 2*np.pi, nx, endpoint=False, dtype=np.float64)
y_points = np.linspace(0, 2*np.pi, ny, endpoint=False, dtype=np.float64)
z_points = np.linspace(0, 2*np.pi, nz, endpoint=False, dtype=np.float64)

points   = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing = 'ij')], dtype = np.float64).T

time = 0
time_end = 0.004
delta_t  = 0.002
option   = [time_end, delta_t]

result, times = getData(dataset, variable, time, temporal_method, spatial_method, spatial_operator, points, option, return_times = True)

write_interpolation_tsv_file(dataset, points, result, output_filename)