def ode_function(t, y):
    l1 = 1 
    l2 = 1 
    m1 = 1 
    m2 = 1 
    g = 10 
    
    z = np.zeros_like(y)
    cosa = np.cos(y[0] - y[1])
    sina = np.sin(y[0] - y[1])
    temp = m1 + m2 * sina ** 2
    c1 = y[2] * y[3] * sina / ( l1 * l2 * temp)
    c2 = (m2*l2**2* y[2]**2 + (m1+m2)*l1**2 * y[3]**2 - 2*m2*l1*l2*y[2]*y[3]*cosa) \
         / (2*l1**2 * l2**2 * temp**2)
    z[0] = (l2 * y[2] - l1 * y[3] * cosa) / ( l1**2 * l2 * temp )
    z[1] = (-m2 * l2 * y[2] * cosa + (m1 + m2) * l1 * y[3]) / ( m2*l1*l2**2 * temp )
    z[2] = - (m1 + m2) * g * l1 * np.sin(y[0]) - c1 + c2 * np.sin(2 * (y[0] - y[1]))
    z[3] = - m2 * g * l2 * np.sin(y[1]) + c1 - c2 * np.sin(2 * (y[0] - y[1]))
    return z

#%%
# Set common parameters
time_step = 0.05 
num_trajectories = 100
num_times = 5
num_parameters = 4

# Generate 1000 initial conditions
initial_conditions = np.zeros((num_trajectories,num_parameters))
initial_conditions[:,0] = np.random.uniform(-np.pi/2, np.pi/2, num_trajectories)
initial_conditions[:,1] = np.random.uniform(-np.pi/2, np.pi/2, num_trajectories)
initial_conditions[:,2] = np.random.uniform(-np.pi, np.pi, num_trajectories)
initial_conditions[:,3] = np.random.uniform(-np.pi, np.pi, num_trajectories)