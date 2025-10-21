import numpy as np
import matplotlib.pyplot as plt
import torch

from Euler_Solver import FD

c = 0.5
# c = 0.99999

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Pri_exact needs to handle numpy input for `x` for now, as it's used with `self.xc.cpu().numpy()`
# If Pri_exact were defined within a torch context, it could take torch tensors directly.
def Pri_exact(x, t):
    rho = 1.0 + c*torch.sin(2.0*np.pi*(x-t))
    v = torch.ones(len(rho)).to(device)
    p = torch.ones(len(rho)).to(device)
    return torch.stack([rho, v, p], dim=1)

# Pri_init will also take numpy x and return numpy array
def Pri_init(x):
    return Pri_exact(x, 0)

# Determine the device
print(f"Using device: {device}")

NN = [10, 20, 40, 80]
error_table_list = [] # Store errors for each batch and N
rate_table = np.zeros((len(NN)-1, 3)) # Will store average rates if batch_size > 1

batch_size_for_test = 1 # We are testing convergence for single cases here, so batch_size=1 makes sense.
                        # If you want to run multiple N simultaneously in a batch, you'd need to adjust NN and batch_size
                        # and interpret error_table accordingly (e.g., mean error over batch).
                        # For convergence study, typically each N is run independently.

for N in NN:
    # Initialize FD with CUDA and float64
    myFD = FD(N, batch_size=batch_size_for_test, device=device, dtype=torch.float64)
    myFD.x_beg = 0
    myFD.x_end = 1
    myFD.test_Convergence = True
    myFD.CFL = 0.4
    myFD.t_end = 0.1

    myFD.Init(Pri_init)
    
    # Printing a value from Cons to ensure it's on the correct device and type
    print(f"N={N}, Initial Cons[0,0] on {myFD.Cons.device} with type {myFD.Cons.dtype}: {myFD.Cons[0,0,0]}")
    
    myFD.Solve()
    
    # Compute_Err now returns a [batch_size, 3] tensor.
    # Since batch_size_for_test is 1, it will be [1, 3].
    current_errors = myFD.Compute_Err(Pri_exact, 0).cpu().numpy() # Convert to numpy for storage/printing
    error_table_list.append(current_errors[0, :]) # Take the first (and only) batch's errors

error_table = np.array(error_table_list) # Convert list of arrays to a single numpy array

# Compute convergence rate
for i in range(len(NN)-1):
    for j in range(3):
        rate_table[i,j] = np.log(error_table[i,j]/error_table[i+1,j])/np.log(NN[i+1]/NN[i])

print("\nError Table:")
print(error_table)
print("\nConvergence Rate Table:")
print(rate_table)

# Formatted Output
print('\nFormatted Output:')
print('%3d &' % NN[0], end = ' ')
for j in range(3):
    print('%-.3e&   --& ' %(error_table[0, j]), end=' ')
print('\\\\')
for i in range(1, len(NN)):
    print('%3d &' % NN[i], end = ' ')
    for j in range(3):
        print('%-.3e& %.2f& ' %(error_table[i, j], rate_table[i-1, j]), end=' ')
    print('\\\\')


# Plotting the error for the last run (highest N)
# Need to move data to CPU for matplotlib
fig, ax = plt.subplots()
ax.plot(myFD.xc.cpu().numpy(), myFD.err[0,:].cpu().numpy()) # Access the first batch's error
ax.set_title(f'Error distribution for N={NN[-1]}')
ax.set_xlabel('x')
ax.set_ylabel('Absolute Error (Density)')
fig.savefig("res_cuda_float64.pdf") # Save with a new name to distinguish
plt.show()
