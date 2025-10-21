import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from base import FNO_ETDRK4
import torch
import random

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize(data, save_as='output.gif', fps=10):
    t, s, _, _ = data.shape
    
    u = data[:,:,:,0]
    v = data[:,:,:,1]
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    ax_u, ax_v = axes
    
    im_u = ax_u.imshow(u[0], cmap='viridis', origin='lower', extent=[0,s,0,s])
    im_v = ax_v.imshow(v[0], cmap='plasma', origin='lower', extent=[0,s,0,s])
    
    ax_u.set_title('u dynamics')
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    ax_v.set_title("v Dynamics")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")

    # Add colorbars
    cbar_u = fig.colorbar(im_u, ax=ax_u)
    cbar_v = fig.colorbar(im_v, ax=ax_v)

    # Update function for animation
    def update(frame):
        # Update data
        im_u.set_data(u[frame])
        im_v.set_data(v[frame])

        # Adjust color limits dynamically
        im_u.set_clim(vmin=np.min(u[frame]), vmax=np.max(u[frame]))
        im_v.set_clim(vmin=np.min(v[frame]), vmax=np.max(v[frame]))

        # Update titles with time step
        ax_u.set_title(f"u Dynamics (t={0.02*frame:.2f})")
        ax_v.set_title(f"v Dynamics (t={0.02*frame:.2f})")
        return [im_u, im_v]

    # Create animation
    anim = FuncAnimation(fig, update, frames=t, interval=1000/fps, blit=True)

    # Save animation
    if save_as.endswith('.gif'):
        anim.save(save_as, writer='imagemagick', fps=fps)
    elif save_as.endswith('.avi'):
        anim.save(save_as, writer='ffmpeg', fps=fps, codec='mpeg4')
    else:
        raise ValueError("Unsupported file format. Use '.gif' or '.avi'.")
    
    plt.close(fig)
    print(f"Animation saved as {save_as}")

model = FNO_ETDRK4(modes1=8, modes2=8, width=40, N=128, device=device).to(device)
# load best_model.pth
model.load_state_dict(torch.load('best_model.pth', map_location=device))
for param in model.parameters():
    param.requires_grad = False

model.eval()

val_input = np.load('../dataset/data/train_fhn_inputs.npy')
val_target = np.load('../dataset/data/train_fhn_outputs.npy')
# val_dt     = np.load('../dataset/data/val_dt.npy')

idx = random.randint(0, val_input.shape[0])

input_data = val_input[idx:idx+1]
target_data = val_target[idx:idx+1]
# dt = val_dt[idx:idx+1]

input_data = torch.tensor(input_data, dtype=torch.float64).to(device)
target_data = torch.tensor(target_data, dtype=torch.float64).to(device)
dt = torch.tensor(1e-2, dtype=torch.float64).to(device) / 2

data = model(input_data, dt)
data = model(data, dt)
# visualize(data, save_as='output.gif', fps=10)

error = torch.abs(data - target_data)
# error = data

x = torch.linspace(0, 1, 128+1)[:-1]
y = torch.linspace(0, 1, 128+1)[:-1]
X, Y = torch.meshgrid(x, y)

# u = data[0,:,:,0] v = data[0,:,:,1]
# 两个subfigures 画图就行
fig, axes = plt.subplots(1,2, figsize=(10,5))
ax_u, ax_v = axes

im_u = ax_u.imshow(error[0,:,:,0].detach().cpu(), cmap='viridis', origin='lower', extent=[0,1,0,1])
im_v = ax_v.imshow(error[0,:,:,1].detach().cpu(), cmap='plasma', origin='lower', extent=[0,1,0,1])

ax_u.set_title('u dynamics')
ax_u.set_xlabel("x")
ax_u.set_ylabel("y")
ax_v.set_title("v Dynamics")
ax_v.set_xlabel("x")
ax_v.set_ylabel("y")

# Add colorbars
cbar_u = fig.colorbar(im_u, ax=ax_u)
cbar_v = fig.colorbar(im_v, ax=ax_v)

plt.savefig('test_error.png')