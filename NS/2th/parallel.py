import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from ANI import N0, ANIBASE, ResidualBlockWithT, MLP, FNO2d
from ANI_NS_2th import ODEPairDataset
from ANI_NS_2th import NavierStokes, A, total_loss  # 假设这些类已经按原来定义好了
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()

def broadcast_model_params(model):
    # Ensure model parameters are the same across all processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def plot_losses(train_losses, val_losses):
    if dist.get_rank() != 0:
        return
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot_small.png")
    plt.close()

def vorticity(u):
    # u: [B, 2, H, W]
    u_y = u[:, 0].roll(shifts=1, dims=1) - u[:, 0]  # du/dy，y 是 dim=1 of u[:, 0]
    v_x = u[:, 1].roll(shifts=1, dims=2) - u[:, 1]  # dv/dx，x 是 dim=2 of u[:, 1]
    return v_x - u_y


def vorticity_loss(pred, target):
    vort_pred = vorticity(pred)
    vort_target = vorticity(target)
    return F.mse_loss(vort_pred, vort_target)


def main_worker(rank, world_size):
    setup_ddp(rank, world_size)

    writer = None
    if rank == 0:
        writer = SummaryWriter()

    batch_size = 8
    data_dir = "../dataset"

    train_dataset = ODEPairDataset(
        os.path.join(data_dir, "train_input.pt"),
        os.path.join(data_dir, "train_output.pt"),
        limit=4000,
    )
    val_dataset = ODEPairDataset(
        os.path.join(data_dir, "val_input.pt"),
        os.path.join(data_dir, "val_output.pt"),
        limit=1000,
    )

    A_model = A(Nx=256, Ny=256, Lx=1.0, Ly=1.0)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)

    device = torch.device(f"cuda:{rank}")
    model = NavierStokes(N0_SCHEME=A(Nx=256, Ny=256, Lx=1.0, Ly=1.0, device=device), modes1=32, modes2=32, width=64, dt=0.01, device=device)
    model.dt_tensor = model.dt_tensor.to(device)
    model = DDP(model, device_ids=[rank])

    broadcast_model_params(model)

    criterion = torch.nn.MSELoss()
    epochs    = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    train_loss_lists = []
    val_loss_lists = []

    os.makedirs("logs", exist_ok=True)
    train_loss_file = open(os.path.join("logs", f"train_loss_rank{rank}.txt"), "w")
    val_loss_file = open(os.path.join("logs", f"val_loss_rank{rank}.txt"), "w")


    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # ✅ VERY IMPORTANT
            targets = targets.to(device)
            # inputs = inputs.permute(0, 3, 1, 2)
            # targets = targets.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            preds = model(inputs)
            # mse_loss = criterion(preds, targets)
            mse_loss = total_loss(preds, targets, A_model)
            loss = mse_loss 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_sampler)
        train_loss_lists.append(train_loss)
        train_loss_file.write(f"{train_loss:.5e}\n")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)   # ✅
                targets = targets.to(device)
                # inputs = inputs.permute(0, 3, 1, 2)
                # targets = targets.permute(0, 3, 1, 2)
                preds = model(inputs)
                val_loss += total_loss(preds, targets, A_model).item() * inputs.size(0) 
        val_loss /= len(val_sampler)
        val_loss_lists.append(val_loss)
        val_loss_file.write(f"{val_loss:.5e}\n")

        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("models", exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join("models", "best_model_fno.pth"))
                print("Saved best model!")

    if writer is not None:
        writer.close()

    train_loss_file.close()
    val_loss_file.close()
    plot_losses(train_loss_lists, val_loss_lists)
    cleanup_ddp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size)