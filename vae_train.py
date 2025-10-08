import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter  # 使用 tensorboardX
from vae_utils import get_timepoint_dirs, DVFDataset
from vae_model import VAE
import random

# --- Config ---
root_dir = r"Augmented_4DCT\00000001"
num_epochs = 500
batch_size = 4
learning_rate = 1e-4
lr_scheduler_type = 'constant'  # 'constant', 'plateau', 'linear_decay'
latent_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1
viz_interval = 1

model_loss_save_path = "./VAE_model_loss_save"
os.makedirs(model_loss_save_path, exist_ok=True)
os.makedirs("runs", exist_ok=True)

# --- Loss functions ---
# def compute_jacobian_loss(dvf):
#     dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
#     dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
#     dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
#     return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

def compute_jacobian_loss(dvf,epsilon=0.01):
    """
    dvf: Tensor of shape [B, 3, D, H, W], representing displacement vector field
    """
    # Compute gradients (partial derivatives)
    dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]

    # Pad to match original shape
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

    # Compute Jacobian matrix components
    du_dx, du_dy, du_dz = dx[:, 0], dy[:, 0], dz[:, 0]
    dv_dx, dv_dy, dv_dz = dx[:, 1], dy[:, 1], dz[:, 1]
    dw_dx, dw_dy, dw_dz = dx[:, 2], dy[:, 2], dz[:, 2]

    # Construct Jacobian determinant
    jac_det = (
        du_dx * (dv_dy * dw_dz - dv_dz * dw_dy) -
        du_dy * (dv_dx * dw_dz - dv_dz * dw_dx) +
        du_dz * (dv_dx * dw_dy - dv_dy * dw_dx)
    )

    # Penalize non-positive determinants
    penalty = F.relu(epsilon - jac_det)

    return penalty.mean()

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.numel()
    jacobian_loss = compute_jacobian_loss(recon_x)
    smooth_loss = torch.mean(torch.abs(recon_x[:, :, 1:, :, :] - recon_x[:, :, :-1, :, :]))
    total_loss = recon_loss + 0.1* kl_loss + 0 * jacobian_loss + 0 * smooth_loss
    return total_loss, recon_loss, kl_loss, jacobian_loss, smooth_loss

# --- Training loop ---
timepoint_dirs = get_timepoint_dirs(root_dir)
for tp, folders in timepoint_dirs.items():
    dataset = DVFDataset(folders)
    if len(dataset) == 0:
        print(f"Skipping {tp}, no DVF data found.")
        continue

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample_shape = (3, *dataset[0].shape[:-1])  # [C, D, H, W]

    model = VAE(sample_shape, latent_dim)
    if multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if lr_scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    elif lr_scheduler_type == 'linear_decay':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs)
    else:
        scheduler = None

    loss_log_path = os.path.join(model_loss_save_path, f"{tp}_loss_log.txt")
    open(loss_log_path, 'w').close()

    # TensorBoard writer
    log_dir = os.path.join("runs", f"tp_{tp}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'jacobian': 0, 'smooth': 0}

        for x in dataloader:
            x = x.permute(0, 4, 1, 2, 3).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            total_loss, recon_loss, kl_loss, jac_loss, smooth_loss = loss_function(recon_x, x, mu, logvar)
            total_loss.backward()
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['jacobian'] += jac_loss.item()
            epoch_losses['smooth'] += smooth_loss.item()

        num_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        writer.add_scalar(f"Loss/Total", epoch_losses['total'], epoch)
        writer.add_scalar(f"Loss/Reconstruction", epoch_losses['recon'], epoch)
        writer.add_scalar(f"Loss/KL", epoch_losses['kl'], epoch)
        writer.add_scalar(f"Loss/Jacobian", epoch_losses['jacobian'], epoch)
        writer.add_scalar(f"Loss/Smoothness", epoch_losses['smooth'], epoch)
        writer.flush()

        with open(loss_log_path, 'a') as log_file:
            log_file.write(f"{epoch + 1}\t{epoch_losses['total']:.6f}\t{epoch_losses['recon']:.6f}\t"
                           f"{epoch_losses['kl']:.6f}\t{epoch_losses['jacobian']:.6f}\t{epoch_losses['smooth']:.6f}\n")

        print(f"[TP {tp}] Epoch {epoch + 1:03d} "
              f"Loss: {epoch_losses['total']:.4f}, Recon: {epoch_losses['recon']:.4f}, "
              f"KL: {epoch_losses['kl']:.4f}, JAC: {epoch_losses['jacobian']:.4f}, SMOOTH: {epoch_losses['smooth']:.4f}")



        if epoch % viz_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_idx = random.randint(0, x.shape[0] - 1)
                input_dvf = x[sample_idx].cpu().detach()  # [C, D, H, W]
                output_dvf = recon_x[sample_idx].cpu().detach()

                z_idx = random.randint(0, input_dvf.shape[1] - 1)

                for i, comp_name in enumerate(["dx", "dy", "dz"]):
                    input_slice = input_dvf[i, z_idx, :, :].numpy()
                    output_slice = output_dvf[i, z_idx, :, :].numpy()

                    def normalize(slice_):
                        min_val = slice_.min()
                        max_val = slice_.max()
                        if max_val > min_val:
                            return (slice_ - min_val) / (max_val - min_val)
                        else:
                            return slice_ * 0


                    input_norm = normalize(input_slice)
                    output_norm = normalize(output_slice)

                    input_img = torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0)
                    output_img = torch.from_numpy(output_norm).unsqueeze(0).unsqueeze(0)

                    writer.add_image(f"{tp}/DVF_Input/{comp_name}", input_img[0], epoch)
                    writer.add_image(f"{tp}/DVF_Recon/{comp_name}", output_img[0], epoch)

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(model_loss_save_path, f"{tp}_epoch_{epoch+1}_VAE_model.pth")
            torch.save(model.state_dict(), model_path)

        if scheduler:
            if lr_scheduler_type == 'plateau':
                scheduler.step(epoch_losses['total'])
            else:
                scheduler.step()

    final_model_path = os.path.join(model_loss_save_path, f"{tp}_VAE_final.pth")
    torch.save(model.state_dict(), final_model_path)

    writer.close()

# tensorboard --logdir=./Datapreprocess/9_VAE_Aug/runs