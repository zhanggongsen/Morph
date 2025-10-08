import os
import torch
import numpy as np
import random

from vae_utils import get_timepoint_dirs, read_dvf, save_dvf
from vae_model import VAE

# --- Config ---
root_dir = r"00000001"
model_loss_save_path = "./VAE_model_loss_save"
latent_dim = 256
sample_number = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1


epoch="400"

# --- Inference ---
timepoint_dirs = get_timepoint_dirs(root_dir)
for tp, folders in timepoint_dirs.items():
    model_path = os.path.join(model_loss_save_path, f"{tp}_epoch_{epoch}_VAE_model.pth")
    # if not os.path.exists(model_path) or len(folders) == 0:
    #     continue

    sample_array, ref_img = read_dvf(os.path.join(folders[0], "DVF.nii"))
    input_shape = (3, *sample_array.shape[:-1])  # [C, D, H, W]

    model = VAE(input_shape, latent_dim)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(sample_number):
            z = torch.randn(1, latent_dim).to(device)
            out = model.module.decode(z) if multi_gpu else model.decode(z)
            out = out.squeeze(0).cpu().numpy().transpose(1, 2, 3, 0)  # [D, H, W, C]

            new_folder = os.path.join(root_dir, f"{tp}_{random.randint(100000,999999)}_VAE")
            os.makedirs(new_folder, exist_ok=True)
            save_dvf(out, ref_img, os.path.join(new_folder, "DVF.nii"))
            print(f"[{tp}] Sample saved to {new_folder}/DVF.nii")
