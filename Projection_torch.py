import SimpleITK as sitk
import torch
import torch.nn.functional as F
import math

def cone_beam_projection(volume, spacing, SAD, SDD,
                         det_rows, det_cols, det_height, det_width,
                         num_samples=2000, chunk_size=1024,
                         angle_deg=0.0):

    volume = volume / 1000
    device = volume.device
    D, H, W = volume.shape[1:]
    sx, sy, sz = spacing

    angle = math.radians(angle_deg)

    det_pixel_u = det_width / det_cols
    det_pixel_v = det_height / det_rows

    u = (torch.arange(det_cols, device=device) - det_cols/2 + 0.5) * det_pixel_u
    v = (torch.arange(det_rows, device=device) - det_rows/2 + 0.5) * det_pixel_v
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    vol_center = torch.tensor([W*sx/2, H*sy/2, D*sz/2], device=device)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = torch.tensor([[cos_a, -sin_a, 0.0],
                      [sin_a,  cos_a, 0.0],
                      [0.0,    0.0,   1.0]], device=device)

    source_pos0 = torch.tensor([0.0, -SAD, 0.0], device=device)
    det_pos0 = torch.stack([
        uu.flatten(),
        torch.full((det_rows*det_cols,), SDD - SAD, device=device),
        vv.flatten()
    ], dim=-1)

    source_pos = vol_center + (R @ source_pos0)
    det_pos = vol_center + (det_pos0 @ R.T)

    ray_dir = det_pos - source_pos
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

    t_vals = torch.linspace(0, 2*SAD, num_samples, device=device)
    proj = torch.zeros((det_rows * det_cols,), device=device)
    vol = volume.unsqueeze(0)  # [1, 1, D, H, W]

    N_rays = det_rows * det_cols
    for start in range(0, N_rays, chunk_size):
        end = min(start + chunk_size, N_rays)
        dir_chunk = ray_dir[start:end]

        ray_points = source_pos[None, None, :] + t_vals[:, None, None] * dir_chunk[None, :, :]

        x_idx = ray_points[..., 0] / sx
        y_idx = ray_points[..., 1] / sy
        z_idx = ray_points[..., 2] / sz

        x_norm = (x_idx / (W-1)) * 2 - 1
        y_norm = (y_idx / (H-1)) * 2 - 1
        z_norm = (z_idx / (D-1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(3)

        mu_vals = F.grid_sample(vol, grid, align_corners=True,
                                mode='bilinear', padding_mode='zeros')
        mu_vals = mu_vals.squeeze()

        dl = torch.norm(dir_chunk, dim=-1) * (t_vals[1]-t_vals[0])
        proj[start:end] = torch.sum(mu_vals, dim=0) * dl

    proj = proj.view(det_rows, det_cols)
    return proj



def projection_mse_loss(volume, spacing, xray_gt,
                        SAD=2700.0, SDD=3700.0,
                        det_rows=768, det_cols=1024,
                        det_height=298.0, det_width=397.0, num_samples=2000):

    proj_pred = cone_beam_projection(volume, spacing, SAD, SDD,
                                     det_rows, det_cols,
                                     det_height, det_width,
                                     num_samples)
    loss = torch.mean((proj_pred - xray_gt)**2)
    return loss


ct_file = r"CT.nii"           # 3D CT
xray_file =r"xray.nii"       # 2D proj

image = sitk.ReadImage(ct_file)
ct_np = sitk.GetArrayFromImage(image)  # z,y,x
spacing = image.GetSpacing()           # (x, y, z)
spacing = (spacing[0], spacing[1], spacing[2])  #  tuple
print(spacing)
#  torch tensor [1, D, H, W]
volume = torch.from_numpy(ct_np).float().unsqueeze(0).cuda()

# #  1/m -> 1/mm
# volume /= 1000.0

xray_img = sitk.ReadImage(xray_file)
xray_np = sitk.GetArrayFromImage(xray_img)
xray_gt = torch.from_numpy(xray_np).float().cuda()

loss = projection_mse_loss(volume, spacing, xray_gt)
print("Testing MSE:", loss.item())

proj_pred = cone_beam_projection(volume, spacing,
                                 SAD=2700.0, SDD=3700.0,
                                 det_rows=768, det_cols=1024,
                                 det_height=298.0, det_width=397.0,
                                 num_samples=1000)

proj_pred = proj_pred.flip(0).flip(1)
proj_pred_np = proj_pred.detach().cpu().numpy()
proj_img = sitk.GetImageFromArray(proj_pred_np)
sitk.WriteImage(proj_img, r"pred_projection.nii")
