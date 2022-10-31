import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(H, W, focal, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)
    directions = torch.stack([(i - W/2) / focal, -(j - H/2) / focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    if directions.ndim == 3:
        assert c2w.ndim == 2
        rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
    elif directions.ndim == 2:
        if c2w.ndim == 3:
            assert c2w.shape[0] == directions.shape[0]
            rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        elif c2w.ndim == 2:
            rays_d = (directions[:,None,:] * c2w[None,:3,:3]).sum(-1) # (N_rays, 3)
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    
    # The origin of all rays is the camera origin in world coordinate
    if c2w.ndim == 2:
        rays_o = c2w[:,3].expand(rays_d.shape) # (H, W, 3)
    elif c2w.ndim == 3:
        rays_o = c2w[:,:,3]

    if not keepdim:
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
