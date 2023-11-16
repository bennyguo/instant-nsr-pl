import torch
import numpy as np
import cv2


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True, k1=0, k2=0, k3=0, k4=0):
    pixel_center = 0.5 if use_pixel_centers else 0
    if k1 == 0:
        i, j= np.meshgrid(
            np.arange(W, dtype=np.float32) + pixel_center,
            np.arange(H, dtype=np.float32) + pixel_center,
            indexing='xy'
        )
        i, j = torch.from_numpy(i), torch.from_numpy(j)
        directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)
    else:
        grid_ij = np.mgrid[0:W, 0:H].transpose(2, 1, 0).astype(np.float32) + pixel_center
        grid_ij = grid_ij.reshape(-1, 1, 2)
        K= np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        D = np.array([k1, k2, k3, k4])
        directions = cv2.fisheye.undistortPoints(grid_ij, K, D).reshape(H, W, 2)
        directions = torch.stack([torch.as_tensor(directions)[...,0], -torch.as_tensor(directions)[...,1], -torch.ones_like(torch.as_tensor(directions)[...,0])], -1)

        """
        r = torch.sqrt((i - cx)**2 + (j - cy)**2)
        theta = torch.atan(r)
        theta2 = theta ** 2
        theta4 = theta2 ** 2
        theta6 = theta4 * theta2
        thetad = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6)
        directions = torch.stack([(i - cx) * thetad / r / fx, -(j - cy) * thetad / r / fy, -torch.ones_like(i)], -1) # (H, W, 3)
        """

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
