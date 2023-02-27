import os
import json
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions



import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from models.ray_utils import get_ray_directions


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    pts = pts[valid]
    center = pts.mean(0)
    return center, pts


def normalize_poses(poses, pts, estimate_ground=True):
    center, pts = get_center(pts)

    if estimate_ground:
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        pts = pts[signed_distance.abs() > 0.01] # remove ground points
        center = pts.mean(0) # estimate new scene center
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    else:
        # use the average camera pose as the up direction
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)

    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    Rc = torch.stack([x, y, z], dim=1)
    tc = center.reshape(3, 1)

    R, t = Rc.T, -Rc.T @ tc

    poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
    inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)

    poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)
    scale = poses_norm[...,3].norm(p=2, dim=-1).min()
    poses_norm[...,3] /= scale

    pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
    pts = pts / scale

    return poses_norm, pts


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w



class ColmapDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = _get_rank()

        camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'sparse/0/cameras.bin'))

        H = int(camdata[1].height)
        W = int(camdata[1].width)

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = int(W / self.config.img_downscale + 0.5), int(H // self.config.img_downscale + 0.5)
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        self.factor = w / W

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.factor
            cx = camdata[1].params[1] * self.factor
            cy = camdata[1].params[2] * self.factor
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.factor
            fy = camdata[1].params[1] * self.factor
            cx = camdata[1].params[2] * self.factor
            cy = camdata[1].params[3] * self.factor
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        
        self.directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)

        imdata = read_images_binary(os.path.join(self.config.root_dir, 'sparse/0/images.bin'))

        mask_dir = os.path.join(self.config.root_dir, 'masks')
        self.use_mask = os.path.exists(mask_dir) and self.config.use_mask
        
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, d in enumerate(imdata.values()):
            R = d.qvec2rotmat()
            t = d.tvec.reshape(3, 1)
            c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
            c2w[:,1:3] *= -1. # COLMAP => OpenGL
            self.all_c2w.append(c2w)
            if self.split in ['train', 'val']:
                img_path = os.path.join(self.config.root_dir, 'images', d.name)
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                if self.use_mask:
                    mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                    mask_paths = list(filter(os.path.exists, mask_paths))
                    assert len(mask_paths) == 1
                    mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                    mask = mask.resize(self.img_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask)[0]
                else:
                    mask = torch.ones_like(img[...,0])
                self.all_fg_masks.append(mask) # (h, w)
                self.all_images.append(img)
        
        self.all_c2w = torch.stack(self.all_c2w, dim=0)   

        pts3d = read_points3d_binary(os.path.join(self.config.root_dir, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()

        self.all_c2w, pts3d = normalize_poses(self.all_c2w, pts3d, estimate_ground=self.config.estimate_ground)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0), torch.stack(self.all_fg_masks, dim=0)  

        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w, self.all_images, self.all_fg_masks = \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_fg_masks.float().to(self.rank)
        

class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
