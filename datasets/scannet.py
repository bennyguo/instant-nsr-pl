import os
import cv2
import json
import math
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from models.ray_utils import get_ray_directions


class ScannetDatasetBase():
    def setup(self, config, split:str):
        self.config = config
        self.split = split

        self.root_path = Path(self.config.root_dir)
        assert self.root_path.exists(), "Data directory is empty"

        image_paths = sorted(self.root_path.glob("*_rgb.png"))
        depth_paths = sorted(self.root_path.glob("*_depth.npy"))
        normal_paths = sorted(self.root_path.glob("*_normal.npy"))

        self.n_images = len(image_paths)

        # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
        # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
        # https://github.com/autonomousvision/monosdf/blob/main/preprocess/scannet_to_monosdf.py
        # TLDR
        # normal/depth needs 384x384, modify images/intrisics to match this

        cam_file = self.root_path/'cameras.npz'
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(None, P)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            if self.fx is None:
                self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        w, h = self.config.img_wh
        
        self.w, self.h = w, h

        self.rank = _get_rank()
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.h, self.w, self.fx, self.fy, self.cx, self.cy, self.config.use_pixel_centers).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, img_path in enumerate(image_paths):
            self.all_c2w.append(self.pose_all[i][:3,:4])

            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.config.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (3, h, w) => (h, w, 3)
            self.all_fg_masks.append(torch.ones_like(img[...,0])) # (h, w)
            self.all_images.append(img)

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)


    def load_K_Rt_from_P(self, filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K/K[2,2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]

        return intrinsics, pose
        

class ScannetDataset(Dataset, ScannetDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ScannetIterableDataset(IterableDataset, ScannetDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('scannet')
class ScannetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        # FIXME: the split is unused
        if stage in [None, 'fit']:
            self.train_dataset = ScannetIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ScannetDataset(self.config, 'val')
        if stage in [None, 'test']:
            self.test_dataset = ScannetDataset(self.config, 'test')
        if stage in [None, 'predict']:
            self.predict_dataset = ScannetDataset(self.config, 'train')

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
