import numpy as np
import os
import gzip
import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank
import cv2
from PIL import Image

import datasets
from datasets.colmap import create_spheric_poses, normalize_poses
from models.ray_utils import get_ray_directions
from datasets.utils import *
from omegaconf import OmegaConf

# Code adapted from https://github.com/POSTECH-CVLab/PeRFception/data_util/co3d.py
def similarity_from_cameras(c2w, fix_rot=False):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if fix_rot:
        R_align = np.eye(3)
        R = np.eye(3)
    else:
        R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale

class Co3dDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = _get_rank()

        self.has_mask = True
        self.apply_mask = self.config.apply_mask
        cam_scale_factor = self.config.cam_scale_factor

        cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
        scene_number = self.config.root_dir.split("/")[-1]
        json_path = os.path.join(self.config.root_dir, "..", "frame_annotations.jgz")
        with gzip.open(json_path, "r") as fp:
            all_frames_data = json.load(fp)

        frame_data, images, intrinsics, extrinsics, image_sizes = [], [], [], [], []
        masks = []

        for temporal_data in all_frames_data:
            if temporal_data["sequence_name"] == scene_number:
                frame_data.append(temporal_data)
            
        used = []
        self.directions = []
        self.all_fg_masks = []
        for (i, frame) in enumerate(frame_data):
            img = cv2.imread(os.path.join(self.config.root_dir, "..", "..", frame["image"]["path"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            H, W = frame["image"]["size"]
            max_hw = max(H, W)
            approx_scale = max(self.config.img_wh[0], self.config.img_wh[1]) / max_hw

            if approx_scale < 1.0:
                H2 = int(approx_scale * H)
                W2 = int(approx_scale * W)
                img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
            else:
                H2 = H
                W2 = W
            self.w, self.h = W2, H2

            image_size = np.array([H2, W2])
            fxy = np.array(frame["viewpoint"]["focal_length"])
            cxy = np.array(frame["viewpoint"]["principal_point"])
            R = np.array(frame["viewpoint"]["R"])
            T = np.array(frame["viewpoint"]["T"])

            if self.config.v2_mode:
                min_HW = min(W2, H2)
                image_size_half = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32) 
                scale_arr = np.array([min_HW * 0.5, min_HW * 0.5], dtype=np.float32)
                fxy_x = fxy * scale_arr
                prp_x = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32) - cxy * scale_arr
                cxy = (image_size_half - prp_x) / image_size_half 
                fxy = fxy_x / image_size_half

            scale_arr = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32) 
            focal = fxy * scale_arr
            prp = -1.0 * (cxy - 1.0) * scale_arr

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3:] = -R @ T[..., None]
            # original camera: x left, y up, z in (Pytorch3D)
            # transformed camera: x right, y down, z in (OpenCV)
            pose = pose @ cam_trans
            intrinsic = np.array(
                [
                    [focal[0], 0.0, prp[0], 0.0],
                    [0.0, focal[1], prp[1], 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            if any([np.all(pose == _pose) for _pose in extrinsics]):
                continue

            used.append(i)
            image_sizes.append(image_size)
            intrinsics.append(intrinsic)
            extrinsics.append(pose)
            images.append(img)
            self.directions.append(get_ray_directions(W, H, focal[0], focal[1], prp[0], prp[1]))

            if self.apply_mask:
                mask = np.array(Image.open(os.path.join(self.config.root_dir, "..", "..", frame["mask"]["path"])))
                mask = mask.astype(np.float32) / 255.0 # (h, w)
            else:
                mask = torch.ones_like(img[..., 0])
            self.all_fg_masks.append(mask)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        image_sizes = np.stack(image_sizes)
        self.directions = torch.stack(self.directions, dim=0)
        self.all_fg_masks = np.stack(self.all_fg_masks, 0)

        H_median, W_median = np.median(
            np.stack([image_size for image_size in image_sizes]), axis=0
        )

        H_inlier = np.abs(image_sizes[:, 0] - H_median) / H_median < 0.1
        W_inlier = np.abs(image_sizes[:, 1] - W_median) / W_median < 0.1
        inlier = np.logical_and(H_inlier, W_inlier)
        dists = np.linalg.norm(
            extrinsics[:, :3, 3] - np.median(extrinsics[:, :3, 3], axis=0), axis=-1
        )
        med = np.median(dists)
        good_mask = dists < (med * 5.0)
        inlier = np.logical_and(inlier, good_mask)

        if inlier.sum() != 0: 
            intrinsics = intrinsics[inlier]
            extrinsics = extrinsics[inlier]
            image_sizes = image_sizes[inlier]
            images = [images[i] for i in range(len(inlier)) if inlier[i]]
            self.directions = self.directions[inlier]
            self.all_fg_masks = self.all_fg_masks[inlier]

        extrinsics = np.stack(extrinsics)
        T, sscale = similarity_from_cameras(extrinsics)
        extrinsics = T @ extrinsics
        
        extrinsics[:, :3, 3] *= sscale * cam_scale_factor

        num_frames = len(extrinsics)

        if self.config.max_num_frames < num_frames:
            num_frames = self.config.max_num_frames
            extrinsics = extrinsics[:num_frames]
            intrinsics = intrinsics[:num_frames]
            image_sizes = image_sizes[:num_frames]
            images = images[:num_frames]
            self.directions = self.directions[:num_frames]
            self.all_fg_masks = self.all_fg_masks[:num_frames]
        
        if self.config.box_crop:
            print('cropping...')
            crop_masks = []
            crop_imgs = []
            crop_directions = []
            crop_xywhs = []
            max_sl = 0
            for i in range(num_frames):
                bbox_xywh = np.array(get_bbox_from_mask(self.all_fg_masks[i], self.config.box_crop_mask_thr))
                clamp_bbox_xywh = get_clamp_bbox(bbox_xywh, self.config.box_crop_context)
                max_sl = max(clamp_bbox_xywh[2] - clamp_bbox_xywh[0], max_sl)
                max_sl = max(clamp_bbox_xywh[3] - clamp_bbox_xywh[1], max_sl)
                mask = crop_around_box(self.all_fg_masks[i][..., None], clamp_bbox_xywh)
                img = crop_around_box(images[i], clamp_bbox_xywh)
                crop_masks.append(mask)
                crop_imgs.append(img)
                crop_xywhs.append(clamp_bbox_xywh)
            # pad all images to the same shape
            for i in range(num_frames):
                uh = (max_sl - crop_imgs[i].shape[0]) // 2 # h
                dh = max_sl - crop_imgs[i].shape[0] - uh
                lw = (max_sl - crop_imgs[i].shape[1]) // 2
                rw = max_sl - crop_imgs[i].shape[1] - lw
                crop_masks[i] = np.pad(crop_masks[i], pad_width=((uh, dh), (lw, rw), (0, 0)), mode='constant', constant_values=0.)
                crop_imgs[i] = np.pad(crop_imgs[i], pad_width=((uh, dh), (lw, rw), (0, 0)), mode='constant', constant_values=1.)
                fx, fy, cx, cy = intrinsics[i][0, 0], intrinsics[i][1, 1], intrinsics[i][0, 2], intrinsics[i][1, 2]
                crop_directions.append(get_ray_directions(max_sl, max_sl, fx, fy, cx - crop_xywhs[i][0] + lw, cy - crop_xywhs[i][1] + uh))
            self.w, self.h = max_sl, max_sl
            images = crop_imgs
            self.all_fg_masks = np.stack(crop_masks, 0)
            self.directions = torch.from_numpy(np.stack(crop_directions, 0))

        
        self.all_c2w = torch.from_numpy((extrinsics @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))[None, ...])[..., :3, :4])
        self.all_images = torch.from_numpy(np.stack(images, axis=0))
        self.img_wh = (self.w, self.h)
        
        # self.all_c2w = []
        # self.all_images = []
        # for i in range(num_frames):
        #     # convert to: x right, y up, z back (OpenGL)
        #     c2w = torch.from_numpy(extrinsics[i] @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32)))[:3, :4]
        #     self.all_c2w.append(c2w)
        #     img = torch.from_numpy(images[i])
        #     self.all_images.append(img)

        # TODO: save data for fast loading next time
        if self.config.load_preprocessed and os.path.exists(self.config.root_dir, 'nerf_preprocessed.npy'):
            pass

        i_all = np.arange(num_frames)
        i_test = i_all[::10]
        i_val = i_test
        i_train = np.array([i for i in i_all if not i in i_test])
        i_split = {
            'train': i_train,
            'val': i_val,
            'test': i_all
        }

        self.all_images, self.all_c2w = self.all_images[i_split[self.split]], self.all_c2w[i_split[self.split]]
        self.directions = self.directions[i_split[self.split]].float().to(self.rank)
        self.all_fg_masks = torch.from_numpy(self.all_fg_masks)[i_split[self.split]].float().to(self.rank)

        near, far = 0., 1.
        ndc_coeffs = (-1., -1.)

        self.directions = self.directions.float().to(self.rank)
        self.all_c2w, self.all_images, self.all_fg_masks = \
                self.all_c2w.float().to(self.rank), \
                self.all_images.float().to(self.rank), \
                self.all_fg_masks.float().to(self.rank)

        

class Co3dDataset(Dataset, Co3dDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class Co3dIterableDataset(IterableDataset, Co3dDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('co3d')
class Co3dDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = Co3dIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = Co3dDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = Co3dDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = Co3dDataset(self.config, self.config.train_split)

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