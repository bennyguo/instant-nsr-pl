import torch
import numpy as np

# Code adapted from https://github.com/eldar/snes/blob/473ff2b1f6/3rdparty/co3d/dataset/co3d_dataset.py
def _get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]

def get_bbox_from_mask(mask, thr, decrease_quant=0.05):
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0

def get_clamp_bbox(bbox, box_crop_context=0.0, impath=""):
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.astype(np.float32)
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        warnings.warn(f"squashed image {impath}!!")
        return None

    # bbox[2:] = np.clip(bbox[2:], 2, )
    bbox[2:] = np.maximum(bbox[2:], 2)
    bbox[2:] += bbox[0:2] + 1  # convert to [xmin, ymin, xmax, ymax]
    # +1 because upper bound is not inclusive

    return bbox

def crop_around_box(tensor, bbox, impath=""):
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, tensor.shape[-2])
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, tensor.shape[-3])
    bbox = bbox.round().astype(np.longlong)
    return tensor[bbox[1] : bbox[3], bbox[0] : bbox[2], ...]
