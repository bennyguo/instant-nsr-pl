# Instant Neural Surface Reconstruction

This repository contains a concise and extensible implementation of NeRF and NeuS for neural surface reconstruction based on Instant-NGP and the Pytorch-Lightning framework. **Training on a NeRF-Synthetic scene takes ~5min for NeRF and ~10min for NeuS on a single RTX3090.**

||NeRF in 5min|NeuS in 10 min|
|---|---|---|
|Rendering|![rendering-nerf](https://user-images.githubusercontent.com/19284678/199078178-b719676b-7e60-47f1-813b-c0b533f5480d.png)|![rendering-neus](https://user-images.githubusercontent.com/19284678/199078300-ebcf249d-b05e-431f-b035-da354705d8db.png)|
|Mesh|![mesh-nerf](https://user-images.githubusercontent.com/19284678/199078661-b5cd569a-c22b-4220-9c11-d5fd13a52fb8.png)|![mesh-neus](https://user-images.githubusercontent.com/19284678/199078481-164e36a6-6d55-45cc-aaf3-795a114e4a38.png)|


## Features
**This repository aims to provide a highly efficient while customizable boilerplate for research projects based on NeRF or NeuS.**

- acceleration techniques from [Instant-NGP](https://github.com/NVlabs/instant-ngp): multiresolution hash encoding and fully fused networks by [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), occupancy grid pruning and rendering by [nerfacc](https://github.com/KAIR-BAIR/nerfacc)
- out-of-the-box multi-GPU and mixed precision training by [PyTorch-Lightning](https://github.com/Lightning-AI/lightning)
- hierarchical project layout that is designed to be easily customized and extended, flexible experiment configuration by [OmegaConf](https://github.com/omry/omegaconf)

**Please subscribe to [#26](https://github.com/bennyguo/instant-nsr-pl/issues/26) for our latest findings on quality improvements!**


## Requirements
**Note:**
- To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.
- Multi-GPU training is currently not supported on Windows (see [#4](https://github.com/bennyguo/instant-nsr-pl/issues/4)).
### Environments
- Install PyTorch>=1.10 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)
- Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- `pip install -r requirements.txt`


## Run
### Training on NeRF-Synthetic
Download the NeRF-Synthetic data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it under `load/`. The file structure should be like `load/nerf_synthetic/lego`.

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scece name:
```bash
# train NeRF
python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=example

# train NeuS with mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
# train NeuS without mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example system.loss.lambda_mask=0.0
```
The code snapshots, checkpoints and experiment outputs are saved to `exp/[name]/[tag]@[timestamp]`, and tensorboard logs can be found at `runs/[name]/[tag]@[timestamp]`. You can change any configuration in the YAML file by specifying arguments without `--`, for exmaple:
```bash
python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=iter50k seed=0 trainer.max_steps=50000
```
### Training on Custom COLMAP Data
To get COLMAP data from custom images, you should have COLMAP installed (see [here](https://colmap.github.io/install.html) for installation instructions). Then put your images in the `images/` folder, and run `scripts/imgs2poses.py` specifying the path containing the `images/` folder. For example:
```bash
python scripts/imgs2poses.py ./load/bmvs_dog # images are in ./load/bmvs_dog/images
```
Existing data following this file structure also works as long as images are store in `images/` and there is a `sparse/` folder for the COLMAP output. An optional `masks/` folder could be provided for mask supervision. To train on COLMAP data, please refer to the example config files `config/*-colmap.yaml`. Some notes:
- Adapt the `root_dir` and `img_wh` option in the config file to your data;
- The scene is normalized so that cameras have an average distance `1.0` to the center of the scene, therefore `radius` is default to `0.5` in the config file. You should consider increase `radius` if the cameras are to close to the object.
- Background model is not yet implemented, so it works best on 360 captures with known foreground masks.

### Testing
The training precedure are by default followed by testing, which computes metrics on test data, generates animations and exports the geometry as triangular meshes. If you want to do testing alone, just resume the pretrained model and replace `--train` with `--test`, for example:
```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test
```


## Benchmarks
All experiments are conducted on a single NVIDIA RTX3090.

|PSNR|Chair|Drums|Ficus|Hotdog|Lego|Materials|Mic|Ship|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|NeRF Paper|33.00|25.01|30.13|36.18|32.54|29.62|32.91|28.65|31.01|
|NeRF Ours|34.80|26.04|33.89|37.42|35.33|29.46|35.22|31.17|32.92|
|NeuS Ours (with mask)|33.14|24.74|28.61|34.39|29.78|26.71|32.60|26.85|29.60|

|Training Time (mm:ss)|Chair|Drums|Ficus|Hotdog|Lego|Materials|Mic|Ship|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|NeRF Ours|04:34|04:35|04:18|04:46|04:39|04:35|04:26|05:41|04:42|
|NeuS Ours (with mask)|08:50|09:01|08:53|09:19|09:37|09:17|08:17|11:53|09:23|


## TODO
- [ ] Support more dataset formats, like ~COLMAP outputs~ and DTU
- [ ] Support background model based on NeRF++ or Mip-NeRF360
- [ ] Support GUI training and interaction
- [ ] More illustrations about the framework

## Related Projects
- [ngp_pl](https://github.com/kwea123/ngp_pl): Great Instant-NGP implementation in PyTorch-Lightning! Background model and GUI supported.
- [Instant-NSR](https://github.com/zhaofuq/Instant-NSR): NeuS implementation using multiresolution hash encoding.
