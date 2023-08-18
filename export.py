import sys
import argparse
import os
import time
import logging
from datetime import datetime
import trimesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args, extras = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    code_dir = os.path.join(args.exp_dir, 'code')
    ckpt_dir = os.path.join(args.exp_dir, 'ckpt')
    latest_ckpt = sorted(os.listdir(ckpt_dir), key=lambda s: int(s.split('-')[0].split('=')[1]), reverse=True)[0]
    latest_ckpt = os.path.join(ckpt_dir, latest_ckpt)
    config_path = os.path.join(args.exp_dir, 'config', 'parsed.yaml')
    sys.path.append(code_dir)
    
    import datasets
    import systems
    import pytorch_lightning as pl
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)

    if 'seed' not in config:
        pl.seed_everything(config.seed)

    system = systems.make(config.system.name, config, load_from_checkpoint=latest_ckpt)
    system.model.cuda()
    mesh = system.model.isosurface()
    mesh = trimesh.Trimesh(
        vertices=v_pos.numpy(),
        faces=t_pos_idx.numpy()
    )
    mesh.export(os.path.join(args.output_dir, 'iso_mesh.ply'))

if __name__ == '__main__':
    main()