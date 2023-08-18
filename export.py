import sys
import argparse
import os
import time
import logging
from datetime import datetime
import trimesh

logging.basicConfig(level=logging.INFO)

def decimate_mesh(mesh: str, decimation_factor: float):
    logging.info(f"Original mesh with {len(mesh.faces)} faces.")

    # Decimate the mesh
    if decimation_factor < 1:
        decimation_factor = int(len(mesh.faces) * decimation_factor)
    else:
        decimation_factor = int(decimation_factor)

    mesh = mesh.simplify_quadratic_decimation(decimation_factor)
    logging.info(f"Decimated mesh to {len(mesh.faces)} faces.")

    return mesh
    
def main():
    logging.info("Start exporting.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--output-dir', required=True)
    
    parser.add_argument('--decimate', type=float, help='Specifies the desired final size of the mesh. \
                        If the number is less than 1, it represents the final size as a percentage of the initial size. \
                        If the number is greater than 1, it represents the desired number of faces.')
    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    code_dir = os.path.join(args.exp_dir, 'code')
    ckpt_dir = os.path.join(args.exp_dir, 'ckpt')
    latest_ckpt = sorted(os.listdir(ckpt_dir), key=lambda s: int(s.split('-')[0].split('=')[1]), reverse=True)[0]
    latest_ckpt = os.path.join(ckpt_dir, latest_ckpt)
    config_path = os.path.join(args.exp_dir, 'config', 'parsed.yaml')
    
    logging.info(f"Importing modules from cached code: {code_dir}")
    sys.path.append(code_dir)
    import datasets
    import systems
    import pytorch_lightning as pl
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    logging.info(f"Loading configuration: {config_path}")
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)

    if 'seed' not in config:
        pl.seed_everything(config.seed)

    logging.info(f"Creating system: {config.system.name}")
    system = systems.make(config.system.name, config, load_from_checkpoint=latest_ckpt)
    system.model.cuda()
    mesh = system.model.isosurface()
    mesh = trimesh.Trimesh(
        vertices=mesh['v_pos'].numpy(),
        faces=mesh['t_pos_idx'].numpy()
    )
    
    if args.decimate > 0:
        logging.info("Decimating mesh.")
        mesh = decimate_mesh(mesh, args.decimate)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Exporting mesh.")
    mesh.export(os.path.join(args.output_dir, 'iso_mesh.ply'))
    logging.info("Export finished successfully.")
    
if __name__ == '__main__':
    main()